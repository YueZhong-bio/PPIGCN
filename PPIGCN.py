import csv
import re
import argparse
import networkx as nx
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import SAGEConv
import secrets
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy import sparse

import random

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feat,
                 num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=F.relu))
        # hidden layers
        for i in range(num_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(n_hidden, out_feat, activation=None))
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, x):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(g, x)
        return x

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(512, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for layer in self.lins:
            layer.reset_parameters()

    def forward(self, x_i, x_j):
        x = torch.cat((x_i,x_j),1)
        # x = x_i*x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

def train(model, predictor, g, x, x2,pos_train_edge,neg_train_edge, optimizer, batch_size, device):
    model.train()
    predictor.train()


    total_loss = total_examples = 0
    for perm in DataLoader(
            range(pos_train_edge.size(0)), batch_size, shuffle=True):

        optimizer.zero_grad()

        h = model(g, x)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]],h[edge[1]])
        # pos_out = predictor(torch.cat((h[edge[0]],x2[edge[0]]),1),torch.cat((h[edge[1]],x2[edge[1]]),1))
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_out, torch.ones(edge.size(1), 1).to(device))

        # Just do some trivial random sampling.
        # edge = torch.randint(
        #     0, x.size(0), torch.Size([2,edge.size(1)]), dtype=torch.long, device = x.device)
        edge = neg_train_edge[perm].t()
        neg_out = predictor(h[edge[0]],h[edge[1]])
        # neg_out = predictor(torch.cat((h[edge[0]],x2[edge[0]]),1),torch.cat((h[edge[1]],x2[edge[1]]),1))
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_out, torch.zeros(edge.size(1), 1).to(device))

        loss = pos_loss + neg_loss
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test(model, predictor, g, x,x2,test_edges,test_neg_edges, batch_size, device):
    model.eval()

    h = model(g, x)

    # Positive test edges
    pos_test_preds = []
    for perm in DataLoader(range(test_edges.size(0)), batch_size=batch_size):
        edge = test_edges[perm].t()
        pos_test_preds += predictor(h[edge[0]], h[edge[1]])
        # pos_test_preds += predictor(torch.cat((h[edge[0]],x2[edge[0]]),1),torch.cat((h[edge[1]],x2[edge[1]]),1))
    pos_test_preds = torch.cat(pos_test_preds, dim=0)
    pos_test_loss = F.binary_cross_entropy_with_logits(
            pos_test_preds, torch.ones(pos_test_preds.size(0)).to(device))

    # Negative test edges
    neg_test_preds = []
    for perm in DataLoader(range(test_neg_edges.size(0)), batch_size=batch_size):
        edge = test_neg_edges[perm].t()
        neg_test_preds += predictor(h[edge[0]], h[edge[1]])
        # neg_test_preds += predictor(torch.cat((h[edge[0]],x2[edge[0]]),1),torch.cat((h[edge[1]],x2[edge[1]]),1))
    neg_test_preds = torch.cat(neg_test_preds, dim=0)
    neg_test_loss = F.binary_cross_entropy_with_logits(
        neg_test_preds, torch.zeros(neg_test_preds.size(0)).to(device))

    test_loss = pos_test_loss + neg_test_loss
    test_preds = torch.cat((pos_test_preds,neg_test_preds),dim=0)

    y_preds = np.where((test_preds.cpu()).numpy()>0.5, 1, 0)
    y_score = np.concatenate((np.ones(pos_test_preds.shape),np.zeros(neg_test_preds.shape)),axis=0)

    recall = recall_score(y_score, y_preds)
    precision = precision_score(y_score, y_preds)
    f1 = 2 * (recall * precision) / (recall + precision)
    acc = accuracy_score(y_score, y_preds)
    auc = metrics.roc_auc_score(y_score,y_preds)

    return test_loss,acc,recall,precision,auc,f1

def load_ppi(fname='./protein.actions.tsv'):
    fin = open(fname)
    print ('Reading: %s' % fname)
    fin.readline()
    edges= []
    neg_edges = []
    pos_edges = []
    for line in fin:
        gene_id1, gene_id2, label= line.strip().split()
        edges +=[[gene_id1,gene_id2]]
        if label =="0":
            neg_edges += [[gene_id1,gene_id2]]
        else:
            pos_edges += [[gene_id1,gene_id2]]
    nodes = set([u for e in edges for u in e])
    print ('Edges: %d' % len(edges))
    print ('Nodes: %d' % len(nodes))

    net = nx.Graph()
    net.add_nodes_from(nodes)
    net.add_edges_from(pos_edges)
    # net.remove_nodes_from(nx.isolates(net))
    # net.remove_edges_from(net.selfloop_edges())

    node2idx = {node: i for i, node in enumerate(net.nodes())}

    a = []
    b = []
    for e in pos_edges[0:4500]:
        a.append(node2idx.get(e[0]))
        b.append(node2idx.get(e[1]))
    g = dgl.graph((a+b,b+a))
    g = dgl.add_self_loop(g)

    nx_g = dgl.to_networkx(g)
    g_adj = sparse.csr_matrix(nx.adjacency_matrix(nx_g))

    x3 = g_adj.dot(g_adj.dot(g_adj))
    g3 = dgl.graph((sparse.find(x3)[0],sparse.find(x3)[1]))

    a = []
    for e in edges:
        a.append([node2idx.get(e[0]),node2idx.get(e[1])])
    e = torch.LongTensor(a)

    #syn_encoder
    d = open('./protein.dictionary.tsv')
    d.readline()
    dic = {}
    for line in d:
        i, s = line.strip().split()
        dic[i] = s

    vocab1 = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
              'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
              '*': 100}

    x1 = np.zeros((2497,1))
    m = 0
    for i in node2idx.keys():
        x = [[vocab1[word] for word in sentence] for sentence in dic.get(i)]
        x = np.array(x).reshape(1,len(x))
        x1[m] = np.sum(x)
        m += 1
    x1 = np.array(x1)
    sss = pd.read_csv('yeastseq.csv',dtype=float)
    sss = np.array(sss)
    print(sss.shape)
    a = np.load('ye.txt.npy')
    return net,node2idx,len(nodes),g3,e,x1,sss,a


def main():
    parser = argparse.ArgumentParser(description='OGBL-PPA (Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    g_nx,node,num_nodes,g,e,qsx,ur,dw= load_ppi()


    x2 = torch.Tensor(dw).to(device)
    x = torch.Tensor(np.concatenate((ur,qsx),axis=1)).to(device)
    print (x.shape)

    g.ndata['feat'] = x
    print(g)

    random.shuffle(e[0:5594,:])
    random.shuffle(e[5594:-1,:])
    train_pos_edges = e[0:4500,:]
    test_pos_edges = e[4500:5594,:]

    train_neg_edges = e[5594:10094,:]
    test_neg_edges = e[10094:-1,:]

    model = GCN(
        x.size(-1),args.hidden_channels, args.hidden_channels,
        args.num_layers, args.dropout).to(device)
    # model = SAGE(
    #     x.size(-1), args.hidden_channels, args.hidden_channels,
    #     args.num_layers, args.dropout).to(device)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    model.reset_parameters()
    predictor.reset_parameters()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr,weight_decay=0.0001)

    for epoch in range(1, 1 + args.epochs):
        model.train()
        predictor.train()
        loss = train(model, predictor, g, x,x2, train_pos_edges,train_neg_edges, optimizer,
                     args.batch_size, device)
        print (f'Loss: {loss:.4f}')
        if epoch % args.eval_steps == 0:
            test_loss,acc,recall,Precision,AUC,f1  = test(model, predictor, g, x, x2,test_pos_edges,test_neg_edges,
                            args.batch_size, device)
            print(
                  f'Epoch: {epoch:02d}, '
                  f'Train_Loss: {loss:.4f}, '
                  f'Test_loss: {test_loss:.4f}, '
                  f'Test_auc: {AUC:.4f}%, '
                  f'Test_recall: {100 * recall:.2f}% '
                  f'Test_precision: {100 * Precision:.2f}% '
                  f'Test_acc: {100 * acc:.2f}% '
                  f'Test_f1: {100 * f1:.2f}% '
            )
if __name__ == "__main__":
    main()
