import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x


class GCN_adv(nn.Module):

    def __init__(self, nfeat, args):
        super(GCN_adv, self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.estimator = GCN(nfeat, args.hidden, 1, dropout)
        self.GCN = GCN_Body(nfeat,args.num_hidden,args.dropout)
        self.classifier = nn.Linear(nhid, 1)
        self.adv = nn.Linear(nhid, 1)

        # G_params = list(self.GNN.parameters()) + list(self.classifier.parameters()) + list(self.estimator.parameters())
        # self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
        # self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        self.args = args
        # self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0
        self.A_loss = 0

    def forward(self, x, edge_index):
        z = self.GCN(x, edge_index)
        y = self.classifier(z)
        return y, z

    def optimize(self, g, x, labels, idx_train, sens, idx_sens_train):
        self.train()

        ### update E, G
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        #s = self.estimator(g, x)
        h = self.GNN(g, x)
        y = self.classifier(h)

        s_g = self.adv(h)

        #s_score = torch.sigmoid(s.detach())
        # s_score = (s_score > 0.5).float()
       # s_score[idx_sens_train] = sens[idx_sens_train].unsqueeze(1).float()
        #y_score = torch.sigmoid(y)
        #self.cov = torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))

        self.cls_loss = self.criterion(y[idx_train], labels[idx_train].unsqueeze(1).float())
        self.adv_loss = self.criterion(s_g[idx_sens_train], sens[idx_sens_train].unsqueeze(1).float())

        self.G_loss = (1-self.args.beta) * self.cls_loss - self.args.beta * self.adv_loss
        self.G_loss.backward()
        self.optimizer_G.step()

        ## update Adv
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        self.A_loss = self.criterion(s_g[idx_sens_train],sens[idx_sens_train].unsqueeze(1).float())
        self.A_loss.backward()
        self.optimizer_A.step()
