
from argparse import ArgumentParser
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics import confusion_matrix

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=True)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # XW
        output = torch.spmm(adj, support)  # AXW
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset_str', default='cora', type=str)

    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=.5)
    #parser.add_argument('--l2_reg', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()

    print('---------------------------------------------')
    print(f'-----------Seed: {args.seed}----------------------')
    path = 'data/'
    adj, features, labels, gender, train_mask, val_mask, test_mask = load_data(path=path,dataset='bail', seed=args.seed)

    model = GCN(features.shape[1], args.hidden_dim, 2, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total = 0
    best_epoch_loss = 0

    epoch_list = []
    loss_list = []
    loss_list_t = []
    loss_list_v = []

    acc_list = []
    acc_list_t = []
    acc_list_v = []

    auc_list = []
    auc_list_t = []
    auc_list_v = []

    auc_f_list = []
    auc_f_list_t = []
    auc_f_list_v = []

    auc_m_list = []
    auc_m_list_t = []
    auc_m_list_v = []

    f1_list = []
    f1_list_t = []
    f1_list_v = []

    sp_list = []
    sp_list_t = []
    sp_list_v = []

    eq_list = []
    eq_list_t = []
    eq_list_v = []

    best_loss = 10000000
    running_loss = 0
    for epoch in range(args.epoch):

        model.train()
        optimizer.zero_grad()
        output = model(features,adj) # logits

        preds = output.max(1)[1].type_as(labels)

        loss =  F.nll_loss(output[train_mask], labels[train_mask])

        loss.backward(retain_graph=True)
        optimizer.step()

        model.eval()
        acc_train = acc_measurements(output[train_mask], labels[train_mask],gender[train_mask])
        auc_roc_train, auc_m, auc_f = auc_measurements(output[train_mask], labels[train_mask],
                                                       gender[train_mask])

        parity, equality = fair_metric(preds[train_mask].numpy(), labels[train_mask].numpy(),
                                       gender[train_mask].numpy())


        if epoch%100== 0:
            print(f'------------epoch {epoch}------')
            print(f'|*|Training: acc : {acc_train[0]} || Auc layer: {auc_roc_train} || AUC Male: {auc_m} || AUC Female: {auc_f} || SP: {parity} || EQ: {equality}|*|')

        acc_test = acc_measurements(output[test_mask], labels[test_mask], gender[test_mask])
        auc_roc_test, auc_m, auc_f = auc_measurements(output[test_mask], labels[test_mask],
                                                      gender[test_mask])

        parity, equality = fair_metric(preds[test_mask].numpy(), labels[test_mask].numpy(),
                                       gender[test_mask].numpy())


        if epoch % 100 == 0:
            print(f'|*|Test: acc layer: {acc_test[0]} || Auc layer : {auc_roc_test} || AUC Male: {auc_m} || AUC Female: {auc_f} || SP: {parity} || EQ: {equality}|*|')
            print(confusion_matrix(labels[test_mask].numpy(), preds[test_mask].numpy()))

        acc_test = acc_measurements(output[val_mask], labels[val_mask], gender[val_mask])
        auc_roc_test, auc_m, auc_f = auc_measurements(output[val_mask], labels[val_mask],
                                                      gender[val_mask])

        parity, equality = fair_metric(preds[val_mask].numpy(), labels[val_mask].numpy(),
                                       gender[val_mask].numpy())
        loss_val = F.nll_loss(output[val_mask], labels[val_mask])

        if epoch % 100 == 0:
            print(f'|*|Validation: acc: {acc_test[0]} || Auc : {auc_roc_test} || AUC Male: {auc_m} || AUC Female: {auc_f} || SP: {parity} || EQ: {equality}|*|')
            print(confusion_matrix(labels[val_mask].numpy(), preds[val_mask].numpy()))

        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            best_epoch = epoch
            torch.save(model.state_dict(), f'weights.pt')


    # Update sample weight
    model.load_state_dict(torch.load(f'weights.pt'))
    model.eval()
    output = model(features,adj)

    acc_test = acc_measurements(output[test_mask], labels[test_mask],gender[test_mask])
    auc_roc_test, auc_m, auc_f = auc_measurements(output[test_mask], labels[test_mask],
                                                  gender[test_mask])

    parity, equality = fair_metric(preds[test_mask].numpy(), labels[test_mask].numpy(),
                                   gender[test_mask].numpy())
    print('-----------------------------------------------------')
    lt = [acc_test[0], auc_roc_test, auc_m, auc_f, parity, equality]
    print(lt)
    print('-----------------------------------------------------')
