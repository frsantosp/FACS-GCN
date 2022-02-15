
from argparse import ArgumentParser
import torch
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import math
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics import confusion_matrix

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset_str', default='cora', type=str)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=100)

    args = parser.parse_args()

    print('---------------------------------------------')
    print(f'-----------Seed: {args.seed}----------------------')
    path = 'data/'
    g, features, labels, gender, train_mask, val_mask, test_mask = load_data_dgl(path,dataset=args.dataset_str, seed=args.seed)

    model = SAGE(in_feats=features.size()[1], hid_feats=32, out_feats=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total = 0
    best_epoch_loss = 0

    best_auc = -1
    #wei = sample_weight.div_(torch.sum(sample_weight))
    # [[idx_train], [idx_val], [idx_test], [labels]]
    running_loss = 0
    for epoch in range(args.epoch):

        model.train()
        optimizer.zero_grad()
        output = model(g,features) # logits

        preds = output.max(1)[1].type_as(labels)

        loss = F.cross_entropy(output[train_mask], labels[train_mask])

        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()

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
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])

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

        if auc_roc_test > best_auc:
            best_auc = auc_roc_test
            best_epoch = epoch
            torch.save(model.state_dict(), f'weights.pt')

    # Update sample weight
    model.load_state_dict(torch.load(f'weights.pt'))
    model.eval()
    output = model(g, features)

    acc_test = acc_measurements(output[test_mask], labels[test_mask],gender[test_mask])
    auc_roc_test, auc_m, auc_f = auc_measurements(output[test_mask], labels[test_mask],
                                                  gender[test_mask])

    parity, equality = fair_metric(preds[test_mask].numpy(), labels[test_mask].numpy(),
                                   gender[test_mask].numpy())

    print('-----------------------------------------------------')
    lt = [acc_test[0],auc_roc_test,auc_m, auc_f, parity, equality]
    print(lt)
    print('-----------------------------------------------------')
