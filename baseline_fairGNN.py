import dgl
import time
import tqdm
import ipdb
import argparse
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from numpy import argmax
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import dropout_adj, convert

import networkx as nx
from fairgnn_nifty import *
from utils_fairgnn_nifty.fairgnn_utils import *
from aif360.sklearn.metrics import consistency_score as cs
from sklearn.metrics import confusion_matrix,roc_curve,plot_roc_curve

def train(model, x, edge_index, labels, idx_train, sens, idx_sens_train):
    model.train()

    train_g_loss = 0
    train_a_loss = 0

    ### update E, G
    model.adv.requires_grad_(False)
    optimizer_G.zero_grad()

    s = model.estimator(x, edge_index)
    h = model.GNN(x, edge_index)
    y = model.classifier(h)

    s_g = model.adv(h)
    s_score_sigmoid = torch.sigmoid(s.detach())
    s_score = s.detach()
    s_score[idx_train]=sens[idx_train].unsqueeze(1).float()
    y_score = torch.sigmoid(y)
    cov =  torch.abs(torch.mean((s_score_sigmoid[idx_train] - torch.mean(s_score_sigmoid[idx_train])) * (y_score[idx_train] - torch.mean(y_score[idx_train]))))
    
    cls_loss = criterion(y[idx_train], labels[idx_train].unsqueeze(1).float())
    adv_loss = criterion(s_g[idx_train], s_score[idx_train])
    G_loss = cls_loss  + args.alpha * cov - args.beta * adv_loss
    G_loss.backward()
    optimizer_G.step()

    ## update Adv
    model.adv.requires_grad_(True)
    optimizer_A.zero_grad()
    s_g = model.adv(h.detach())
    A_loss = criterion(s_g[idx_train], s_score[idx_train])
    A_loss.backward()
    optimizer_A.step()
    return G_loss.detach(), A_loss.detach()


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


def f1_sens(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    f1_fem = f1_score(labels[idx_s0],pred[idx_s0])
    f1_male = f1_score(labels[idx_s1],pred[idx_s1])
    return f1_fem,f1_male

def auc_sens(labels, pred, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    auc_f = roc_auc_score(labels[idx_s0], pred[idx_s0])
    auc_m = roc_auc_score(labels[idx_s1], pred[idx_s1])
    return auc_f,auc_m


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units of the sensitive attribute estimator')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=.01,
                        help='The hyperparameter of alpha') #4
    parser.add_argument('--beta', type=float, default=5,
                        help='The hyperparameter of beta') #.5
    parser.add_argument('--model', type=str, default="GAT",
                        help='the type of model GCN/GAT')
    parser.add_argument('--dataset', type=str, default='pokec_n',
                        choices=['pokec_z','pokec_n','nba', 'german', 'bail', 'credit','facebook','tagged','tagged_2','region_job'])
    parser.add_argument('--num-hidden', type=int, default=32,
                        help='Number of hidden units of classifier.')
    parser.add_argument("--num-heads", type=int, default=1,
                            help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--attn-drop", type=float, default=.0,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--acc', type=float, default=0.688,
                        help='the selected FairGNN accuracy on val would be at least this high')
    parser.add_argument('--roc', type=float, default=0.745,
                        help='the selected FairGNN ROC score on val would be at least this high')
    parser.add_argument('--sens_number', type=int, default=200,
                        help="the number of sensitive attributes")
    parser.add_argument('--label_number', type=int, default=500,
                        help="the number of labels")
    parser.add_argument('--run', type=int, default=0,
                        help="kth run of the model")
    parser.add_argument('--pretrained', type=bool, default=False,
                        help="load a pretrained model")

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(args)

    #%%
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.run)

    # Load data
    # print(args.dataset)

    if args.dataset == 'german':
        dataset = 'german'
        sens_attr = "Gender"
        predict_attr = "GoodCustomer"
        label_number = 100
        sens_number = 100
        path = "./dataset/german"
        test_idx = True
        adj, features, labels, idx_train, idx_val, idx_test,sens, idx_sens_train = load_german(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,
                                                                                    path=path,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number)
    # Load credit_scoring dataset
    elif args.dataset == 'credit':
        sens_attr = "Age"  # column number after feature process is 1
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        label_number = 30000
        sens_number = 30000
        path_credit = "./dataset/credit"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_credit(args.dataset, sens_attr,
                                                                                     predict_attr, path=path_credit,
                                                                                     label_number=label_number,
                                                                                     sens_number=sens_number)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    # Load bail dataset
    elif args.dataset == 'bail':
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        sens_number = 100
        path_bail = "./dataset/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_bail(args.dataset, sens_attr,
                                                                                    predict_attr, path=path_bail,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    elif args.dataset == 'facebook':
        sens_attr = "Gender"  # column number after feature process is 0
        sens_idx = 1
        predict_attr = "GoodCustomer"
        label_number = 1042
        path_german = "./dataset/facebook"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_facebook(args.dataset, sens_attr,
                                                                                predict_attr, path=path_german,
                                                                                label_number=label_number,
                                                                                )

    elif args.dataset == 'tagged':
        sens_attr = "gender"  # column number after feature process is 0
        sens_idx = 71128
        predict_attr = "label"
        label_number = 71128 #relation 7
        #label_number = 1052065 #relation 1
        #label_number = 795083 #relation 2
        path_german = "./dataset/tagged"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_tagged(args.dataset, sens_attr,
                                                                                predict_attr, path=path_german,
                                                                                label_number=label_number,
                                                                                )

    elif args.dataset == 'pokec_n':

        dataset = 'region_job_2'
        sens_attr = "region"
        predict_attr = "my_active_sports_indicator"
        label_number = 66569
        sens_number = 66569
        seed = 20
        path = "./dataset/pokec/"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset, sens_attr,
                                                                                                predict_attr,
                                                                                                path=path,
                                                                                                label_number=label_number,
                                                                                                )

    #elif args.dataset == "tagged":
    #    sens_idx = 0
    #    adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_data(dataset=args.dataset, seed=args.seed)
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]


    # Model and optimizer
    model = FairGNN(nfeat=features.shape[1], args=args).to(device)
    if args.cuda:
        model.cuda()
        features = features.cuda()
        edge_index = edge_index.cuda()
        labels = labels.cuda()
        sens = sens.cuda()
        idx_sens_train = idx_sens_train.cuda()

    from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score

    # Train model
    t_total = time.time()
    best_result = {}
    best_fair = 100
    G_params = list(model.GNN.parameters()) + list(model.classifier.parameters()) + list(model.estimator.parameters())
    optimizer_G = torch.optim.Adam(G_params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_A = torch.optim.Adam(model.adv.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_roc_val = 0

    f = open('dataset/credit/Results/log5_fairgnn.txt', 'w')
    for epoch in range(args.epochs):
        t = time.time()

        # model.train()
        loss = train(model, features, edge_index, labels, idx_train, sens, idx_sens_train)
        model.eval()
        output, ss, z = model(features, edge_index)
        output_preds = (output.squeeze() > 0).type_as(labels)
        ss_preds = (ss.squeeze() > 0).type_as(labels)

        auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train],multi_class='ovr')
        f1_train = f1_score(labels[idx_train].cpu().numpy(), output_preds[idx_train].cpu().numpy())
        auc_f, auc_m = auc_sens(labels.cpu().numpy()[idx_train.cpu()], output.detach().cpu().numpy()[idx_train],
                                sens[idx_train].numpy())
        parity, equality = fair_metric(output_preds[idx_train].cpu().numpy(), labels[idx_train].cpu().numpy(),
                                               sens[idx_train].cpu().numpy())


        s_train = f"|Training| Epoch {epoch:05d} | Loss {loss[0]:.5f} | AUC {auc_roc_train:.5f} | Overall F1 {f1_train:.5f} | Male AUC {auc_m:.5f} | Female AUC {auc_f:.5f} | SP {parity:.5f} | EQ {equality:.5F}"

        # evaluate test
        #loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test])
        auc_roc_t = roc_auc_score(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test],multi_class='ovr')
        f1_t = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
        auc_f, auc_m = auc_sens(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()],
                                sens[idx_test].numpy())
        parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                                               sens[idx_test].cpu().numpy())



        s_test = f"|TEST| Epoch {epoch:05d} | Loss {loss[0]:.5f} | AUC {auc_roc_t:.5f} | Overall F1 {f1_t:.5f} | Male AUC {auc_m:.5f} | Female AUC {auc_f:5f} | SP {parity:.5f} | EQ {equality:.5F}\n"
        preds = (output.squeeze() > 0).type_as(labels)
        #loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val])

        auc_roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy(),multi_class='ovr')
        f1_val = f1_score(labels[idx_val].cpu().numpy(), output_preds[idx_val].cpu().numpy())
        auc_f, auc_m = auc_sens(labels.cpu().numpy()[idx_val.cpu()], output.detach().cpu().numpy()[idx_val.cpu()],
                                sens[idx_val].numpy())
        parity, equality = fair_metric(output_preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(),
                                               sens[idx_val].cpu().numpy())


        s_val = f"|Validation| Epoch {epoch:05d} | Loss {loss[0]:.5f} | AUC {auc_roc_val:.5f} | Overall F1 {f1_val:.5f} | Male AUC {auc_m:.5f} | Female AUC {auc_f:5f} | SP {parity:.5f} | EQ {equality:.5F}"

        # Store accuracy
        acc_val = accuracy(output_preds[idx_val], labels[idx_val]).item()
        roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy(),multi_class='ovr')
        acc_sens = accuracy(ss_preds[idx_test], sens[idx_test]).item()
        parity_val, equality_val = fair_metric(output_preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(),
                                               sens[idx_val].cpu().numpy())

        acc_test = accuracy(output_preds[idx_test], labels[idx_test]).item()
        roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
        parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                                       sens[idx_test].cpu().numpy())

        s_log = s_train + s_val + s_test
        f.write(s_log)
        if epoch % 100 == 0:
            print('Epoch: {:04d}'.format(epoch+1), 'acc_val: {:.4f}'.format(acc_val), "roc_val: {:.4f}".format(roc_val), "parity_val: {:.4f}".format(parity_val), "equality: {:.4f}".format(equality_val), 'loss: {:.4f}'.format(loss[0]))

        if roc_val > best_roc_val:
            best_roc_val = roc_val
            best_epoch = epoch
            best_loss = loss[0]
            best_result['acc'] = acc_test
            best_result['roc'] = roc_test
            best_result['parity'] = parity
            best_result['equality'] = equality

            # SaVE models
            num_str =  '{:02d}'.format(args.run+1)
            str_model = "./fairgnn_model.pth"
            #print(str_model)
            torch.save(model.state_dict(), str_model)

            out_preds = output.squeeze()
            out_preds = (out_preds > 0).type_as(labels)

            # print("=================================")
            # print('Epoch: {:04d}'.format(epoch+1),
            #     'cov: {:.4f}'.format(cov.item()),
            #     'cls: {:.4f}'.format(cls_loss.item()),
            #     'adv: {:.4f}'.format(adv_loss.item()),
            #     'acc_val: {:.4f}'.format(acc_val.item()),
            #     "roc_val: {:.4f}".format(roc_val),
            #     "parity_val: {:.4f}".format(parity_val),
            #     "equality: {:.4f}".format(equality_val))
           # print("Test:",
           #          "accuracy: {:.4f}".format(acc_test.item()),
           #          "roc: {:.4f}".format(roc_test),
           #          "acc_sens: {:.4f}".format(acc_sens),
           #          "parity: {:.4f}".format(parity),
           #          "equality: {:.4f}".format(equality))

    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # print('============performace on test set=============')
    if len(best_result) > 0:
        f.close()
        # Load best weights
        num_str = '{:02d}'.format(args.run+1)
        str_model = "./fairgnn_model.pth"
        #print(str_model)
        model.load_state_dict(torch.load(str_model))
        model.eval()
        output, _, _ = model(features, edge_index)
        fpr, tpr, thresholds = roc_curve(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test])
        # get the best threshold
        J = tpr - fpr
        ix = argmax(J)
        best_thresh = thresholds[ix]
        print(thresholds[ix])

        output_preds = (output.squeeze() > 0).type_as(labels)
        ss_preds = (ss.squeeze() > 0).type_as(labels)
        noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
        noisy_output, _, _ = model(noisy_features.to(device), edge_index.to(device))
        noisy_output_preds = (noisy_output.squeeze() > 0).type_as(labels)
        auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test])
        robustness_score = 1 - (output_preds.eq(noisy_output_preds)[idx_test].sum().item() / idx_test.shape[0])
        parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                                       sens[idx_test].cpu().numpy())

        acc_test = accuracy(output_preds[idx_test], labels[idx_test]).item()
        f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
        f1_f, f1_m = f1_sens(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                             sens[idx_test].numpy())
        auc_f, auc_m = auc_sens(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()],
                                sens[idx_test].numpy())
        confusion_mat = confusion_matrix(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())

        print('Best Threshold=%f' % (best_thresh))
        print('acc_train: {:.4f}'.format(acc_test))
        print('acc_test: {:.4f}'.format(acc_test))
        print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
        print('Parity: {:.4f} | Equality: {:.4f}'.format(parity,equality))
        print('F1-score: {:.4f}'.format(f1_s))
        print('CounterFactual Fairness: N/A')
        print('Robustness Score: {:.4f}'.format(robustness_score))
        print('Confusion matrix:', confusion_mat)
        print('-----------------------------------------------------')
        print(f'F1-score female: {f1_f}')
        print(f'F1-score male: {f1_m}')
        print("The AUCROC of estimator female: {:.4f}".format(auc_f))
        print("The AUCROC of estimator male: {:.4f}".format(auc_m))
        print('-----------------------------------------------------')
        print('-----------------------------------------------------')
        lt = [best_epoch, loss[0].item(), auc_roc_test, auc_m, auc_f, parity, equality]
        print(lt)
        print('-----------------------------------------------------')

    else:
        print("Please set smaller acc/roc thresholds")
