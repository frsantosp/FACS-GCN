#%%
import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from utils_fairgnn_nifty.nifty_utils import *
from fairgnn_nifty import *
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
from sklearn.metrics import confusion_matrix,roc_curve
from numpy import argmax
import matplotlib.pyplot as plt


def average_fair(preds,labels,sens,thr):
    sum_sp = 0
    sum_eq = 0
    for i in thr:
        out_preds =  (preds.squeeze()>i).type_as(labels)
        sp,eq = fair_metric(out_preds.cpu().numpy(), labels.cpu().numpy(), sens.numpy())
        sum_sp+=sp
        sum_eq+=eq
    sum_sp = (sum_sp/len(thr))
    sum_eq = (sum_eq/len(thr))
    return sum_sp,sum_eq

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
    return f1_male,f1_fem

def auc_sens(labels, pred, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    auc_f = roc_auc_score(labels[idx_s0], pred[idx_s0])
    auc_m = roc_auc_score(labels[idx_s1], pred[idx_s1])
    return auc_m,auc_f



def ssf_validation(model, x_1, edge_index_1, x_2, edge_index_2, y):
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    # projector
    p1 = model.projection(z1)
    p2 = model.projection(z2)

    # predictor
    h1 = model.prediction(p1)
    h2 = model.prediction(p2)

    l1 = model.D(h1[idx_val], p2[idx_val])/2
    l2 = model.D(h2[idx_val], p1[idx_val])/2
    sim_loss = args.sim_coeff*(l1+l2)

    # classifier
    c1 = model.classifier(z1)
    c2 = model.classifier(z2)

    # Binary Cross-Entropy
    l3 = F.binary_cross_entropy_with_logits(c1[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2
    l4 = F.binary_cross_entropy_with_logits(c2[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2

    return sim_loss, l3+l4


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--proj_hidden', type=int, default=16,
                    help='Number of hidden units in the projection layer of encoder.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--drop_edge_rate_1', type=float, default=0.1,
                    help='drop edge for first augmented graph')
parser.add_argument('--drop_edge_rate_2', type=float, default=0.1,
                    help='drop edge for second augmented graph')
parser.add_argument('--drop_feature_rate_1', type=float, default=0.1,
                    help='drop feature for first augmented graph')
parser.add_argument('--drop_feature_rate_2', type=float, default=0.1,
                    help='drop feature for second augmented graph')
parser.add_argument('--sim_coeff', type=float, default=0.6,
                    help='regularization coeff for the self-supervised task')
parser.add_argument('--dataset', type=str, default='loan',
                    choices=['nba','bail','loan', 'credit', 'german','facebook','tagged','pokec','tagged_2','region_job'])
parser.add_argument("--num_heads", type=int, default=1,
                        help="number of hidden attention heads")
parser.add_argument("--num_out_heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'rogcn'])
parser.add_argument('--encoder', type=str, default='gcn')


args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
# print(args.dataset)

# Load credit_scoring dataset
if args.dataset == 'credit':
    sens_attr = "Age"  # column number after feature process is 1
    sens_idx = 1
    predict_attr = 'NoDefaultNextMonth'
    label_number = 30000
    path_credit = "./dataset/credit"
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(args.dataset, sens_attr,
                                                                            predict_attr, path=path_credit,
                                                                            label_number=label_number
                                                                            )
    norm_features = feature_norm(features)
    norm_features[:, sens_idx] = features[:, sens_idx]
    features = norm_features

# Load german dataset
elif args.dataset == 'german':
    sens_attr = "Gender"  # column number after feature process is 0
    sens_idx = 0
    predict_attr = "GoodCustomer"
    label_number = 100
    path_german = "./dataset/german"
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(args.dataset, sens_attr,
                                                                            predict_attr, path=path_german,
                                                                            label_number=label_number,
                                                                            )
# Load bail dataset
elif args.dataset == 'bail':
    sens_attr = "WHITE"  # column number after feature process is 0
    sens_idx = 0
    predict_attr = "RECID"
    label_number = 100
    path_bail = "./dataset/bail"
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(args.dataset, sens_attr,
                                                                            predict_attr, path=path_bail,
                                                                            label_number=label_number,
                                                                            )
    norm_features = feature_norm(features)
    norm_features[:, sens_idx] = features[:, sens_idx]
    features = norm_features

elif args.dataset == 'facebook':
    sens_attr = "Gender"  # column number after feature process is 0
    sens_idx = 1
    predict_attr = "GoodCustomer"
    label_number = 1046
    path_german = "./dataset/facebook"
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_facebook(args.dataset, sens_attr,
                                                                            predict_attr, path=path_german,
                                                                            label_number=label_number,
                                                                            )

elif args.dataset == 'tagged':
    sens_attr = "gender"  # column number after feature process is 0
    sens_idx = 0
    predict_attr = "label"
    label_number =71128  #relation 17
    # label_number = 1052065 #relation 1
    #label_number = 1052065 #relation 2
    path_german = "./dataset/tagged"
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_tagged(args.dataset, sens_attr,
                                                                            predict_attr, path=path_german,
                                                                            label_number=label_number,
                                                                            )

elif args.dataset == 'pokec':

    dataset = 'region_job_2'
    sens_attr = "region"
    predict_attr = "my_active_sports_indicator"
    label_number = 66569
    sens_idx = 0
    #sens_number = args.sens_number
    path = "./dataset/pokec/"
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec(dataset, sens_attr,
                                                                            predict_attr, path=path,
                                                                            label_number=label_number,
                                                                            )

#elif args.dataset == "tagged":
#    sens_idx = 0
#    adj, features, labels, idx_train, idx_val, idx_test, sens = load_data(dataset= args.dataset, seed=seed)
else:
    print('Invalid dataset name!!')
    exit(0)

edge_index = convert.from_scipy_sparse_matrix(adj)[0]

#%%
# Model and optimizer
num_class = labels.unique().shape[0]-1
if args.model == 'gcn':
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=num_class,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)

elif args.model == 'sage':
    model = SAGE(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=num_class,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)

elif args.model == 'gin':
    model = GIN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=num_class,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)

elif args.model == 'jk':
    model = JK(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=num_class,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)

elif args.model == 'infomax':
    enc_dgi = Encoder_DGI(nfeat=features.shape[1], nhid=args.hidden)
    enc_cls = Encoder_CLS(nhid=args.hidden, nclass=num_class)
    model = GraphInfoMax(enc_dgi=enc_dgi, enc_cls=enc_cls)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)

elif args.model == 'rogcn':
    model = RobustGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nhid=args.hidden, nclass=num_class, dropout=args.dropout, device=device, seed=args.seed)

elif args.model == 'ssf':
    encoder = Encoder(in_channels=features.shape[1], out_channels=args.hidden, base_model=args.encoder).to(device)
    model = SSF(encoder=encoder, num_hidden=args.hidden, num_proj_hidden=args.proj_hidden, sim_coeff=args.sim_coeff, nclass=num_class).to(device)
    val_edge_index_1 = dropout_adj(edge_index.to(device), p=args.drop_edge_rate_1)[0]
    val_edge_index_2 = dropout_adj(edge_index.to(device), p=args.drop_edge_rate_2)[0]
    val_x_1 = drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx, sens_flag=False)
    val_x_2 = drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx)
    test_edge_index_1 = dropout_adj(edge_index.to(device), p=args.drop_edge_rate_1)[0]
    test_edge_index_2 = dropout_adj(edge_index.to(device), p=args.drop_edge_rate_2)[0]
    test_x_1 = drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx, sens_flag=False)
    test_x_2 = drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx)
    par_1 = list(model.encoder.parameters()) + list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.fc3.parameters()) + list(model.fc4.parameters())
    par_2 = list(model.c1.parameters()) + list(model.encoder.parameters())
    optimizer_1 = optim.Adam(par_1, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_2 = optim.Adam(par_2, lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)


# Train model
t_total = time.time()
best_loss = 1000000000
best_acc = 0
features = features.to(device)
edge_index = edge_index.to(device)
labels = labels.to(device)

if args.model == 'rogcn':
    model.fit(features, adj, labels, idx_train, idx_val=idx_val, idx_test=idx_test, verbose=True, attention=False, train_iters=args.epochs)

f = open('log1_nifty.txt', 'w')
for epoch in range(args.epochs+1):
    t = time.time()

    if args.model in ['gcn', 'sage', 'gin', 'jk', 'infomax']:
        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index)

        # Binary Cross-Entropy
        preds = (output.squeeze()>0).type_as(labels)
        loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))

        loss_train.backward()
        optimizer.step()

        auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
        f1_train = f1_score(labels[idx_train].cpu().numpy(), preds[idx_train].cpu().numpy())
        auc_m, auc_f = auc_sens(labels.cpu().numpy()[idx_train.cpu()], output.detach().cpu().numpy()[idx_train.cpu()],
                                sens[idx_train].numpy())
        parity, equality = fair_metric(preds[idx_train].cpu().numpy(), labels[idx_train].cpu().numpy(),
                                       sens[idx_train].numpy())

        #evaluate test
        model.eval()
        output = model(features, edge_index)

        preds = (output.squeeze() > 0).type_as(labels)
        loss_test = F.binary_cross_entropy_with_logits(output[idx_test],
                                                        labels[idx_test].unsqueeze(1).float().to(device))

        auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test])
        f1_t = f1_score(labels[idx_test].cpu().numpy(), preds[idx_test].cpu().numpy())
        auc_m, auc_f = auc_sens(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()],
                                sens[idx_test].numpy())
        parity, equality = fair_metric(preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                                       sens[idx_test].numpy())


        # Evaluate validation set performance separately,
        model.eval()
        output = model(features, edge_index)


        # Binary Cross-Entropy
        preds = (output.squeeze()>0).type_as(labels)
        loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float().to(device))

        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])
        f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
        auc_m, auc_f = auc_sens(labels.cpu().numpy()[idx_val.cpu()], output.detach().cpu().numpy()[idx_val.cpu()],
                                sens[idx_val].numpy())
        parity, equality = fair_metric(preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(),
                                       sens[idx_val].numpy())


        if epoch % 100 == 0:
            print(f"[Train] Epoch {epoch}:train_loss: {loss_train.item():.4f} | train_auc_roc: {auc_roc_train:.4f} | val_loss: {loss_val.item():.4f} | val_auc_roc: {auc_roc_val:.4f}")

        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            best_epoch = epoch
            torch.save(model.state_dict(), 'weights_vanilla.pt')

    elif args.model == 'ssf':
        sim_loss = 0
        cl_loss = 0
        rep = 1
        for _ in range(rep):
            model.train()
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            edge_index_1 = dropout_adj(edge_index, p=args.drop_edge_rate_1)[0]
            edge_index_2 = dropout_adj(edge_index, p=args.drop_edge_rate_2)[0]
            x_1 = drop_feature(features, args.drop_feature_rate_2, sens_idx, sens_flag=False)
            x_2 = drop_feature(features, args.drop_feature_rate_2, sens_idx)
            z1 = model(x_1, edge_index_1)
            z2 = model(x_2, edge_index_2)

            # projector
            p1 = model.projection(z1)
            p2 = model.projection(z2)

            # predictor
            h1 = model.prediction(p1)
            h2 = model.prediction(p2)

            l1 = model.D(h1[idx_train], p2[idx_train])/2
            l2 = model.D(h2[idx_train], p1[idx_train])/2
            sim_loss += args.sim_coeff*(l1+l2)

        (sim_loss/rep).backward()
        optimizer_1.step()

        # classifier
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        c1 = model.classifier(z1)
        c2 = model.classifier(z2)

        # Binary Cross-Entropy
        l3 = F.binary_cross_entropy_with_logits(c1[idx_train], labels[idx_train].unsqueeze(1).float().to(device))/2
        l4 = F.binary_cross_entropy_with_logits(c2[idx_train], labels[idx_train].unsqueeze(1).float().to(device))/2

        cl_loss = (1-args.sim_coeff)*(l3+l4)
        cl_loss.backward()
        optimizer_2.step()
        loss = (sim_loss/rep + cl_loss)

        # Validation
        model.eval()
        val_s_loss, val_c_loss = ssf_validation(model, val_x_1, val_edge_index_1, val_x_2, val_edge_index_2, labels)
        emb = model(val_x_1, val_edge_index_1)
        output = model.predict(emb)
        preds = (output.squeeze()>0).type_as(labels)
        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])
        #f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
        auc_m, auc_f = auc_sens(labels.cpu().numpy()[idx_val.cpu()], output.detach().cpu().numpy()[idx_val.cpu()],
                                sens[idx_val].numpy())
        parity, equality = fair_metric(preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(),
                                       sens[idx_val].numpy())


        emb = model(features.to(device), edge_index.to(device))
        output = model.predict(emb)
        preds = (output.squeeze() > 0).type_as(labels)
        auc_roc_t = roc_auc_score(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test])
        f1_t = f1_score(labels[idx_test].cpu().numpy(), preds[idx_test].cpu().numpy())
        auc_m, auc_f = auc_sens(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()],
                                sens[idx_test].numpy())
        parity, equality = fair_metric(preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                                       sens[idx_test].numpy())

        #loss_list_t.append(test_c_loss + test_s_loss)


        s_test = "|TEST| Epoch {:05d} | Loss {:.5f} | AUC {:.5f} | Overall F1 {:.5f} |" \
                 "Male AUC {:.5f} | Female AUC {:5f} | SP {:.5f} | EQ {:.5F}\n".format(epoch,loss, auc_roc_t, f1_t,auc_m,auc_f ,parity, equality)

        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])
        f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
        auc_m, auc_f = auc_sens(labels.cpu().numpy()[idx_val.cpu()], output.detach().cpu().numpy()[idx_val.cpu()],
                                sens[idx_val].numpy())
        parity, equality = fair_metric(preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(),
                                       sens[idx_val].numpy())



        s_val = "|Validation| Epoch {:05d} | Loss {:.5f} | AUC {:.5f} | Overall F1 {:.5f} |" \
                 "Male AUC {:.5f} | Female AUC {:5f} | SP {:.5f} | EQ {:.5F}".format(epoch, loss, auc_roc_val, f1_val,
                                                                                     auc_m, auc_f, parity, equality)

        auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
        f1_train = f1_score(labels[idx_train].cpu().numpy(), preds[idx_train].cpu().numpy())
        auc_m, auc_f = auc_sens(labels.cpu().numpy()[idx_train.cpu()], output.detach().cpu().numpy()[idx_train.cpu()],
                                sens[idx_train].numpy())
        parity, equality = fair_metric(preds[idx_train].cpu().numpy(), labels[idx_train].cpu().numpy(),
                                       sens[idx_train].numpy())



        s_train = "|Training| Epoch {:05d} | Loss {:.5f} | AUC {:.5f} | Overall F1 {:.5f} |" \
                "Male AUC {:.5f} | Female AUC {:5f} | SP {:.5f} | EQ {:.5F}".format(epoch, loss, auc_roc_train, f1_train,
                                                                                    auc_m, auc_f, parity, equality)



        s_log = s_train+s_val+s_test
        f.write(s_log)
        if epoch % 100 == 0:
            print(f"[Train] Epoch {epoch}:train_s_loss: {(sim_loss/rep):.4f} | train_c_loss: {cl_loss:.4f} | val_s_loss: {val_s_loss:.4f} | val_c_loss: {val_c_loss:.4f} | val_auc_roc: {auc_roc_val:.4f}")
            fpr, tpr, thresholds = roc_curve(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])
            #plt.plot(fpr, tpr)
            #plt.plot([0, 1], [0, 1])
            #plt.show()

        if (val_c_loss + val_s_loss) < best_loss:
            # print(f'{epoch} | {val_s_loss:.4f} | {val_c_loss:.4f}')
            best_loss = val_c_loss + val_s_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'weights_ssf_{args.encoder}_1.pt')

print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
f.close()
if args.model in ['gcn', 'sage', 'gin', 'jk', 'infomax']:
    model.load_state_dict(torch.load('weights_vanilla.pt'))
    model.eval()
    output = model(features.to(device), edge_index.to(device))
    counter_features = features.clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model(counter_features.to(device), edge_index.to(device))
    noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
    noisy_output = model(noisy_features.to(device), edge_index.to(device))

elif args.model == 'rogcn':
    model.load_state_dict(torch.load(f'weights_rogcn_{args.seed}.pt'))
    model.eval()
    model = model.to('cpu')
    output = model.predict(features.to('cpu'))
    counter_features = features.to('cpu').clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model.predict(counter_features.to('cpu'))
    noisy_features = features.clone().to('cpu') + torch.ones(features.shape).normal_(0, 1).to('cpu')
    noisy_output = model.predict(noisy_features)

else:
    model.load_state_dict(torch.load(f'weights_ssf_{args.encoder}_1.pt'))
    model.eval()
    emb = model(features.to(device), edge_index.to(device))
    output = model.predict(emb)
    counter_features = features.clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model.predict(model(counter_features.to(device), edge_index.to(device)))
    noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
    noisy_output = model.predict(model(noisy_features.to(device), edge_index.to(device)))

# Report
fpr, tpr, thresholds = roc_curve(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test])
        # get the best threshold
#J = tpr - fpr
#ix = argmax(J)
#best_thresh = thresholds[ix]
#print(thresholds[ix])
#output_preds = (output.squeeze()>thresholds[ix]).type_as(labels)
output_preds = (output.squeeze()>0).type_as(labels)
counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
noisy_output_preds = (noisy_output.squeeze()>0).type_as(labels)
auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])
robustness_score = 1 - (output_preds.eq(noisy_output_preds)[idx_test].sum().item()/idx_test.shape[0])

parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
f1_m,f1_f= f1_sens(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
auc_m,auc_f = auc_sens(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()],sens[idx_test].numpy())
confusion_mat = confusion_matrix(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())

#fpr, tpr, thresholds = roc_curve(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
#avg_parity,avg_eq = average_fair(output[idx_train],labels[idx_train],sens[idx_train],thresholds)

#fpr, tpr, thresholds_v = roc_curve(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])
#avg_parity_v,avg_eq_v = average_fair(output[idx_val],labels[idx_val],sens[idx_val],thresholds)

#fpr, tpr, thresholds_t = roc_curve(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test])
#avg_parity_t,avg_eq_t = average_fair(output[idx_test],labels[idx_test],sens[idx_test],thresholds)
# print report
#
print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
print(f'Parity: {parity} | Equality: {equality}')
print(f'F1-score: {f1_s}')
print(f'CounterFactual Fairness: {counterfactual_fairness}')
print(f'Robustness Score: {robustness_score}')
print('Confusion matrix:',confusion_mat)
print('-----------------------------------------------------')
print(f'F1-score female: {f1_f}')
print(f'F1-score male: {f1_m}')
print("The AUCROC of estimator female: {:.4f}".format(auc_f))
print("The AUCROC of estimator male: {:.4f}".format(auc_m))
print('-----------------------------------------------------')
#print('Average_sp: {:.4f}  Average EQ: {.4f}'.format(avg_parity,avg_eq))
print('-----------------------------------------------------')
lt = [best_epoch, loss.item(),auc_roc_test,auc_m, auc_f, parity, equality]
print(lt)
print('-----------------------------------------------------')
