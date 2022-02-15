import random
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize
import dgl


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()


def sample_mask(idx, length):
    """Create mask."""
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def aug_normalized_adjacency(adj):
    """ A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2 """
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def load_data(path='data/', dataset='tagged', seed=80):
    """Load user network dataset (Tagged only for now)"""
    print('Loading {} dataset...'.format(dataset))
    node_features = pd.read_csv(path + str(dataset) + '_features.csv', header=0)
    if dataset == 'german':
        node_features = node_features.drop(['PurposeOfLoan'], axis=1)
    labels = torch.from_numpy(node_features['label'].to_numpy())  # label tensor
    sensitive_attribute = torch.from_numpy(node_features['sensitive'].to_numpy()).type(torch.LongTensor)  # gender tensor

    # last three columns are userIds, sensitives and labels
    features = node_features[node_features.columns[:-3]].to_numpy()
    relations = pd.read_csv(path + str(dataset) + '_edges.csv', header=0)
    # build graph
    try:
        adj = sp.coo_matrix((relations['weight'].to_numpy(), (relations['src'].to_numpy(),
                                                              relations['dst'].to_numpy())),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
    except KeyError:
        adj = sp.coo_matrix((np.ones(relations.shape[0]), (relations['src'].to_numpy(),
                                                           relations['dst'].to_numpy())),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = torch.FloatTensor(normalize(features))  # normalize feature matrix
    aug_adj = aug_normalized_adjacency(sp.csr_matrix(adj)) # normalize adjacency matrix and add self loop
    aug_adj = sparse_mx_to_torch_sparse_tensor(aug_adj)

    # randomize dataset selection
    random.seed(seed)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:int(0.6 * len(label_idx_0))], label_idx_1[:int(0.6 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.6 * len(label_idx_0)):int(0.8 * len(label_idx_0))],
                        label_idx_1[int(0.6 * len(label_idx_1)):int(0.8 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.8 * len(label_idx_0)):], label_idx_1[int(0.8 * len(label_idx_1)):])

    train_mask = torch.from_numpy(sample_mask(idx_train, node_features.shape[0]))
    val_mask = torch.from_numpy(sample_mask(idx_val, node_features.shape[0]))
    test_mask = torch.from_numpy(sample_mask(idx_test, node_features.shape[0]))

    return aug_adj, features, labels, sensitive_attribute, train_mask, val_mask, test_mask


def acc_measurements(output, labels, gender_label):
    output = torch.argmax(output, dim=-1)
    female_mask = gender_label > 0
    male_mask = gender_label < 1
    overall_acc_score = accuracy_score(y_true=labels, y_pred=output)
    male_acc_score = accuracy_score(y_true=labels[male_mask], y_pred=output[male_mask])
    female_acc_score = accuracy_score(y_true=labels[female_mask], y_pred=output[female_mask])
    return [overall_acc_score, male_acc_score, female_acc_score]


def auc_measurements(output, labels, gender_label):
    output = output[:, 1].detach().numpy()
    female_mask = gender_label > 0
    male_mask = gender_label < 1
    overall_auc_score = roc_auc_score(y_true=labels, y_score=output)
    male_auc_score = roc_auc_score(y_true=labels[male_mask], y_score=output[male_mask])
    female_auc_score = roc_auc_score(y_true=labels[female_mask], y_score=output[female_mask])
    return [overall_auc_score, male_auc_score, female_auc_score]


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

def load_data_dgl(path='/Users/franciscosantos/Documents/AdaGCN/Data/', dataset='tagged', seed=80):
    """Load user network dataset (Tagged only for now)"""
    """Load user network dataset (Tagged only for now)"""
    print('Loading {} dataset...'.format(dataset))
    node_features = pd.read_csv(path + str(dataset) + '_features.csv', header=0)
    size = node_features.shape
    if dataset == 'german':
        node_features = node_features.drop(['PurposeOfLoan'], axis=1)
    labels = torch.from_numpy(node_features['label'].to_numpy())  # label tensor
    sensitive_attribute = torch.from_numpy(node_features['sensitive'].to_numpy()).type(
        torch.LongTensor)  # gender tensor

    # last three columns are userIds, sensitives and labels
    features = node_features[node_features.columns[:-3]].to_numpy()
    relations = pd.read_csv(path + str(dataset) + '_edges.csv', header=0)
    # build graph
    g = dgl.graph((relations['src'].to_numpy(), relations['dst'].to_numpy()), num_nodes=node_features.shape[0])

    features = torch.FloatTensor(normalize(features))  # normalize feature matrix

    # randomize dataset selection
    random.seed(seed)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:int(0.6 * len(label_idx_0))], label_idx_1[:int(0.6 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.6 * len(label_idx_0)):int(0.8 * len(label_idx_0))],
                        label_idx_1[int(0.6 * len(label_idx_1)):int(0.8 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.8 * len(label_idx_0)):], label_idx_1[int(0.8 * len(label_idx_1)):])

    train_mask = torch.from_numpy(sample_mask(idx_train, node_features.shape[0]))
    val_mask = torch.from_numpy(sample_mask(idx_val, node_features.shape[0]))
    test_mask = torch.from_numpy(sample_mask(idx_test, node_features.shape[0]))

    return g, features, labels, sensitive_attribute, train_mask, val_mask, test_mask