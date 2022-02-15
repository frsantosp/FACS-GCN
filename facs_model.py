import torch.nn.functional as F
import math
from utils import *
import torch.nn as nn
from adagcn_utils import MixedDropout, MixedLinear
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class SparseMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.save_for_backward(sparse, dense)
        return torch.mm(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        sparse, dense = ctx.saved_tensors
        grad_sparse = grad_dense = None
        if ctx.needs_input_grad[0]:
            grad_sparse = torch.mm(grad_output, dense.t())
        if ctx.needs_input_grad[1]:
            grad_dense = torch.mm(sparse.t(), grad_output)
        return grad_sparse, grad_dense


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # initialization
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):  # softmax(A * ReLU(A X W^0) * W^1), input: X, adj: A
        support = torch.mm(input, self.weight)  # dense matrix multiplication： X * W
        output = SparseMM.apply(adj, support) # modification

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, dropout_adj, layer=2):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid) # n_feature: C, n_hidden: H
        self.gc_layer = GraphConvolution(nhid, nhid)  # n_hidden: H, n_hidden: H
        self.gc2 = GraphConvolution(nhid, nclass) # n_hidden: H, n_classes: F
        self.dropout = dropout
        self.layer = layer
        self.dropout_adj = MixedDropout(dropout_adj)

    def forward(self, x, adj): # X, A
        x = F.relu(self.gc1(x, self.dropout_adj(adj))) # for APPNP paper
        for i in range(self.layer - 2):
            x = F.relu(self.gc_layer(x, adj))  # middle conv
            x = F.dropout(x, self.dropout, training=self.training)
        if self.layer > 1:
            x = self.gc2(x, adj) # 2th conv
        return F.log_softmax(x, dim=1) # N * F

class AdaGCN(nn.Module):
    def __init__(self, nfeat,  nhid, nclass, dropout, dropout_adj):
        super().__init__()
        fcs = [MixedLinear(nfeat, nhid, bias=False)]
        fcs.append(nn.Linear(nhid, nclass, bias=False))
        self.fcs = nn.ModuleList(fcs)
        self.reg_params = list(self.fcs[0].parameters())

        if dropout == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(dropout) # p: drop rate
        if dropout_adj == 0:
            self.dropout_adj = lambda x: x
        else:
            self.dropout_adj = MixedDropout(dropout_adj) # p: drop rate
        self.act_fn = nn.ReLU()

    def _transform_features(self, x):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(x)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.act_fn(self.fcs[-1](self.dropout_adj(layer_inner)))
        return res

    def forward(self, x, adj, idx):  # X, A
        logits = self._transform_features(x) # MLP: X->H, Mixed-layer + some layers FC
        return F.log_softmax(logits, dim=-1)