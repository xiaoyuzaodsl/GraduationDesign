import math
import numpy as np
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

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
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GG(Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, func, bias=True):
        super(GG, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.func = func
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.random_base = Parameter(torch.FloatTensor(1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.random_base.data.uniform_(0.001,0.001)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, my_adj):
        support = torch.mm(input, self.weight)
        l_x = len(adj)
        l_y = len(support[0])
        if self.func=="GCN":
            output = torch.mm(adj, support)
            #noise = torch.randn(input.size())
            #noise = torch.mul(noise,self.random_base)
            #noisy = torch.mm(adj, torch.mm(noise,self.weight))
            #output = output + noisy
        # output = torch.mm(adj, support)

        #这里是经过我自己修改后的一些模式，会发现比原先的要慢非常多
        elif self.func=="rand":
            output = torch.mm(adj, support)
            noise = torch.rand(input.size())
            noise = torch.mul(noise, self.random_base)
            noisy = torch.mm(adj, torch.mm(noise, self.weight))
            output = output + noisy
        elif self.func=="randn":
            output = torch.mm(adj, support)
            noise = torch.randn(input.size())
            noise = torch.mul(noise, self.random_base)
            noisy = torch.mm(adj, torch.mm(noise, self.weight))
            output = output + noisy
        else:
            print("没有输入需要的模型，自动转为GCN方法")
            output = torch.mm(adj.support)
        # output = torch.mm(adj,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'