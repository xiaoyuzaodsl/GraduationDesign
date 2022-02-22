import math
import numpy as np
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import random
from random import shuffle

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

    def forward(self, input, adj, graph):
        support = torch.mm(input, self.weight)

        #这里是论文最初的方式，上面是为了确认tensor矩阵如何拆分
        l_x = len(adj)
        l_y = len(support[0])
        if self.func=="GCN":
            output = torch.mm(adj, support)
        # output = torch.mm(adj, support)

        #这里是经过我自己修改后的一些模式，会发现比原先的要慢非常多
        elif self.func=="MEAN":
            output = torch.empty((l_x,l_y))
            #这里是全部的
            # for i in range(len(graph)):
            #     paper1 = int(i)
            #     x = support[paper1]
            #     for j in range(len(graph[i])):
            #         paper2 = int(graph[i][j])
            #         x = torch.add(x,support[paper2])
            #     num = len(graph[i]) + 1
            #     x = torch.div(x,num)
            #     output[i] = x

            #这里是抽样的
            sample_num = 100
            for i in range(len(graph)):
                paper1 = int(i)
                x = support[paper1]
                tmp_list = graph[i]
                random.shuffle(tmp_list)
                if len(graph[i]) < sample_num:
                    for j in range(len(tmp_list)):
                        paper2 = int(tmp_list[j])
                        x = torch.add(x,support[paper2])
                    num = len(tmp_list) + 1
                    x = torch.div(x,num)
                    output[i] = x
                else:
                    for j in range(sample_num):
                        paper2 = int(tmp_list[j])
                        x = torch.add(x,support[paper2])
                    num = sample_num + 1
                    x = torch.div(x,num)
                    output[i] = x

        elif self.func=="MAX":
            output = torch.empty((l_x, l_y))
            # for i in range(len(graph)):
            #     paper1 = int(i)
            #     x = support[paper1]
            #     for j in range(len(graph[i])):
            #         paper2 = int(graph[i][j])
            #         x = torch.max(x, support[paper2])
            #     output[i] = x
            # 这里是抽样的
            sample_num = 20
            for i in range(len(graph)):
                paper1 = int(i)
                x = support[paper1]
                tmp_list = graph[i]
                random.shuffle(tmp_list)
                if len(graph[i]) < sample_num:
                    for j in range(len(tmp_list)):
                        paper2 = int(tmp_list[j])
                        x = torch.max(x, support[paper2])
                    output[i] = x
                else:
                    for j in range(sample_num):
                        paper2 = int(tmp_list[j])
                        x = torch.max(x, support[paper2])
                    output[i] = x

        elif self.func=="MIN":
            output = torch.empty((l_x, l_y))
            for i in range(len(graph)):
                paper1 = int(i)
                x = support[paper1]
                for j in range(len(graph[i])):
                    paper2 = int(graph[i][j])
                    x = torch.min(x, support[paper2])
                output[i] = x
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