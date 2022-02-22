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

    def forward(self, input, adj, my_adj):
        support = torch.mm(input, self.weight)
        # 在这里要将普通的矩阵乘法转为需要的aggregation function
        # out0 = torch.spmm(adj,support)
        # s0 = support[:,0]
        # s0 = s0.view(1,-1)
        # # print("s0:", s0.size())
        # s0 = s0.transpose(0,1)
        # # print("s0^T",s0.size())
        # output = torch.spmm(adj, s0)
        # for i in range(len(support[0])):
        #     if i > 0:
        #         tmp = support[:, i]
        #         tmp = tmp.view(1,-1)
        #         # print("tmp:",tmp.size())
        #         tmp = tmp.transpose(0,1)
        #         tmp_out = torch.spmm(adj, tmp)
        #         output = torch.cat((output,tmp_out),1)
        # print("my_adjjjjjjjj\n\n",my_adj)

        #for paper1, link in my_adj.items():
        #    print(paper1, link)

        # 第一次尝试
        # me = torch.empty(len(support[0]))
        # for j in my_adj.get(0):
        #     me = torch.add(me, support[int(j)])
        # me = torch.add(me, support[0])
        # me = torch.div(me, (len(my_adj.get(0))+1))
        # for i in range(len(input)):
        #     if i > 0:
        #         my_set = my_adj.get(i)
        #         x = torch.empty(len(support[0]))
        #         for j in my_set:
        #             x = torch.add(x, support[int(j)])
        #         x = torch.add(x, support[int(i)])
        #         # print(x)
        #         x = torch.div(x, (len(my_set)+1))
        #         me = torch.cat((me,x),0)
        # print(me.size())
        # me = me.view(len(support), -1)
        # print(me.size())
        # output = me

        # 因为无法理解前一种方法效果差的原因，这里放弃了原来sparse_matrix的结构
        # tt = adj[0]
        # tt = tt.view(1,-1)
        # me = torch.mm(tt, support)
        # for i in range(1, len(adj)):
        #     tt = adj[i]
        #     tt = tt.view(1,-1)
        #     x = torch.mm(tt, support)
        #     me = torch.cat((me,x),0)
        # me.view(len(support),-1)
        # output = me

        #这里是论文最初的方式，上面是为了确认tensor矩阵如何拆分
        l_x = len(adj)
        l_y = len(support[0])
        if self.func=="GCN":
            output = torch.mm(adj, support)
        # output = torch.mm(adj, support)

        #这里是经过我自己修改后的一些模式，会发现比原先的要慢非常多
        elif self.func=="MEAN":
            output = torch.empty((l_x, l_y))
            # out = np.empty((int(len(adj)), int(len(support[0]))))

            for i in range(l_x):
                for j in range(l_y):
                    my_neighbour = adj[i]
                    my_feature = support[:, j]
                    feature_list = torch.mul(my_neighbour,my_feature)
                    feature_list, will_no_use = torch.sort(feature_list,descending=True)
                    feature_not_zero = torch.count_nonzero(feature_list)
                    no_zero_num = int(feature_not_zero)

                    # #方法一：邻居取样
                    # sample_list = feature_list[:no_zero_num]
                    # sample_list_rand = sample_list[(torch.randperm(sample_list.size(0)))]
                    # num = int(sample_list.size(0))
                    # if num > 5:
                    #     num = 5
                    #     sample = sample_list_rand[:6]
                    # elif num == 0:
                    #     num = 1
                    #     sample = feature_list[:1]
                    # else:
                    #     sample = sample_list_rand
                    # output[i][j] = torch.div((torch.sum(sample)),num)

                    #方法二：直接计算

                    output[i][j] = torch.div((torch.sum(feature_list)), no_zero_num)

        elif self.func=="MAX":
            output = torch.empty((l_x, l_y))
            for i in range(l_x):
                for j in range(l_y):
                    output[i][j] = torch.max(torch.mul(adj[i],support[:,j]))

        elif self.func=="MIN":
            output = torch.empty((l_x, l_y))
            for i in range(l_x):
                for j in range(l_y):
                    my_neighbour = adj[i]
                    my_feature = support[:, j]
                    #print(my_neighbour.size(), my_feature.size())
                    feature_list = torch.mul(my_neighbour, my_feature)

                    #这里是随机取样邻居的版本
                    # feature_list, not_be_used_min = torch.sort(feature_list)
                    # cut_feature_list = feature_list[:5]
                    # now_list = cut_feature_list[(torch.randperm(cut_feature_list.size(0)))]
                    # output[i][j] = torch.min(now_list)
                    feature_list, not_be_used = torch.sort(feature_list,descending=True)
                    feature_not_zero = torch.count_nonzero(feature_list)
                    no_zero_num = int(feature_not_zero)
                    if no_zero_num > 0:
                        output[i][j] = torch.min(feature_list[:no_zero_num])
                    else:
                        output[i][j] = torch.min(feature_list)
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