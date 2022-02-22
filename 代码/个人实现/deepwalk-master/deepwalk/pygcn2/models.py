import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution,GG

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, func):
        super(GCN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.gc1 = GG(nfeat, nhid, func)
        self.gc2 = GG(nhid, nclass, func)
        self.dropout = dropout

    def forward(self, x, adj, my_adj):
        x = F.relu(self.gc1(x, adj, my_adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, my_adj)
        return F.log_softmax(x, dim=1)

