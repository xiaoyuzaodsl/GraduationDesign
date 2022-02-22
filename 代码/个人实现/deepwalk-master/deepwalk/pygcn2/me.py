import numpy as np
import scipy.sparse as sp
import torch
import pickle

a = torch.rand(3,5)
print(a)
b = a[0]
b = b[torch.randperm(b.size(0))]
print(b)
b, x = torch.sort(b,descending=True)
print(b)
print(torch.count_nonzero(b))
print(torch.min(b[:(torch.count_nonzero(b))]))
xm = int(torch.count_nonzero(b))
print(xm)
c = b[:2]
print(c)
print(c.size())