import numpy as np
import scipy.sparse as sp
import torch
import pickle

a = torch.rand(3,5)
print(a)
b = torch.rand(1)
print(b)
c = torch.mul(a,b)
print(c)