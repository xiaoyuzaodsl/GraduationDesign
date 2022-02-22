import numpy as np
import scipy.sparse as sp
import torch
import pickle

a = torch.rand(1,5)
b = torch.rand(1,5)
print(a)
print(b)
print(torch.max(a,b))