from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import my_load_data, data_load, accuracy
from models import GCN
# 可以接受的命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='输入随机seed')
parser.add_argument('--dataset', type=str, default="cora", help='输入数据集')
parser.add_argument('--epochs', type=int, default=200, help='训练轮数epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4, help = 'Weight Decay（L2参数')
parser.add_argument('--lr', type=float, default=0.01, help='初始学习率')
parser.add_argument('--hidden', type=int, default=16, help='hidden unit数量')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率(1-留存率)')
args=parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = data_load(args.dataset)
print(adj)
#print(features)