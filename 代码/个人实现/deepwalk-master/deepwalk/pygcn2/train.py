from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import my_load_data, load_data2, accuracy
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
parser.add_argument('--func', type=str, default="GCN", help='选择聚合方法，有GCN，MEAN, MAX, MIN')

args=parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
#
# path = "../data/" + args.dataset + "/"
# print("path={}".format(path))

# 输入数据
if args.dataset=="cora":
    adj, features, labels, idx_train, idx_val, idx_test, my_adj = my_load_data(args.dataset, args.func)
else:
    adj, features, labels, idx_train, idx_val, idx_test, my_adj = load_data2(args.dataset, args.func)

# 构建GCN模型
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item()+1,
            dropout=args.dropout,
            func=args.func
)

# 确定优化器optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj, my_adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj, my_adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# 训练模型
time_initial = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("训练结束！")
print("总计时间为:{:.4f}".format(time.time() - time_initial))

# 测试模型
test()
