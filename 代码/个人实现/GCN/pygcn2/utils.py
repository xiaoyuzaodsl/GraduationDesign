import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
import networkx as nx
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import scipy.io as sio
import random
from sklearn import preprocessing
from collections import defaultdict

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



def my_load_data(dataset, func):
    """"
    数据集在当前目录的 ../data/dataset位置给出，需要dataset.content和dataset.cites
    两个文件，
    content文件：id(int) + feature(array) + label(str，表示类别的字符串)
    cites文件：(id, id)对，表示文件引用关系
    """
    path = "../data/" + dataset + "/"
    # print("path={}".format(path))
    print("正在载入数据集{}".format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path,dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # 提取图中的所有文章的idx标记，由于本身不是依次排序的，所以使用hash map修改对应
    # 的类别标记, 方便后面处理
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 这里用这个没有order过的edges做一个neubourhood list

    print(edges)
    '''
    my_adj = defaultdict(set)
    for line in edges:
        paper1 = line[0]
        paper2 = line[1]
        my_adj[paper1].add(paper2)
        my_adj[paper2].add(paper1)
    print("my_adj:",my_adj)
    '''
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    tmp_adj = np.array(adj.todense())
    # print(tmp_adj)
    my_adj = defaultdict(set)
    for i, line in enumerate(tmp_adj):
        for j, paper in enumerate(line):
            paper1 = i
            paper2 = j
            if float(paper) > 0.:
                my_adj[paper1].add(paper2)
                my_adj[paper2].add(paper1)
                # print("there is one paper link")
    # print("my_adj:", my_adj)
    # 建立稀疏邻接矩阵，注意这里是单向图
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # 建立循环邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize(features)
    # 加上自身以后正则化，是按照文章的优化写的
    if func=="GCN":
        adj = normalize_all(adj + sp.eye(adj.shape[0]))
    else:
        adj = (adj + sp.eye(adj.shape[0]))
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    # adj = adj.todense()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, my_adj

def load_data2(dataset_source, func):
    data = sio.loadmat("../data/{}.mat".format(dataset_source))
    features = data["Attributes"]
    adj = data["Network"]
    labels = data["Label"]
    #print(adj)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)
    tmp_adj = np.array(adj.todense())

    # print(tmp_adj)
    my_adj = defaultdict(set)
    for i, line in enumerate(tmp_adj):
        for j, paper in enumerate(line):
            paper1 = i
            paper2 = j
            if float(paper) > 0.:
                my_adj[paper1].add(paper2)
                my_adj[paper2].add(paper1)
                # print("there is one paper link")
    # print("my_adj:", my_adj)

    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    if func=="GCN":
        adj = normalize_all(adj + sp.eye(adj.shape[0]))
    else:
        adj = (adj + sp.eye(adj.shape[0]))

    # features = preprocessing.normalize(features, norm='l2', axis=0)
    adj = torch.FloatTensor(np.array(adj.todense()))

    node_perm = np.random.permutation(labels.shape[0])
    num_train = int(0.05 * adj.shape[0])
    num_val = int(0.1 * adj.shape[0])
    idx_train = node_perm[:num_train]
    idx_val = node_perm[num_train:num_train + num_val]
    idx_test = node_perm[num_train + num_val:]

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    #adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, my_adj


def normalize(mx):
    """对于行正则化"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_line(mx):
    """对于列正则化"""
    linesum = np.array(mx.sum(0))
    l_inv = np.power(linesum, -1). flatten()
    l_inv[np.isinf(l_inv)] = 0.
    l_mat_inv = sp.diags(l_inv)
    mx = l_mat_inv.dot(mx)
    return mx


def normalize_all(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    linesum = np.array(mx.sum(0))
    l_inv = np.power(linesum, -0.5).flatten()
    l_inv[np.isinf(l_inv)] = 0.
    l_mat_inv = sp.diags(l_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(l_mat_inv)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)