import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


def encode_onehot(labels):
    classes = set(labels)
    print("classes:\n", classes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    print("class_dict:\n", classes_dict)
    print("enumerate:\n", list(enumerate(classes)))
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


path="./pygcn-master/data/cora/"
dataset="cora"
idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
# print(idx_features_labels)

features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
labels = encode_onehot(idx_features_labels[:, -1])
#print(features)
print("idx_features_labels:\n", idx_features_labels[:, -1])
print("labels\n", labels)

idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                dtype=np.int32)
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                 dtype=np.int32).reshape(edges_unordered.shape)
print("edges:\n",edges)

adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(labels.shape[0], labels.shape[0]),
                    dtype=np.float32)

print("adj:\n",adj)
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = adj + sp.eye(adj.shape[0])

print("adj:\n",adj)
idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)
print("idx:\n",list(idx_train))