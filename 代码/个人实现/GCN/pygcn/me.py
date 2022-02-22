import numpy as np
import scipy.sparse as sp
import torch
import pickle

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

dataset="citeseer"
path = "../data/" + dataset + "/"
    # print("path={}".format(path))
print("正在载入数据集{}".format(dataset))
idx_features_labels = np.genfromtxt("{}{}.content".format(path,dataset),
                                        dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
labels = encode_onehot(idx_features_labels[:, -1])

# 提取图中的所有文章的idx标记，由于本身不是依次排序的，所以使用hash map修改对应
# 的类别标记, 方便后面处理
idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
idx_map = {j: i for i, j in enumerate(idx)}
oldlist = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.dtype(str))
# edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
newlist = [i for j in range(len(oldlist)) for i in oldlist[j]]
mylist = list(map(idx_map.get, newlist))
my_list = np.array(mylist).reshape(-1,2)
print(my_list)

file = open('newciteseer.cites','w')
for i in range(len(my_list)):
    file.write(str(my_list[i, 0]))
    file.write("\t")
    file.write(str(my_list[i, 1]))
    file.write("\n")
file.close()

for i in range(len(idx)):
    idx_features_labels[i,0] = i

file = open("newciteseer.content","w")
for i in range(len(idx_features_labels)):
    for j in range(len(idx_features_labels[0])):
        file.write(str(idx_features_labels[i,j]))
        file.write(" ")
    file.write("\t")
file.close()