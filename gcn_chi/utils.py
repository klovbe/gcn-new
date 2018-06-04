#encoding=utf-8

import codecs
import sklearn
import os
import sklearn.preprocessing
from os.path import join
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf

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

def load_cell_x(path, gene_orders=None):
    df = pd.read_csv(path)
    if gene_orders is not None:
        df = df[gene_orders]
    return df.values

def MinMaxNormalize(a, scaler=None):
    if scaler is not None:
        norm_a = scaler.transform(a)
    else:
        scaler = sklearn.preprocessing.MinMaxScaler()
        norm_a = scaler.fit_transform(a)
    return norm_a, scaler

def RowNormalize(a):
    row_sum = a.sum(axis=1)
    row_normal = np.zeros_like(a)
    for i, sum in enumerate(row_sum):
        if np.nonzero(sum):
            row_normal[i] = a[i] / sum
    return row_normal

def read_cell_gene_all_data(data_dir, norm=True):
    df_list = []
    for i in range(1, 5):
        path = join(data_dir, "human{}.csv".format(i))
        # path = "./baron/human{}.csv".format(i)
        df = pd.read_csv(path)
        row_sum = np.sum(df, axis=1)
        row_sum = np.expand_dims(row_sum, 1)
        div = np.divide(df, row_sum)
        div = np.log(1 + 1e6 * div)
        df = pd.DataFrame(div, columns=df.columns, dtype=np.float32)
        df_list.append(df)
    label_list = []
    for i in range(1, 5):
        path = join(data_dir, "human{}_label.csv".format(i))
        # path = "./baron/human{}_label.csv".format(i)
        label = pd.read_csv(path, header=None).values.flatten().tolist()
        label_list.append(label)
    return df_list, label_list

def read_dense_ppi_graph(path, gene_names=None):
    ppi_df = pd.read_csv(path, index_col=0)
    return ppi_df

    # if gene_names is not None:
    #     ppi_df = ppi_df.ix[gene_names, gene_names]
    # else:
    #     gene_names = list(ppi_df.columns)
    # ppi_adj = ppi_df.values
    # return ppi_adj
def construct_train_val_test(df_list, label_list):
    val_x = df_list[0].values
    val_y = label_list[0]
    test_x = df_list[3].values
    test_y = label_list[3]
    train_x = pd.concat(df_list[1:3], ignore_index=True).values
    train_y = label_list[1] + label_list[2]
    return train_x, train_y, val_x, val_y, test_x, test_y


def load_cell_gene_data_variable(data_dir, ppi_dense_path, cell_row_norm=True, gene_norm=True):
    ppi_df = pd.read_csv(ppi_dense_path, index_col=0)
    ppi_gene_names = list(ppi_df.columns)
    df_list, label_list = read_cell_gene_all_data(data_dir, norm=True)
    cell_gene_names = list(df_list[0].columns)
    gene_names = []
    for col in ppi_gene_names:
        if col not in cell_gene_names:
            continue
        gene_names.append(col)
    ppi_df = ppi_df.ix[gene_names,gene_names]
    df_list = [df[gene_names] for df in df_list]
    cell_x = [df.values for df in df_list]
    cell_x = np.concatenate(cell_x)
    all_df = pd.concat(df_list, ignore_index=True)
    gene_feature = all_df.T
    labels = []
    for label in label_list:
        labels += label
    cell_set_size = [len(df) for df in df_list]

    cell_train_x, cell_train_y, cell_val_x, cell_val_y, cell_test_x, cell_test_y = construct_train_val_test(df_list, label_list)

    cell_labels = set(labels)
    cell_labels_to_id = {label: i for i, label in enumerate(cell_labels)}
    cell_labels_num = len(cell_labels)
    cell_names_num = len(labels)

    gene_labels = gene_names
    gene_labels_to_id = {label: i for i, label in enumerate(gene_labels)}
    gene_labels_num = len(gene_labels)

    if gene_norm:
        gene_feature, scaler = MinMaxNormalize(gene_feature)
    print("cell label num: {}, gene label num: {}".format(cell_labels_num, gene_labels_num))
    print("gene_feature shape: ", gene_feature.shape)
    coo_gene_feature = sp.coo_matrix(gene_feature, dtype=np.float32)

    if cell_row_norm:
        cell_x = RowNormalize(cell_x)

    def cat_ylabel(label_list):
        y = [cell_labels_to_id[v.strip()] for v in label_list]
        a = np.zeros((len(y), len(cell_labels)), dtype=np.int32)
        for s, yi in enumerate(y):
            a[s, yi] = 1
        return a

    cell_y = cat_ylabel(labels)

    assert len(cell_x) == len(cell_y)

    print("cell shape: ", cell_x.shape)

    # # build symmetric adjacency matrix
    # gene_adj = gene_adj + gene_adj.T.multiply(gene_adj.T > gene_adj) - gene_adj.multiply(gene_adj.T > gene_adj)

    return (ppi_df.values, coo_gene_feature, gene_labels, cell_names_num, cell_set_size, cell_x, cell_y)

def load_cell_gene_data(data_dir, ppi_dense_path, cell_row_norm=False, gene_norm=True):
    ppi_df = pd.read_csv(ppi_dense_path, index_col=0)
    ppi_gene_names = list(ppi_df.columns)
    df_list, label_list = read_cell_gene_all_data(data_dir, norm=True)
    cell_gene_names = list(df_list[0].columns)
    gene_names = []
    for col in ppi_gene_names:
        if col not in cell_gene_names:
            continue
        gene_names.append(col)
    ppi_df = ppi_df.ix[gene_names,gene_names]
    df_list = [df[gene_names] for df in df_list]
    all_df = pd.concat(df_list, ignore_index=True)
    gene_feature = all_df.T

    cell_train_x, cell_train_y, cell_val_x, cell_val_y, cell_test_x, cell_test_y = construct_train_val_test(df_list, label_list)

    cell_labels = set(cell_train_y + cell_val_y + cell_test_y)
    cell_labels_to_id = {label: i for i, label in enumerate(cell_labels)}
    cell_labels_num = len(cell_labels)
    cell_names_num = len(cell_train_y + cell_val_y + cell_test_y)

    gene_labels = gene_names
    gene_labels_to_id = {label: i for i, label in enumerate(gene_labels)}
    gene_labels_num = len(gene_labels)

    if gene_norm:
        gene_feature, scaler = MinMaxNormalize(gene_feature)
    print("cell label num: {}, gene label num: {}".format(cell_labels_num, gene_labels_num))
    print("gene_feature shape: ", gene_feature.shape)
    coo_gene_feature = sp.coo_matrix(gene_feature, dtype=np.float32)

    def load_cell_x(path, gene_orders=None):
        print(path)
        df = pd.read_csv(path)
        if gene_orders is not None:
            df = df[gene_orders]
        return df.values

    if cell_row_norm:
        cell_train_x = RowNormalize(cell_train_x)
        cell_val_x = RowNormalize(cell_val_x)
        cell_test_x = RowNormalize(cell_test_x)

    def cat_ylabel(label_list):
        y = [cell_labels_to_id[v.strip()] for v in label_list]
        a = np.zeros((len(y), len(cell_labels)), dtype=np.int32)
        for s, yi in enumerate(y):
            a[s, yi] = 1
        return a

    cell_train_y = cat_ylabel(cell_train_y)
    cell_val_y = cat_ylabel(cell_val_y)
    cell_test_y = cat_ylabel(cell_test_y)

    assert len(cell_train_x) == len(cell_train_y)
    assert len(cell_val_x) == len(cell_val_y)
    assert len(cell_test_x) == len(cell_test_y)

    print("train cell shape: ", cell_train_x.shape)
    print("val cell shape: ", cell_val_x.shape)
    print("test cell shape: ", cell_test_x.shape)

    # # build symmetric adjacency matrix
    # gene_adj = gene_adj + gene_adj.T.multiply(gene_adj.T > gene_adj) - gene_adj.multiply(gene_adj.T > gene_adj)

    return (ppi_df.values, coo_gene_feature, gene_labels, cell_names_num,
           cell_train_x, cell_train_y,
           cell_val_x, cell_val_y,
           cell_test_x, cell_test_y)


# def load_cell_gene_data_variable(data_dir, ppi_dense_path, cell_row_norm=False, gene_norm=True):
#     ppi_df = pd.read_csv(ppi_dense_path, index_col=0)
#     ppi_gene_names = list(ppi_df.columns)
#     df_list, label_list = read_cell_gene_all_data(data_dir, norm=True)
#     cell_gene_names = list(df_list[0].columns)
#     gene_names = []
#     for col in ppi_gene_names:
#         if col not in cell_gene_names:
#             continue
#         gene_names.append(col)
#     ppi_df = ppi_df.ix[gene_names,gene_names]
#     df_list = [df[gene_names] for df in df_list]
#     all_df = pd.concat(df_list, ignore_index=True)
#     gene_feature = all_df.T
#
#     cell_train_x, cell_train_y, cell_val_x, cell_val_y, cell_test_x, cell_test_y = construct_train_val_test(df_list, label_list)
#
#     cell_labels = set(cell_train_y + cell_val_y + cell_test_y)
#     cell_labels_to_id = {label: i for i, label in enumerate(cell_labels)}
#     cell_labels_num = len(cell_labels)
#     cell_names_num = len(cell_train_y + cell_val_y + cell_test_y)
#
#     gene_labels = gene_names
#     gene_labels_to_id = {label: i for i, label in enumerate(gene_labels)}
#     gene_labels_num = len(gene_labels)
#
#     if gene_norm:
#         gene_feature, scaler = MinMaxNormalize(gene_feature)
#     print("cell label num: {}, gene label num: {}".format(cell_labels_num, gene_labels_num))
#     print("gene_feature shape: ", gene_feature.shape)
#     coo_gene_feature = sp.coo_matrix(gene_feature, dtype=np.float32)
#
#     def load_cell_x(path, gene_orders=None):
#         print(path)
#         df = pd.read_csv(path)
#         if gene_orders is not None:
#             df = df[gene_orders]
#         return df.values
#
#     if cell_row_norm:
#         cell_train_x = RowNormalize(cell_train_x)
#         cell_val_x = RowNormalize(cell_val_x)
#         cell_test_x = RowNormalize(cell_test_x)
#
#     def cat_ylabel(label_list):
#         y = [cell_labels_to_id[v.strip()] for v in label_list]
#         a = np.zeros((len(y), len(cell_labels)), dtype=np.int32)
#         for s, yi in enumerate(y):
#             a[s, yi] = 1
#         return a
#
#     cell_train_y = cat_ylabel(cell_train_y)
#     cell_val_y = cat_ylabel(cell_val_y)
#     cell_test_y = cat_ylabel(cell_test_y)
#
#     assert len(cell_train_x) == len(cell_train_y)
#     assert len(cell_val_x) == len(cell_val_y)
#     assert len(cell_test_x) == len(cell_test_y)
#
#     print("train cell shape: ", cell_train_x.shape)
#     print("val cell shape: ", cell_val_x.shape)
#     print("test cell shape: ", cell_test_x.shape)
#
#     # # build symmetric adjacency matrix
#     # gene_adj = gene_adj + gene_adj.T.multiply(gene_adj.T > gene_adj) - gene_adj.multiply(gene_adj.T > gene_adj)
#
#     return (ppi_df.values, coo_gene_feature, gene_labels, cell_names_num,
#            cell_train_x, cell_train_y,
#            cell_val_x, cell_val_y,
#            cell_test_x, cell_test_y)

def random_gene_data(data_dir, ppi_dense_path, save_dir, random_gene_num=1000):
    # data_dir = "./baron/"
    # ppi_dense_path = "./gene_gcn/gcn/data/ppi_dense.csv"
    # save_dir = "./baron_random"
    # random_gene_num = 1000
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    ppi_df = pd.read_csv(ppi_dense_path, index_col=0)
    ppi_gene_names = list(ppi_df.columns)
    df_list, label_list = read_cell_gene_all_data(data_dir, norm=True)
    cell_gene_names = list(df_list[0].columns)
    gene_names = []
    for col in ppi_gene_names:
        if col not in cell_gene_names:
            continue
        gene_names.append(col)
    ppi_df = ppi_df.ix[gene_names, gene_names]
    df_list = [df[gene_names] for df in df_list]

    random_genes = np.random.choice(gene_names, (random_gene_num)).tolist()
    random_genes_path = os.path.join(save_dir, "gene_labels.csv")
    with codecs.open(random_genes_path, "w", encoding="utf-8") as f:
        for la in random_genes:
            f.write(str(la) + "\n")
    sample_ppi_df = ppi_df.ix[random_genes, random_genes]
    sample_ppi_path = join(save_dir, "ppi_dense.csv")
    sample_ppi_df.to_csv(sample_ppi_path)

    for i, df in enumerate(df_list):
        sdf = df[random_genes]
        path = os.path.join(save_dir, "human{}.csv".format(i + 1))
        sdf.to_csv(path, index=False)
        label = label_list[i]
        label_path = os.path.join(save_dir, "human{}_label.csv".format(i + 1))
        with codecs.open(label_path, "w", encoding="utf-8") as f:
            for la in label:
                f.write(str(la) + "\n")

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        return mx.row, mx.col, mx.data
        # coords = np.vstack((mx.row, mx.col)).transpose()
        # values = mx.data
        # shape = mx.shape
        # return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj, dtype=np.float32)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def load_sparse_data(name, sparse_input):
    placeholders = {
        'coo_row_ind': tf.placeholder(tf.int32),
        'coo_col_ind': tf.placeholder(tf.int32),
        'coo_val': tf.placeholder(tf.float32)
    }
    with tf.variable_scope(name):
        input_data = {
            'coo_row_ind': tf.Variable(placeholders['coo_row_ind'], trainable=False, name='coo_row_ind',
                                       validate_shape=False, dtype=tf.int32, collections=[]),
            'coo_col_ind': tf.Variable(placeholders['coo_col_ind'], trainable=False, name='coo_col_ind',
                                       validate_shape=False, dtype=tf.int32, collections=[]),
            'coo_val': tf.Variable(placeholders['coo_val'], trainable=False, name='coo_val',
                                   validate_shape=False, dtype=tf.float32, collections=[])
        }
    feed_dict = {
        # placeholders['coo_row_ind']: np.int32(sparse_input[0][:,0]),
        # placeholders['coo_col_ind']: np.int32(sparse_input[0][:,1]),
        # placeholders['coo_val']: np.float32(sparse_input[1])
        placeholders['coo_row_ind']: np.int32(sparse_input[0]),
        placeholders['coo_col_ind']: np.int32(sparse_input[1]),
        placeholders['coo_val']: np.float32(sparse_input[2])
    }
    return input_data, feed_dict


def load_dense_data(cell_num, gene_names_num, cell_label_num, data):
    placeholders = {
        "cell_gene_weight": tf.placeholder(tf.float32, shape=(cell_num, gene_names_num)),
        'cell_labels': tf.placeholder(tf.int32, shape=(cell_num, cell_label_num)),
        'labels_mask': tf.placeholder(tf.float32),
        # 'dropout': tf.placeholder(tf.float32, shape=1),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    input_data = {
        'cell_gene_weight': tf.Variable(placeholders['cell_gene_weight'], trainable=False, name='cell_gene_weight',
                                   validate_shape=False, dtype=tf.float32, collections=[]),
        'cell_labels': tf.Variable(placeholders['cell_labels'], trainable=False, name='cell_labels',
                                   validate_shape=False, dtype=tf.int32, collections=[]),
        'labels_mask': tf.Variable(placeholders['labels_mask'], trainable=False, name='labels_mask',
                               validate_shape=False, dtype=tf.float32, collections=[]),
        'dropout': tf.Variable(placeholders['dropout'], trainable=False, name='dropout',
                               validate_shape=False, dtype=tf.float32, collections=[]),
        'num_features_nonzero': tf.Variable(placeholders['num_features_nonzero'], trainable=False, name='num_features_nonzero',
                               validate_shape=False, dtype=tf.int32, collections=[])
    }
    feed_dict = {
        placeholders['cell_gene_weight']: data[0],
        placeholders['cell_labels']: data[1],
        placeholders['labels_mask']: data[2],
        placeholders['dropout']: data[3],
        placeholders['num_features_nonzero']: data[4]
    }
    return input_data, feed_dict
