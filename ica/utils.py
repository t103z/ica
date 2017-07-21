"""" This implementation is largely based on and adapted from:
 https://github.com/sskhandle/Iterative-Classification """
from __future__ import print_function
import networkx as nx
import cPickle as pkl
import numpy as np
import scipy.sparse as sp
import os.path
from graph import UndirectedGraph, Node, Edge
from aggregators import Count, Prop
import sys


def build_graph(adj, features, labels):
    edges = np.array(adj.nonzero()).T
    y_values = np.array(labels.nonzero()).T  # nLabeled * 2 matrix, with (nodeID, label) in each line
    y_values = dict(zip(y_values[:, 0], y_values[:, 1]))  # dict of nodeID:label

    domain_labels = []
    for i in range(labels.shape[1]):
        domain_labels.append("c" + str(i))
    domain_labels.append("none")  # node without labels, TODO: is this correct?

    # create graph
    graph = UndirectedGraph()
    id_obj_map = []
    for i in range(adj.shape[0]):
        n = Node(i, features[i, :], domain_labels[y_values[i] if i in y_values else -1])
        graph.add_node(n)
        id_obj_map.append(n)
    for e in edges:
        graph.add_edge(Edge(id_obj_map[e[1]], id_obj_map[e[0]]))

    return graph, domain_labels


def pick_aggregator(agg, domain_labels):
    if agg == 'count':
        aggregator = Count(domain_labels)
    elif agg == 'prop':
        aggregator = Prop(domain_labels)
    else:
        raise ValueError('Invalid argument for agg (aggregation operator): ' + str(agg))
    return aggregator


def create_map(graph, train_indices):
    conditional_map = {}
    for i in train_indices:
        conditional_map[graph.node_list[i]] = graph.node_list[i].label
    return conditional_map


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype=np.float32)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def to_one_hot(labels, N):
    """In: list of (nodeId, label) tuples, #nodes N
       Out: N * |label| matrix"""
    ids, labels = zip(*labels)
    encoded = np.zeros([N, len(np.unique(labels))])
    # label mapping
    lMap = dict()
    nu = 0
    for label in np.unique(labels):
        lMap[label] = nu
        nu += 1
    for i in range(len(ids)):
        encoded[ids[i], lMap[labels[i]]] = 1
    return encoded


def load_data(dataset_str):
    """Load data."""
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            objects.append(pkl.load(open("data/ind.{}.{}".format(dataset_str, names[i]))))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1] - 1))
            ty_extended_ = np.ones((len(test_idx_range_full), 1))  # add dummy labels
            ty_extended = np.hstack([ty_extended, ty_extended_])
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        # types in homogeneous network should be consistent
        types = np.zeros(adj.shape[0])
    else:
        # since lca works on homogeneous network, label information is not important
        # for now, we'll leave them for extention
        print('Loading HIN')
        # load node type info
        types = dict()
        with open('data/{}/node_info.txt'.format(dataset_str), 'r') as f:
            for line in f:
                tok = line.strip().split()
                nodeId, typeId = int(tok[0]), int(tok[1])
                types[nodeId] = typeId
        nNodes = len(types)
        # turn into numpy vector
        types = np.array([typeId for (nodeId, typeId) in sorted(types.items())])

        # load features
        if dataset_str in ['dblp', 'cora_hin']:
            features = sp.eye(nNodes, format='csr')  # one hot encoding
        elif dataset_str in ['dblp-feat']:
            with open('data/{}/features.pkl'.format(dataset_str), 'rb') as f:
                features = pkl.load(f)
        else:
            pass   # TODO: dataset with features
        assert features.shape[0] == nNodes

        # load adjacency matrix
        DG = nx.DiGraph()
        with open('data/{}/links.txt'.format(dataset_str), 'rb') as f:
            nu = 0
            for line in f:
                nu += 1
                if nu == 1:  # first line should be nNodes \t nEdges
                    nNodes, nEdges = [int(x) for x in line.strip().split()]
                    continue
                v1, v2 = [int(x) for x in line.strip().split()]
                DG.add_edge(v1, v2)
        adj = nx.adjacency_matrix(DG)

        # split train / test / validation set
        labelRate = 0.3
        testSplit = 0.4
        valSplit = 0.2

        # load labels
        labels = []  # a list of (nodeID, label)
        with open('data/{}/labels.txt'.format(dataset_str), 'rb') as f:
            for line in f:
                nodeId, label = [int(x) for x in line.strip().split()]
                labels.append((nodeId, label))
        nLabeled = len(labels)
        labels = to_one_hot(labels, nNodes)  # turn into N * L matrix

        # load indices from file
        idx_train = np.loadtxt('data/{}/train.ind'.format(dataset_str), dtype=np.int32).tolist()
        idx_val = np.loadtxt('data/{}/val.ind'.format(dataset_str), dtype=np.int32).tolist()
        idx_test = np.loadtxt('data/{}/test.ind'.format(dataset_str), dtype=np.int32).tolist()

    print('Shape adj', adj.shape)
    print('Shape types', types.shape)
    print('Shape features', features.shape)
    print('Shape labels', labels.shape)

    return adj, types, features, labels, idx_train, idx_val, idx_test
