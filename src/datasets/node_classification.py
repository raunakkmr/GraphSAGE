import os

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

class Cora(Dataset):

    def __init__(self, path, mode, num_layers,
                 self_loop=False, normalize_adj=False, transductive=False):
        """
        Parameters
        ----------
        path : str
            Path to the cora dataset with cora.cites and cora.content files.
        mode : str
            train / val / test.
        num_layers : int
            Depth of the model.
        self_loop : Boolean
            Whether to add self loops, default: False.
        normalize_adj : Boolean
            Whether to use symmetric normalization on the adjacency matrix, default: False.
        transductive : Boolean
            Whether to use all node features while training, as in a transductive setting, default: False.
        """
        super(Cora, self).__init__()

        self.path = path
        self.mode = mode
        self.num_layers = num_layers
        self.self_loop = self_loop
        self.normalize_adj = normalize_adj
        self.transductive = transductive
        self.idx = {
            'train' : np.array(range(140)),
            'val' : np.array(range(200, 500)),
            'test' : np.array(range(500, 1500))
        }

        print('--------------------------------')
        print('Reading cora dataset from {}'.format(path))
        citations = np.loadtxt(os.path.join(path, 'cora.cites'), dtype=np.int64)
        content = np.loadtxt(os.path.join(path, 'cora.content'), dtype=str)
        print('Finished reading data.')

        print('Setting up data structures.')
        if transductive:
            idx = np.arange(content.shape[0])
        else:
            if mode == 'train':
                idx = self.idx['train']
            elif mode == 'val':
                idx = np.hstack((self.idx['train'], self.idx['val']))
            elif mode == 'test':
                idx = np.hstack((self.idx['train'], self.idx['test']))
        features, labels = content[idx, 1:-1].astype(np.float32), content[idx, -1]
        d = {j : i for (i,j) in enumerate(sorted(set(labels)))}
        labels = np.array([d[l] for l in labels])

        vertices = np.array(content[idx, 0], dtype=np.int64)
        d = {j : i for (i,j) in enumerate(vertices)}
        edges = np.array([e for e in citations if e[0] in d.keys() and e[1] in d.keys()])
        edges = np.array([d[v] for v in edges.flatten()]).reshape(edges.shape)
        n, m = labels.shape[0], edges.shape[0]
        u, v = edges[:, 0], edges[:, 1]
        adj = sp.coo_matrix((np.ones(m), (u, v)),
                            shape=(n, n),
                            dtype=np.float32)
        adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        if self_loop:
            adj += sp.eye(n)
        if normalize_adj:
            degrees = np.power(np.array(np.sum(adj, axis=1)), -0.5).flatten()
            degrees = sp.diags(degrees)
            adj = (degrees.dot(adj.dot(degrees)))
        print('Finished setting up data structures.')
        print('--------------------------------')

        self.features = features
        self.labels = labels
        self.adj = adj.tolil()

    def __len__(self):
        return len(self.idx[self.mode])

    def __getitem__(self, idx):
        if self.transductive:
            idx += int(self.idx[self.mode][0])
        else:
            if self.mode != 'train':
                idx += len(self.idx['train'])
        node_layers, mappings = self._form_computation_graph(idx)
        rows = self.adj.rows[node_layers[0]]
        features = self.features[node_layers[0], :]
        labels = self.labels[node_layers[-1]]
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        return features, node_layers, mappings, rows, labels

    def collate_wrapper(self, batch):
        """
        Parameters
        ----------
        batch : list
            A list of examples from this dataset.

        Returns
        -------
        features : torch.FloatTensor
            An (n' x input_dim) tensor of input node features.
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        rows : list
        labels : torch.LongTensor
            An (n') length tensor of node labels.
        """
        idx = [node_layers[-1][0] for node_layers in [sample[1] for sample in batch]]

        node_layers, mappings = self._form_computation_graph(idx)
        rows = self.adj.rows[node_layers[0]]
        features = self.features[node_layers[0], :]
        labels = self.labels[node_layers[-1]]
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        return features, node_layers, mappings, rows, labels

    def get_dims(self):
        """
        Returns
        -------
        dimension of input features, dimension of output features
        """
        return self.features.shape[1], len(set(self.labels))

    def _form_computation_graph(self, idx):
        """
        Parameters
        ----------
        idx : int
            Index of the node for which the forward pass needs to be computed.

        Returns
        -------
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        """
        _list, _set = list, set
        rows = self.adj.rows
        if type(idx) is int:
            node_layers = [np.array([idx], dtype=np.int64)]
        elif type(idx) is list:
            node_layers = [np.array(idx, dtype=np.int64)]
        for _ in range(self.num_layers):
            prev = node_layers[-1]
            arr = [node for node in prev]
            arr.extend([v for node in arr for v in rows[node]])
            arr = np.array(_list(_set(arr)), dtype=np.int64)
            node_layers.append(arr)
        node_layers.reverse()

        mappings = [{j : i for (i,j) in enumerate(arr)} for arr in node_layers]

        return node_layers, mappings