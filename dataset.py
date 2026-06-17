from ogb.utils.features import (atom_to_feature_vector,bond_to_feature_vector) # Converts RDKit atom/bond objects into fixed-length number arrays. Like turning "Carbon atom" into [6, 0, 1, ...] 
from torch_geometric.data import InMemoryDataset 
from torch_geometric.data import Data
from rdkit import Chem # Chemistry toolkit.
from rdkit.Chem import AllChem # Chemistry toolkit
from torch_geometric.datasets import TUDataset, MNISTSuperpixels # added 
from tqdm import tqdm
import os
import pathlib
import os.path as osp
import pandas as pd
import numpy as np
import torch
import copy
import networkx as nx #added


class apply_k_core(): #added (but will we use this?) Simple clustering ? Graph clustering models we need to implement (Example : lovine, metis ) or use l core for clustreing ? after this, find the "best subgraphs" and then use those subgraphs for learning 

  def k_core_reduction(edge_index, num_nodes, k): # Translates PyTorch into NetworkX (remove hardcoded k=2)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    # edge_index is [2, num_edges] COO format
    G.add_edges_from(edge_index.T.tolist())
    
    core = nx.k_core(G, k=k) #yay # this line finds any node with fewer than k connections
    kept_nodes = sorted(core.nodes()) # removing those nodes changes the connection count of their neighbors, it checks everyone again (more might have to be deleted)

    # fallback if reduction empties the graph
    if len(kept_nodes) == 0: # Safety Net
        return list(range(num_nodes)), edge_index, {i: i for i in range(num_nodes)}
    
    node_map = {old: new for new, old in enumerate(kept_nodes)} # When we delete nodes, we create gaps in your indexing, we can't pass that to a PyTorch tensor because tensors expect sequential indices starting at 0
    kept_set = set(kept_nodes) 
    
    new_edges = [
        [node_map[u], node_map[v]]
        for u, v in edge_index.T
        if u in kept_set and v in kept_set
    ]
    new_edge_index = np.array(new_edges, dtype=np.int64).T if new_edges else np.empty((2,0), dtype=np.int64)
    
    return kept_nodes, new_edge_index, node_map

# SMILES stands for Simplified Molecular Input Line Entry System — it's basically a way to write a 3D molecule as a simple text string.
# [H]   = hydrogen
# *     = polymer end point (where chain continues)
# C(C)  = carbon with a branch to another carbon
# =     = double bond
class PolymerRegDataset(InMemoryDataset): # this class is a data preparation class that reads CSV file and converts each row into a graph that the GNN can understand.
    def __init__(self, name='o2_prop', root ='data', transform=None, pre_transform = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        ''' 
        self.name = name
        self.dir_name = '_'.join(name.split('-'))
        root = osp.join(root,name,'raw')
        self.original_root = root
        self.processed_root = osp.join(osp.dirname(osp.abspath(root)))

        self.num_tasks = 1 # predicting one property (o2 permeability)
        self.eval_metric = 'rmse' # how we measure performance
        self.task_type = 'regression'
        self.__num_classes__ = '-1'
        self.binary = 'False'

        super(PolymerRegDataset, self).__init__(self.processed_root, transform, pre_transform) # calls the parent class (InMemoryDataset) __init__ — this triggers PyG to either load the cached processed file if it exists, or call process() to build it from scratch.

        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def processed_file_names(self): # output
        return 'geometric_data_processed.pt' 

    def process(self): # Processes data
        read_path = osp.join(self.original_root, self.name.split('_')[0]+'_raw.csv')
        data_list = self.read_graph_pyg(read_path)
        print(data_list[:3])
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


    def csv2graphs(self, raw_dir): # Read and Clean Up the Data Tables
        '''
            - raw_dir: the position where gas property csv stored, 
            the name of the file is the gas name,  
            each file contains two columns: one for smiles, one for property value 
        '''
        dfs = []
        path_suffix = pathlib.Path(raw_dir).suffix
        if path_suffix == '': #is path
            for file_name in os.listdir(raw_dir):
                if len(file_name)<=10:
                    df_temp = pd.read_csv('{}/{}'.format(raw_dir, file_name), engine='python')
                    df_temp.set_index('SMILES', inplace=True)
                    dfs.append(df_temp)
                    print(file_name,':',len(df_temp.index))
            df_full = pd.concat(dfs).groupby(level=0).mean().fillna(-1)
        elif path_suffix == '.csv':
            df_full = pd.read_csv(raw_dir, engine='python')
            df_full.set_index('SMILES', inplace=True)
            print(df_full[:5])
        graph_list = []
        for smiles_idx in df_full.index[:]:
            graph_dict = smiles2graph(smiles_idx)
            props = df_full.loc[smiles_idx]
            for (name,value) in props.items(): # added line (iteritems() is deprecated in pandas 1.5.0, so we use items() instead)
                graph_dict[name] = np.array([[value]])
            graph_list.append(graph_dict)
        return graph_list

    def read_graph_pyg(self, raw_dir): # This function maps unstructured dictionary keys to standardized, fast deep-learning tensor spaces
        print('raw_dir', raw_dir)
        graph_list = self.csv2graphs(raw_dir)
        pyg_graph_list = []
        print('Converting graphs into PyG objects...')
        print(type(graph_list))
        for graph in tqdm(graph_list):
            g = Data()
            g.__num_nodes__ = graph['num_nodes']
            g.edge_index = torch.from_numpy(graph['edge_index'])

            del graph['num_nodes']
            del graph['edge_index']

            if graph['edge_feat'] is not None:
                g.edge_attr = torch.from_numpy(graph['edge_feat'])
                del graph['edge_feat']

            if graph['node_feat'] is not None:
                g.x = torch.from_numpy(graph['node_feat'])
                del graph['node_feat']

            addition_prop = copy.deepcopy(graph)
            for key in addition_prop.keys():
                g[key] = torch.tensor(graph[key])
                del graph[key]

            pyg_graph_list.append(g)

        return pyg_graph_list
    
TU_DATASETS = ['MUTAG', 'PROTEINS', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB']

class _DegreeFeatures:
    def __init__(self, max_degree):
        self.max_degree = max_degree
    def __call__(self, data):
        row = data.edge_index[0]
        deg = torch.zeros(data.num_nodes, dtype=torch.long)
        deg.scatter_add_(0, row, torch.ones_like(row))
        deg = torch.clamp(deg, max=self.max_degree - 1)
        data.x = torch.nn.functional.one_hot(deg, self.max_degree).float()
        return data

def get_tu_dataset(name, root='data'):
    dataset = TUDataset(root=root, name=name, use_node_attr=True)
    if dataset.num_node_features == 0:
        max_degree = 0
        for data in dataset:
            row = data.edge_index[0]
            deg = torch.zeros(data.num_nodes, dtype=torch.long)
            deg.scatter_add_(0, row, torch.ones_like(row))
            max_degree = max(max_degree, deg.max().item())
        max_degree = int(max_degree) + 1
        dataset = TUDataset(root=root, name=name, transform=_DegreeFeatures(max_degree))
    dataset.task_type = 'classification'
    dataset.eval_metric = 'accuracy'
    dataset.num_tasks = dataset.num_classes
    return dataset

def get_mnist_dataset(root='data'):
    train_dataset = MNISTSuperpixels(root=root, train=True)
    test_dataset = MNISTSuperpixels(root=root, train=False)
    dataset = train_dataset + test_dataset
    dataset.task_type = 'classification'
    dataset.eval_metric = 'accuracy'
    dataset.num_tasks = 10
    return dataset, train_dataset, test_dataset


def smiles2graph(smiles_string): # Only for SMILES
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    atom_label = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
        atom_label.append(atom.GetSymbol())

    x = np.array(atom_features_list, dtype = np.int64)
    atom_label = np.array(atom_label, dtype = str) # added line (np.str is deprecated in numpy 1.20, so we use np.str_ instead)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    return graph 


#def pixel2graph() # needed ?