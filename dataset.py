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
import community as community_louvain #added


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
    def __init__(self, name='o2_prop', root ='data', transform=None, pre_transform = None, clustering = False):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        ''' 
        self.clustering = clustering
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

        if clustering and pre_transform is None:
            pre_transform = apply_louvain_clustering

        super(PolymerRegDataset, self).__init__(self.processed_root, transform, pre_transform) # calls the parent class (InMemoryDataset) __init__ — this triggers PyG to either load the cached processed file if it exists, or call process() to build it from scratch.

        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def processed_file_names(self): # output
        if self.clustering:
            return 'geometric_data_processed_clustered.pt'
        else:
            return 'geometric_data_processed_atomic.pt'

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
        for i, smiles_idx in enumerate(df_full.index):
            graph_dict = smiles2graph(smiles_idx, clustering=self.clustering)
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

            # Make sure it adds edge_attr IF it exists...
            if graph['edge_feat'] is not None:
                g.edge_attr = torch.from_numpy(graph['edge_feat'])
            # ...but ALWAYS delete it from the dictionary so it doesn't crash later!
            if 'edge_feat' in graph:
                del graph['edge_feat']

            if graph['node_feat'] is not None:
                g.x = torch.from_numpy(graph['node_feat'])
            if 'node_feat' in graph:
                del graph['node_feat']

            addition_prop = copy.deepcopy(graph)
            for key in addition_prop.keys():
                g[key] = torch.tensor(graph[key])
                del graph[key]

            pyg_graph_list.append(g)

        return pyg_graph_list

TU_DATASETS = ['MUTAG', 'PROTEINS', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'COLLAB', 'DD', 'GITHUB_STARGAZERS']

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

def apply_louvain_clustering(data):
    # 1. Handle Features vs Featureless
    if getattr(data, 'x', None) is not None:
        x = data.x.numpy()
    else:
        # Featureless graphs (like IMDB-BINARY) get a baseline weight of 1 (dummy weights)
        x = np.ones((data.num_nodes, 1), dtype=np.float32)
        
    edge_index = data.edge_index.numpy()
    
    # 2. Build Base NetworkX Graph
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    if edge_index.size > 0:
        G.add_edges_from(zip(edge_index[0], edge_index[1]))

    # 3. Run Louvain Clustering
    partition = community_louvain.best_partition(G)

    # Grouping nodes into communities and mapping safely to 0, 1, 2...
    unique_comms = sorted(list(set(partition.values())))
    comm_to_idx = {comm_id: i for i, comm_id in enumerate(unique_comms)}

    # 4. Feature Aggregation
    # Options: 'mean' or 'sum' — switch by commenting/uncommenting below
    num_supernodes = len(unique_comms)
    supernode_features = np.zeros((num_supernodes, x.shape[1]), dtype=np.float32)
    cluster_sizes = np.zeros(num_supernodes, dtype=np.float32)

    for original_node, comm_id in partition.items():
        new_idx = comm_to_idx[comm_id]
        supernode_features[new_idx] += x[original_node]
        cluster_sizes[new_idx] += 1

    # --- Mean pooling (active) ---
    supernode_features /= cluster_sizes[:, None]
    # --- Sum pooling (uncomment to switch back) ---
    # pass

    # 5. Extracting edge features for the compressed graph (Safe Topological Rewiring)
    new_edges = set()
    if edge_index.size > 0:
        for i in range(edge_index.shape[1]):
            u = edge_index[0, i]
            v = edge_index[1, i]
            
            su = comm_to_idx[partition[u]]
            sv = comm_to_idx[partition[v]]
            
            # Avoid self-loops
            if su != sv:
                new_edges.add((su, sv))
                new_edges.add((sv, su))

    if len(new_edges) > 0:
        new_edge_index = np.array(list(new_edges), dtype=np.int64).T
    else:
        new_edge_index = np.empty((2, 0), dtype=np.int64)

    # 6. Overwrite the PyG Data object
    data.x = torch.from_numpy(supernode_features)
    data.edge_index = torch.from_numpy(new_edge_index)
    data.num_nodes = num_supernodes
    
    # Drop edge features if they exist (crucial for OGB and Polymers so it doesn't crash)
    if hasattr(data, 'edge_attr'):
        data.edge_attr = None 
        
    return data

def smiles2graph(smiles_string, clustering=False): # Only for SMILES
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
 
    if not clustering:
        # Return the original graph
        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)
        return graph
 
    # Build a temporary PyG Data object, reuse apply_louvain_clustering,
    # then convert back to the graph dict format that read_graph_pyg expects.
    tmp = Data()
    tmp.x = torch.from_numpy(x)
    tmp.edge_index = torch.from_numpy(edge_index)
    tmp.num_nodes = len(x)
    tmp = apply_louvain_clustering(tmp)
 
    graph = dict()
    graph['edge_index'] = tmp.edge_index.numpy()
    graph['edge_feat'] = None  # edge features not preserved after clustering
    graph['node_feat'] = tmp.x.numpy()
    graph['num_nodes'] = tmp.num_nodes
    return graph
    


#def pixel2graph() # needed ?
