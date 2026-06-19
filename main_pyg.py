import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import numpy as np
from tqdm import tqdm

## dataset
from sklearn.model_selection import train_test_split
from dataset import PolymerRegDataset, get_tu_dataset, get_mnist_dataset, TU_DATASETS
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

## Clustering/Compression
from dataset import apply_louvain_clustering

## training
from model import GraphEnvAug
from utils import init_weights, get_args, train, eval


def main(args):
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.dataset.startswith('ogbg'):
        dataset = PygGraphPropPredDataset(name = args.dataset, root='data')
        
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = 0)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = 0)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = 0)
        evaluator = Evaluator(args.dataset)

    elif args.dataset.startswith('plym'):
        dataset = PolymerRegDataset(name = args.dataset.split('-')[1], root='data', clustering=args.clustering) # PolymerRegDataset
        full_idx = list(range(len(dataset)))
        train_ratio = 0.6
        valid_ratio = 0.1
        test_ratio = 0.3
        train_index, test_index, _, _ = train_test_split(full_idx, full_idx, test_size=test_ratio, random_state=42)
        train_index, val_index, _, _ = train_test_split(train_index, train_index, test_size=valid_ratio/(valid_ratio+train_ratio), random_state=42)

        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)

        train_loader = DataLoader(dataset[train_index], batch_size=args.batch_size, shuffle=True, num_workers = 0)
        valid_loader = DataLoader(dataset[val_index], batch_size=args.batch_size, shuffle=False, num_workers = 0)
        test_loader = DataLoader(dataset[test_index], batch_size=args.batch_size, shuffle=False, num_workers = 0)
        evaluator = Evaluator('ogbg-molesol') # RMSE metric

    elif args.dataset == 'mnist':
        dataset, train_ds, test_ds = get_mnist_dataset(root='data')
        full_train_idx = list(range(len(train_ds)))
        train_index, val_index = train_test_split(full_train_idx, test_size=0.1, random_state=42)
        train_loader = DataLoader(train_ds[torch.LongTensor(train_index)], batch_size=args.batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(train_ds[torch.LongTensor(val_index)], batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        evaluator = None

    elif args.dataset in TU_DATASETS:
        dataset = get_tu_dataset(name=args.dataset, root='data')
        if args.clustering:
            args.task_type = dataset.task_type      # save before dataset is overwritten
            args.num_tasks = dataset.num_tasks      # save before dataset is overwritten
            dataset = [apply_louvain_clustering(data) for data in dataset]
        full_idx = list(range(len(dataset)))
        train_index, test_index = train_test_split(full_idx, test_size=0.2, random_state=42)
        train_index, val_index = train_test_split(train_index, test_size=0.125, random_state=42)
        train_loader = DataLoader(dataset[torch.LongTensor(train_index)] if not args.clustering else [dataset[i] for i in train_index], batch_size=args.batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(dataset[torch.LongTensor(val_index)] if not args.clustering else [dataset[i] for i in val_index], batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(dataset[torch.LongTensor(test_index)] if not args.clustering else [dataset[i] for i in test_index], batch_size=args.batch_size, shuffle=False, num_workers=0)
        evaluator = None
        
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    n_train_data, n_val_data, n_test_data = len(train_loader.dataset), len(valid_loader.dataset), float(len(test_loader.dataset))
    print(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")

    args.task_type = dataset.task_type if not isinstance(dataset, list) else args.task_type

    # Detect node/edge feature dimensions for non-molecular datasets
    is_chemical_dataset = args.dataset.startswith('ogbg') or args.dataset.startswith('plym')
    atom_encode = is_chemical_dataset and not args.clustering
    node_dim, edge_dim = None, None
    if not atom_encode:
        sample = next(iter(train_loader))
        node_dim = sample.x.size(-1)
        edge_dim = sample.edge_attr.size(-1) if sample.edge_attr is not None else None

    num_tasks = dataset.num_tasks if not isinstance(dataset, list) else args.num_tasks
    model = GraphEnvAug(gnn_type = args.gnn, num_tasks = num_tasks, num_layer = args.num_layer,
                         emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, gamma=args.gamma, use_linear_predictor = args.use_linear_predictor,
                         atom_encode=atom_encode, node_dim=node_dim, edge_dim=edge_dim).to(device)
    init_weights(model, args.initw_name, init_gain=0.02)
    opt_separator = optim.Adam(model.separator.parameters(), lr=args.lr, weight_decay=args.l2reg)
    opt_predictor = optim.Adam(list(model.graph_encoder.parameters())+list(model.predictor.parameters()), lr=args.lr, weight_decay=args.l2reg)
    optimizers = {'separator': opt_separator, 'predictor': opt_predictor}
    if args.use_lr_scheduler:
        schedulers = {}
        for opt_name, opt in optimizers.items():
            schedulers[opt_name] = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)
    else:
        schedulers = None
    cnt_wait = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        print("=====Epoch {}".format(epoch))
        path = epoch % int(args.path_list[-1])
        if path in list(range(int(args.path_list[0]))):
            optimizer_name = 'separator' 
        elif path in list(range(int(args.path_list[0]), int(args.path_list[1]))):
            optimizer_name = 'predictor'

        task_type = dataset.task_type if not isinstance(dataset, list) else args.task_type
        train(args, model, device, train_loader, optimizers, task_type, optimizer_name)

        if schedulers != None:
            schedulers[optimizer_name].step()
        train_perf = eval(args, model, device, train_loader, evaluator)[0]
        valid_perf = eval(args, model, device, valid_loader, evaluator)[0]
        update_test = False
        if epoch != 0:
            task_type = dataset.task_type if not isinstance(dataset, list) else args.task_type
            if 'classification' in task_type and valid_perf >  best_valid_perf:
                update_test = True
            elif 'classification' not in task_type and valid_perf <  best_valid_perf:
                update_test = True
        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            cnt_wait = 0
            best_epoch = epoch
            test_perfs = eval(args, model, device, test_loader, evaluator)
            if args.dataset.startswith('ogbg'):
                test_auc  = test_perfs[0]
                print({'Metric': 'AUC', 'Train': train_perf, 'Validation': valid_perf, 'Test': test_auc})
            elif args.dataset.startswith('plym'):
                test_rmse, test_r2 = test_perfs[0], test_perfs[1]
                print({'Metric': 'RMSE', 'Train': train_perf, 'Validation': valid_perf, 'Test': test_rmse, 'Test R2': test_r2})
            else:
                test_acc = test_perfs[0]
                print({'Metric': 'Accuracy', 'Train': train_perf, 'Validation': valid_perf, 'Test': test_acc})
        else:
            print({'Train': train_perf, 'Validation': valid_perf})
            cnt_wait += 1
            if cnt_wait > args.patience:
                break
    print('Finished training! Results from epoch {} with best validation {}.'.format(best_epoch, best_valid_perf))
    if args.dataset.startswith('ogbg'):
        print('Test auc: {}'.format(test_auc))
        return [best_valid_perf, test_auc]
    elif args.dataset.startswith('plym'):
        print('Test rmse: {}, Test r2: {} \n'.format(test_rmse, test_r2))
        return [best_valid_perf, test_rmse, test_r2]
    else:
        print('Test accuracy: {}'.format(test_acc))
        return [best_valid_perf, test_acc]

def config_and_run(args):
    
    if args.by_default:
        if args.dataset == 'plym-o2_prop':
            # oxygen permeability
            args.gamma = 0.2
            args.epochs = 400
            args.num_layer = 3
            args.drop_ratio = 0.1
            args.batch_size = 32
            args.l2reg = 1e-4
            args.lr = 1e-2
            if args.gnn == 'gcn-virtual':
                args.lr = 1e-3
                args.l2reg = 1e-5
                args.patience = 100
        if args.dataset == 'plym-mt_prop':
            # melting temperature
            args.epochs = 400
            args.l2reg = 1e-5
            args.gamma = 0.05
            args.num_layer = 3
            args.drop_ratio = 0.1
            args.batch_size = 32
            args.lr = 1e-2
            if args.gnn == 'gcn-virtual':
                args.lr = 1e-3
            args.patience = 50
        if args.dataset == 'plym-tg_prop':
            # glass temperature
            args.epochs = 400
            args.l2reg = 1e-5 
            args.gamma = 0.05
            args.num_layer = 3
            args.drop_ratio = 0.1
            args.initw_name = 'orthogonal'
            args.batch_size = 256
            args.lr = 1e-2
            args.patience = 50
        if args.dataset == 'plym-density_prop':
            # polymer density
            args.epochs = 400
            args.l2reg = 1e-5
            args.gamma = 0.3
            args.num_layer = 3
            args.drop_ratio = 0.5
            if args.gnn == 'gcn-virtual':
                args.l2reg = 1e-4
            args.batch_size = 32
            args.lr = 1e-3
            args.patience = 50
            args.use_clip_norm = True
        
        if args.dataset == 'ogbg-molhiv':
            args.gamma = 0.1
            args.batch_size = 512
            args.initw_name = 'orthogonal'
            if args.gnn == 'gcn-virtual':
                args.lr = 1e-3
                args.l2reg = 1e-5
                args.epochs = 100
                args.num_layer = 3
                args.use_clip_norm = True
                args.path_list=[2, 4]
        if args.dataset == 'ogbg-molbace':
            if args.gnn == 'gin-virtual' or args.gnn == 'gin':
                args.gnn = 'gin'
                args.l2reg = 7e-4
                args.gamma = 0.55
                args.num_layer = 4  
                args.batch_size = 64
                args.emb_dim = 64
                args.use_lr_scheduler = True
                args.patience = 100
                args.drop_ratio = 0.3
                args.initw_name = 'orthogonal' 
            if args.gnn == 'gcn-virtual' or args.gnn == 'gcn':
                args.gnn = 'gcn'
                args.patience = 100
                args.initw_name = 'orthogonal' 
                args.num_layer = 2
                args.emb_dim = 64
                args.batch_size = 128
        if args.dataset == 'ogbg-molbbbp':
            args.l2reg = 5e-6
            args.initw_name = 'orthogonal'
            args.num_layer = 2
            args.emb_dim = 64
            args.batch_size = 256 
            args.use_lr_scheduler = True 
            args.gamma = 0.2
            if args.gnn == 'gcn-virtual' or args.gnn == 'gcn':
                args.gnn = 'gcn-virtual'
                args.gamma = 0.4
                args.emb_dim = 128
                args.use_lr_scheduler = False 
        if args.dataset == 'ogbg-molsider':
            if args.gnn == 'gin-virtual' or args.gnn == 'gin':
                args.gnn = 'gin'
            if args.gnn == 'gcn-virtual' or args.gnn == 'gcn':
                args.gnn = 'gcn'
            args.l2reg = 1e-4
            args.patience = 100
            args.gamma = 0.65
            args.num_layer =  5
            args.epochs = 400
        if args.dataset == 'ogbg-molclintox':
            if args.gnn == 'gin-virtual' or args.gnn == 'gin':
                args.gnn = 'gin'
            if args.gnn == 'gcn-virtual' or args.gnn == 'gcn':
                args.gnn = 'gcn'
            args.use_linear_predictor = True
            args.use_clip_norm = True
            args.gamma = 0.2
            args.patience = 100
            args.batch_size = 64 
            args.num_layer = 5
            args.emb_dim = 300
            args.l2reg = 1e-4
            args.epochs = 400
            args.drop_ratio=0.5
        if args.dataset == 'ogbg-moltox21':
            args.gamma = 0.8 
        if args.dataset == 'ogbg-moltoxcast':
            if args.gnn == 'gin-virtual' or args.gnn == 'gin':
                args.gnn = 'gin'
            if args.gnn == 'gcn-virtual' or args.gnn == 'gcn':
                args.gnn = 'gcn'
            args.patience = 50
            args.epochs = 150
            args.l2reg = 1e-5
            args.gamma = 0.7
            args.num_layer = 2

        if args.dataset in TU_DATASETS:
            args.gnn = 'gin'
            args.epochs = 200
            args.num_layer = 3
            args.emb_dim = 64
            args.batch_size = 32
            args.lr = 1e-3
            args.l2reg = 1e-5
            args.gamma = 0.4
            args.drop_ratio = 0.3
            args.patience = 50

        if args.dataset == 'mnist':
            args.gnn = 'gin'
            args.epochs = 100
            args.num_layer = 3
            args.emb_dim = 64
            args.batch_size = 128
            args.lr = 1e-3
            args.l2reg = 1e-5
            args.gamma = 0.4
            args.drop_ratio = 0.3
            args.patience = 20

    if args.dataset.startswith('plym'):
        args.plym_prop = args.dataset.split('-')[1].split('_')[0]
    else:
        args.plym_prop = 'none'

    if args.dataset.startswith('ogbg'):
        results = {'valid_auc': [], 'test_auc': []}
    elif args.dataset.startswith('plym'):
        results = {'valid_rmse': [], 'test_rmse': [], 'test_r2':[]}
    else:
        results = {'valid_acc': [], 'test_acc': []}
    for _ in range(args.trails):
        if args.dataset.startswith('plym'):
            valid_rmse, test_rmse, test_r2 = main(args)
            results['test_r2'].append(test_r2)
            results['test_rmse'].append(test_rmse)
            results['valid_rmse'].append(valid_rmse)
        elif args.dataset.startswith('ogbg'):
            valid_auc, test_auc = main(args)
            results['valid_auc'].append(valid_auc)
            results['test_auc'].append(test_auc)
        else:
            valid_acc, test_acc = main(args)
            results['valid_acc'].append(valid_acc)
            results['test_acc'].append(test_acc)
    for mode, nums in results.items():
        print('{}: {:.4f}+-{:.4f} {}'.format(
            mode, np.mean(nums), np.std(nums), nums))

if __name__ == "__main__":
    args = get_args()
    config_and_run(args)

    # current code is built for graph-level regression tasks, but it can be easily adapted for classification tasks by changing the loss function and evaluation metric.
    # torch_geometric.datasets import TUDataset 
    # Sugar datasets might make more sense for testing and comparing ?