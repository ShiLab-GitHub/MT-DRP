import numpy as np
from rdkit import Chem
import networkx as nx
import csv
import argparse
import pickle
from sklearn import preprocessing
import random
import torch.nn as nn
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import os
from torch_geometric.nn import GINConv, SAGEConv, GCNConv, GATConv, JumpingKnowledge, global_max_pool, global_mean_pool
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from math import sqrt
from scipy import stats
import time
import warnings
warnings.filterwarnings("ignore")

# 设置随机种子函数
def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 数据集类
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='set',xds=None,
                 xcm=None, xcc=None, xcg=None, xcr=None, y=None,
                 transform=None, pre_transform=None, smile_graph=None):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xds, xcm, xcc, xcg, xcr, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xds, xcm, xcc, xcg, xcr, y, smile_graph):
        assert (len(xds) == len(xcm)
                and len(xcm) == len(xcc) and len(xcc) == len(xcg) and len(xcg) == len(xcr) and len(xcr) == len(y)), "The five lists must be the same length!"
        data_list = []
        data_len = len(xds)
        for i in range(data_len):
            smiles = xds[i]
            meth = xcm[i]
            copynumber = xcc[i]
            mut = xcg[i]
            RNAseq = xcr[i]
            labels = y[i]
            # Check if smile exists in smile_graph
            if smiles not in smile_graph:
                print(f"Warning: SMILES {smiles} not found in smile_graph, skipping")
                continue
                
            c_size, features, edge_index = smile_graph[smiles]
            
            # Handle empty edge_index case (single atom molecules)
            if len(edge_index) == 0:
                # Create a self-loop for single atom molecules
                edge_index = [[0, 0]]
                
            processedData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            processedData.meth = torch.FloatTensor([meth])
            processedData.copynumber = torch.FloatTensor([copynumber])
            processedData.mut = torch.FloatTensor([mut])
            processedData.RNAseq = torch.FloatTensor([RNAseq])
            processedData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(processedData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [data for data in data_list if self.pre_transform(data)]
            
        if len(data_list) == 0:
            raise ValueError("No valid data found for dataset")
            
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def getXD(self):
        return self.xd


class BBBPDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='bbbp_set', smiles_list=None, labels=None,
                 transform=None, pre_transform=None, smile_graph=None):
        super(BBBPDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed BBBP data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed BBBP data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(smiles_list, labels, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, smiles_list, labels, smile_graph):
        assert len(smiles_list) == len(labels), "SMILES and labels must be the same length!"
        data_list = []
        data_len = len(smiles_list)
        for i in range(data_len):
            smiles = smiles_list[i]
            label = labels[i]
            if smiles in smile_graph:
                c_size, features, edge_index = smile_graph[smiles]
                
                # Handle empty edge_index case (single atom molecules)
                if len(edge_index) == 0:
                    # Create a self-loop for single atom molecules
                    edge_index = [[0, 0]]
                    
                processedData = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([label]))
                processedData.__setitem__('c_size', torch.LongTensor([c_size]))
                data_list.append(processedData)
            else:
                print(f"Warning: SMILES {smiles} not found in smile_graph, skipping")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [data for data in data_list if self.pre_transform(data)]
            
        if len(data_list) == 0:
            raise ValueError(f"No valid BBBP data found for dataset {self.dataset}")
            
        print(f'BBBP graph construction done. Saving to file. Found {len(data_list)} valid samples.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# 评估指标函数
def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse


def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse


def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp


def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def calculate_pcc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.corrcoef(y_true, y_pred)[0, 1]


# ========== MT-DRPNet模型定义 ==========
class MT_DRPNet(torch.nn.Module):
    """MT-DRPNet: A Multi-Task Deep Learning Framework for Joint Prediction of Drug Response and Blood-Brain Barrier Permeability"""
    def __init__(self, n_filters=4, output_dim=256, dropout=0.5, 
                 meth_dim=378, mut_dim=2028, copynumber_dim=512, rnaseq_dim=512,
                 cell_feat_dim=128, cell_conv_dim=64, atom_embedding_dim=128, 
                 drug_gnn_layers=3, drug_gnn_dim=512, drug_gnn_dropout=0.2,
                 bbbp_hidden_dim=256, alpha=1.0, drp_gin_layers=3, bbbp_gin_layers=2,
                 cell_features=None):
        super(MT_DRPNet, self).__init__()
        
        # 默认特征组合：meth + rnaseq (根据PDF中(e)部分的"Optimal: meth + rnaseq")
        if cell_features is None:
            cell_features = ['meth', 'rnaseq']
        self.cell_features = cell_features
        
        # 检查输入特征组合是否有效
        valid_features = ['meth', 'copynumber', 'mut', 'rnaseq']
        for feat in self.cell_features:
            if feat not in valid_features:
                raise ValueError(f"Invalid cell feature: {feat}. Must be one of {valid_features}")
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.layer_drug = drug_gnn_layers
        self.dim_drug = drug_gnn_dim
        self.dropout_ratio = dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_ratio)
        
        # Store dimension parameters
        self.meth_dim = meth_dim
        self.mut_dim = mut_dim
        self.copynumber_dim = copynumber_dim
        self.rnaseq_dim = rnaseq_dim
        self.cell_feat_dim = cell_feat_dim
        self.cell_conv_dim = cell_conv_dim
        self.atom_embedding_dim = atom_embedding_dim
        self.drug_gnn_dropout = drug_gnn_dropout
        self.bbbp_hidden_dim = bbbp_hidden_dim
        self.alpha = alpha
        self.drp_gin_layers = drp_gin_layers
        self.bbbp_gin_layers = bbbp_gin_layers
        
        # 计算GNN输出维度
        self.drp_gnn_output_dim = (drp_gin_layers + 2) * drug_gnn_dim
        self.bbbp_gnn_output_dim = (bbbp_gin_layers + 2) * drug_gnn_dim

        # (a). Shared Multi-Task Molecular Graph Base Feature Extraction
        self.shared_molecular_feature_extractor = SharedMolecularFeatureExtractor(
            self.layer_drug, self.dim_drug, self.atom_embedding_dim, self.drug_gnn_dropout)
        
        # DRP-specific layers - 对应(b)部分的分层多尺度图表示细化器
        self.drp_molecular_refiner = HierarchicalGraphNet(
            self.dim_drug, self.atom_embedding_dim, self.drug_gnn_dropout, self.drp_gin_layers)
        self.drp_drug_emb = nn.Sequential(
            nn.Linear(self.drp_gnn_output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # BBBP-specific layers - 对应(b)部分的分层多尺度图表示细化器
        self.bbbp_molecular_refiner = HierarchicalGraphNet(
            self.dim_drug, self.atom_embedding_dim, self.drug_gnn_dropout, self.bbbp_gin_layers)
        self.bbbp_drug_emb = nn.Sequential(
            nn.Linear(self.bbbp_gnn_output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # (e). DRP Cell Multi-Omics Features Dual-Channel Processing and Fusion Module
        # First channel for each cell feature (3 linear layers) - 根据cell_features初始化
        if 'meth' in self.cell_features:
            self.meth_fc1 = nn.Linear(meth_dim, 512)
            self.meth_fc2 = nn.Linear(512, 256)
            self.meth_fc3 = nn.Linear(256, cell_feat_dim)
            self.meth_bn1 = nn.BatchNorm1d(512)
            self.meth_bn2 = nn.BatchNorm1d(256)
            self.meth_bn3 = nn.BatchNorm1d(cell_feat_dim)
            self.meth_dropout1 = nn.Dropout(p=self.dropout_ratio)
            self.meth_dropout2 = nn.Dropout(p=self.dropout_ratio)
        
        if 'copynumber' in self.cell_features:
            self.copynumber_fc1 = nn.Linear(copynumber_dim, 512)
            self.copynumber_fc2 = nn.Linear(512, 256)
            self.copynumber_fc3 = nn.Linear(256, cell_feat_dim)
            self.copynumber_bn1 = nn.BatchNorm1d(512)
            self.copynumber_bn2 = nn.BatchNorm1d(256)
            self.copynumber_bn3 = nn.BatchNorm1d(cell_feat_dim)
            self.copynumber_dropout1 = nn.Dropout(p=self.dropout_ratio)
            self.copynumber_dropout2 = nn.Dropout(p=self.dropout_ratio)
        
        if 'mut' in self.cell_features:
            self.mut_fc1 = nn.Linear(mut_dim, 512)
            self.mut_fc2 = nn.Linear(512, 256)
            self.mut_fc3 = nn.Linear(256, cell_feat_dim)
            self.mut_bn1 = nn.BatchNorm1d(512)
            self.mut_bn2 = nn.BatchNorm1d(256)
            self.mut_bn3 = nn.BatchNorm1d(cell_feat_dim)
            self.mut_dropout1 = nn.Dropout(p=self.dropout_ratio)
            self.mut_dropout2 = nn.Dropout(p=self.dropout_ratio)
        
        if 'rnaseq' in self.cell_features:
            self.rnaseq_fc1 = nn.Linear(rnaseq_dim, 512)
            self.rnaseq_fc2 = nn.Linear(512, 256)
            self.rnaseq_fc3 = nn.Linear(256, cell_feat_dim)
            self.rnaseq_bn1 = nn.BatchNorm1d(512)
            self.rnaseq_bn2 = nn.BatchNorm1d(256)
            self.rnaseq_bn3 = nn.BatchNorm1d(cell_feat_dim)
            self.rnaseq_dropout1 = nn.Dropout(p=self.dropout_ratio)
            self.rnaseq_dropout2 = nn.Dropout(p=self.dropout_ratio)

        # Second channel for each cell feature (1 linear layer to unified dimension) - 根据cell_features初始化
        if 'meth' in self.cell_features:
            self.meth_fc_single = nn.Linear(meth_dim, cell_conv_dim)
        if 'copynumber' in self.cell_features:
            self.copynumber_fc_single = nn.Linear(copynumber_dim, cell_conv_dim)
        if 'mut' in self.cell_features:
            self.mut_fc_single = nn.Linear(mut_dim, cell_conv_dim)
        if 'rnaseq' in self.cell_features:
            self.rnaseq_fc_single = nn.Linear(rnaseq_dim, cell_conv_dim)
        
        # 2D CNN for processing the concatenated cell features
        # 输入通道数为使用的特征数量
        self.cell_conv1 = nn.Conv2d(1, 16, kernel_size=(2, 2), padding=1)
        self.cell_bn1 = nn.BatchNorm2d(16)
        self.cell_pool1 = nn.MaxPool2d(2)
        self.cell_dropout1 = nn.Dropout2d(p=self.dropout_ratio)
        
        self.cell_conv2 = nn.Conv2d(16, 32, kernel_size=(2, 2), padding=1)
        self.cell_bn2 = nn.BatchNorm2d(32)
        self.cell_pool2 = nn.MaxPool2d(2)
        self.cell_dropout2 = nn.Dropout2d(p=self.dropout_ratio)
        
        self.cell_conv3 = nn.Conv2d(32, 64, kernel_size=(2, 2), padding=1)
        self.cell_bn3 = nn.BatchNorm2d(64)
        self.cell_pool3 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.cell_fc = nn.Linear(64, cell_feat_dim)
        self.cell_bn4 = nn.BatchNorm1d(cell_feat_dim)

        # (f). DRP Fusion Prediction layers with residual connections
        # 输入维度根据使用的特征数量变化
        comb_input_dim = len(self.cell_features) * cell_feat_dim + cell_feat_dim + output_dim
        self.drp_fusion_prediction = DRPFusionPrediction(
            comb_input_dim, self.dropout_ratio)

        # (c). BBBP Prediction head
        self.bbbp_prediction = BBBPPrediction(
            output_dim, self.bbbp_hidden_dim, self.dropout_ratio)

    def forward(self, data, task='drp'):
        # (a). Shared Multi-Task Molecular Graph Base Feature Extraction
        shared_features = self.shared_molecular_feature_extractor(data)
        
        if task == 'drp':
            meth = data.meth
            copynumber = data.copynumber
            mut = data.mut
            RNAseq = data.RNAseq
            
            # (b). Hierarchical Multiscale Graph Representation Refiner for DRP
            x_drug = self.drp_molecular_refiner(shared_features, data)
            x_drug = self.drp_drug_emb(x_drug)

            # (e). DRP Cell Multi-Omics Features Dual-Channel Processing and Fusion Module
            # First channel for each cell feature - 只处理使用的特征
            first_channel_features = []
            
            if 'meth' in self.cell_features:
                xcm1 = self.meth_fc1(meth)
                xcm1 = self.meth_bn1(xcm1)
                xcm1 = self.relu(xcm1)
                xcm1 = self.meth_dropout1(xcm1)
                
                xcm1 = self.meth_fc2(xcm1)
                xcm1 = self.meth_bn2(xcm1)
                xcm1 = self.relu(xcm1)
                xcm1 = self.meth_dropout2(xcm1)
                
                xcm1 = self.meth_fc3(xcm1)
                xcm1 = self.meth_bn3(xcm1)
                xcm1 = self.relu(xcm1)
                first_channel_features.append(xcm1)
            
            if 'copynumber' in self.cell_features:
                xcc1 = self.copynumber_fc1(copynumber)
                xcc1 = self.copynumber_bn1(xcc1)
                xcc1 = self.relu(xcc1)
                xcc1 = self.copynumber_dropout1(xcc1)
                
                xcc1 = self.copynumber_fc2(xcc1)
                xcc1 = self.copynumber_bn2(xcc1)
                xcc1 = self.relu(xcc1)
                xcc1 = self.copynumber_dropout2(xcc1)
                
                xcc1 = self.copynumber_fc3(xcc1)
                xcc1 = self.copynumber_bn3(xcc1)
                xcc1 = self.relu(xcc1)
                first_channel_features.append(xcc1)
            
            if 'mut' in self.cell_features:
                xcg1 = self.mut_fc1(mut)
                xcg1 = self.mut_bn1(xcg1)
                xcg1 = self.relu(xcg1)
                xcg1 = self.mut_dropout1(xcg1)
                
                xcg1 = self.mut_fc2(xcg1)
                xcg1 = self.mut_bn2(xcg1)
                xcg1 = self.relu(xcg1)
                xcg1 = self.mut_dropout2(xcg1)
                
                xcg1 = self.mut_fc3(xcg1)
                xcg1 = self.mut_bn3(xcg1)
                xcg1 = self.relu(xcg1)
                first_channel_features.append(xcg1)
            
            if 'rnaseq' in self.cell_features:
                xcr1 = self.rnaseq_fc1(RNAseq)
                xcr1 = self.rnaseq_bn1(xcr1)
                xcr1 = self.relu(xcr1)
                xcr1 = self.rnaseq_dropout1(xcr1)
                
                xcr1 = self.rnaseq_fc2(xcr1)
                xcr1 = self.rnaseq_bn2(xcr1)
                xcr1 = self.relu(xcr1)
                xcr1 = self.rnaseq_dropout2(xcr1)
                
                xcr1 = self.rnaseq_fc3(xcr1)
                xcr1 = self.rnaseq_bn3(xcr1)
                xcr1 = self.relu(xcr1)
                first_channel_features.append(xcr1)
            
            # Second channel for each cell feature - 只处理使用的特征
            second_channel_features = []
            
            if 'meth' in self.cell_features:
                xcm2 = self.meth_fc_single(meth)
                xcm2 = self.relu(xcm2)
                second_channel_features.append(xcm2)
            
            if 'copynumber' in self.cell_features:
                xcc2 = self.copynumber_fc_single(copynumber)
                xcc2 = self.relu(xcc2)
                second_channel_features.append(xcc2)
            
            if 'mut' in self.cell_features:
                xcg2 = self.mut_fc_single(mut)
                xcg2 = self.relu(xcg2)
                second_channel_features.append(xcg2)
            
            if 'rnaseq' in self.cell_features:
                xcr2 = self.rnaseq_fc_single(RNAseq)
                xcr2 = self.relu(xcr2)
                second_channel_features.append(xcr2)
            
            # Concatenate second channel features and reshape for 2D CNN
            if second_channel_features:
                cell_features_2d = torch.stack(second_channel_features, dim=2)
                cell_features_2d = cell_features_2d.unsqueeze(1)
                
                # Apply 2D CNN
                x_cell_2d = self.cell_conv1(cell_features_2d)
                x_cell_2d = self.cell_bn1(x_cell_2d)
                x_cell_2d = self.relu(x_cell_2d)
                x_cell_2d = self.cell_dropout1(x_cell_2d)
                x_cell_2d = self.cell_pool1(x_cell_2d)
                
                x_cell_2d = self.cell_conv2(x_cell_2d)
                x_cell_2d = self.cell_bn2(x_cell_2d)
                x_cell_2d = self.relu(x_cell_2d)
                x_cell_2d = self.cell_dropout2(x_cell_2d)
                x_cell_2d = self.cell_pool2(x_cell_2d)
                
                x_cell_2d = self.cell_conv3(x_cell_2d)
                x_cell_2d = self.cell_bn3(x_cell_2d)
                x_cell_2d = self.relu(x_cell_2d)
                x_cell_2d = self.cell_pool3(x_cell_2d)
                
                x_cell_2d = x_cell_2d.view(x_cell_2d.size(0), -1)
                x_cell_2d = self.cell_fc(x_cell_2d)
                x_cell_2d = self.cell_bn4(x_cell_2d)
                x_cell_2d = self.relu(x_cell_2d)
            else:
                x_cell_2d = torch.zeros((meth.size(0), self.cell_feat_dim)).to(meth.device)
            
            # Concatenate first channel features
            if first_channel_features:
                x_cell_1d = torch.cat(first_channel_features, dim=1)
            else:
                x_cell_1d = torch.zeros((meth.size(0), len(self.cell_features) * self.cell_feat_dim)).to(meth.device)
            
            # Concatenate all features
            xfusion = torch.cat([x_drug, x_cell_1d, x_cell_2d], dim=1)
            
            # (f). DRP Fusion Prediction
            out = self.drp_fusion_prediction(xfusion)
            out = self.sigmoid(out)
            out = out.view(-1, 1)
            return out
        
        elif task == 'bbbp':
            # (b). Hierarchical Multiscale Graph Representation Refiner for BBBP
            x_drug = self.bbbp_molecular_refiner(shared_features, data)
            x_drug = self.bbbp_drug_emb(x_drug)
            
            # (c). BBBP Prediction
            out = self.bbbp_prediction(x_drug)
            # 不应用激活函数，返回logits
            return out


# ========== 模型组件定义 ==========
class SharedMolecularFeatureExtractor(torch.nn.Module):
    """(a). Shared Multi-Task Molecular Graph Base Feature Extraction"""
    def __init__(self, layer_drug, dim_drug, atom_embedding_dim, dropout_ratio=0.2):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.atom_embedding_dim = atom_embedding_dim
        self.dropout_ratio = dropout_ratio
        
        # Atom type embedding layer (atom feature bank)
        self.atom_type_embedding = nn.Embedding(44, atom_embedding_dim)
        
        # Linear layer to combine embedded atom type and fixed features
        self.linear = nn.Linear(atom_embedding_dim + 34, dim_drug)
        self.linear_dropout = nn.Dropout(p=self.dropout_ratio)
        
        # Shared GCN layers
        self.gcn1 = GCNConv(dim_drug, self.dim_drug)
        self.gcn2 = GCNConv(self.dim_drug, self.dim_drug)
        self.relu = nn.ReLU()
        self.gcn_dropout = nn.Dropout(p=self.dropout_ratio)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Separate atom type (first 44 dimensions) and fixed features (last 34 dimensions)
        atom_type_one_hot = x[:, :44]
        fixed_features = x[:, 44:]
        
        # Convert one-hot encoded atom types to indices for embedding lookup
        atom_type_indices = torch.argmax(atom_type_one_hot, dim=1)
        
        # Get atom type embeddings from the feature bank
        atom_type_embedded = self.atom_type_embedding(atom_type_indices)
        
        # Concatenate atom type embeddings with fixed features
        x = torch.cat([atom_type_embedded, fixed_features], dim=1)
        
        # Apply linear layer to get the initial node representations
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear_dropout(x)
        
        # Shared GCN processing
        x_gcn = self.gcn1(x, edge_index)
        x_gcn1 = self.relu(x_gcn)
        x_gcn1 = self.gcn_dropout(x_gcn1)
        
        x_gcn2 = self.gcn2(x_gcn1, edge_index)
        x_gcn2 = self.relu(x_gcn2)
        x_gcn2 = self.gcn_dropout(x_gcn2)
        
        return x_gcn2


class HierarchicalGraphNet(torch.nn.Module):
    """Hierarchical Multiscale Graph Representation Refiner with GIN+GRU+JumpingKnowledge"""
    def __init__(self, dim_drug, atom_embedding_dim, dropout_ratio=0.2, num_layers=3):
        super().__init__()
        self.dim_drug = dim_drug
        self.atom_embedding_dim = atom_embedding_dim
        self.dropout_ratio = dropout_ratio
        self.num_layers = num_layers
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        self.gru_cells = torch.nn.ModuleList()
        
        # GIN layers for hierarchical graph processing
        for i in range(self.num_layers):
            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(), 
                                      nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)
            gru_cell = nn.GRUCell(self.dim_drug, self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)
            self.gru_cells.append(gru_cell)

        self.gnn_dropout = nn.Dropout(p=self.dropout_ratio)
        self.layer_norm = nn.LayerNorm(self.dim_drug)

    def forward(self, shared_features, data):
        x, edge_index, batch = shared_features, data.edge_index, data.batch
        
        x_drug_list = []
        x_drug_m = 1
        x_drug_e = 0
        
        # Hierarchical GIN processing with GRU updates
        h = x  # Initial hidden state
        for i in range(len(self.convs_drug)):
            # GIN convolution
            x_gin = F.relu(self.convs_drug[i](h, edge_index))
            x_gin = self.bns_drug[i](x_gin)
            x_gin = self.gnn_dropout(x_gin)
            
            # Use GRU to update node features
            h = self.gru_cells[i](x_gin, h)
            h = self.layer_norm(h)
            
            # Multi-scale feature aggregation
            x_drug_m *= h
            x_drug_e += h
            x_drug_list.append(h)

        # Jumping Knowledge aggregation
        node_representation = self.JK(x_drug_list)
        node_representation = torch.cat([node_representation, x_drug_m, x_drug_e], dim=-1)
        x_drug = global_max_pool(node_representation, batch)
        return x_drug


class DRPFusionPrediction(torch.nn.Module):
    """(f). DRP Fusion Prediction with residual connections"""
    def __init__(self, input_dim, dropout_ratio):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.relu = nn.ReLU()
        
        self.comb_fc1 = nn.Linear(input_dim, 1024)
        self.comb_bn1 = nn.BatchNorm1d(1024)
        self.comb_dropout1 = nn.Dropout(p=self.dropout_ratio)
        
        self.comb_fc2 = nn.Linear(1024, 1024)
        self.comb_bn2 = nn.BatchNorm1d(1024)
        self.comb_dropout2 = nn.Dropout(p=self.dropout_ratio)
        
        self.comb_fc3 = nn.Linear(1024, 1024)
        self.comb_bn3 = nn.BatchNorm1d(1024)
        self.comb_dropout3 = nn.Dropout(p=self.dropout_ratio)
        
        self.comb_fc4 = nn.Linear(1024, 128)
        self.comb_bn4 = nn.BatchNorm1d(128)
        self.comb_dropout4 = nn.Dropout(p=self.dropout_ratio)
        self.comb_out = nn.Linear(128, 1)

    def forward(self, xfusion):
        # Final fusion layers with residual connections
        x1 = self.comb_fc1(xfusion)
        x1 = self.comb_bn1(x1)
        x1 = self.relu(x1)
        x1 = self.comb_dropout1(x1)
        
        x2 = self.comb_fc2(x1)
        x2 = self.comb_bn2(x2)
        x2 = self.relu(x2)
        x2 = self.comb_dropout2(x2)
        
        x2 = x2 + x1  # Residual connection
        
        x3 = self.comb_fc3(x2)
        x3 = self.comb_bn3(x3)
        x3 = self.relu(x3)
        x3 = self.comb_dropout3(x3)
        
        x3 = x3 + x2  # Residual connection
        
        x4 = self.comb_fc4(x3)
        x4 = self.comb_bn4(x4)
        x4 = self.relu(x4)
        x4 = self.comb_dropout4(x4)

        out = self.comb_out(x4)
        return out


class BBBPPrediction(torch.nn.Module):
    """(c). BBBP Prediction head"""
    def __init__(self, input_dim, hidden_dim, dropout_ratio):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.relu = nn.ReLU()
        
        self.bbbp_fc1 = nn.Linear(input_dim, hidden_dim)
        self.bbbp_bn1 = nn.BatchNorm1d(hidden_dim)
        self.bbbp_dropout1 = nn.Dropout(p=self.dropout_ratio)
        
        self.bbbp_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bbbp_bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bbbp_dropout2 = nn.Dropout(p=self.dropout_ratio)
        
        # 输出两个值用于二分类
        self.bbbp_fc3 = nn.Linear(hidden_dim // 2, 2)

    def forward(self, x_drug):
        x_bbbp = self.bbbp_fc1(x_drug)
        x_bbbp = self.bbbp_bn1(x_bbbp)
        x_bbbp = self.relu(x_bbbp)
        x_bbbp = self.bbbp_dropout1(x_bbbp)
        
        x_bbbp = self.bbbp_fc2(x_bbbp)
        x_bbbp = self.bbbp_bn2(x_bbbp)
        x_bbbp = self.relu(x_bbbp)
        x_bbbp = self.bbbp_dropout2(x_bbbp)
        
        # 输出两个值
        out = self.bbbp_fc3(x_bbbp)
        return out


# 训练和评估函数
def train_step_multi_task(model, device, train_loader_drp, train_loader_bbbp, optimizer, epoch, alpha):
    model.train()
    loss_fun_drp = nn.MSELoss()
    loss_fun_bbbp = nn.CrossEntropyLoss()  # 修改为CrossEntropyLoss
    losses_drp = []
    losses_bbbp = []
    total_losses = []

    # Determine the number of batches to use (the larger of the two)
    num_batches = max(len(train_loader_drp), len(train_loader_bbbp))
    
    # Create iterators for both loaders
    drp_iter = iter(train_loader_drp)
    bbbp_iter = iter(train_loader_bbbp)
    
    for batch_idx in range(num_batches):
        optimizer.zero_grad()
        
        # Get DRP batch
        try:
            data_drp = next(drp_iter)
        except StopIteration:
            drp_iter = iter(train_loader_drp)
            data_drp = next(drp_iter)
        
        data_drp = data_drp.to(device)
        output_drp = model(data_drp, task='drp')
        loss_drp = loss_fun_drp(output_drp, data_drp.y.view(-1, 1).float().to(device))
        
        # Get BBBP batch
        try:
            data_bbbp = next(bbbp_iter)
        except StopIteration:
            bbbp_iter = iter(train_loader_bbbp)
            data_bbbp = next(bbbp_iter)
        
        data_bbbp = data_bbbp.to(device)
        output_bbbp = model(data_bbbp, task='bbbp')
        
        # 修改为使用CrossEntropyLoss，需要将标签转换为长整型
        labels_bbbp = data_bbbp.y.view(-1).long()
        loss_bbbp = loss_fun_bbbp(output_bbbp, labels_bbbp)
        
        # Combined loss
        total_loss = loss_drp + alpha * loss_bbbp
        total_loss.backward()
        optimizer.step()
        
        losses_drp.append(loss_drp.item())
        losses_bbbp.append(loss_bbbp.item())
        total_losses.append(total_loss.item())

    return sum(losses_drp) / len(losses_drp), sum(losses_bbbp) / len(losses_bbbp), sum(total_losses) / len(total_losses)


def predict_drp(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data, task='drp')
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def predict_drp_with_probs(model, device, data_loader):
    """预测DRP任务，返回标签、预测值和概率"""
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_probs = torch.Tensor()  # 添加概率输出

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data, task='drp')
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            total_probs = torch.cat((total_probs, output.cpu()), 0)  # 模型输出本身就是概率

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_probs.numpy().flatten()


def predict_bbbp(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_probs = torch.Tensor()

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data, task='bbbp')
            
            # 使用softmax获取概率
            probs = F.softmax(output, dim=1)
            # 取概率最大的类别作为预测
            preds = torch.argmax(probs, dim=1)
            
            total_preds = torch.cat((total_preds, preds.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1).cpu()), 0)
            total_probs = torch.cat((total_probs, probs.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_probs.numpy()


def calculate_drp_auc(y_true, y_pred, threshold=0.5):
    """计算DRP任务的AUC，将连续值二分类"""
    # 将真实值和预测值二分类
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # 计算AUC
    try:
        auc = roc_auc_score(y_true_binary, y_pred)
    except:
        # 如果只有一类，返回0.5
        auc = 0.5
    
    # 计算其他分类指标
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    return auc, accuracy, f1, y_true_binary, y_pred_binary


def train_custom(modeling, train_batch, val_batch, test_batch, lr, epoch_num, cuda_name, i, 
                 fold_results, args, cell_features, combo_name):
    model_st = modeling.__name__
    train_losses_drp = []
    train_losses_bbbp = []
    train_total_losses = []
    val_losses_drp = []
    val_pccs = []
    val_r2s = []
    val_accs_bbbp = []
    val_aucs_bbbp = []

    # Load datasets
    train_data_drp = TestbedDataset(root='data', dataset='train_set{num}'.format(num=i))
    val_data_drp = TestbedDataset(root='data', dataset='val_set{num}'.format(num=i))
    test_data_drp = TestbedDataset(root='data', dataset='test_set{num}'.format(num=i))
    
    # Check if BBBP datasets exist
    bbbp_train_path = os.path.join('data', 'processed', 'bbbp_train.pt')
    bbbp_val_path = os.path.join('data', 'processed', 'bbbp_val.pt')
    bbbp_test_path = os.path.join('data', 'processed', 'bbbp_test.pt')
    
    if os.path.exists(bbbp_train_path) and os.path.exists(bbbp_val_path) and os.path.exists(bbbp_test_path):
        train_data_bbbp = BBBPDataset(root='data', dataset='bbbp_train')
        val_data_bbbp = BBBPDataset(root='data', dataset='bbbp_val')
        test_data_bbbp = BBBPDataset(root='data', dataset='bbbp_test')
        
        # Create data loaders for BBBP
        train_loader_bbbp = DataLoader(train_data_bbbp, batch_size=train_batch, shuffle=True)
        val_loader_bbbp = DataLoader(val_data_bbbp, batch_size=val_batch, shuffle=False)
        test_loader_bbbp = DataLoader(test_data_bbbp, batch_size=test_batch, shuffle=False)
        has_bbbp = True
    else:
        print("BBBP datasets not found, skipping BBBP training")
        has_bbbp = False
        train_loader_bbbp = None
        val_loader_bbbp = None
        test_loader_bbbp = None

    # Create data loaders for DRP
    train_loader_drp = DataLoader(train_data_drp, batch_size=train_batch, shuffle=True)
    val_loader_drp = DataLoader(val_data_drp, batch_size=val_batch, shuffle=False)
    test_loader_drp = DataLoader(test_data_drp, batch_size=test_batch, shuffle=False)

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model with specific feature combination
    model = modeling(
        meth_dim=args.meth_dim,
        mut_dim=args.mut_dim,
        copynumber_dim=args.copynumber_dim,
        rnaseq_dim=args.rnaseq_dim,
        cell_feat_dim=args.cell_feat_dim,
        cell_conv_dim=args.cell_conv_dim,
        atom_embedding_dim=args.atom_embedding_dim,
        drug_gnn_layers=args.drug_gnn_layers,
        drug_gnn_dim=args.drug_gnn_dim,
        dropout=args.dropout,
        drug_gnn_dropout=args.drug_gnn_dropout,
        bbbp_hidden_dim=args.bbbp_hidden_dim,
        alpha=args.alpha,
        drp_gin_layers=args.drp_gin_layers,
        bbbp_gin_layers=args.bbbp_gin_layers,
        cell_features=cell_features
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_mse = float('inf')
    best_pcc = 0
    best_epoch = -1

    # Create directory for this feature combination
    combo_dir = f'result/custom/{combo_name}'
    os.makedirs(combo_dir, exist_ok=True)

    with tqdm(total=epoch_num, desc=f"{combo_name} - Fold {i + 1}/5") as epoch_progress:
        for epoch in range(1, epoch_num + 1):
            # Multi-task training if BBBP data is available
            if has_bbbp:
                train_loss_drp, train_loss_bbbp, train_total_loss = train_step_multi_task(
                    model, device, train_loader_drp, train_loader_bbbp, optimizer, epoch, args.alpha)
            else:
                # Single-task training for DRP only
                model.train()
                loss_fun_drp = nn.MSELoss()
                losses_drp = []
                
                for data_drp in train_loader_drp:
                    optimizer.zero_grad()
                    data_drp = data_drp.to(device)
                    output_drp = model(data_drp, task='drp')
                    loss_drp = loss_fun_drp(output_drp, data_drp.y.view(-1, 1).float().to(device))
                    loss_drp.backward()
                    optimizer.step()
                    losses_drp.append(loss_drp.item())
                
                train_loss_drp = sum(losses_drp) / len(losses_drp)
                train_loss_bbbp = 0
                train_total_loss = train_loss_drp

            # Validation for DRP
            G_val_drp, P_val_drp = predict_drp(model, device, val_loader_drp)
            val_mse = mean_squared_error(G_val_drp, P_val_drp)
            val_pcc = calculate_pcc(G_val_drp, P_val_drp)
            val_r2 = r2_score(G_val_drp, P_val_drp)

            # Validation for BBBP if available
            if has_bbbp:
                G_val_bbbp, P_val_bbbp, Probs_val_bbbp = predict_bbbp(model, device, val_loader_bbbp)
                val_acc_bbbp = accuracy_score(G_val_bbbp, P_val_bbbp)
                # 计算AUC时使用正类的概率（第二类）
                val_auc_bbbp = roc_auc_score(G_val_bbbp, Probs_val_bbbp[:, 1])
            else:
                val_acc_bbbp = 0
                val_auc_bbbp = 0

            # Test for DRP (回归指标)
            G_test_drp, P_test_drp, Probs_test_drp = predict_drp_with_probs(model, device, test_loader_drp)
            test_rmse = rmse(G_test_drp, P_test_drp)
            test_mse = mean_squared_error(G_test_drp, P_test_drp)
            test_pcc = calculate_pcc(G_test_drp, P_test_drp)
            test_spearman = spearman(G_test_drp, P_test_drp)
            test_r2 = r2_score(G_test_drp, P_test_drp)
            
            # 新增：计算DRP二分类指标
            # 使用阈值0.5进行二分类
            test_auc_drp, test_acc_drp, test_f1_drp, G_test_drp_binary, P_test_drp_binary = calculate_drp_auc(G_test_drp, P_test_drp, threshold=0.5)

            # Test for BBBP if available
            if has_bbbp:
                G_test_bbbp, P_test_bbbp, Probs_test_bbbp = predict_bbbp(model, device, test_loader_bbbp)
                test_acc_bbbp = accuracy_score(G_test_bbbp, P_test_bbbp)
                test_auc_bbbp = roc_auc_score(G_test_bbbp, Probs_test_bbbp[:, 1])
                test_f1_bbbp = f1_score(G_test_bbbp, P_test_bbbp)
            else:
                test_acc_bbbp = 0
                test_auc_bbbp = 0
                test_f1_bbbp = 0

            train_losses_drp.append(train_loss_drp)
            train_losses_bbbp.append(train_loss_bbbp)
            train_total_losses.append(train_total_loss)
            val_losses_drp.append(val_mse)
            val_pccs.append(val_pcc)
            val_r2s.append(val_r2)
            val_accs_bbbp.append(val_acc_bbbp)
            val_aucs_bbbp.append(val_auc_bbbp)

            epoch_progress.set_postfix({
                'train_loss_drp': f'{train_loss_drp:.6f}',
                'train_loss_bbbp': f'{train_loss_bbbp:.6f}',
                'val_mse': f'{val_mse:.6f}',
                'val_pcc': f'{val_pcc:.6f}',
                'val_acc_bbbp': f'{val_acc_bbbp:.4f}'
            })
            epoch_progress.update(1)

            if val_mse < best_mse:
                torch.save(model.state_dict(), f'{combo_dir}/model_fold{i + 1}.pt')
                best_epoch = epoch
                best_mse = val_mse
                best_pcc = val_pcc

    print(f"\n===== {combo_name} - Fold {i + 1}/5 Results =====")
    print(f"Best Epoch: {best_epoch}")
    print(f"DRP Test RMSE: {test_rmse:.6f}")
    print(f"DRP Test MSE: {test_mse:.6f}")
    print(f"DRP Test PCC: {test_pcc:.6f}")
    print(f"DRP Test Spearman: {test_spearman:.6f}")
    print(f"DRP Test R²: {test_r2:.6f}")
    print(f"DRP Test AUC (binary): {test_auc_drp:.4f}")
    print(f"DRP Test Accuracy (binary): {test_acc_drp:.4f}")
    print(f"DRP Test F1 (binary): {test_f1_drp:.4f}")
    
    if has_bbbp:
        print(f"BBBP Test Accuracy: {test_acc_bbbp:.4f}")
        print(f"BBBP Test AUC: {test_auc_bbbp:.4f}")
        print(f"BBBP Test F1: {test_f1_bbbp:.4f}")

    # Save results for this fold
    with open(f'{combo_dir}/result_fold{i + 1}.csv', 'w') as f:
        f.write("Task,RMSE,MSE,PCC,Spearman,R2,AUC,Accuracy,F1\n")
        f.write(f"DRP,{test_rmse},{test_mse},{test_pcc},{test_spearman},{test_r2},{test_auc_drp},{test_acc_drp},{test_f1_drp}\n")
        if has_bbbp:
            f.write("\nTask,Accuracy,AUC,F1\n")
            f.write(f"BBBP,{test_acc_bbbp},{test_auc_bbbp},{test_f1_bbbp}\n")

    fold_results.append({
        'fold': i + 1,
        'drp_rmse': test_rmse,
        'drp_mse': test_mse,
        'drp_pcc': test_pcc,
        'drp_spearman': test_spearman,
        'drp_r2': test_r2,
        'drp_auc': test_auc_drp,
        'drp_acc': test_acc_drp,
        'drp_f1': test_f1_drp,
        'bbbp_acc': test_acc_bbbp,
        'bbbp_auc': test_auc_bbbp,
        'bbbp_f1': test_f1_bbbp
    })


# 主函数 - MT-DRPNet模型
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MT-DRPNet: A Multi-Task Deep Learning Framework for Joint Prediction of Drug Response and Blood-Brain Barrier Permeability')
    
    # 实验配置参数
    parser.add_argument('--train_batch', type=int, required=False, default=256, help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=1024, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=1024, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=300, help='Number of epoch')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    
    # 细胞特征组合参数
    parser.add_argument('--cell_features', nargs='+', default=['meth', 'rnaseq'],
                       choices=['meth', 'copynumber', 'mut', 'rnaseq'],
                       help='List of cell features to use (default: meth rnaseq)')
    
    # Dropout参数
    parser.add_argument('--dropout', type=float, default=0.3, help='Main dropout rate')
    parser.add_argument('--drug_gnn_dropout', type=float, default=0.25, help='Drug GNN dropout rate')
    
    # GIN层数参数
    parser.add_argument('--drp_gin_layers', type=int, default=3, help='Number of GIN layers for DRP task')
    parser.add_argument('--bbbp_gin_layers', type=int, default=3, help='Number of GIN layers for BBBP task')
    
    # BBBP参数
    parser.add_argument('--bbbp_hidden_dim', type=int, default=256, help='BBBP隐藏层维度')
    parser.add_argument('--alpha', type=float, default=0.01, help='BBBP损失权重')
    
    # 特征维度参数
    parser.add_argument('--meth_dim', type=int, default=378, help='甲基化特征维度')
    parser.add_argument('--mut_dim', type=int, default=2028, help='突变特征维度')
    parser.add_argument('--copynumber_dim', type=int, default=512, help='拷贝数特征维度')
    parser.add_argument('--rnaseq_dim', type=int, default=512, help='RNAseq特征维度')
    parser.add_argument('--drug_gnn_layers', type=int, default=2, help='药物共享GNN层数')
    parser.add_argument('--drug_gnn_dim', type=int, default=512, help='药物GNN特征维度')
    parser.add_argument('--cell_feat_dim', type=int, default=256, help='细胞特征统一映射的维度')
    parser.add_argument('--cell_conv_dim', type=int, default=512, help='细胞特征卷积前的统一维度')
    parser.add_argument('--atom_embedding_dim', type=int, default=128, help='原子类型嵌入维度')
    
    args = parser.parse_args()

    modeling = MT_DRPNet
    train_batch = args.train_batch
    val_batch = args.val_batch
    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    cuda_name = args.cuda_name
    cell_features = args.cell_features

    # 设置随机种子
    set_seed(args.seed)

    # 创建结果目录
    if not os.path.exists('result/no_SADE'):
        os.makedirs('result/no_SADE')
    if not os.path.exists('result/no_SADE/custom'):
        os.makedirs('result/no_SADE/custom')

    # 生成组合名称
    combo_name = '_'.join(sorted(cell_features))
    
    print(f"\n{'='*80}")
    print(f"MT-DRPNet: A Multi-Task Deep Learning Framework for Joint Prediction of Drug Response and Blood-Brain Barrier Permeability")
    print(f"Starting training with custom cell feature combination")
    print(f"Cell features to use: {cell_features}")
    print(f"Combination name: {combo_name}")
    print(f"Number of features: {len(cell_features)}")
    print(f"{'='*80}")
    
    # 存储当前组合的所有折的结果
    combo_fold_results = []
    
    # 训练每一折
    with tqdm(total=5, desc=f"{combo_name} - 5-Fold CV") as fold_progress:
        for i in range(5):
            print(f"\n===== Starting training for fold {i + 1}/5 =====")
            train_custom(modeling, train_batch, val_batch, test_batch, lr, num_epoch, 
                          cuda_name, i, combo_fold_results, args, cell_features, combo_name)
            fold_progress.update(1)
    
    # 计算当前组合的平均值
    avg_drp_rmse = sum(r['drp_rmse'] for r in combo_fold_results) / len(combo_fold_results)
    avg_drp_mse = sum(r['drp_mse'] for r in combo_fold_results) / len(combo_fold_results)
    avg_drp_pcc = sum(r['drp_pcc'] for r in combo_fold_results) / len(combo_fold_results)
    avg_drp_spearman = sum(r['drp_spearman'] for r in combo_fold_results) / len(combo_fold_results)
    avg_drp_r2 = sum(r['drp_r2'] for r in combo_fold_results) / len(combo_fold_results)
    avg_drp_auc = sum(r['drp_auc'] for r in combo_fold_results) / len(combo_fold_results)
    avg_drp_acc = sum(r['drp_acc'] for r in combo_fold_results) / len(combo_fold_results)
    avg_drp_f1 = sum(r['drp_f1'] for r in combo_fold_results) / len(combo_fold_results)
    
    # Only calculate BBBP averages if we have BBBP results
    bbbp_results = [r for r in combo_fold_results if r['bbbp_acc'] > 0]
    if len(bbbp_results) > 0:
        avg_bbbp_acc = sum(r['bbbp_acc'] for r in bbbp_results) / len(bbbp_results)
        avg_bbbp_auc = sum(r['bbbp_auc'] for r in bbbp_results) / len(bbbp_results)
        avg_bbbp_f1 = sum(r['bbbp_f1'] for r in bbbp_results) / len(bbbp_results)
    else:
        avg_bbbp_acc = 0
        avg_bbbp_auc = 0
        avg_bbbp_f1 = 0
    
    # 保存当前组合的平均结果
    combo_result = {
        'combo_name': combo_name,
        'features': cell_features,
        'num_features': len(cell_features),
        'avg_drp_rmse': avg_drp_rmse,
        'avg_drp_mse': avg_drp_mse,
        'avg_drp_pcc': avg_drp_pcc,
        'avg_drp_spearman': avg_drp_spearman,
        'avg_drp_r2': avg_drp_r2,
        'avg_drp_auc': avg_drp_auc,
        'avg_drp_acc': avg_drp_acc,
        'avg_drp_f1': avg_drp_f1,
        'avg_bbbp_acc': avg_bbbp_acc,
        'avg_bbbp_auc': avg_bbbp_auc,
        'avg_bbbp_f1': avg_bbbp_f1,
        'fold_results': combo_fold_results
    }
    
    # 打印当前组合的结果
    print(f"\n{'='*80}")
    print(f"MT-DRPNet Summary for feature combination: {combo_name}")
    print(f"Features used: {combo_result['features']}")
    print(f"Number of features: {combo_result['num_features']}")
    print(f"DRP Average RMSE: {avg_drp_rmse:.6f}")
    print(f"DRP Average MSE: {avg_drp_mse:.6f}")
    print(f"DRP Average PCC: {avg_drp_pcc:.6f}")
    print(f"DRP Average Spearman: {avg_drp_spearman:.6f}")
    print(f"DRP Average R²: {avg_drp_r2:.6f}")
    print(f"DRP Average AUC: {avg_drp_auc:.4f}")
    print(f"DRP Average Accuracy: {avg_drp_acc:.4f}")
    print(f"DRP Average F1: {avg_drp_f1:.4f}")
    
    if avg_bbbp_acc > 0:
        print(f"BBBP Average Accuracy: {avg_bbbp_acc:.4f}")
        print(f"BBBP Average AUC: {avg_bbbp_auc:.4f}")
        print(f"BBBP Average F1: {avg_bbbp_f1:.4f}")
    print(f"{'='*80}")
    
    # 保存当前组合的详细结果
    combo_dir = f'result/custom/{combo_name}'
    
    # 保存平均结果
    with open(f'{combo_dir}/average_results.csv', 'w') as f:
        f.write("Feature Combination,Features,RMSE,MSE,PCC,Spearman,R2,AUC,Accuracy,F1,BBBP_Accuracy,BBBP_AUC,BBBP_F1\n")
        f.write(f"{combo_name},{'-'.join(combo_result['features'])},{avg_drp_rmse},{avg_drp_mse},{avg_drp_pcc},{avg_drp_spearman},{avg_drp_r2},{avg_drp_auc},{avg_drp_acc},{avg_drp_f1},{avg_bbbp_acc},{avg_bbbp_auc},{avg_bbbp_f1}\n")
    
    # 保存当前组合的所有折的结果
    with open(f'{combo_dir}/all_folds_results.csv', 'w') as f:
        f.write("Fold,RMSE,MSE,PCC,Spearman,R2,AUC,Accuracy,F1,BBBP_Accuracy,BBBP_AUC,BBBP_F1\n")
        for fold_result in combo_fold_results:
            f.write(f"{fold_result['fold']},{fold_result['drp_rmse']},{fold_result['drp_mse']},{fold_result['drp_pcc']},{fold_result['drp_spearman']},{fold_result['drp_r2']},{fold_result['drp_auc']},{fold_result['drp_acc']},{fold_result['drp_f1']},{fold_result['bbbp_acc']},{fold_result['bbbp_auc']},{fold_result['bbbp_f1']}\n")
    
    # 保存详细配置
    with open(f'{combo_dir}/config.txt', 'w') as f:
        f.write("MT-DRPNet Configuration Summary\n")
        f.write("================================\n")
        f.write(f"Model: MT-DRPNet: A Multi-Task Deep Learning Framework for Joint Prediction of Drug Response and Blood-Brain Barrier Permeability\n")
        f.write(f"Cell Features: {cell_features}\n")
        f.write(f"Combination Name: {combo_name}\n")
        f.write(f"Number of Features: {len(cell_features)}\n")
        f.write(f"\nTraining Parameters:\n")
        f.write(f"  Train Batch Size: {train_batch}\n")
        f.write(f"  Validation Batch Size: {val_batch}\n")
        f.write(f"  Test Batch Size: {test_batch}\n")
        f.write(f"  Learning Rate: {lr}\n")
        f.write(f"  Number of Epochs: {num_epoch}\n")
        f.write(f"  Dropout Rate: {args.dropout}\n")
        f.write(f"  Drug GNN Dropout: {args.drug_gnn_dropout}\n")
        f.write(f"  Random Seed: {args.seed}\n")
        f.write(f"\nModel Architecture:\n")
        f.write(f"  (a) Shared Molecular Graph Base Feature Extractor\n")
        f.write(f"  (b) HierarchicalGraphNet (GIN+GRU+JumpingKnowledge)\n")
        f.write(f"  (c) BBBP Prediction Head\n")
        f.write(f"  (e) DRP Cell Multi-Omics Features Dual-Channel Processing and Fusion Module\n")
        f.write(f"  (f) DRP Fusion Prediction with Residual Connections\n")
        f.write(f"\nFeature Dimensions:\n")
        f.write(f"  Meth Dimension: {args.meth_dim}\n")
        f.write(f"  Mut Dimension: {args.mut_dim}\n")
        f.write(f"  Copynumber Dimension: {args.copynumber_dim}\n")
        f.write(f"  RNAseq Dimension: {args.rnaseq_dim}\n")
        f.write(f"  Cell Feature Dimension: {args.cell_feat_dim}\n")
        f.write(f"  Cell Conv Dimension: {args.cell_conv_dim}\n")
        f.write(f"  Atom Embedding Dimension: {args.atom_embedding_dim}\n")
        f.write(f"  Drug GNN Layers: {args.drug_gnn_layers}\n")
        f.write(f"  Drug GNN Dimension: {args.drug_gnn_dim}\n")
        f.write(f"  DRP GIN Layers: {args.drp_gin_layers}\n")
        f.write(f"  BBBP GIN Layers: {args.bbbp_gin_layers}\n")
        f.write(f"  BBBP Hidden Dimension: {args.bbbp_hidden_dim}\n")
        f.write(f"  BBBP Loss Weight (alpha): {args.alpha}\n")
    
    # 保存所有结果汇总
    with open(f'result/custom/all_combinations_summary.csv', 'a') as f:
        # 如果文件不存在，先写表头
        if os.path.getsize(f'result/custom/all_combinations_summary.csv') == 0:
            f.write("Combination,Features,NumFeatures,RMSE,MSE,PCC,Spearman,R2,AUC,Accuracy,F1,BBBP_Accuracy,BBBP_AUC,BBBP_F1\n")
        f.write(f"{combo_name},{'-'.join(cell_features)},{len(cell_features)},"
               f"{avg_drp_rmse},{avg_drp_mse},{avg_drp_pcc},"
               f"{avg_drp_spearman},{avg_drp_r2},{avg_drp_auc},"
               f"{avg_drp_acc},{avg_drp_f1},{avg_bbbp_acc},"
               f"{avg_bbbp_auc},{avg_bbbp_f1}\n")
    
    print(f"\nMT-DRPNet training completed!")
    print(f"Results saved in: {combo_dir}")
    print(f"Configuration saved in: {combo_dir}/config.txt")
    print(f"All combinations summary saved in: result/custom/all_combinations_summary.csv")
    print(f"\nTo run with different feature combinations, use:")
    print(f"  python script.py --cell_features meth copynumber")
    print(f"  python script.py --cell_features meth mut rnaseq")
    print(f"  python script.py --cell_features meth copynumber mut rnaseq")
    print(f"\nMT-DRPNet successfully implemented and trained!")
