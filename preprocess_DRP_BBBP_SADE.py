import csv
from smiles2graph import smile_to_graph
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random
import numpy as np
from functions import TestbedDataset
import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from rdkit import Chem
import argparse


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
            if smiles in smile_graph:  # Only process if we have a graph for this smile
                c_size, features, edge_index = smile_graph[smiles]
                
                # Handle empty edge_index case (single atom molecules)
                if len(edge_index) == 0:
                    # Create a self-loop for single atom molecules
                    edge_index = [[0, 0]]  # Fixed: Use [0, 0] instead of [[0], [0]]
                    
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
            data_list = [self.pre_transform(data) for data in data_list]
            
        if len(data_list) == 0:
            raise ValueError(f"No valid BBBP data found for dataset {self.dataset}")
            
        print(f'BBBP graph construction done. Saving to file. Found {len(data_list)} valid samples.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


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


def is_valid_smiles(smiles):
    """检查SMILES字符串是否有效"""
    if not smiles or len(smiles) == 0:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def read_drug_list(filename, seed=42):
    """加载药物及其物理化学性质，按固定顺序排序以确保可复现性"""
    set_seed(seed)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        lines = [line for line in reader]
    
    # 按药物名称排序以确保可复现性
    lines.sort(key=lambda x: x[3])
    
    drug_dict = {}
    for index, line in enumerate(lines):
        drug_dict[line[3]] = index
    
    return drug_dict


def read_drug_smiles(filename, drug_dict, seed=42):
    """加载药物的SMILES，按固定顺序处理"""
    set_seed(seed)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        lines = [line for line in reader]
    
    # 按药物名称排序以确保可复现性
    lines.sort(key=lambda x: x[3])
    
    drug_smiles = [""] * len(drug_dict)
    for line in lines:
        if line[3] in drug_dict:
            drug_smiles[drug_dict[line[3]]] = line[9]
    
    return drug_smiles


def read_cell_line_list(filename, seed=42):
    """加载细胞系并构建字典，按固定顺序排序"""
    set_seed(seed)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        lines = [line for line in reader]
    
    # 按细胞系名称排序以确保可复现性
    lines.sort(key=lambda x: x[0])
    
    cell_line_dict = {}
    for index, line in enumerate(lines):
        cell_line_dict[line[0]] = index
    
    return cell_line_dict


def read_cell_line_mut(filename, cell_line_dict, seed=42):
    """加载细胞系的mut特征，按固定顺序处理"""
    set_seed(seed)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        lines = [line for line in reader if line[0] in cell_line_dict]
    
    # 按细胞系名称排序以确保可复现性
    lines.sort(key=lambda x: x[0])
    
    mut = [list() for _ in range(len(cell_line_dict))]
    for line in lines:
        idx = cell_line_dict[line[0]]
        # 处理空字符串，将其转换为0.0
        values = []
        for x in line[1:]:
            if x == '':
                values.append(0.0)
            else:
                try:
                    values.append(float(x))
                except ValueError:
                    print(f"Warning: Invalid value '{x}' found in MUT data for cell line {line[0]}, replaced with 0.0")
                    values.append(0.0)
        mut[idx] = values
    
    return mut


def read_cell_line_meth(filename, cell_line_dict, seed=42):
    """加载细胞系的meth特征，按固定顺序处理"""
    set_seed(seed)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        lines = [line for line in reader if line[0] in cell_line_dict]
    
    # 按细胞系名称排序以确保可复现性
    lines.sort(key=lambda x: x[0])
    
    meth = [list() for _ in range(len(cell_line_dict))]
    for line in lines:
        idx = cell_line_dict[line[0]]
        # 处理空字符串，将其转换为0.0
        values = []
        for x in line[1:]:
            if x == '':
                values.append(0.0)
            else:
                try:
                    values.append(float(x))
                except ValueError:
                    print(f"Warning: Invalid value '{x}' found in METH data for cell line {line[0]}, replaced with 0.0")
                    values.append(0.0)
        meth[idx] = values
    
    return meth


def read_cell_line_cnv_from_csv(filename, cell_line_dict, seed=42):
    """从CSV文件加载细胞系的CNV特征，按固定顺序处理"""
    set_seed(seed)
    
    # 先确定特征维度
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) > 1:
            expected_dim = len(header) - 1  # 减去细胞系名称列
        else:
            # 如果没有特征列，设置为0
            expected_dim = 0
    
    print(f"CNV特征预期维度: {expected_dim}")
    
    # 读取所有行
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        lines = []
        for line in reader:
            if line[0] in cell_line_dict:
                lines.append(line)
    
    # 按细胞系名称排序以确保可复现性
    lines.sort(key=lambda x: x[0])
    
    cnv = [list() for _ in range(len(cell_line_dict))]
    
    for line in lines:
        idx = cell_line_dict[line[0]]
        # 处理空字符串，将其转换为0.0
        values = []
        for i, x in enumerate(line[1:]):
            if x == '':
                values.append(0.0)
            else:
                try:
                    values.append(float(x))
                except ValueError:
                    print(f"Warning: Invalid value '{x}' found in CNV data for cell line {line[0]}, replaced with 0.0")
                    values.append(0.0)
        
        # 检查特征维度是否匹配
        if len(values) != expected_dim:
            print(f"Warning: CNV data for cell line {line[0]} has {len(values)} features, expected {expected_dim}")
            # 如果特征数量不足，用0补齐
            if len(values) < expected_dim:
                values.extend([0.0] * (expected_dim - len(values)))
            # 如果特征数量过多，截断
            else:
                values = values[:expected_dim]
        
        cnv[idx] = values
    
    # 检查是否有细胞系没有数据
    empty_cell_lines = []
    for cell_line, idx in cell_line_dict.items():
        if len(cnv[idx]) == 0:
            empty_cell_lines.append(cell_line)
    
    if empty_cell_lines:
        print(f"Warning: {len(empty_cell_lines)} cell lines have no CNV data: {empty_cell_lines}")
        # 为没有数据的细胞系填充0值
        for cell_line in empty_cell_lines:
            idx = cell_line_dict[cell_line]
            cnv[idx] = [0.0] * expected_dim
    
    # 将列表转换为numpy数组
    cnv_array = np.array(cnv, dtype=np.float32)
    print(f"CNV数据形状: {cnv_array.shape}")
    
    return cnv_array


def read_cell_line_rnaseq_from_csv(filename, cell_line_dict, seed=42):
    """从CSV文件加载细胞系的RNAseq特征，按固定顺序处理"""
    set_seed(seed)
    
    # 先确定特征维度
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) > 1:
            expected_dim = len(header) - 1  # 减去细胞系名称列
        else:
            # 如果没有特征列，设置为0
            expected_dim = 0
    
    print(f"RNAseq特征预期维度: {expected_dim}")
    
    # 读取所有行
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        lines = []
        for line in reader:
            if line[0] in cell_line_dict:
                lines.append(line)
    
    # 按细胞系名称排序以确保可复现性
    lines.sort(key=lambda x: x[0])
    
    rnaseq = [list() for _ in range(len(cell_line_dict))]
    
    for line in lines:
        idx = cell_line_dict[line[0]]
        # 处理空字符串，将其转换为0.0
        values = []
        for i, x in enumerate(line[1:]):
            if x == '':
                values.append(0.0)
            else:
                try:
                    values.append(float(x))
                except ValueError:
                    print(f"Warning: Invalid value '{x}' found in RNAseq data for cell line {line[0]}, replaced with 0.0")
                    values.append(0.0)
        
        # 检查特征维度是否匹配
        if len(values) != expected_dim:
            print(f"Warning: RNAseq data for cell line {line[0]} has {len(values)} features, expected {expected_dim}")
            # 如果特征数量不足，用0补齐
            if len(values) < expected_dim:
                values.extend([0.0] * (expected_dim - len(values)))
            # 如果特征数量过多，截断
            else:
                values = values[:expected_dim]
        
        rnaseq[idx] = values
    
    # 检查是否有细胞系没有数据
    empty_cell_lines = []
    for cell_line, idx in cell_line_dict.items():
        if len(rnaseq[idx]) == 0:
            empty_cell_lines.append(cell_line)
    
    if empty_cell_lines:
        print(f"Warning: {len(empty_cell_lines)} cell lines have no RNAseq data: {empty_cell_lines}")
        # 为没有数据的细胞系填充0值
        for cell_line in empty_cell_lines:
            idx = cell_line_dict[cell_line]
            rnaseq[idx] = [0.0] * expected_dim
    
    # 将列表转换为numpy数组
    rnaseq_array = np.array(rnaseq, dtype=np.float32)
    print(f"RNAseq数据形状: {rnaseq_array.shape}")
    
    return rnaseq_array


def min_max_normalization(data, min_val=None, max_val=None):
    """Min-max标准化"""
    if min_val is None:
        min_val = min(data)
    if max_val is None:
        max_val = max(data)
    
    return [(x - min_val) / (max_val - min_val) for x in data]


def get_all_graph(drug_smiles, seed=42):
    """为所有药物SMILES生成图表示"""
    set_seed(seed)
    smile_graph = {}
    # 先过滤掉无效的SMILES
    valid_smiles = [s for s in drug_smiles if is_valid_smiles(s)]
    print(f"Found {len(valid_smiles)} valid SMILES out of {len(drug_smiles)} total")
    
    # 按SMILES排序以确保可复现性
    sorted_smiles = sorted(valid_smiles)
    for smile in sorted_smiles:
        try:
            graph = smile_to_graph(smile)
            if graph is not None:
                smile_graph[smile] = graph
            else:
                print(f"Failed to generate graph for SMILES: {smile}")
        except Exception as e:
            print(f"Error processing SMILES {smile}: {e}")
    
    return smile_graph


def read_bbbp_data(bbbp_file, smile_graph, seed=42):
    """读取BBBP数据集并进行分层划分，确保正负样本比例一致"""
    set_seed(seed)
    
    with open(bbbp_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        smiles_list = []
        labels = []
        valid_smiles = set(smile_graph.keys())
        
        for line in reader:
            if len(line) < 2:
                continue
                
            smile = line[0].strip()
            label = int(line[1])
            
            # 只包含可以处理的SMILES
            if smile in valid_smiles:
                smiles_list.append(smile)
                labels.append(label)
            else:
                print(f"BBBP SMILES {smile} not in smile_graph, skipping")
    
    print(f"Loaded {len(smiles_list)} valid BBBP samples")
    
    if len(smiles_list) == 0:
        raise ValueError("No valid BBBP samples found!")
    
    # 分层划分数据集，确保正负样本比例一致
    # 首先划分训练集和临时测试集（80%/20%）
    smiles_train, smiles_temp, labels_train, labels_temp = train_test_split(
        smiles_list, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    
    # 然后将临时测试集划分为验证集和测试集（各占10%）
    smiles_val, smiles_test, labels_val, labels_test = train_test_split(
        smiles_temp, labels_temp, test_size=0.5, stratify=labels_temp, random_state=seed
    )
    
    # 打印每个集合的正负样本比例
    def print_class_ratio(name, labels):
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        print(f"{name}: {len(labels)} samples, positive: {pos_count} ({pos_count/len(labels)*100:.2f}%), "
              f"negative: {neg_count} ({neg_count/len(labels)*100:.2f}%)")
    
    print_class_ratio("BBBP Training set", labels_train)
    print_class_ratio("BBBP Validation set", labels_val)
    print_class_ratio("BBBP Test set", labels_test)
    
    return smiles_train, smiles_val, smiles_test, labels_train, labels_val, labels_test


def read_response_data_and_process(filename, bbbp_file=None, cnv_file=None, rnaseq_file=None, 
                                  use_pretrained_cnv=True, use_pretrained_rnaseq=True, seed=42):
    """读取响应数据并进行处理，确保完全可复现"""
    set_seed(seed)
    
    # 加载特征
    drug_dict = read_drug_list('data/drug/smile_inchi.csv', seed)
    smile = read_drug_smiles('data/drug/smile_inchi.csv', drug_dict, seed)
    
    # 读取BBBP数据（如果提供了文件路径）
    bbbp_smiles_all = []
    if bbbp_file and os.path.exists(bbbp_file):
        # 先读取BBBP文件获取所有SMILES
        with open(bbbp_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for line in reader:
                if len(line) > 0:
                    bbbp_smiles_all.append(line[0].strip())
    
    # 合并药物和BBBP的SMILES来构建图
    all_smiles = list(smile) + bbbp_smiles_all
    smile_graph = get_all_graph(all_smiles, seed)
    
    cell_line_dict = read_cell_line_list('data/cellline/cellline_listwithACH_80cellline.csv', seed)
    
    print(f"读取细胞系特征...")
    meth = read_cell_line_meth('data/cellline/METH_84cellline_378dim.csv', cell_line_dict, seed)
    mut = read_cell_line_mut('data/cellline/MUT_85dim_2028dim.csv', cell_line_dict, seed)
    
    # 将meth和mut转换为numpy数组
    meth = np.array(meth, dtype=np.float32)
    mut = np.array(mut, dtype=np.float32)
    print(f"METH数据形状: {meth.shape}")
    print(f"MUT数据形状: {mut.shape}")
    
    # 加载CNV特征：可以选择使用预训练好的pkl文件或从CSV读取
    # 使用预训练脚本使用的CNV文件路径
    cnv_csv_path = cnv_file if cnv_file else 'data/cellline/CNV_714cellline_24254dim_shu.csv'
    
    if use_pretrained_cnv:
        print("Loading CNV features from pretrained pkl file...")
        copynumber = pickle.load(open('data/cellline/512dim_copynumber.pkl', 'rb'))
    elif cnv_csv_path and os.path.exists(cnv_csv_path):
        print(f"Loading CNV features from CSV file: {cnv_csv_path}")
        copynumber = read_cell_line_cnv_from_csv(cnv_csv_path, cell_line_dict, seed)
    else:
        raise ValueError(f"CNV feature file not found: {cnv_csv_path}")
    
    # 加载RNAseq特征：可以选择使用预训练好的pkl文件或从CSV读取
    # 使用预训练脚本使用的RNAseq文件路径
    rnaseq_csv_path = rnaseq_file if rnaseq_file else 'data/cellline/903cellline_17737dim_scRNAseq.csv'
    
    if use_pretrained_rnaseq:
        print("Loading RNAseq features from pretrained pkl file...")
        RNAseq = pickle.load(open('data/cellline/512dim_RNAseq.pkl', 'rb'))
    elif rnaseq_csv_path and os.path.exists(rnaseq_csv_path):
        print(f"Loading RNAseq features from CSV file: {rnaseq_csv_path}")
        RNAseq = read_cell_line_rnaseq_from_csv(rnaseq_csv_path, cell_line_dict, seed)
    else:
        raise ValueError(f"RNAseq feature file not found: {rnaseq_csv_path}")
    
    print(f"所有特征数据形状:")
    print(f"  METH: {meth.shape}")
    print(f"  MUT: {mut.shape}")
    print(f"  CNV: {copynumber.shape}")
    print(f"  RNAseq: {RNAseq.shape}")
    
    # 特征归一化
    print("归一化特征...")
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    meth = min_max_scaler.fit_transform(meth)
    mut = min_max_scaler.fit_transform(mut)
    
    # 只有在从CSV读取时才需要归一化，因为pkl文件已经处理过了
    if not use_pretrained_cnv:
        copynumber = min_max_scaler.fit_transform(copynumber)
    else:
        # 即使是从pkl加载，也确保进行归一化
        copynumber = min_max_scaler.fit_transform(copynumber)
    
    if not use_pretrained_rnaseq:
        RNAseq = min_max_scaler.fit_transform(RNAseq)
    else:
        # 即使是从pkl加载，也确保进行归一化
        RNAseq = min_max_scaler.fit_transform(RNAseq)
    
    # 读取响应数据
    print("读取响应数据...")
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        data = []
        for line in reader:
            drug = line[0]
            cell_line = line[2]
            ic50 = float(line[7])
            data.append((drug, cell_line, ic50))
    
    # 按药物和细胞系排序以确保可复现性
    data.sort(key=lambda x: (x[0], x[1]))
    
    # 使用固定种子打乱数据
    random.Random(seed).shuffle(data)
    
    # 匹配特征和标签
    drug_smile = []
    cell_meth = []
    cell_copy = []
    cell_mut = []
    cell_RNAseq = []
    label = []
    
    for item in data:
        drug, cell_line, ic50 = item
        if drug in drug_dict and cell_line in cell_line_dict:
            drug_smile.append(smile[drug_dict[drug]])
            cell_meth.append(meth[cell_line_dict[cell_line]])
            cell_copy.append(copynumber[cell_line_dict[cell_line]])
            cell_mut.append(mut[cell_line_dict[cell_line]])
            cell_RNAseq.append(RNAseq[cell_line_dict[cell_line]])
            label.append(ic50)
        else:
            print(f"Warning: Drug {drug} or cell line {cell_line} not found in dictionaries")
    
    print(f"匹配到 {len(drug_smile)} 个样本")
    
    if len(drug_smile) == 0:
        raise ValueError("No matching data found!")
    
    # 标签归一化
    label = min_max_normalization(label)
    
    # 转换为numpy数组
    drug_smile = np.array(drug_smile)
    cell_meth = np.array(cell_meth)
    cell_copy = np.array(cell_copy)
    cell_mut = np.array(cell_mut)
    cell_RNAseq = np.array(cell_RNAseq)
    label = np.array(label)
    
    # 创建5折交叉验证分割
    print("创建5折交叉验证数据集...")
    for i in range(5):
        total_size = drug_smile.shape[0]
        size_0 = int(total_size * 0.2 * i)
        size_1 = size_0 + int(total_size * 0.1)
        size_2 = int(total_size * 0.2 * (i + 1))
        
        # 分割特征
        drugsmile_test = drug_smile[size_0:size_1]
        drugsmile_val = drug_smile[size_1:size_2]
        drugsmile_train = np.concatenate((drug_smile[:size_0], drug_smile[size_2:]), axis=0)
        
        cellmeth_test = cell_meth[size_0:size_1]
        cellmeth_val = cell_meth[size_1:size_2]
        cellmeth_train = np.concatenate((cell_meth[:size_0], cell_meth[size_2:]), axis=0)
        
        cellcopy_test = cell_copy[size_0:size_1]
        cellcopy_val = cell_copy[size_1:size_2]
        cellcopy_train = np.concatenate((cell_copy[:size_0], cell_copy[size_2:]), axis=0)
        
        cellmut_test = cell_mut[size_0:size_1]
        cellmut_val = cell_mut[size_1:size_2]
        cellmut_train = np.concatenate((cell_mut[:size_0], cell_mut[size_2:]), axis=0)
        
        cellRNAseq_test = cell_RNAseq[size_0:size_1]
        cellRNAseq_val = cell_RNAseq[size_1:size_2]
        cellRNAseq_train = np.concatenate((cell_RNAseq[:size_0], cell_RNAseq[size_2:]), axis=0)
        
        # 分割标签
        label_test = label[size_0:size_1]
        label_val = label[size_1:size_2]
        label_train = np.concatenate((label[:size_0], label[size_2:]), axis=0)
        
        # 创建数据集
        TestbedDataset(root='data', dataset=f'train_set{i}',
                       xds=drugsmile_train,
                       xcm=cellmeth_train, xcc=cellcopy_train, xcg=cellmut_train, xcr=cellRNAseq_train,
                       y=label_train, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset=f'val_set{i}',
                       xds=drugsmile_val,
                       xcm=cellmeth_val, xcc=cellcopy_val, xcg=cellmut_val, xcr=cellRNAseq_val,
                       y=label_val, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset=f'test_set{i}',
                       xds=drugsmile_test,
                       xcm=cellmeth_test, xcc=cellcopy_test, xcg=cellmut_test, xcr=cellRNAseq_test,
                       y=label_test, smile_graph=smile_graph)
    
    # 处理BBBP数据（如果提供了文件路径）
    if bbbp_file and os.path.exists(bbbp_file):
        bbbp_smiles_train, bbbp_smiles_val, bbbp_smiles_test, bbbp_labels_train, bbbp_labels_val, bbbp_labels_test = read_bbbp_data(bbbp_file, smile_graph, seed)
        
        # 创建BBBP数据集
        BBBPDataset(root='data', dataset=f'bbbp_train', 
                    smiles_list=bbbp_smiles_train, labels=bbbp_labels_train, smile_graph=smile_graph)
        BBBPDataset(root='data', dataset=f'bbbp_val', 
                    smiles_list=bbbp_smiles_val, labels=bbbp_labels_val, smile_graph=smile_graph)
        BBBPDataset(root='data', dataset=f'bbbp_test', 
                    smiles_list=bbbp_smiles_test, labels=bbbp_labels_test, smile_graph=smile_graph)
    
    print("Data processing completed successfully!")
    return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process drug response data with optional feature sources')
    
    # 数据文件路径
    parser.add_argument('--response_file', type=str, default='data/ic50/80cell_line_ic50.csv', 
                       help='响应数据文件路径')
    parser.add_argument('--bbbp_file', type=str, default='drug/dataset_merged.csv', 
                       help='BBBP数据集文件路径')
    
    # 使用预训练脚本中的文件路径
    parser.add_argument('--cnv_file', type=str, default='data/cellline/CNV_714cellline_24254dim_shu.csv', 
                       help='CNV特征CSV文件路径（使用预训练脚本的默认文件）')
    parser.add_argument('--rnaseq_file', type=str, default='data/cellline/903cellline_17737dim_scRNAseq.csv', 
                       help='RNAseq特征CSV文件路径（使用预训练脚本的默认文件）')
    
    # 特征来源选择
    parser.add_argument('--use_pretrained_cnv', action='store_true', default=False,
                       help='是否使用预训练的CNV特征(pkl文件)')
    parser.add_argument('--use_pretrained_rnaseq', action='store_true', default=False,
                       help='是否使用预训练的RNAseq特征(pkl文件)')
    
    # 随机种子
    parser.add_argument('--seed', type=int, default=11, 
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置所有可能的随机种子
    set_seed(args.seed)
    
    # 检查文件是否存在
    if not os.path.exists(args.response_file):
        raise FileNotFoundError(f"Response file not found: {args.response_file}")
    
    if args.bbbp_file and not os.path.exists(args.bbbp_file):
        print(f"Warning: BBBP file not found: {args.bbbp_file}")
        args.bbbp_file = None
    
    # 如果使用预训练特征，检查pkl文件是否存在
    if args.use_pretrained_cnv and not os.path.exists('data/cellline/512dim_copynumber.pkl'):
        print("Warning: Pretrained CNV pkl file not found, will try to use CSV file if provided")
        args.use_pretrained_cnv = False
    
    if args.use_pretrained_rnaseq and not os.path.exists('data/cellline/512dim_RNAseq.pkl'):
        print("Warning: Pretrained RNAseq pkl file not found, will try to use CSV file if provided")
        args.use_pretrained_rnaseq = False
    
    # 如果不使用预训练特征，检查CSV文件是否存在
    if not args.use_pretrained_cnv and not (args.cnv_file and os.path.exists(args.cnv_file)):
        raise FileNotFoundError(f"CNV CSV file not found and pretrained pkl not used: {args.cnv_file}")
    
    if not args.use_pretrained_rnaseq and not (args.rnaseq_file and os.path.exists(args.rnaseq_file)):
        raise FileNotFoundError(f"RNAseq CSV file not found and pretrained pkl not used: {args.rnaseq_file}")
    
    print(f"Using parameters:")
    print(f"  Response file: {args.response_file}")
    print(f"  BBBP file: {args.bbbp_file}")
    print(f"  CNV file: {args.cnv_file}")
    print(f"  RNAseq file: {args.rnaseq_file}")
    print(f"  Use pretrained CNV: {args.use_pretrained_cnv}")
    print(f"  Use pretrained RNAseq: {args.use_pretrained_rnaseq}")
    print(f"  Seed: {args.seed}")
    
    # 处理数据
    read_response_data_and_process(
        filename=args.response_file,
        bbbp_file=args.bbbp_file,
        cnv_file=args.cnv_file,
        rnaseq_file=args.rnaseq_file,
        use_pretrained_cnv=args.use_pretrained_cnv,
        use_pretrained_rnaseq=args.use_pretrained_rnaseq,
        seed=args.seed
    )
