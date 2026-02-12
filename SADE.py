import csv
import torch.nn as nn
from sklearn import preprocessing
import torch
import torch.utils.data as Data
from scipy.stats import pearsonr
import pickle
import numpy as np
import argparse
import os

# 超参数
EPOCH = 500
LR = 1e-4

# 数据类型的配置
DATA_CONFIG = {
    'CNV': {
        'data_file': 'data/cellline/CNV_714cellline_24254dim_shu.csv',
        'input_size': 24254,
        'hidden_size': 256,
        'output_dim': 512,
        'batch_size': 80,
        'output_file': 'data/cellline/512dim_copynumber.pkl'
    },
    'RNAseq': {
        'data_file': 'data/cellline/903cellline_17737dim_scRNAseq.csv',
        'input_size': 17737,
        'hidden_size': 256,
        'output_dim': 512,
        'batch_size': 714,
        'output_file': 'data/cellline/512dim_RNAseq.pkl'
    }
}

def read_cell_line_list(filename):
    """读取细胞系列表并构建字典"""
    cell_line_dict = {}
    index = 0
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"标题行: {header}")
        
        for line in reader:
            cell_line_dict[line[0]] = index
            index += 1
    
    return cell_line_dict

def read_cell_line_data(filename, cell_line_dict, data_type):
    """读取细胞系数据"""
    data = [list() for _ in range(len(cell_line_dict))]
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"标题行: {header}")
        
        for line in reader:
            if line[0] in cell_line_dict:
                processed_line = []
                for value in line[1:]:
                    if value.strip() == '':
                        processed_line.append(0.0)
                    else:
                        try:
                            processed_line.append(float(value))
                        except ValueError:
                            print(f"警告: 无法将值 '{value}' 转换为浮点数，已替换为0")
                            processed_line.append(0.0)
                data[cell_line_dict[line[0]]] = processed_line
    
    return data

# Denoising Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_noisy = x + 0.1 * torch.randn_like(x)
        x_encoded = self.encoder(x_noisy)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded

# Sparse Autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded

class CombinedModel(nn.Module):
    def __init__(self, denoising_ae, sparse_ae):
        super(CombinedModel, self).__init__()
        self.denoising_ae = denoising_ae
        self.sparse_ae = sparse_ae

    def forward(self, x):
        x_denoising_ae, x_denoising_dae = self.denoising_ae(x)
        x_sparse_ae, x_sparse_dae = self.sparse_ae(x)
        x_combined = torch.cat([x_denoising_ae, x_sparse_ae], dim=1)
        x_combined_decoder = x_denoising_dae + x_sparse_dae
        return x_combined, x_combined_decoder

def load_data(data_type):
    """加载数据"""
    config = DATA_CONFIG[data_type]
    
    # 读取细胞系列表
    cell_line_list_file = 'data/cellline/cellline_listwithACH_714cellline.csv'
    if not os.path.exists(cell_line_list_file):
        print(f"错误: 细胞系列表文件 '{cell_line_list_file}' 不存在!")
        exit(1)
    
    cell_line_dict = read_cell_line_list(cell_line_list_file)
    
    # 读取特定类型的数据
    data_file = config['data_file']
    if not os.path.exists(data_file):
        print(f"错误: 数据文件 '{data_file}' 不存在!")
        exit(1)
    
    data = read_cell_line_data(data_file, cell_line_dict, data_type)
    
    # 验证数据
    print(f"数据类型: {data_type}")
    print(f"数据维度: {len(data)} 个样本")
    if data:
        print(f"第一个样本长度: {len(data[0])}")
    
    # 转换为numpy数组以便于检查
    data_array = np.array(data)
    print(f"数组形状: {data_array.shape}")
    print(f"是否包含NaN: {np.isnan(data_array).any()}")
    
    # 归一化数据
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    data_normalized = min_max_scaler.fit_transform(data)
    train_data = torch.FloatTensor(data_normalized)
    
    # 划分训练集和测试集
    train_size = int(train_data.shape[0] * 0.8)
    data_train = train_data[:train_size]
    data_test = train_data[train_size:]
    
    return data_train, data_test, train_data, config

def train(train_data, test_data, data_all, config):
    """训练模型"""
    BATCH_SIZE = config['batch_size']
    input_size = config['input_size']
    hidden_size = config['hidden_size']
    output_file = config['output_file']
    
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型
    denoising_ae = DenoisingAutoencoder(input_size, hidden_size)
    sparse_ae = SparseAutoencoder(input_size, hidden_size)
    combined_model = CombinedModel(denoising_ae, sparse_ae)
    
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    best_loss = float('inf')
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\n开始训练 {config['output_file'].split('/')[-1]}...")
    print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
    print(f"输入维度: {input_size}, 输出维度: {config['output_dim']}")
    print(f"批量大小: {BATCH_SIZE}, 总迭代次数: {EPOCH}")
    
    for epoch in range(EPOCH):
        # 训练阶段
        combined_model.train()
        for step, data in enumerate(train_loader):
            _, decoded = combined_model(data)
            loss = loss_func(decoded, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 测试阶段
        combined_model.eval()
        with torch.no_grad():
            _, test_de = combined_model(test_data)
            test_loss = loss_func(test_de, test_data)
            
            # 计算Pearson相关系数
            pearson = pearsonr(test_de.view(-1).tolist(), test_data.view(-1))[0]
            
            if test_loss < best_loss:
                best_loss = test_loss
                # 在整个数据集上获取编码结果
                res, _ = combined_model(data_all)
                # 保存结果
                pickle.dump(res.data.numpy(), open(output_file, 'wb'))
                print(f'Epoch: {epoch} | 最佳测试损失: {best_loss:.4f} | Pearson: {pearson:.4f}')
                print(f'模型已保存到: {output_file}')
        
        if epoch % 50 == 0:
            print(f'Epoch: {epoch} | 测试损失: {test_loss:.4f}')
    
    print(f"训练完成! 最佳测试损失: {best_loss:.4f}")
    return

def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='训练SAE和DAE组合模型')
    parser.add_argument('--data_type', type=str, choices=['CNV', 'RNAseq'], 
                       default='CNV', help='数据类型: CNV 或 RNAseq')
    parser.add_argument('--epochs', type=int, default=EPOCH, help='训练迭代次数')
    parser.add_argument('--lr', type=float, default=LR, help='学习率')
    parser.add_argument('--batch_size', type=int, help='批量大小（可选，默认根据数据类型设置）')
    
    args = parser.parse_args()
    
    # 更新超参数
    global EPOCH, LR
    EPOCH = args.epochs
    LR = args.lr
    
    # 如果指定了batch_size，则覆盖配置
    if args.batch_size:
        DATA_CONFIG[args.data_type]['batch_size'] = args.batch_size
    
    print(f"训练配置:")
    print(f"  数据类型: {args.data_type}")
    print(f"  迭代次数: {EPOCH}")
    print(f"  学习率: {LR}")
    print(f"  批量大小: {DATA_CONFIG[args.data_type]['batch_size']}")
    
    # 加载数据
    train_data, test_data, all_data, config = load_data(args.data_type)
    
    # 训练模型
    train(train_data, test_data, all_data, config)

if __name__ == "__main__":
    main()
