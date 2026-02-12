# MT-DRPNet 项目

## 项目描述
本项目实现了一个基于多任务学习的药物反应预测模型（MT-DRPNet），用于预测脑肿瘤细胞系对药物的敏感性。该模型整合了多组学数据和药物特征，以提高药物反应预测的准确性。

## 数据集来源

### 1. 脑肿瘤数据集
- **来源**: GitHub 仓库
- **链接**: https://github.com/ShiLab-GitHub/DeepMoDRP
- **描述**: 本研究整理和使用的脑肿瘤数据集

### 2. 细胞系-药物反应数据集
- **来源**: 癌症药物敏感性基因组学 (GDSC) 数据库
- **链接**: https://www.cancerrxgene.org
- **描述**: 药物敏感性数据

### 3. 多组学数据集
- **来源**: 
  - GDSC 数据库 (https://www.cancerrxgene.org)
  - 癌症细胞系百科全书 (CCLE) (https://sites.broadinstitute.org/ccle/datasets)
- **描述**: 多组学特征数据

### 4. 源代码与整合数据集
- **来源**: GitHub 仓库
- **链接**: https://github.com/ShiLab-GitHub/DeepMoDRP
- **说明**: 本项目使用的完整、整合好的数据集可从此仓库获取

## 环境要求

### Python 版本
- Python 3.7+

### 依赖库
请安装 `requirements.txt` 中列出的所有依赖：

```bash
pip install -r requirements.txt
```

## 运行步骤

请按以下顺序执行代码：

### 步骤 1: 运行 SADE.py
```bash
python SADE.py
```
**作用**: 处理药物特征或执行相关特征提取。

### 步骤 2: 运行预处理脚本
```bash
python preprocess_DRP_BBBP_SADE.py
```
**作用**: 预处理药物反应预测和BBB渗透性相关数据。

### 步骤 3: 运行主模型
```bash
python MT-DRPNet.py
```
**作用**: 训练和评估多任务药物反应预测模型。

## 数据准备

1. 从源代码仓库下载整合好的完整数据集：https://github.com/ShiLab-GitHub/DeepMoDRP
2. 将数据放置在 `data/` 目录下的相应子文件夹中
3. 根据预处理脚本的要求整理数据格式

## 注意事项

1. 确保所有依赖库已正确安装
2. 按照顺序执行三个脚本
3. 根据实际情况调整文件路径和参数
4. 确保有足够的存储空间和计算资源（特别是GPU内存）


## 引用

如果使用本项目代码或数据，请引用原始数据源和相关论文：

1. DeepMoDRP 项目：https://github.com/ShiLab-GitHub/DeepMoDRP
2. GDSC 数据库：https://www.cancerrxgene.org
3. CCLE 数据库：https://sites.broadinstitute.org/ccle

## 联系方式

如有问题，请参考原始代码仓库：https://github.com/ShiLab-GitHub/DeepMoDRP

