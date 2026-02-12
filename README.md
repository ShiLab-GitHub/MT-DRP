# MT-DRPNet: A Multi-Task Deep Learning Framework for Brain Tumor Drug Response and BBB Permeability Prediction
This project implements a **multi-task learning model (MT-DRPNet)** for simultaneous prediction of brain tumor cell line drug sensitivity and blood-brain barrier permeability (BBBP). The model integrates multi-omics cellular profiles and drug molecular structural features to improve prediction accuracy, built on curated DRP and BBBP benchmark datasets.

## Dataset Sources
### 1. Drug Response Prediction (DRP) Dataset
- **Sources**: GDSC2 Database, CCLE Database
- **Hosted Repository**: https://github.com/ShiLab-GitHub/DeepMoDRP
- **Key Content**: 80 brain tumor cell lines (LGG, GBM, DLBCL) with complete multi-omics features (RNAseq, METH, CNV, MUT), GDSC2 IC50 drug response labels, and drug SMILES representations.

### 2. Blood-Brain Barrier Permeability (BBBP) Dataset
- **Name**: DL-BBBP
- **Source Repository**: https://github.com/ShiLab-GitHub/DL-BBBP
- **Key Content**: Binary BBB permeability labels (BBB+/BBB-), standardized drug molecular features and SMILES, curated for deep learning-based prediction (supporting ICBBE'2024 paper).

## Environment Requirements
### Python Version
Python 3.7+ (recommended 3.7.10 for dependency compatibility)

### Dependencies
Install all required packages via the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Running Steps
Run the scripts in the **exact order** for full pipeline execution:
1. **Drug Feature Extraction**
   ```bash
   python SADE.py
   ```
   *Function*: Extract and process drug molecular features from SMILES sequences for multi-modal fusion.

2. **Multi-Task Data Preprocessing**
   ```bash
   python preprocess_DRP_BBBP_SADE.py
   ```
   *Function*: Unify DRP/BBBP data formats, convert to PyTorch tensor, and split training/validation/test sets.

3. **Model Training & Evaluation**
   ```bash
   python MT-DRPNet.py
   ```
   *Function*: Train the multi-task model, predict brain tumor DRP (IC50 regression) and BBBP (binary classification), and output performance metrics.

## Data Preparation
1. Download DRP dataset: https://github.com/ShiLab-GitHub/DeepMoDRP
2. Download BBBP dataset: https://github.com/ShiLab-GitHub/DL-BBBP
3. Place DRP data in the project's `data/` directory following the original DeepMoDRP file structure
4. Create `data/bbbp/` subfolder and place all DL-BBBP data/files into it
5. Ensure file paths in all scripts match local storage structure

## Key Notes
1. Follow the strict script execution order to avoid missing feature files
2. Adjust hyperparameters (batch size, learning rate, epochs) in `MT-DRPNet.py` based on local computing resources (GPU/CPU)
3. Sufficient GPU memory (≥8G) is recommended for high-dimensional multi-omics and molecular graph feature processing
4. For DL-BBBP data preprocessing, run `CreateData.py` from the DL-BBBP repository first to generate standardized features

## Citation
If using this project's code/datasets, cite the original repositories:
1. DeepMoDRP: https://github.com/ShiLab-GitHub/DeepMoDRP
2. DL-BBBP: https://github.com/ShiLab-GitHub/DL-BBBP
3. GDSC: https://www.cancerrxgene.org
4. CCLE: https://sites.broadinstitute.org/ccle/datasets

## Contact & Issues
For technical questions/bugs, refer to the original repositories:
- DeepMoDRP/MT-DRPNet: https://github.com/ShiLab-GitHub/DeepMoDRP
- DL-BBBP: https://github.com/ShiLab-GitHub/DL-BBBP

