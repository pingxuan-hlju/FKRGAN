# FKRGAN

## Introduction

Fourier transform and Kolmogorov–Arnold Network enhanced relation-aware generative and adversarial network for miRNA-disease association prediction.

## Environment

The FKRGAN code has been implemented and tested in the following development environment:

| Environment name | Version                                                      |
| ---------------- | ------------------------------------------------------------ |
| pytroch          | ![Static Badge](https://img.shields.io/badge/pytorch-2.2.1%2Bcu121-red) |
| matplotlib       | ![Static Badge](https://img.shields.io/badge/matplotlib-3.9.2-pink) |
| numpy            | ![Static Badge](https://img.shields.io/badge/numpy-1.26.3-green) |
| python           | ![Static Badge](https://img.shields.io/badge/python-3.9.18-blue) |


## File information

### Catalogs

- `./data`: Contains the dataset used in our method.
- `dataloade.py`:Processes miRNA-disease similarity and association data to generate embeddings and adjacency matrices, compiling them into a dataset named `common_set.pkl`. Then, splits the dataset into training, validation, and test sets, and saves them as `train_set.pkl` and `test_set.pkl`, respectively.
- `parameters.py`:Defines hyperparameters for the FKRGAN model.
- `model0527.py`: Defines the FKRGAN model architecture.
- `feature_Block_train.py`:Optimizes the topological features and biological features (miRNA family characteristics and cluster features) for miRNA and disease nodes, as well as the miRNA family characteristics and miRNA cluster features.
- `relgan_train.py`:Trains the relation-aware adversarial network to generate enhanced feature representations.
- `train.py`:Trains the FKRGAN model.
- `tools4roc_pr.py`: Evaluates the performance of the FKRGAN model.
- `Supplementary Table ST1.xlsx`: Lists the top 50 candidate miRNAs for each disease.

### Data

path : `./data/data.rar`

| file_name             | size          | description                                                  |
| --------------------- | ------------- | ------------------------------------------------------------ |
| `miRNA_name.npy`      | `(1245,)`     | Contains names of 1245 miRNAs as strings.                    |
| `disease_name.npy`    | `(2077,1206)` | Contains names of 2077 diseases as strings.                  |
| `disease_disease.npy` | `(2077,2077)` | Contains semantic similarity scores between pairs of 2077 diseases. |
| `miRNA_miRNA.npy`     | `(1245,1245)` | Contains sequence similarity scores between pairs of 1245 miRNAs. |
| `miRNA_rfam.npy`      | `(1245,)`     | Contains family features for 1245 miRNAs.                    |
| `miRNA_cluster.npy`   | `(1245,1245)` | Contains cluster similarity scores for 1245 miRNAs.          |
| `miRNA_targets.npy`   | `(1245,5357)` | Contains associations between 1245 miRNAs and 5357 protein targets. |
| `miRNA_disease.npy`   | `(1245,2077)` | Contains associations between 1245 miRNAs and 2077 diseases. |

### Original Databases of Dataset Sources

| file_name         | source                                             | description                                                  |
| ----------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| `miRNA.dat`       | [mirbase](https://mirbase.org/)                    | miRBase v22 (version 2019): A comprehensive archive of microRNA sequences and annotations. Used to generate `miRNA_miRNA.npy`,`miRNA_rfam.npy`,`miRNA_cluster.npy`,`miRNA_disease.npy`. |
| `desc2024.xml`    | [MeSH](https://www.nlm.nih.gov/mesh/meshhome.html) | Medical Subject Headings (MeSH) thesaurus (version 2024):  A controlled and hierarchically-organized vocabulary from the National Library of Medicine. Used to generate `disease_disease.npy`. |
| `alldata_v4.xlsx` | [HMDD](http://www.cuilab.cn/hmdd)             | HMDD v4.0 (version 2023.07): A comprehensive dataset of miRNA-disease association data. Used to generate `miRNA_name.npy`, `disease_name.npy` and `miRNA_disease.npy`. |

## How to Run the Code

Before running the code, create two directories: `mdd` and `savedata`. The `mdd` directory stores `.pkl` files generated during the first step of Data Preprocessing, while the `savedata` directory stores model parameters.

1. **Data Preprocessing:** Process miRNA-disease similarity and association to generate embeddings, adjacency matrices, and compile them into a dataset file named `common_set.pkl`. Then, split the data into training, validation, and test sets, saved as `train_set.pkl` and `test_set.pkl`, respectively.

   ```
   python dataloade.py
   ```

2. **Feature optimization**: Optimize the topological features of miRNA and disease nodes, as well as the miRNA family characteristics and miRNA cluster features.

   ```
   python feature_Block_train.py
   ```

3. **Relation-aware adversarial network**: Optimize the feature representations of miRNA and disease nodes using a relation-aware adversarial network generator.

   ```
   python relgan_train.py
   ```

4. Train and test model.

   ```
   python train.py
   ```

## Contributor

This study was a collaborative effort, with significant contribution during the data collection phase, which was conducted in an organized and coordinated manner. The main contributors to the data collection process are:

- Siyuan Lu
- Dongliang Chen
- Mengxia Wang
