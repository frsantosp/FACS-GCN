# FACS-GCN: A Fairness Aware Cost Sensitive Boosting Graph Convolutional Network

This repository contains the source code for FACS-GCN. Link to paper: https://ieeexplore.ieee.org/document/9892919

## 1. Datasets

Preprocessed datasets are contained in the Data folder. Raw data are in Dataset folder.

## 2. Usage

The main scripts to run FACS-GCN is facs_gcn.py. The code containing the model can be found in facs_model.py. The implementation uses PyTorch. 

An example to run the model:

`python facs_gcn.py --dataset_str facebook --lr 0.001 --reward_class_2 2.0 --epoch 500
`
## 3. Questions?
If you have questions/suggestions, please feel free to email santosf3 @ msu dot edu
