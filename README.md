
# Fine-grained CIFAR-100 Classification with MobileViT

This project provides a pipeline to train and evaluate a MobileViT model on a focused subset of the CIFAR-100 dataset (consisting of superclasses "vehicles 1" and "vehicles 2"). It includes scripts for dataset preparation, model training, and evaluation/visualization.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Using the scripts
The project is divided into three main scripts:
 1. prepare_dataset.py
 2. MobileViT_training.py
 3. MobileViT_evaluation.py

Scripts 1 downloads the full CIFAR-100 datasets and creates a three-way testplit. Script 2 trains the MobileViT model and script 3 evaulates the model on a held out test dataset and caluclates top1 accuracy as well as generating loss/accuracy vs epochs plots and a confusion matrix.
