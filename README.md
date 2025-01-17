# DeepAdapter
## A self-adaptive and versatile tool for eliminating multiple undesirable variations from transcriptome
Codes and tutorial for [A self-adaptive and versatile tool for eliminating multiple undesirable variations from transcriptome](https://www.biorxiv.org/content/10.1101/2024.02.04.578839v1).


## Installation
We add scripts for fine-tuning. Please install it with
```sh
$ pip install deepadapter==1.0.1
```
**Note: only deepadapter (v1.0.1) supports the fine-tuning**

# Get started

## Running environment configuration
**Step 1**: create a new conda environment
```sh
$ # Create a new conda environment
$ conda create -n DA python=3.9
$ # Activate environment
$ conda activate DA
```
**Step 2**: install the package with `pip`
```sh
$ # Install the our package
$ pip install deepadapter
```
**Step 3**: launch jupyter notebook and double-click to open tutorials
```sh
$ # Launch jupyter notebook
$ jupyter notebook
```
**After opening the tutorials, please press Shift-Enter to execute a "cell" in `.ipynb`.**

## Train DeepAdapter with the provided example datsets or your own dataset
Before runing the codes, download our tutorials.
* `DA-Example-Tutorial.ipynb`: the tutorial of re-training DeepAdapter using the example dataset ([click here to download](https://github.com/mjDelta/DeepAdapter/blob/main/DA-Example-Tutorial.ipynb));
* `DA-YourOwnData-Tutorial.ipynb`: the tutorial of training DeepAdapter using your own dataset ([click here to download](https://github.com/mjDelta/DeepAdapter/blob/main/DA-YourOwnData-Tutorial.ipynb)).

## Fine-tune DeepAdapter with limited data samples
Before fine-tuning, make sure that the gene set in the small dataset the same as the the gene set used in pretrained models.
The gene set used in pretrained models can be found in `trained_models/[model]/genes.csv`. **The order of gene set does not matter.**

Double-click to open tutorials after launching jupyter notebook
* `DA-Example-finetune.ipynb`: the tutorial of fine-tuning DeepAdapter using the example dataset ([click here to download](https://github.com/mjDelta/DeepAdapter/blob/main/DA-Example-Finetune.ipynb));;
* `DA-YourOwnData-finetune.ipynb`: the tutorial of fine-tuning DeepAdapter using your own dataset ([click here to download](https://github.com/mjDelta/DeepAdapter/blob/main/DA-YourOwnData-Finetune.ipynb));.

**After opening the tutorials, please press Shift-Enter to execute a "cell" in `.ipynb`.**

## Run the benchmarking methods
The benchmarking methods can be found in `Benchmarking-methods.ipynb` ([click here to download](https://github.com/mjDelta/DeepAdapter/blob/main/Benchmarking-methods.ipynb)) and `Benchmarking-MNN.py` ([click here to download](https://github.com/mjDelta/DeepAdapter/blob/main/Benchmarking-MNN.py));.

**Step 1**: run methods except MNN
* installation instructions: find the installation cmds in `Benchmarking-methods.ipynb`
* run benchmarking methods: choose the benchmarking method and run it

**Step 2**: run MNN with the followsing cmds in a new shell

Before running, ensure that the codes in `mnn_utils/` ([click here to download](https://github.com/mjDelta/DeepAdapter/blob/main/mnn_utils)) for loading the dataset are the same hierarchy as this tutorial. To run **MNN**, please create the environment with th following codes:
```sh
$ conda create -n py3.8 python=3.8
$ conda activate py3.8
$ pip install mnnpy==0.1.9.5 matplotlib tqdm umap-learn openpyxl scipy==1.5.4
$ python Benchmarking-MNN.py
```

# Resources
## Datasets
Please download the open datasets in [Zenodo](https://zenodo.org/records/10494751).
These datasets are collected from literatures to demonstrate multiple unwanted variations, including:
* batch datasets: LINCS-DToxS ([van Hasselt et al. Nature Communications, 2020](https://www.nature.com/articles/s41467-020-18396-7)) and Quartet project ([Yu, Y. et al. Nature Biotechnology, 2023](https://www.nature.com/articles/s41587-023-01867-9)).
* platform datasets: profiles from microarray ([Iorio, F. et al. Cell, 2016](https://www.cell.com/cell/pdf/S0092-8674(16)30746-2.pdf)) and RNA-seq ([Ghandi, M. et al. Nature, 2019](https://www.nature.com/articles/s41586-019-1186-3)).
* purity datasets: profiles from cancer cell lines ([Ghandi, M. et al. Nature, 2019](https://www.nature.com/articles/s41586-019-1186-3)) and tissues ([Weinstein, J.N. et al. Nature genetics, 2013](https://www.nature.com/articles/ng.2764)).

After downloading, place the datasets in the `data/` directory located in the same hierarchy as this tutorial.
* batch datasets: `data/batch_data/`
* platform datasets: `data/platform_data/`
* purity datasets: `data/purity_data/`

## Pretrained models
Please find the pretrained models in folder `models` ([click here to download](https://zenodo.org/records/14664454)).
* batch integration: `models/batch_LINCS` and `models/batch_Quartet`
* platform integration: `models/platform`
* purity integration: `models/purity`

After downloading, place the models in the `models/` directory located in the same hierarchy as this tutorial.
