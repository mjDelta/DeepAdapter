# # Benchmarking methods comparison
# ## 1. Installation and requirements
# ### 1.1 Installation
# #### 1.1.1 Extra installation for Windows
# Before installing the packages, please make sure **Microsoft Visual C++ 14.0 or greater** installed. The official installation link can be found [VC++](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
# #### 1.1.2 Python packages
# Before running, please ensure the packages of related methods are installed. These benchmarking methods are: **Quantile Normalization**, **Combat**, **MNN**, **Harmony**, **PRPS**, **Scanorama**, **BBKNN**, **AutoClss**, and **scVI**.

# These benchmarking methods rely on different running environment:

# A. Methods that are compatible to DA environment include **Quantile Normalization**, **Combat**, **Harmony**, **PRPS**, **Scanorama**, **BBKNN**, **AutoClss**, and **scVI**.

# ```sh
# $ conda activate DA
# $ pip install qnorm==0.8.1 combat==0.3.3 scanorama==1.7.4 anndata bbknn==1.6.0 scanpy harmonypy scvi==0.6.8 tensorflow
# $ git clone https://github.com/datapplab/AutoClass
# $ jupyter notebook
# ```
# In jupyter notebook, open `Benchmarking methods.ipynb` to run the methods that are compatible to DA environment.

# B. Method that is not compatible to DA environment includes **MNN**.

# Firstly, to run **MNN**, please create the environment with th following codes:
# ```sh
# $ conda create -n py3.8 python=3.8
# $ conda activate py3.8
# $ pip install mnnpy==0.1.9.5 matplotlib tqdm umap-learn openpyxl scipy==1.5.4
# $ python Benchmarking-MNN.py
# ```
# Secondly, download the codes in `mnn_utils/` for loading the dataset and put this folder in the same hierarchy as this tutorial.

# *Noting: some packages don't support separate training and testing. For these packages, we will load the whole data as the training set. In other words, these methods will be trained with more samples used by **DeepAdapter**. The comparison will be unfiar to **DeepAdapter** which uses less samples. Yet, **DeepAdapter** wins.*

# #### 1.1.3 Potential environmental errors and solutions for installation
# After running the codes to install packages for **MNN**, there might be some environmental errors with running the **multiprocessing** package required by **mnnpy**. If the errors appear, we would suggest you comment the codes with multiprocessing acceleration. The multiprocessing works for accelerating the calculation speed and does not affect the aligned performances.

# **Solution:** 
# 1. Open the folder that you installed with `pip install mnnpy==0.1.9.5`. It might be in `~/user_name/anaconda3/envs/py3.8/Lib/site-packages/mnnpy`.
# 2. Comment the `mnn.py` in line 191 and 192 and add one-line code. The commented codes should look like:
#    ```
#    191 # with Pool(n_jobs) as p_n:
#    192 #     angle_out = p_n.map(find_subspace_job, correction_in)
#    193 angle_out = find_subspace_job(correction_in)
#    ```
# 3. Open `settings.py` and replace **parallel** with **nonparallel**.


# ### 1.2 Datasets
# Please download the open datasets in [Zenodo](https://zenodo.org/records/10494751).
# These datasets are collected from literatures to demonstrate multiple unwanted variations, including:
# * batch datasets: LINCS-DToxS ([van Hasselt et al. Nature Communications, 2020](https://www.nature.com/articles/s41467-020-18396-7)) and Quartet project ([Yu, Y. et al. Nature Biotechnology, 2023](https://www.nature.com/articles/s41587-023-01867-9)).
# * platform datasets: profiles from microarray ([Iorio, F. et al. Cell, 2016](https://www.cell.com/cell/pdf/S0092-8674(16)30746-2.pdf)) and RNA-seq ([Ghandi, M. et al. Nature, 2019](https://www.nature.com/articles/s41586-019-1186-3)).
# * purity datasets: profiles from cancer cell lines ([Ghandi, M. et al. Nature, 2019](https://www.nature.com/articles/s41586-019-1186-3)) and tissues ([Weinstein, J.N. et al. Nature genetics, 2013](https://www.nature.com/articles/ng.2764)).

# After downloading, place the datasets in the `data/` directory located in the same hierarchy as this tutorial.
# * batch datasets: `data/batch_data/`
# * platform datasets: `data/platform_data/`
# * purity datasets: `data/purity_data/`
  
# **Putting datasets in the right directory is important for loading the example datasets successfully.**



# ## 2. Load the datasets and preprocess
# ### 2.1. load the modules
import os, sys, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
from mnn_utils import data_utils as DT
from mnn_utils import utils as UT


# ### 2.2. Load the demonstrated datasets
# We ultilize Batch-LINCS for demonstration. To load datasets of platform and purity variations, please download them in Zenodo (https://zenodo.org/records/10494751).
#   * In the tutorial, we have **data** for gene expression, **batches** for unwanted variations, and **donors** for biological signals.
loadTransData = DT.LoadTransData()
data, batches, wells, donors, infos, test_infos = loadTransData.load_lincs_lds1593()
ids = np.arange(len(data))


# ### 2.3. Preprocess the transcriptomic data
# The gene expression profiles are preprocessed by sample normalization, gene ranking, and log normalization. Let $S_i = \sum_l x_{i l}$ denote the sum over all genes. In sample normalization, we divide $S_i$ for every sample and multiply a constant 10000 ([Xiaokang Yu et al. Nature communications, 2023](https://www.nature.com/articles/s41467-023-36635-5)):
# $$x_{i l} = \frac{x_{i l}}{S_i} 10^4.$$
# Then, we sort genes by their expression levels and perform the log transformation $x_{i l} = \log {(x_{i l} + 1)}$.
prepTransData = DT.PrepTransData()
raw_df = prepTransData.sample_norm(data)
raw_df, sorted_cols = prepTransData.sort_genes_sgl_df(raw_df)
input_arr = prepTransData.sample_log(raw_df)
bat2label, label2bat, unwanted_labels, unwanted_onehot = prepTransData.label2onehot(batches)


# ## 3. Run the benchmarking methods
# ### 3.1 Split dataset
train_data, train_labels, train_labels_hot, \
    val_data, val_labels, val_labels_hot, \
    test_data, test_labels, test_labels_hot, \
    train_ids, val_ids, test_ids, \
    tot_train_val_idxs, tot_train_idxs, tot_val_idxs, tot_test_idxs = DT.data_split_lds1593(input_arr, unwanted_labels, unwanted_onehot, ids, infos, test_infos)


# ### 3.2 Align the datasets by benchmarking methods
baseline = "mnn"
out_dir = f"./baselines_out/{baseline}/"
os.makedirs(out_dir, exist_ok = True)

if baseline == "mnn":
    import mnnpy
    dat1_mask = unwanted_labels == 0
    dat2_mask = unwanted_labels == 1
    dat3_mask = unwanted_labels == 2
    dat4_mask = unwanted_labels == 3
    dat1_name, dat2_name, dat3_name, dat4_name = "b1", "b2", "b3", "b4"
    corrected = mnnpy.mnn_correct(
        input_arr[dat1_mask], input_arr[dat2_mask], input_arr[dat3_mask], input_arr[dat4_mask],
        var_index = sorted_cols, batch_categories = [dat1_name, dat2_name, dat3_name, dat4_name], k = 20)
    labels = list(unwanted_labels[dat1_mask]) + list(unwanted_labels[dat2_mask]) + list(unwanted_labels[dat3_mask]) + list(unwanted_labels[dat4_mask])
    labels = np.array(labels)
    normed_data = corrected[0]
else:
    raise("This script is for MNN only. If you want to run other benchmarking methods, please refer to Benchmarking methods.ipynb")


# ### Visualization of aligned dataset
# - BBKNN will return the decomposed data. Thus, there is no need to perform data decomposition for BBKNN.
# - Perform decomposition for other methods
def decom_plot(data, labels, save_path, baseline, colors, perplexity = 30, label2name = None, min_dist = 0.99, size = 20, metric = "euclidean", n_neighbors = 15):
    import umap
    import matplotlib.pyplot as plt
    label_set = sorted(set(labels))
    if baseline != "bbknn":
        fitter = umap.UMAP(random_state = 42, min_dist = min_dist, metric = metric, n_neighbors = n_neighbors)
        trans_data = fitter.fit_transform(data)
    else:
        trans_data = data
    align_score = UT.alignment_score(trans_data, labels)
    
    fig = plt.figure(figsize = (7, 5))
    for l, c in zip(label_set, colors):
        mask = labels == l
        plt.scatter(trans_data[mask][:, 0], trans_data[mask][:, 1], edgecolor = c, color = c, 
                    s = size,
                    linewidths = 0.5, label = label2name[l], alpha = 0.8)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.title("Align score of {}".format(align_score))
    plt.savefig(save_path, bbox_inches = "tight")  
    return trans_data
    
colors = ["#57904B", "violet",  "#C93C2A", "#372A8F"]
trans_aligned = decom_plot(
        normed_data, labels, os.path.join(out_dir, "aligned.png"), 
        baseline, colors = colors, label2name = label2bat)


