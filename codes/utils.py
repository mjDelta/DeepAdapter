import os, math, csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as mcolors
from tqdm import tqdm
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, f_oneway, bartlett
from scipy.spatial import distance
from scipy.special import kl_div
from sklearn.manifold import TSNE
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score
from sklearn.model_selection import StratifiedKFold

def scatter_plot_diseases_cl_tumor(batches, diseases, tsne_data, colors_dict, save_path):
    diseases = np.array(diseases)
    dis2label, label2dis = {"Others": 0}, {0: "Others"}
    dis_set = sorted(set(diseases))
    for dis in dis_set:
        if dis == "Others":
            continue
        dis_mask = diseases == dis
        label = len(dis2label)
        dis2label[dis] = label
        label2dis[label] = dis
    labels = np.array([dis2label[dis] if dis in dis2label else dis2label["Others"] for dis in diseases])

    fig = plt.figure(figsize = (8, 6), dpi = 300)
    label_set = sorted(set(labels))
    for l in label_set:
        l_mask = labels == l
        dis_data = tsne_data[l_mask]
        dis_batches = batches[l_mask]
        b_mask = dis_batches == 0
        dis = label2dis[l]
        plt.scatter(dis_data[~b_mask, 0], dis_data[~b_mask, 1], 
                    s = 5, marker = "o", color = colors_dict[dis],
                    edgecolor = "white",  alpha = 1, lw = 0.4,
                    label = dis
                   )
        plt.scatter(
            dis_data[b_mask, 0], dis_data[b_mask, 1], 
#             label = dis, 
            s = 6, color = colors_dict[dis], 
            edgecolor = "black", alpha = 1, lw = 0.4)

#     plt.legend()
#     plt.grid(False)
    plt.savefig(save_path, bbox_inches = "tight")
    return len(dis2label) - 1

def scatter_plot_diseases(batches, diseases, tsne_data, colors_dict, save_path):
    diseases = np.array(diseases)
    dis2label, label2dis = {"Others": 0}, {0: "Others"} ## for cell line of RNA-seq and microarray
    dis_set = sorted(set(diseases))
    for dis in dis_set:
        dis_mask = diseases == dis
        if dis_mask.sum() <= 44:
            continue
        label = len(dis2label)
        dis2label[dis] = label
        label2dis[label] = dis
#     print(dis2label)
    labels = np.array([dis2label[dis] if dis in dis2label else dis2label["Others"] for dis in diseases])
    
    fig = plt.figure(figsize = (7, 5), dpi = 300)
    label_set = sorted(set(labels))
#     print(label_set)
    for l in label_set:
        l_mask = labels == l
        dis_data = tsne_data[l_mask]
        dis_batches = batches[l_mask]
        b_mask = dis_batches == 0
        
        dis = label2dis[l]
        if dis in ["Lymphoma", "Myeloma", "Leukemia"]:
            dis = "Blood cancer"

        plt.scatter(
            dis_data[b_mask, 0], dis_data[b_mask, 1], 
            label = dis, 
            s = 13, color = colors_dict[dis], 
#             edgecolor = "none", 
            alpha = 0.6)
        
        plt.scatter(dis_data[~b_mask, 0], dis_data[~b_mask, 1], 
                    s = 13, 
#                     marker = "x",
                    color = colors_dict[dis],
#                     edgecolor = colors_dict[dis], 
                    alpha = 0.6, 
#                     lw = 1
                   )
#     plt.legend()
    plt.savefig(save_path, bbox_inches = "tight")
    
    return len(dis2label) - 2

def get_cosmic_id(cl_id_df, id_):
    mask = cl_id_df["ID"] == id_
    this_cl_df = cl_id_df[mask]
    assert len(this_cl_df) == 1
    cosmic_id = this_cl_df["COSMIC_ID"].values[0]
    return cosmic_id

def sort_genes(df):
    mean = df.mean(axis = 0).values.squeeze()
    cols = np.array(list(df.columns))
    argidxs = np.argsort(mean)
    return cols[argidxs] 

def rearrange_genes_by_single_data(df1, df2, sort_by, dat1_name):
    if sort_by == dat1_name:
        new_cols = sort_genes(df1)
    else:
        new_cols = sort_genes(df2)
    return df1[new_cols], df2[new_cols], new_cols

def kl_divergence(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64) + 1e-9

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))    
    
def topk_neighbors_labels(pnter, neighbors, labels, k):
    distances = np.array([(neighbors[:, i] - pnter[i])**2 for i in range(len(pnter))])
    distances = np.sqrt(np.sum(distances, axis = 0))
    argmins = np.argpartition(distances, k)[:k]
    return labels[argmins]

def alignment_score(trans_data, labels, neighbor = 0.01):
    sample_num = len(labels)
    k = int(np.ceil(neighbor*sample_num))
#     print(k, neighbor, sample_num)
    if k <= 4:
        k = 4
    labels_set = sorted(set(labels))
    percentages = [(labels == l).sum()/sample_num for l in labels_set]
        
    hits = np.zeros((len(trans_data), ))
    for i, pnter in enumerate(trans_data):
        nei_labels = topk_neighbors_labels(pnter, trans_data, labels, k)
        pnter_label = labels[i]
        hit = (nei_labels == pnter_label).sum()
        hits[i] = hit

    xs = []
    for l in labels_set:
        mask = labels == l
        xs.append(hits[mask].mean())
#     print(labels_set, percentages, xs, k)

    score = 0.
    for x, percentage in zip(xs, percentages):
        dist_score = (x - percentage*k)/(k - percentage*k)
        s = percentage*(1 - dist_score)
        score += s
    return score
