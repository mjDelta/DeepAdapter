import os, sys
import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

sys.setrecursionlimit(100000)

class HeatmapDrawer:
    def __init__(self, out_dir):
        self.out_dir = out_dir
    
    def draw_clustermap(self, labels, data, out_name, colors = ["#C93C2A", "#372A8F"]):
        print(set(labels))
        fig = plt.figure(dpi = 300)
        clustergrid = sns.clustermap(data)
        plt.title(out_name)
        plt.savefig(os.path.join(self.out_dir, "{}.png".format(out_name)), bbox_inches = "tight") 
        
        reordered_rows = clustergrid.dendrogram_row.reordered_ind
        reordered_cols = clustergrid.dendrogram_col.reordered_ind

        fig = plt.figure(dpi = 300, figsize = (1, 2))
        labels = labels[reordered_rows]
        labels = np.tile(labels, (50, 1))
        plt.imshow(labels, cmap = mpl.colors.ListedColormap(colors))
        plt.imshow(labels, cmap = mpl.colors.ListedColormap(colors))
        plt.axis("off")
        plt.savefig(self.out_dir + "variation_bar.png", bbox_inches = "tight")
        
        return reordered_rows, reordered_cols
    
    def draw_heatmap(self, data, out_name, rows, cols):
        fig = plt.figure(dpi = 300)
        data1 = data[rows]
        data2 = data1[:, cols]
        sns.heatmap(
            data2,
            xticklabels = False,
            yticklabels = False
        )
        plt.title(out_name)
        plt.savefig(os.path.join(self.out_dir, "{}.png".format(out_name)), bbox_inches = "tight") 
        
