import umap
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from deepadapter.utils.utils import alignment_score

def decom_plot_nosplit(data, labels, save_path, title = "", fitter = "tsne", colors = ["red", "blue"], perplexity = 30, label2name = None, min_dist = 0.99, size = 20, metric = "euclidean", n_neighbors = 15):
    label_set = sorted(set(labels))
    if fitter == "tsne":
        fitter = TSNE(random_state = 42, perplexity = perplexity)
    elif fitter == "umap":
        fitter = umap.UMAP(random_state = 42, min_dist = min_dist, metric = metric, n_neighbors = n_neighbors)
    else:
        raise("Unk fitter of {}".format(fitter))
        
    trans_data = fitter.fit_transform(data)
    align_score = alignment_score(trans_data, labels)
    print(label2name, set(labels))
    fig = plt.figure(figsize = (7, 5))
    for l, c in zip(label_set, colors):
        mask = labels == l
        plt.scatter(trans_data[mask][:, 0], trans_data[mask][:, 1], edgecolor = c, color = c, 
                    s = size,
                    linewidths = 0.5, label = label2name[l], alpha = 0.8)

    legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(legend_labels, legend_handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.title("n = {}, {}".format(len(data), align_score))
    plt.savefig(save_path, bbox_inches = "tight")
    
    # print(save_path, align_score)
    return trans_data, align_score

