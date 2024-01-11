import matplotlib, collections, umap
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from utils import alignment_score

def decom_plot_moredata_notrain(data, labels, save_path, title = "", fitter = "tsne", colors = ["red", "blue"], perplexity = 30, label2name = None, min_dist = 0.99, n_neighbors = 15, marker_size = 10, lw = 1):
    label_set = sorted(set(labels))
    if fitter == "tsne":
        fitter = TSNE(random_state = 42, perplexity = perplexity)
    elif fitter == "umap":
        fitter = umap.UMAP(random_state = 42, min_dist = min_dist, n_neighbors = n_neighbors)
    else:
        raise("Unk fitter of {}".format(fitter))
        
    trans_data = fitter.fit_transform(data)
    align_score = alignment_score(trans_data, labels)    
    fig = plt.figure(dpi = 300, figsize = (7, 5))
    ### plot train val with open circles
    for l, c in zip(label_set, colors):
        mask = labels == l
        plt.scatter(data[mask][:, 0], data[mask][:, 1], edgecolor = c, color = "none", 
                    s = marker_size, linewidths = lw, label = label2name[l], alpha = 0.8)

#     legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(legend_labels, legend_handles))
#     plt.legend(by_label.values(), by_label.keys())
    
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.title("n = {}".format(len(data)))
    plt.savefig(save_path, bbox_inches = "tight")
    
    print(save_path, align_score)
    return trans_data

def decom_plot_moredata(data, labels, train_val_idxs, test_idxs, save_path, title = "", fitter = "tsne", colors = ["red", "blue"], perplexity = 30, label2name = None, min_dist = 0.99, n_neighbors = 15, marker_size = 10, lw = 1, metric = "euclidean"):
    label_set = sorted(set(labels))
    if fitter == "tsne":
        fitter = TSNE(random_state = 42, perplexity = perplexity)
    elif fitter == "umap":
        fitter = umap.UMAP(random_state = 42, min_dist = min_dist, n_neighbors = n_neighbors, metric = metric)
    else:
        raise("Unk fitter of {}".format(fitter))
        
    trans_data = fitter.fit_transform(data)
    align_score = alignment_score(trans_data, labels)
    align_score_train = alignment_score(trans_data[train_val_idxs], labels[train_val_idxs])
    align_score_test = alignment_score(trans_data[test_idxs], labels[test_idxs])
    
    fig = plt.figure(dpi = 300, figsize = (7, 5))
    ### plot train val with open circles
    train_val_data, train_val_labels = trans_data[train_val_idxs], labels[train_val_idxs]
    for l, c in zip(label_set, colors):
        mask = train_val_labels == l
        plt.scatter(train_val_data[mask][:, 0], train_val_data[mask][:, 1], edgecolor = c, color = "none", 
                    s = marker_size, linewidths = lw, label = label2name[l], alpha = 0.8)
    
    test_data, test_labels = trans_data[test_idxs], labels[test_idxs]        
    for l, c in zip(label_set, colors):
        mask = test_labels == l
        plt.scatter(test_data[mask][:, 0], test_data[mask][:, 1], edgecolor = c, color = c, 
                    s = marker_size,  alpha = 0.8, lw = lw, label = label2name[l])

#     legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(legend_labels, legend_handles))
#     plt.legend(by_label.values(), by_label.keys())
    
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.title("n = {}".format(len(data)))
    plt.savefig(save_path, bbox_inches = "tight")
    
    print(save_path, align_score, align_score_train, align_score_test)
    return trans_data

def decom_plot_nosplit(data, labels, save_path, title = "", fitter = "tsne", colors = ["red", "blue"], perplexity = 30, label2name = None, min_dist = 0.99, size = 20, metric = "euclidean"):
    label_set = sorted(set(labels))
    if fitter == "tsne":
        fitter = TSNE(random_state = 42, perplexity = perplexity)
    elif fitter == "umap":
        fitter = umap.UMAP(random_state = 42, min_dist = min_dist, metric = metric)
    else:
        raise("Unk fitter of {}".format(fitter))
        
    trans_data = fitter.fit_transform(data)
    align_score = alignment_score(trans_data, labels)
    
    fig = plt.figure(dpi = 300, figsize = (7, 5))
    for l, c in zip(label_set, colors):
        print(l, c)
        mask = labels == l
        plt.scatter(trans_data[mask][:, 0], trans_data[mask][:, 1], edgecolor = c, color = c, 
                    s = size,
                    linewidths = 0.5, label = label2name[l], alpha = 0.8)

#     legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(legend_labels, legend_handles))
#     plt.legend(by_label.values(), by_label.keys())
    
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.title("n = {}".format(len(data)))
    plt.savefig(save_path, bbox_inches = "tight")
    
    print(save_path, align_score)
    return trans_data

def decom_plot(data, labels, train_val_idxs, test_idxs, save_path, title = "", fitter = "tsne", colors = ["red", "blue"], perplexity = 30, label2name = None, min_dist = 0.99):
    label_set = sorted(set(labels))
    if fitter == "tsne":
        fitter = TSNE(random_state = 42, perplexity = perplexity)
    elif fitter == "umap":
        fitter = umap.UMAP(random_state = 42, min_dist = min_dist)
    else:
        raise("Unk fitter of {}".format(fitter))
        
    trans_data = fitter.fit_transform(data)
    align_score = alignment_score(trans_data, labels)
    align_score_train = alignment_score(trans_data[train_val_idxs], labels[train_val_idxs])
    align_score_test = alignment_score(trans_data[test_idxs], labels[test_idxs])
    
#     fig = plt.figure(dpi = 300, figsize = (7, 5))
    fig = plt.figure(dpi = 300, figsize = (5.5, 5))
    ### plot train val with open circles
    train_val_data, train_val_labels = trans_data[train_val_idxs], labels[train_val_idxs]
    for l, c in zip(label_set, colors):
        mask = train_val_labels == l
        plt.scatter(train_val_data[mask][:, 0], train_val_data[mask][:, 1], edgecolor = "white", color = c, 
                    s = 20,
#                     s = 10,
                    linewidths = 0.5, label = label2name[l], alpha = 0.8)
    
    test_data, test_labels = trans_data[test_idxs], labels[test_idxs]        
    for l, c in zip(label_set, colors):
        mask = test_labels == l
        plt.scatter(test_data[mask][:, 0], test_data[mask][:, 1], edgecolor = "black", color = c, 
                    s = 20, 
#                     s = 10, 
                    alpha = 0.8, lw = 1, label = label2name[l])

#     legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(legend_labels, legend_handles))
#     plt.legend(by_label.values(), by_label.keys())
    
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.title("n = {}".format(len(data)))
    plt.savefig(save_path, bbox_inches = "tight")
    
    print(save_path, align_score, align_score_train, align_score_test)
    return trans_data

def decom_plot_colored_test(data, labels, train_val_idxs, test_idxs, save_path, title = "", colors = ["red", "blue"], perplexity = 30, min_dist = 0.99, fitter = "tsne"):
    label_set = sorted(set(labels))
    if fitter == "tsne":
        fitter = TSNE(random_state = 42, perplexity = perplexity)
    elif fitter == "umap":
        fitter = umap.UMAP(random_state = 42, min_dist = min_dist)
    else:
        raise("Unk fitter of {}".format(fitter))
        
    trans_data = fitter.fit_transform(data)
    align_score = alignment_score(trans_data, labels)
    align_score_train = alignment_score(trans_data[train_val_idxs], labels[train_val_idxs])
    align_score_test = alignment_score(trans_data[test_idxs], labels[test_idxs])
#     fig = plt.figure(dpi = 300, figsize = (7, 5))
    fig = plt.figure(dpi = 300, figsize = (5.5, 5))
    ## plot train val with grey circles
    train_val_data, train_val_labels = trans_data[train_val_idxs], labels[train_val_idxs]
    plt.scatter(train_val_data[:, 0], train_val_data[:, 1], 
                    edgecolor = "none", color = "gray", s = 20, linewidths = 0.01, label = "Training sample", alpha = 0.5)
    
    test_data, test_labels = trans_data[test_idxs], labels[test_idxs]
    print(set(test_labels))
    for l, c in zip(set(test_labels), colors):
        mask = test_labels == l
        print(l, c, mask.sum())
        plt.scatter(test_data[mask][:, 0], test_data[mask][:, 1], 
                    edgecolor = "black", color = c, s = 20, alpha = 0.9, lw = 1.5, label = l)

#     legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(legend_labels, legend_handles))
#     plt.legend(by_label.values(), by_label.keys())
    
#     ## plot lines for separated clusters
#     plt.vlines(7, -5, 20, color = "red")
#     plt.hlines(14.5, -3, 20, color = "red")
#     plt.vlines(8, -5, 20, color = "blue")
#     plt.hlines(4, -3, 20, color = "blue")
#     plt.vlines(10, -5, 20, color = "green")
#     plt.hlines(-1, -3, 20, color = "green")
    
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.title("{}, n = {}".format(title, len(data)))
    plt.savefig(save_path, bbox_inches = "tight")
    
    print(save_path, align_score, align_score_train, align_score_test)
    return trans_data

def decom_plot_all(data, labels, save_path, title = "cell line & tumor", colors = ["red", "blue"], trans = True, fitter = "tsne", min_dist = 0.99):
    label_set = sorted(set(labels))
    if trans:
        if len(data) <= 30:
            perplexity = 10
        else:
            perplexity = 30
        print(len(data), perplexity)
        if fitter == "tsne":
            fitter = TSNE(random_state = 42, perplexity = perplexity)
        elif fitter == "umap":
            fitter = umap.UMAP(random_state = 42, min_dist = min_dist)
        else:
            raise("Unk fitter of {}".format(fitter))
        trans_data = fitter.fit_transform(data)
        print("TSNE finished...")    
        align_score = alignment_score(trans_data, labels)
        print(save_path, align_score)
    else:
        trans_data = data
    
#     fig = plt.figure(dpi = 300, figsize = (7, 5))
    fig = plt.figure(dpi = 300, figsize = (5.5, 5))
    ### plot train val with open circles
    for l, c in tqdm(zip(label_set, colors), ncols = 80):
        mask = labels == l
        plt.scatter(trans_data[mask][:, 0], trans_data[mask][:, 1], 
                    edgecolor = c, color = c, alpha = 0.8,
                    s = 20, label = l)
    plt.title(title)
    
    legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(legend_labels, legend_handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(save_path, bbox_inches = "tight")    
    return trans_data