import os
import numpy as np
import pandas as pd

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



