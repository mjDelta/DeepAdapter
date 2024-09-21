import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

def find_nn(vec, arr):
    dist = np.linalg.norm(vec-arr, axis = 1)
    argmin = np.argmin(dist)
    return argmin
	
def find_MNN_cosine_kSources(raw_data, raw_labels, raw_ids):
    '''
    data: [sample_number x gene_number]
    '''
    ## normalization
    data = raw_data.copy()
    norm = np.linalg.norm(data, ord = 2, axis = 1, keepdims = True)
    data /= norm
    ## find mutual pairs for k sources
    label_set = sorted(set(raw_labels))
    data_list = {l: raw_data[raw_labels == l] for l in label_set}
    ids_list = {l: raw_ids[raw_labels == l] for l in label_set}
    
    mutuals_list = {l: {} for l in label_set}
    
    for i in tqdm(label_set, ncols = 80):
        ## assign pivot data
        pivot_data, pivot_ids = data_list[i], ids_list[i]
        ## calculate mnn pairs
        for j in range(len(pivot_data)):
            i_idx = label_set.index(i)
            for k in label_set[i_idx+1:]:
                argmin2 = find_nn(pivot_data[j], data_list[k])
                argmin1 = find_nn(data_list[k][argmin2], pivot_data)
                if argmin1 == j:
                    arg_id1 = pivot_ids[argmin1]
                    arg_id2 = ids_list[k][argmin2]
                    
                    if arg_id1 not in mutuals_list[i]:
                        mutuals_list[i][arg_id1] = [arg_id2]
                    else:
                        mutuals_list[i][arg_id1].append(arg_id2)
                    
                    if arg_id2 not in mutuals_list[k]:
                        mutuals_list[k][arg_id2] = [arg_id1]
                    else:
                        mutuals_list[k][arg_id2].append(arg_id1)
    return mutuals_list

def get_triplet4_kSource(rng, labels, ids, mutuals_list, per_num = 2):
    '''
    labels: batch labels，数据源标签
    ids: batch ids，病人标签
    '''
    ## An --> anchor; Pn --> positive, assigned by mutual NN; Am --> negative = 同数据源，不同样本（可以同癌种，可以不同癌种）
    anc_list, pos_list, neg_list = [], [], []
    for i, id_ in enumerate(ids):
        ## generate positives
        pos_idxs = get_positive_mutualkSource(id_, labels[i], mutuals_list, ids, labels)
        if len(pos_idxs) == 0:
            continue
        ## generate negatives
        for pos_idx in pos_idxs:
            ### same data source & different sample
            l = labels[i]
            l_mask = (labels == l).astype("bool")
            id_mask = (ids != id_).astype("bool")
            mask = l_mask & id_mask
            mask = mask.astype("bool")
            neg_idxs = list(np.where(mask)[0])        
            if len(neg_idxs) > per_num:
                neg_idxs = rng.choice(neg_idxs, per_num, replace = False)
            neg_len = len(neg_idxs)
            
            anc_list.extend([i]*neg_len)
            pos_list.extend([pos_idx]*neg_len)
            neg_list.extend(neg_idxs)

    anc_list = np.array(anc_list)
    pos_list = np.array(pos_list)
    neg_list = np.array(neg_list)

    return torch.LongTensor(anc_list).cuda(), torch.LongTensor(pos_list).cuda(), torch.LongTensor(neg_list).cuda()


def get_positive_mutualkSource(target_id, target_label, mutuals_list, ids, labels):        
    if target_id not in mutuals_list[target_label]:
        return []
    
    pos_ids = mutuals_list[target_label][target_id]

    pos_idx_list, markers = [], []
    for pos_id in pos_ids:
        if pos_id not in ids:
            pos_idx_list.append(-1)
            markers.append(False)
        else:
            mask = ids == pos_id
            pos_idxs = np.where(mask)[0]
            pos_idx = -1
            for idx in pos_idxs:
                if labels[idx] != target_label:
                    pos_idx = idx
            if pos_idx == -1:
                pos_idx_list.append(-1)
                markers.append(False)
            else:
                pos_idx_list.append(pos_idx)
                markers.append(True)
    pos_idx_list, markers = np.array(pos_idx_list), np.array(markers)
    return pos_idx_list[markers]
