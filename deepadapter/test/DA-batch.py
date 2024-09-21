import os, sys
import pandas as pd
import numpy as np

from deepadapter.utils import data_utils as DT
from deepadapter.utils import utils as UT
from deepadapter.utils import triplet as TRP
from deepadapter.models.trainer import Trainer
from deepadapter.models.data_loader import TransData, DataLoader
from deepadapter.models.dl_utils import AE, FBatch
from deepadapter.params import dl_params as DLPARAM

### load data
loadTransData = DT.LoadTransData()
data, batches, wells, donors, infos, test_infos = loadTransData.load_lincs_lds1593()
ids = np.arange(len(data))

### preprocessing
prepTransData = DT.PrepTransData()
raw_df = prepTransData.sample_norm(data)
raw_df, sorted_cols = prepTransData.sort_genes_sgl_df(raw_df)
input_arr = prepTransData.sample_log(raw_df)
bat2label, label2bat, unwanted_labels, unwanted_onehot = prepTransData.label2onehot(batches)


### out paths
db_name = "LDS1593"
out_dir = os.path.join("../model_testcodes/batch_data", "deepAligner_LINCS_batch/ln_pre_15K_{}/".format(db_name))
os.makedirs(out_dir, exist_ok = True)


### train & val & test for LINCS LDS-1593
train_data, train_labels, train_labels_hot, \
    val_data, val_labels, val_labels_hot, \
    test_data, test_labels, test_labels_hot, \
    train_ids, val_ids, test_ids, \
    tot_train_val_idxs, tot_train_idxs, tot_val_idxs, tot_test_idxs = DT.data_split_lds1593(input_arr, unwanted_labels, unwanted_onehot, ids, infos, test_infos)

train_bios, val_bios, test_bios = donors[tot_train_idxs], donors[tot_val_idxs], donors[tot_test_idxs]

net_args = DLPARAM.load_dl_params()

## initialize models
in_dim = input_arr.shape[1]
num_unw_vars = len(bat2label)
ae = AE(in_dim, net_args.hidden_dim, num_unw_vars, net_args.z_dim, net_args.drop).cuda()
fbatch = FBatch(net_args.hidden_dim, num_unw_vars, net_args.z_dim, net_args.drop).cuda()

## initialize dataloader
train_trans = TransData(train_data, train_labels, train_bios, train_ids, train_labels_hot)
train_loader = DataLoader(train_trans, batch_size = net_args.batch_size, collate_fn = train_trans.collate_fn, shuffle = True, drop_last = False)
val_trans = TransData(val_data, val_labels, val_bios, val_ids, val_labels_hot)
val_loader = DataLoader(val_trans, batch_size = net_args.batch_size, collate_fn = val_trans.collate_fn, shuffle = False, drop_last = False)
test_trans = TransData(test_data, test_labels, test_bios, test_ids, test_labels_hot)
test_loader = DataLoader(test_trans, batch_size = net_args.batch_size, collate_fn = test_trans.collate_fn, shuffle = False, drop_last = False)

## initialize trainer
bio_label2bat = {t:t for t in set(train_bios)}
trainer = Trainer(train_loader, val_loader, test_loader, ae, fbatch, bio_label2bat, label2bat, net_args, out_dir)

## initialize mutual neighbors
train_mutuals = TRP.find_MNN_cosine_kSources(train_data, train_labels, train_ids)
val_mutuals = TRP.find_MNN_cosine_kSources(val_data, val_labels, val_ids)
print(len(train_bios), len(set(train_bios)), len(train_ids), len(set(train_ids)))

## training
trainer.fit(train_mutuals, val_mutuals)

### load trained model & test
trainer.load_trained_ae(os.path.join(out_dir, "ae.tar"))

record_path = os.path.join(out_dir, "res.csv")
trainer.evaluate(record_path, db_name, test_loader)
