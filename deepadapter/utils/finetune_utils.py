import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from deepadapter.utils import triplet as TRP
from deepadapter.models.trainer import Trainer
from deepadapter.models.data_loader import TransData
from deepadapter.utils.data_utils import data_split_random
from deepadapter.models.dl_utils import AE, FBatch

def test_pretrained(test_num, trainer, test_name, out_dir, n_test = 1000):
	os.makedirs(out_dir, exist_ok = True)
	record_path = os.path.join(out_dir, "res_test_distibution.csv")

	trainer._record_res(record_path, "test_name,test_num,test_i,align,asw_var,nmi,ari\n")
	data, normed_data, participants, labels = trainer.test()

	rng = np.random.RandomState(0)
	all_idxs = np.arange(len(data))
	rng.shuffle(all_idxs)
	for i in tqdm(range(n_test), ncols = 80):
		test_idxs = rng.choice(all_idxs, test_num, replace = False)
		test_data, test_labels, test_participants = normed_data[test_idxs], labels[test_idxs], participants[test_idxs]
		align, asw_var, nmi, ari = trainer.evaluate_quantative(test_data, test_participants, test_labels)
		trainer._record_res(record_path, f"{test_name},{len(test_participants)},{i},{align},{asw_var},{nmi},{ari}\n")


def test_finetune(data, labels, labels_hot, participants, ids, label2bat, load_dir, out_dir, net_args, num_platform, test_num, n_test = 1000):
	rng = np.random.RandomState(0)
	all_idxs = np.arange(len(data))
	rng.shuffle(all_idxs)
	record_path = os.path.join(out_dir, f"finetune_{n_test}_res.csv")
	for i in tqdm(range(n_test), ncols = 80):
		ft_idxs = rng.choice(all_idxs, test_num, replace = False)
		ft_data, ft_labels, ft_participants, ft_labels_hot, ft_ids = data[ft_idxs], labels[ft_idxs], participants[ft_idxs], labels_hot[ft_idxs], ids[ft_idxs]
		# # ## train & val & test split with random split
		train_data, train_labels, train_labels_hot, \
			val_data, val_labels, val_labels_hot, \
			test_data, test_labels, test_labels_hot, \
			train_ids, val_ids, test_ids, \
			tot_train_val_idxs, tot_train_idxs, tot_val_idxs, tot_test_idxs = data_split_random(ft_data, ft_labels, ft_labels_hot, ft_ids, train_val_ratio = 0.5)
		
		train_mutuals = TRP.find_MNN_cosine_kSources(train_data, train_labels, train_ids)
		val_mutuals = TRP.find_MNN_cosine_kSources(val_data, val_labels, val_ids)
		print(len(participants), len(set(participants)), len(ids), len(set(ids)))

		train_bios, val_bios, test_bios = ft_participants[tot_train_idxs], ft_participants[tot_val_idxs], ft_participants[tot_test_idxs]
		train_trans = TransData(train_data, train_labels, train_bios, train_ids, train_labels_hot)
		train_loader = DataLoader(train_trans, batch_size = net_args.batch_size, collate_fn = train_trans.collate_fn, shuffle = True, drop_last = False)
		val_trans = TransData(val_data, val_labels, val_bios, val_ids, val_labels_hot)
		val_loader = DataLoader(val_trans, batch_size = net_args.batch_size, collate_fn = val_trans.collate_fn, shuffle = False, drop_last = False)
		test_trans = TransData(test_data, test_labels, test_bios, test_ids, test_labels_hot)
		test_loader = DataLoader(test_trans, batch_size = net_args.batch_size, collate_fn = test_trans.collate_fn, shuffle = False, drop_last = False)

		in_dim = ft_data.shape[1]
		ae = AE(in_dim, net_args.hidden_dim, num_platform, net_args.z_dim, net_args.drop).cuda()
		fbatch = FBatch(net_args.hidden_dim, num_platform, net_args.z_dim, net_args.drop).cuda()

		bio_label2bat = {t:t for t in set(participants)}
		sub_out = os.path.join(out_dir, f"finetune")
		trainer = Trainer(train_loader, val_loader, test_loader, ae, fbatch, bio_label2bat, label2bat, net_args, sub_out)
		trainer.load_freeze_ae(os.path.join(load_dir, "ae.tar"))
		# trainer.check_unfreeze_params(trainer.ae)
		# trainer.check_unfreeze_params(trainer.fbatch)

		trainer.fit(train_mutuals, val_mutuals)
		trainer.evaluate(record_path, f"finetune{i}")
		