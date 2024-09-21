from deepadapter.utils import triplet as TRP
from deepadapter.models.trainer import Trainer
from deepadapter.models.data_loader import TransData, DataLoader
from deepadapter.models.dl_utils import AE, FBatch

def train(train_list, val_list, test_list, label2unw, label2wnt, net_args, out_dir):
	train_data, train_labels, train_bios, train_ids, train_labels_hot = train_list
	val_data, val_labels, val_bios, val_ids, val_labels_hot = val_list
	test_data, test_labels, test_bios, test_ids, test_labels_hot = test_list

	## initialize models
	in_dim = train_data.shape[1]
	num_unw_vars = len(label2unw)
	ae = AE(in_dim, net_args.hidden_dim, num_unw_vars, net_args.z_dim, net_args.drop).cuda()
	fbatch = FBatch(net_args.hidden_dim, num_unw_vars, net_args.z_dim, net_args.drop).cuda()

	## intialize dataloader
	train_trans = TransData(train_data, train_labels, train_bios, train_ids, train_labels_hot)
	train_loader = DataLoader(train_trans, batch_size = net_args.batch_size, collate_fn = train_trans.collate_fn, shuffle = True, drop_last = False)
	val_trans = TransData(val_data, val_labels, val_bios, val_ids, val_labels_hot)
	val_loader = DataLoader(val_trans, batch_size = net_args.batch_size, collate_fn = val_trans.collate_fn, shuffle = False, drop_last = False)
	test_trans = TransData(test_data, test_labels, test_bios, test_ids, test_labels_hot)
	test_loader = DataLoader(test_trans, batch_size = net_args.batch_size, collate_fn = test_trans.collate_fn, shuffle = False, drop_last = False)

	## intialize trainer
	trainer = Trainer(train_loader, val_loader, test_loader, ae, fbatch, label2wnt, label2unw, net_args, out_dir)

	## intialize MNN
	train_mutuals = TRP.find_MNN_cosine_kSources(train_data, train_labels, train_ids)
	val_mutuals = TRP.find_MNN_cosine_kSources(val_data, val_labels, val_ids)

	## begin training
	trainer.fit(train_mutuals, val_mutuals)

	return trainer