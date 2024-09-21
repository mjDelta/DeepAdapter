import os
import torch
import matplotlib
from tqdm import tqdm
from torch import nn, optim

import deepadapter.utils.utils as UT
import deepadapter.utils.triplet as TRP
import deepadapter.utils.decomposition_utils as DPU
from deepadapter.models.dl_utils import *

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, train_loader, val_loader, test_loader, ae, fbatch, wnt_label2name, unw_label2name, net_args, out_dir):
		super(Trainer, self).__init__()
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader

		self.ae = ae
		self.fbatch = fbatch
		self.net_args = net_args

		self.optimizers = {
			'ae': ScheduledOptim(optim.Adam(ae.parameters(), betas=(0.9, 0.98), eps=1e-09),
				self.net_args.lr_lower_ae, self.net_args.lr_upper_ae, 70),
			"fbatch": ScheduledOptim(optim.Adam(fbatch.parameters(), betas=(0.9, 0.98), eps=1e-09),
				self.net_args.lr_lower_batch, self.net_args.lr_upper_batch, 10)
			}

		self.criterion_ae_rec = nn.L1Loss()
		self.criterion_ae_tri = nn.TripletMarginLoss(p = 2)
		self.criterion_fbatch = nn.CrossEntropyLoss()

		self.ae.apply(weight_init)
		self.fbatch.apply(weight_init)
		print(f"Params of AE: {net_param(self.ae)}; Params of Discriminator: {net_param(self.fbatch)}")

		self.wnt_label2name = wnt_label2name
		self.unw_label2name = unw_label2name

		self.out_dir = out_dir
		self.loss_path = os.path.join(self.out_dir, "loss.csv")
		self.loss_png = os.path.join(self.out_dir, "loss.png")

		# # get the triplet function by num_unw_infs
		# if len(self.unw_label2name) == 2:
		# 	self.trip_func = XXXXX
		# elif len(self.unw_label2name) > 2:
		# 	self.trip_func = TRP.get_triplet4_kSource

		### kSource works for 2 sources
		self.trip_func = TRP.get_triplet4_kSource

	def load_freeze_ae(self, pretrain_path):
		fz_keys = self.load_pretrained_ae(pretrain_path)
		## choose to freeze the pretrained params or not
		self.freeze_params_ae([])
		print("Load the pretrained AE")

	def load_pretrained_ae(self, pretrained_path):
		fintuned_keys = ["map"]
		pretrained_dict = torch.load(pretrained_path) 
		model_dict = self.ae.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k[:3] not in fintuned_keys}
		model_dict.update(pretrained_dict) 
		self.ae.load_state_dict(model_dict)
		return list(pretrained_dict.keys())

	def load_trained_ae(self, trained_path):
		ae_sd = torch.load(trained_path)			
		self.ae.load_state_dict(ae_sd)
		self.ae.eval()

	def freeze_params_ae(self, freeze_keys):
		for name, param in self.ae.named_parameters():
			if name in freeze_keys:
				param.requires_grad = False
			else:
				param.requires_grad = True

	def check_unfreeze_params(self, model):
		for name, param in model.named_parameters():
			if param.requires_grad:
				print(name, param.requires_grad, param.shape)
			
	def _train_ae_warmup(self, rng, train_data_tensor, train_labels_hot_tensor, train_labels, train_ids, train_mutuals):
		self.optimizers["ae"].zero_grad()
		with torch.enable_grad():
			hidden, rec_x = self.ae(train_data_tensor, train_labels_hot_tensor)
			## reconstruction loss
			rec_loss = self.criterion_ae_rec(rec_x, train_data_tensor)
			## triplet loss with mutual nearest neighbors as the positive samples
			anc_idx, pos_idx, neg_idx = self.trip_func(rng, train_labels, train_ids, train_mutuals)		
			if len(anc_idx) != 0:
				anc, pos, neg = hidden[anc_idx], hidden[pos_idx], hidden[neg_idx]
				tri_loss = self.criterion_ae_tri(anc, pos, neg)
				## total ae loss
				ae_loss = rec_loss + tri_loss
			else:
				ae_loss = rec_loss
			ae_loss.backward()
			self.optimizers["ae"].step_and_update_lr()
			ae_l = ae_loss.item()
		return ae_l

	def _train_dis(self, train_data_tensor, train_labels_hot_tensor, train_labels_tensor):
		self.optimizers["fbatch"].zero_grad()
		with torch.no_grad():
			hidden, rec_x = self.ae(train_data_tensor, train_labels_hot_tensor)
		with torch.enable_grad():
			batch_pred = self.fbatch(hidden)
			batch_loss = self.criterion_fbatch(batch_pred, train_labels_tensor)
			batch_loss.backward()
			self.optimizers["fbatch"].step_and_update_lr()
			b_l = batch_loss.item()	
		return b_l

	def _train_ae(self, rng, train_data_tensor, train_labels_hot_tensor, train_labels_tensor, train_labels, train_ids, train_mutuals):
		self.optimizers["ae"].zero_grad()
		with torch.enable_grad():
			hidden, rec_x = self.ae(train_data_tensor, train_labels_hot_tensor)
			## reconstruction loss
			rec_loss = self.criterion_ae_rec(rec_x, train_data_tensor)
			## triplet loss
			anc_idx, pos_idx, neg_idx = self.trip_func(rng, train_labels, train_ids, train_mutuals)
			if len(anc_idx) != 0:
				anc, pos, neg = hidden[anc_idx], hidden[pos_idx], hidden[neg_idx]
				tri_loss = self.criterion_ae_tri(anc, pos, neg)
				## total ae loss
				ae_loss = rec_loss + tri_loss
			else:
				ae_loss = rec_loss
			ae_loss *= self.net_args.lambda_ae
			batch_pred = self.fbatch(hidden)
			batch_loss = self.net_args.lambda_batch*self.criterion_fbatch(batch_pred, train_labels_tensor)
			ae_loss -= batch_loss
			ae_loss.backward()
			self.optimizers["ae"].step_and_update_lr()
			ae_l = ae_loss.item()
		return ae_l		

	def train_on_step(self, e, rng, train_data_tensor, train_labels_hot_tensor, train_labels_tensor, train_labels, train_ids, train_mutuals):
		ae_l, b_l = 0, 0
		if e < self.net_args.ae_epochs:
			ae_l = self._train_ae_warmup(rng, train_data_tensor, train_labels_hot_tensor, train_labels, train_ids, train_mutuals)

		if e >= self.net_args.ae_epochs:
			b_l = self._train_dis(train_data_tensor, train_labels_hot_tensor, train_labels_tensor)

		if e > self.net_args.ae_epochs + self.net_args.batch_epochs:
			ae_l = self._train_ae(rng, train_data_tensor, train_labels_hot_tensor, train_labels_tensor, train_labels, train_ids, train_mutuals)
		return ae_l, b_l

	def val_on_step(self, rng, val_data_tensor, val_labels_hot_tensor, val_labels_tensor, val_labels, val_ids, val_mutuals):
		val_ae = None
		with torch.no_grad():
			hidden, rec_x = self.ae(val_data_tensor, val_labels_hot_tensor)
			## reconstruction loss
			rec_loss = self.criterion_ae_rec(rec_x, val_data_tensor)
			## triplet loss
			anc_idx, pos_idx, neg_idx = self.trip_func(rng, val_labels, val_ids, val_mutuals)
			if len(anc_idx) != 0:
				anc, pos, neg = hidden[anc_idx], hidden[pos_idx], hidden[neg_idx]
				tri_loss = self.criterion_ae_tri(anc, pos, neg)
				## total ae loss
				ae_loss = rec_loss + tri_loss
			else:
				ae_loss = rec_loss
			ae_loss *= self.net_args.lambda_ae  
				
			batch_pred = self.fbatch(hidden)
			batch_loss = self.net_args.lambda_batch*self.criterion_fbatch(batch_pred, val_labels_tensor)
			ae_loss -= batch_loss
			val_ae = ae_loss.item()
		return val_ae

	def test_on_step(self, test_data_tensor):
		rec_x = self.ae.dec(self.ae.enc(test_data_tensor))
		return rec_x

	def _record_res(self, path, line, mode = "a+"):
		with open(path, mode) as f:
			f.write(line)
			f.flush()

	def _save_model(self):
		torch.save(self.ae.state_dict(), os.path.join(self.out_dir, "ae.tar"))
		torch.save(self.fbatch.state_dict(), os.path.join(self.out_dir, "fbatch.tar"))		

	def fit(self, train_mutuals, val_mutuals):
		os.makedirs(self.out_dir, exist_ok = True)
		self._record_res(self.loss_path, "epoch,ae_loss,batch_loss,val_ae\n", "w+")

		best_loss = 9999999
		rng = np.random.RandomState(0)
		pbar = tqdm(range(self.net_args.epochs), ncols = 100)
		for e in pbar:
			self.ae.train()
			self.fbatch.train()
			total_ae, total_b = 0, 0
			for data in self.train_loader:
				train_data, train_labels, train_bios, train_ids, train_labels_hot = data
				train_data_tensor = torch.FloatTensor(train_data).cuda()
				train_labels_tensor = torch.LongTensor(train_labels).cuda()
				train_labels_hot_tensor = torch.FloatTensor(train_labels_hot).cuda()
				ae_l, b_l = self.train_on_step(e, rng, 
					train_data_tensor, train_labels_hot_tensor, train_labels_tensor, train_labels, train_ids, train_mutuals)
				total_ae += ae_l
				total_b += b_l
			### to do: find the batch number of dataloader
			# total_ae /= iteration
			# total_b /= iteration

			self.ae.eval()
			self.fbatch.eval()
			val_ae = 0
			for data in self.val_loader:
				val_data, val_labels, val_bios, val_ids, val_labels_hot = data
				val_data_tensor = torch.FloatTensor(val_data).cuda()
				val_labels_tensor = torch.LongTensor(val_labels).cuda()
				val_labels_hot_tensor = torch.FloatTensor(val_labels_hot).cuda()
				val_ae += self.val_on_step(rng, 
					val_data_tensor, val_labels_hot_tensor, val_labels_tensor, val_labels, val_ids, val_mutuals)
			### to do: find the batch number of dataloader
			# val_ae /= iteration		
			
			self._record_res(self.loss_path, "{},{},{},{}\n".format(e, total_ae, total_b, val_ae))
			pbar.set_postfix({"e": e, "tot_ae": total_ae, "tot_b": total_b, "val_ae":val_ae})
			
			if e > self.net_args.ae_epochs + self.net_args.batch_epochs:
				if val_ae < best_loss:
					self._save_model()
					best_loss = val_ae
					print(f"Update model, loss of {val_ae}")

	def test(self, dataloader):
		test_rec_arrs, test_arrs, test_wnt_infs, test_unw_infs = [], [], [], []
		self.ae.eval()
		self.fbatch.eval()
		for data in dataloader:
			test_data, test_labels, test_bios, test_ids, test_labels_hot = data
			test_data_tensor = torch.FloatTensor(test_data).cuda()
			test_rec_tensor = self.test_on_step(test_data_tensor)
			test_rec_arrs.append(test_rec_tensor.detach().cpu().numpy())
			test_arrs.append(test_data)
			test_wnt_infs.extend(list(test_bios))
			test_unw_infs.extend(list(test_labels))
		data = np.vstack(test_arrs)
		normed_data = np.vstack(test_rec_arrs)
		test_wnt_infs = np.array(test_wnt_infs)
		test_unw_infs = np.array(test_unw_infs)
		return data, normed_data, test_wnt_infs, test_unw_infs

	def evaluate(self, record_path, test_name, dataloader):
		print("trainer:  " + self.out_dir)
		data, normed_data, test_wnt_infs, test_unw_infs = self.test(dataloader)

		## quantatitive results analysis
		self._record_res(record_path, "test_name,test_num,align,asw_var,nmi,ari\n")
		align, asw_var, nmi, ari = self.evaluate_quantative(normed_data, test_wnt_infs, test_unw_infs)
		self._record_res(record_path, 
						 f"{test_name},{len(test_wnt_infs)},{align},{asw_var},{nmi},{ari}\n")
		return data, normed_data, test_wnt_infs, test_unw_infs

	def evaluate_quantative(self, data, test_wnt_infs, test_unw_infs):
		trans_aligned, align = self._tsne_analysis(data, test_unw_infs, self.unw_label2name, "test_normed_variation")
		_, _ = self._tsne_analysis(data, test_wnt_infs, self.wnt_label2name, "test_normed_signal")

		n_cluster = len(set(test_wnt_infs))
		nmi = UT.NMI_index(trans_aligned, test_wnt_infs, n_cluster = n_cluster)
		ari = UT.ARI_index(trans_aligned, test_wnt_infs, n_cluster = n_cluster)
		asw_var = 1 - UT.ASW_index(trans_aligned, test_unw_infs)
		return align, asw_var, nmi, ari

	def _tsne_analysis(self, data, labels, label2name, png_name):
		colors = self._get_umap_colors(len(set(labels)))
		fitter = "umap"
		trans_aligned, align = DPU.decom_plot_nosplit(
			data, labels,
			os.path.join(self.out_dir, "{}_{}.png".format(fitter, png_name)), colors = colors, title = png_name, 
			label2name = label2name, fitter = fitter)
		return trans_aligned, align

	def _get_umap_colors(self, num_color):
		rng = np.random.RandomState(0)
		colors = ["#57904B", "violet",  "#C93C2A", "#372A8F"]
		tot_colors = list(matplotlib.colors.cnames.keys())
		while len(colors) < num_color:
			c = rng.choice(tot_colors, 1)
			if c not in colors:
				colors.append(c)
		colors = colors[:num_color]
		return colors

	def draw_loss(self):
		df = pd.read_csv(self.loss_path)
		xs = df["epoch"].values.squeeze()
		ys = df["ae_loss"].values.squeeze()
		fig = plt.figure(dpi = 300, figsize = (5, 4))
		plt.plot(xs, ys)
		plt.savefig(self.loss_png, bbox_inches = "tight")
