import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TransData(Dataset):
	"""docstring for TransData"""
	def __init__(self, gene_exps, unw_infs, bio_infs, ids, unw_inf_hots):
		super(TransData, self).__init__()
		self.gene_exps = gene_exps
		self.unw_infs = unw_infs
		self.bio_infs = bio_infs
		self.ids = ids
		self.unw_inf_hots = unw_inf_hots

	def __len__(self):
		return len(self.gene_exps)

	def __getitem__(self, idx):
		outs = [
		self.gene_exps[idx], self.unw_infs[idx], self.bio_infs[idx],
		self.ids[idx], self.unw_inf_hots[idx]]
		return outs

	def collate_fn(self, data):
		b_gene_exps = np.array([t[0] for t in data])
		b_unw_infs = np.array([t[1] for t in data])
		b_bio_infs = np.array([t[2] for t in data])
		b_ids = np.array([t[3] for t in data])
		b_unw_inf_hots = np.array([t[4] for t in data])
		return b_gene_exps, b_unw_infs, b_bio_infs, b_ids, b_unw_inf_hots
