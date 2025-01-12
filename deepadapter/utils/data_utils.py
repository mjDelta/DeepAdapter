import os, sys, matplotlib
import pandas as pd
import numpy as np
from tqdm import tqdm

class LoadTransData:
	"""docstring for ClassName"""
	def __init__(self):
		super(LoadTransData, self).__init__()

		self.quartet_dir = "data/batch_data/"
		self.lincs_dir = "data/batch_data/LDS-1593/Data"
		self.platform_dir = "data/platform_data"
		self.purity_dir = "data/purity_data"

		self.platform_bioSig_path = "data/platform_data/cl_ids.csv"
		self.purity_bioSig_path = "data/purity_data/Celligner_info.csv"

		self.ensg_path = "data/batch_data/ensg2symbol.csv"

	def load_ensg2symbol(self):
		ensg_df = pd.read_csv(self.ensg_path)
		ensg2symbol = dict(ensg_df.values)
		return ensg2symbol

	def load_bioSig_platform(self):
		bio_df = pd.read_csv(self.platform_bioSig_path)
		bio_df["ID"] = bio_df["ID"].astype("int")

		disease_set = sorted(set(bio_df["primary_disease"].values.tolist()))
		disease2label, label2disease = {}, {}
		for i, d in enumerate(disease_set):
			disease2label[d] = i
			label2disease[i] = d
		if "Unknown" not in disease2label:
			_num_dis = len(disease2label)
			disease2label["Unknown"] = _num_dis
			label2disease[_num_dis] = "Unknown"

		return bio_df, disease2label, label2disease

	def load_bioLabel_platform(self, ids1, ids2):
		bio_df, disease2label, label2disease = self.load_bioSig_platform()

		dis_labels, diseases = [], []
		for id_ in list(ids1) + list(ids2):
			mask = bio_df["ID"] == int(id_)
			if np.sum(mask) == 0:
				dis_labels.append(disease2label["Unknown"])
				diseases.append("Unknown")
			else:
				disease = bio_df[mask]["primary_disease"].values[0]
				dis_labels.append(disease2label[disease])
				diseases.append(disease)
		dis_labels = np.array(dis_labels)
		diseases = np.array(diseases)
		return dis_labels, diseases, disease2label, label2disease

	def load_data_platform(self, platform):
		assert platform in ["rna", "arr"]
		path = os.path.join(self.platform_dir, "cl_{}.csv".format(platform))
		data_raw = pd.read_csv(path)
		# ## delete the duplicated samples
		data_raw = data_raw[~data_raw.duplicated(["ID"], keep = "first")]
		data_raw.sort_values(by = ["ID"], inplace = True)
		ids = data_raw.pop("ID").values.squeeze().tolist()
		return data_raw, ids

	def load_data_purity(self, purity):
		path = os.path.join(self.purity_dir, "{}_rna.csv".format(purity))
		data_raw = pd.read_csv(path)
		ids = data_raw.pop("ID").values.squeeze().tolist()
		return data_raw, ids

	def load_bioSig_purity(self):
		bio_df = pd.read_csv(self.purity_bioSig_path)
		return bio_df

	def load_bioLabel_purity(self, ids1, ids2):
		bio_df = self.load_bioSig_purity()
		dises1 = [self.get_disease_by_id_purity(bio_df, id_, "CL_tumor_class") for id_ in ids1]
		dises2 = [self.get_disease_by_id_purity(bio_df, id_, "lineage") for id_ in ids2]
		all_dis_set = sorted(set(dises1 + dises2))
		dis2label, label2dis = {}, {}
		for i, d in enumerate(all_dis_set):
			dis2label[d] = i
			label2dis[i] = d
		diseases = np.array(dises1 + dises2)
		dis_labels = np.array([dis2label[t] for t in diseases])
		return dis_labels, diseases, dis2label, label2dis

	def get_disease_by_id_purity(self, df, id_, col_name):
		mask = df["sampleID"] == id_
		this_df = df[mask]
		dis = this_df[col_name].values[0]
		dis = str(dis)
		return dis

	def load_data_quartet(self, renameENSG = True):
		info_path = os.path.join(self.quartet_dir, "OMIX002254_studydesign.csv")
		data_path = os.path.join(self.quartet_dir, "OMIX002254-01.csv")
		info_df = pd.read_csv(info_path, sep = ",")
		info_df["sequencing_id"] = info_df["sequencing_id"].apply(lambda x: x[21:])

		data_raw = pd.read_csv(data_path, sep = ",").transpose()
		data_raw.columns = data_raw.iloc[0]
		data_raw = data_raw.iloc[1:]
		data_raw.reset_index(inplace = True)
		data_raw.rename({"index": "ID"}, axis = 1, inplace = True)

		## extract batch information & disease information
		ids = data_raw.pop("ID").values.squeeze()
		batches = [info_df[info_df["sequencing_id"] == id_]["batch"].values[0] for id_ in ids]
		donors = [info_df[info_df["sequencing_id"] == id_]["biospecimen_name"].values[0] for id_ in ids]
		print(f"Load quartet, size of {data_raw.shape}")

		## rename quartet columns
		if renameENSG:
			ensg2symbol = self.load_ensg2symbol()
			data_raw = data_raw.rename(columns=ensg2symbol)
			print(f"Rename quartet's columns to ENSG symbol")
		return data_raw, ids, batches, np.array(donors)

	def load_lincs_lds1593(self):
		info_path = os.path.join(self.lincs_dir, "PromoCell-DGE-RNAseq-Experiment-Design.DCIC.xlsx")
		sheets = ["SR1", "SR2", "SR4", "SR5"]
		info_dfs = []
		for i, sheet in enumerate(sheets):
			tmp = pd.read_excel(info_path, sheet_name = sheet)
			tmp["batch"] = i
			info_dfs.append(tmp)
		info_df = pd.concat(info_dfs)
		info_df["des1"] = info_df["State"].str.cat(info_df["Drug.Conc.1"].map(str) + info_df["Drug.Conc.Unit.1"].map(str), sep = "_").str.cat(info_df["Drug.Conc.2"].map(str) + info_df["Drug.Conc.Unit.2"].map(str), sep = "_")
		info_df["des"] = info_df["Cell"].str.cat(info_df["des1"], sep = "_")

		data_dfs = []
		bat2label = {
			"Sequencing Experiment Number 1": 0,
			"Sequencing Experiment Number 2": 1,
			"Sequencing Experiment Number 4": 2,
			"Sequencing Experiment Number 5": 3}
		trans_data_dir = os.path.join(self.lincs_dir, "Level3/Level3")
		for dir_ in os.listdir(trans_data_dir):
			trans_data_dir_ = os.path.join(trans_data_dir, dir_)
			for file in os.listdir(trans_data_dir_):
				if dir_ == "Sequencing Experiment Number 5":
					for file_ in os.listdir(os.path.join(trans_data_dir_, file)):
						if "Human" not in file_:
							continue
						file_path = os.path.join(trans_data_dir_, file, file_)
						batch = bat2label[dir_]
						data_df = self.read_data_lds1593(file_path, batch)
						data_dfs.append(data_df)
				else:	   
					if "Human" not in file:
						continue
					file_path = os.path.join(trans_data_dir_, file)
					batch = bat2label[dir_]
					data_df = self.read_data_lds1593(file_path, batch)
					data_dfs.append(data_df)
		data_raw = pd.concat(data_dfs)
		data_raw = data_raw.dropna(axis='columns')
		data_raw = data_raw.drop_duplicates()

		batches = data_raw.pop("batch").values.squeeze()
		wells = data_raw.pop("Well").values.squeeze()
		cells = data_raw.pop("Cell").values.squeeze()
		data_raw.pop("info")

		infos = []
		for cell, well, batch in zip(cells, wells, batches):
			this_info1 = info_df[info_df["batch"] == batch]
			this_info2 = this_info1[this_info1["Well"] == well]
			if len(this_info2) == 0:
				print(cell, well, batch)
			infos.append(this_info2["des"].values[0])
		infos = np.array(infos)
		print(f"Load LINCS LDS 1593, size of {data_raw.shape}")

		test_infos = []
		for info in sorted(set(infos)):
			bs = batches[infos == info]
			if len(set(bs)) != 1:
				test_infos.append(info)
		return data_raw, batches, wells, cells, infos, test_infos
		
	def read_data_lds1593(self, file_path, batch):
		data_df = pd.read_csv(file_path, sep = "\t").transpose().reset_index()
		data_df.columns = data_df.iloc[0]
		data_df = data_df.iloc[1:]
		data_df.rename({"Gene": "info"}, inplace = True, axis = 1)
		data_df["batch"] = batch
		data_df["Well"] = data_df["info"].apply(lambda x: x.split(".")[-1])
		data_df["Cell"] = data_df["info"].apply(lambda x: x.split(".")[1])
		## exclude normalized data
		data_df = data_df[data_df["Well"] != "Norm"]
		return data_df

	def load_data(self, expression_path, annotate_path, sample_id = "SampleID", uwn_col = "Unwanted_var", wnt_col = "Biological_sig"):
		exp_df = pd.read_csv(expression_path)
		ann_df = pd.read_csv(annotate_path)
		data_df = exp_df.join(ann_df, how = "inner", on = sample_id)
		ids = data_df.pop(sample_id).values.squeeze()
		unwanted_labels = data_df.pop(uwn_col).values.squeeze()
		wanted_labels = data_df.pop(wnt_col).values.squeeze()
		return data, ids, unwanted_labels, wanted_labels
		
class PrepTransData:
	"""docstring for PrepTransData"""
	def __init__(self):
		super(PrepTransData, self).__init__()

	def sample_log(self, raw_data):
		raw_data_plus_1 = raw_data.values + 1
		raw_data_plus_1 = raw_data_plus_1.astype("float64")
		data = np.log(raw_data_plus_1)
		return data

	def sample_norm(self, df):
		data_sum = np.expand_dims(df.sum(axis = 1).values, axis = 1)
		df = df / data_sum * 10000.
		return df

	def sort_genes_return_cols(self, df):
		mean = df.mean(axis = 0).values.squeeze()
		cols = np.array(list(df.columns))
		argidxs = np.argsort(mean)
		sorted_cols = cols[argidxs]
		return sorted_cols

	def sort_genes(self, df_pvt, df_oth):
		sorted_cols = self.sort_genes_return_cols(df_pvt)
		df_pvt, df_oth = df_pvt[sorted_cols], df_oth[sorted_cols]
		return df_pvt, df_oth, sorted_cols

	def sort_genes_sgl_df(self, df):
		sorted_cols = self.sort_genes_return_cols(df)
		return df[sorted_cols], sorted_cols

	def label2onehot(self, batches):
		bat2label = {b:i for i, b in enumerate(sorted(set(batches)))}
		label2bat = {i:b for i, b in enumerate(sorted(set(batches)))}
		unwanted_labels = np.array([bat2label[t] for t in batches])
		unwanted_onehot = transform_hot(unwanted_labels)
		return bat2label, label2bat, unwanted_labels, unwanted_onehot

def extract_intersected_genes(df1, df2):
	df1_cols, df2_cols = list(df1.columns), list(df2.columns)
	intersect_gene = list(set(df1_cols).intersection(set(df2_cols)))
	df1, df2 = df1[intersect_gene], df2[intersect_gene]
	print(f"Extract intersected genes: df1'size {df1.shape}, df2'size{df2.shape}")
	return df1, df2

def transform_hot(ys):
	ys_set = set(ys)
	hots = np.eye(len(ys_set))[ys]
	return hots	

def data_split_random_notest(data, labels, labels_onehot, ids, val_ratio = 0.2):
	rng = np.random.RandomState(0)
	all_ids = np.arange(len(data))
	all_idxs = np.arange(len(data))
	rng.shuffle(all_idxs)

	tot_val_idxs = rng.choice(all_idxs, size = int(val_ratio*len(all_idxs)), replace = False)
	tot_train_idxs = np.array([t for t in all_idxs if t not in tot_val_idxs])

	print(tot_train_idxs.shape, tot_val_idxs.shape)

	train_data, train_labels, train_labels_hot = data[tot_train_idxs], labels[tot_train_idxs], labels_onehot[tot_train_idxs]
	val_data, val_labels, val_labels_hot = data[tot_val_idxs], labels[tot_val_idxs], labels_onehot[tot_val_idxs]
	train_ids, val_ids = ids[tot_train_idxs], ids[tot_val_idxs]
	print(train_data.shape, train_labels_hot.shape)
	print(val_data.shape, val_labels_hot.shape)
	return train_data, train_labels, train_labels_hot, \
	val_data, val_labels, val_labels_hot, \
	train_ids, val_ids, \
	tot_train_idxs, tot_val_idxs
	
def data_split_random(data, labels, labels_onehot, ids, train_val_ratio = 0.8, val_ratio = 0.2):
	rng = np.random.RandomState(0)
	all_ids = np.arange(len(data))
	all_idxs = np.arange(len(data))
	rng.shuffle(all_idxs)

	tot_train_val_idxs = rng.choice(all_idxs, size = int(train_val_ratio*len(all_idxs)), replace = False)
	tot_test_idxs = np.array([t for t in all_idxs if t not in tot_train_val_idxs])
	tot_val_idxs = rng.choice(tot_train_val_idxs, size = int(val_ratio*len(tot_train_val_idxs)), replace = False)
	tot_train_idxs = np.array([t for t in tot_train_val_idxs if t not in tot_val_idxs])

	print(tot_train_idxs.shape, tot_val_idxs.shape, tot_test_idxs.shape)

	train_data, train_labels, train_labels_hot = data[tot_train_idxs], labels[tot_train_idxs], labels_onehot[tot_train_idxs]
	val_data, val_labels, val_labels_hot = data[tot_val_idxs], labels[tot_val_idxs], labels_onehot[tot_val_idxs]
	test_data, test_labels, test_labels_hot = data[tot_test_idxs], labels[tot_test_idxs], labels_onehot[tot_test_idxs]
	train_ids, val_ids, test_ids = ids[tot_train_idxs], ids[tot_val_idxs], ids[tot_test_idxs]
	print(train_data.shape, train_labels_hot.shape)
	print(val_data.shape, val_labels_hot.shape)
	print(test_data.shape, test_labels_hot.shape)
	return train_data, train_labels, train_labels_hot, \
	val_data, val_labels, val_labels_hot, \
	test_data, test_labels, test_labels_hot, \
	train_ids, val_ids, test_ids, \
	tot_train_val_idxs, tot_train_idxs, tot_val_idxs, tot_test_idxs

def data_split_lds1593(data, labels, labels_onehot, ids, infos, test_infos, val_ratio = 0.2):
	rng = np.random.RandomState(0)
	tot_test_idxs = np.array([t for t, info in enumerate(infos) if info in test_infos])
	tot_train_val_idxs = np.array([t for t, info in enumerate(infos) if info not in test_infos])

	tot_val_idxs = rng.choice(tot_train_val_idxs, size = int(val_ratio*len(tot_train_val_idxs)), replace = False)
	tot_train_idxs = np.array([t for t in tot_train_val_idxs if t not in tot_val_idxs])

	train_data, train_labels, train_labels_hot = data[tot_train_idxs], labels[tot_train_idxs], labels_onehot[tot_train_idxs]
	val_data, val_labels, val_labels_hot = data[tot_val_idxs], labels[tot_val_idxs], labels_onehot[tot_val_idxs]
	test_data, test_labels, test_labels_hot= data[tot_test_idxs], labels[tot_test_idxs], labels_onehot[tot_test_idxs]
	train_ids, val_ids, test_ids = infos[tot_train_idxs], infos[tot_val_idxs], infos[tot_test_idxs]
	print(train_data.shape, train_labels_hot.shape)
	print(val_data.shape, val_labels_hot.shape)
	print(test_data.shape, test_labels_hot.shape)

	return train_data, train_labels, train_labels_hot, \
	val_data, val_labels, val_labels_hot, \
	test_data, test_labels, test_labels_hot, \
	train_ids, val_ids, test_ids, \
	tot_train_val_idxs, tot_train_idxs, tot_val_idxs, tot_test_idxs

def data_split_platform(data, labels, labels_onehot, dis_labels, ids1, ids2, train_ratio = 0.8):
	rng = np.random.RandomState(0)
	## paired data
	inte_ids = sorted(set(ids1).intersection(set(ids2)))
	tot_ids = np.array(list(ids1) + list(ids2))
	tot_train_val_idxs, tot_test_idxs = [], []
	for idx, id_ in enumerate(tot_ids):
		if id_ in inte_ids:
			tot_test_idxs.append(idx)
		else:
			tot_train_val_idxs.append(idx)

	tot_train_idxs = rng.choice(tot_train_val_idxs, size = int(train_ratio*len(tot_train_val_idxs)), replace = False)
	tot_val_idxs = np.array([t for t in tot_train_val_idxs if t not in tot_train_idxs])
	tot_test_idxs = np.array(tot_test_idxs)
	train_data, train_labels, train_labels_hot, train_dis_labels, train_ids = data[tot_train_idxs], labels[tot_train_idxs], labels_onehot[tot_train_idxs], dis_labels[tot_train_idxs], tot_ids[tot_train_idxs]
	val_data, val_labels, val_labels_hot, val_dis_labels, val_ids = data[tot_val_idxs], labels[tot_val_idxs], labels_onehot[tot_val_idxs], dis_labels[tot_val_idxs], tot_ids[tot_val_idxs]
	test_data, test_labels, test_labels_hot, test_dis_labels, test_ids = data[tot_test_idxs], labels[tot_test_idxs], labels_onehot[tot_test_idxs], dis_labels[tot_test_idxs], tot_ids[tot_test_idxs]
	print(train_data.shape, train_labels_hot.shape)
	print(val_data.shape, val_labels_hot.shape)
	print(test_data.shape, test_labels_hot.shape)

	return train_data, train_labels, train_labels_hot, \
	val_data, val_labels, val_labels_hot, \
	test_data, test_labels, test_labels_hot, \
	train_ids, val_ids, test_ids, \
	train_dis_labels, val_dis_labels, test_dis_labels, \
	tot_train_val_idxs, tot_train_idxs, tot_val_idxs, tot_test_idxs
