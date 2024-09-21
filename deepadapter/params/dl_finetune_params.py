import argparse

def load_dl_params():
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default = 5000, type = int) ## 15K
	parser.add_argument("--ae_epochs", default = 400, type = int)
	parser.add_argument("--batch_epochs", default = 50, type = int)
	parser.add_argument("--batch_size", default = 256, type = int)
	parser.add_argument("--hidden_dim", default = 256, type = int)
	parser.add_argument("--z_dim", default = 128, type = int)
	parser.add_argument("--drop", default = 0.3, type = float)
	parser.add_argument("--lr_lower_ae", default = 1e-5, type = float)
	parser.add_argument("--lr_upper_ae", default = 5e-4, type = float)
	parser.add_argument("--lr_lower_batch", default = 1e-5, type = float)
	parser.add_argument("--lr_upper_batch", default = 5e-4, type = float)
	parser.add_argument("--lambda_batch", default = 0.01, type = float)
	parser.add_argument("--lambda_ae", default = 1, type = float)
	parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
	args = parser.parse_args()
	return args
