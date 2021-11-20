import argparse
from utils.utils import load_config
from train import * 


def main():
	parser = argparse.ArgumentParser(description = "Training ...")
	parser.add_argument('--data_path',type=str,default='./data/experiment-i',help='Path to your experiment-i folder')
	parser.add_argument('--training_type',type=str,default='loso',help='Training type, solo or kfold')
	parser.add_argument('--model',type=str,default='resnet_50',help='What model u want to train')
	parser.add_argument('--loss',type=str,default='contrastive_center_loss',help='loss name')
	parser.add_argument('--epoch_n',type=int,default=5)
	parser.add_argument('--lr',type=float,default=0.0001)
	parser.add_argument('l2_regularization',type=float,default=0.002)
	parser.add_argument('lambda',type=float,default=1)
	parser.add_argument('batch_size',type=int,default=64)
	parser.add_argument('save_checkpoint',type=bool,default=False)
	args = vars(parser.parse_args())
	loss_config = load_config('./config.yml')
	if args['training_type'] == 'loso':
		train_loso(args,loss_config)
	else:
		train_kfold(args,loss_config)






