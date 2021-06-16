import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=0,help='number of threads for data loading')

# Preprocess parameters
parser.add_argument('--n_labels', type=int, default=2,help='number of classes')
parser.add_argument('--weight', type=str, default=None, help='model init weight')
parser.add_argument('--upper', type=int, default=1000, help='')
parser.add_argument('--lower', type=int, default=-200, help='')
#parser.add_argument('--norm_factor', type=float, default=200.0, help='')
#parser.add_argument('--expand_slice', type=int, default=20, help='')
#parser.add_argument('--min_slices', type=int, default=48, help='')
#parser.add_argument('--slice_down_scale', type=float, default=1.0, help='')
#parser.add_argument('--valid_rate', type=float, default=0.2, help='')

# data in/out and dataset
parser.add_argument('--data_path', default = 'raw_dataset', help='dataset root path')
parser.add_argument('--pred_path', default = 'output/pred', help='pred path')
parser.add_argument('--save', default='output', help='save path of trained model')
parser.add_argument('--batch_size', type=int, default=6, help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=1, metavar='N',help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.001)')
parser.add_argument('--early-stop', default=30, type=int, help='early stopping (default: 30)')
parser.add_argument('--crop_size', type=int, default=64)
#parser.add_argument('--val_crop_max_size', type=int, default=96)

# test
#parser.add_argument('--test_cut_size', type=int, default=48, help='size of sliding window')
#parser.add_argument('--test_cut_stride', type=int, default=24, help='stride of sliding window')
parser.add_argument('--postprocess', type=bool, default=True, help='post process')


args = parser.parse_args()


