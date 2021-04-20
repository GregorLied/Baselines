import os
import argparse
import tensorflow as tf
import numpy as np
from time import time
from data_loader import load_data
from train import train

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)
        
def get_filepath(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return file_path

np.random.seed(555)

parser = argparse.ArgumentParser()

# movie
parser.add_argument('--dataset', type=str, default='movielens', help='Choose dataset from {movielens, lastfm}')
parser.add_argument('--kg', type=str, default='wikidata', help='Choose knowledge graph from {wikidata, dbpedia}')
parser.add_argument('--dim', type=int, default=4, help='dimension of user and entity embeddings')
parser.add_argument('--rel_dim', type=int, default=4, help='dimension of relation embeddings')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of l2 regularization')
parser.add_argument('--kge_weight', type=float, default=1e-5, help='weight of l2 regularization')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='CF batch size')
parser.add_argument('--show_topk', type=bool, default=False, help='Use topk eval or not')
# Note: batch_size_kg will be setup accordingly based on the batch_size and size of the training set

# Logging
args = parser.parse_args()
save_dir = 'log/{}-{}/dim{}_reldim{}__l2{}_l2kge{}_epochs{}_lr{}_batch{}/'.format(args.dataset, args.kg, args.dim, args.rel_dim, args.l2_weight, args.kge_weight, args.n_epochs, args.lr, args.batch_size)
ensureDir(save_dir)
args.log = get_filepath(save_dir)

# Verbose
show_loss = False
show_time = False
show_topk = args.show_topk

# Training
t = time()
data = load_data(args)
train(args, data, show_loss, show_topk)

if show_time:
    print('time used: %d s' % (time() - t))