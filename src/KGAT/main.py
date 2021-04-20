import os
import argparse
import tensorflow as tf
import numpy as np
import random
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
random.seed(555)

parser = argparse.ArgumentParser()

# movie
parser.add_argument('--dataset', type=str, default='movielens', help='Choose dataset from {movielens, lastfm}')
parser.add_argument('--kg', type=str, default='wikidata', help='Choose knowledge graph from {wikidata, dbpedia}')
parser.add_argument('--adj_type', type=str, default='si', help='Choose normalization technique from {si, bi}')
parser.add_argument('--layer_size', nargs='?', default='[64, 32, 16]', help='dimension of every layer. len(*) is number of layers used')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--rel_dim', type=int, default=64, help='dimension of relation embeddings')
parser.add_argument('--aggregator', type=str, default='bi', help='which aggregator to use from {bi, gcn, graphsage}')
parser.add_argument('--node_dropout', type=float, default= 0.1, help='node dropout ratio')
parser.add_argument('--mess_dropout', type=float, default= 0.1, help='message dropout ratio')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of l2 regularization')
parser.add_argument('--kge_weight', type=float, default=1e-5, help='weight of l2 regularization')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=10240, help='CF batch size')
parser.add_argument('--batch_size_kg', type=int, default=20480, help='KG batch size')
parser.add_argument('--show_topk', type=bool, default=False, help='Use topk eval or not')

# Logging
args = parser.parse_args()
save_dir = 'log/{}-{}/layer{}_dim{}_reldim{}_{}-aggregator_l2{}_l2kge{}_epochs{}_lr{}_batch{}_l2batch{}/'.format(args.dataset, args.kg, '-'.join(args.layer_size), args.dim, args.rel_dim, args.aggregator, args.l2_weight, args.kge_weight, args.n_epochs, args.lr, args.batch_size, args.batch_size_kg)
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