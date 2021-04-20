import os
import argparse
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

parser.add_argument('--dataset', type=str, default='movielens', help='Choose dataset from {movielens, lastfm}')
parser.add_argument('--kg', type=str, default='dbpedia', help='Choose knowledge graph from {wikidata, dbpedia}')
parser.add_argument('--neighbor_sample_size', type=int, default=16, help='the number of neighbors to be sampled')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_iter', type=int, default=1, help='number of layers / number of iterations when computing entity representation')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--show_topk', type=bool, default=False, help='Use topk eval or not')


# Logging
args = parser.parse_args()
save_dir = 'log/{}-{}-neighborsample{}/iter{}_dim{}_{}-aggregator_l2{}_epochs{}_batch{}_lr{}/'.format(args.dataset, args.kg, args.neighbor_sample_size, args.n_iter, args.dim, args.aggregator, args.l2_weight , args.n_epochs, args.batch_size, args.lr)
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
