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

# Movie
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movielens', help='Choose dataset from {movielens, lastfm}')
parser.add_argument('--kg', type=str, default='wikidata', help='Choose knowledge graph from {wikidata, dbpedia}')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop / neighborhood sampling') # [8, 16, 32, 64] 32
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops / layers') #H [2, 3] 2
parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings') #d [8, 16, 32]
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term') #lambda1 [10-7, 10-6, 10âˆ’5, 10-4, 10-3, 10-2]
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term') #lambda2 [10-7, 10-6, 10âˆ’5, 10-4, 10-3, 10-2]
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
parser.add_argument('--n_epochs', type=int, default=1000, help='the number of epochs') 
parser.add_argument('--lr', type=float, default=0.01, help='learning rate') #n [0.05, 0.01, 0.005, 0.001]
parser.add_argument('--batch_size', type=int, default=2048, help='batch size') # [256, 512, 1024, 2048, 4096]
parser.add_argument('--show_topk', type=bool, default=False, help='Use topk eval or not')

# Logging
args = parser.parse_args()
save_dir = 'log/{}-{}-memories{}/hops{}_dim{}_l2{}_kge{}_epochs{}_lr{}_batch{}/'.format(args.dataset, args.kg, args.n_memory, args.n_hop, args.dim, args.l2_weight, args.kge_weight, args.n_epochs, args.lr, args.batch_size)
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
