import os
import argparse
import numpy as np
from time import time
from data_loader import load_data
from train import train
import gc

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

def sw_early_stopping(cur_sw_auc, cur_sw_epoch, best_sw_auc, best_sw_epoch, stopping=3):
    if cur_sw_auc > best_sw_auc:
        best_sw_auc = cur_sw_auc
        best_sw_epoch = cur_sw_epoch
    if cur_sw_epoch - best_sw_epoch >= stopping:
        print("Stage-Wise Early Stopping triggered. No improvements since %d stage-wise epochs." % stopping)
        should_stop_sw = True
    else:
        should_stop_sw = False
    return best_sw_auc, best_sw_epoch, should_stop_sw

np.random.seed(555)

# Movie
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movielens', help='Choose dataset from {movielens, lastfm}')
parser.add_argument('--kg', type=str, default='wikidata', help='Choose knowledge graph from {wikidata, dbpedia}')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--p_hop', type=int, default=2, help='number of iterations when computing user representation')
parser.add_argument('--n_mix_hop', type=int, default=2, help='number of wide layer mixing iterations when computing entity representation')
parser.add_argument('--h_hop', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--l2_agg_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate') 
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--show_topk', type=bool, default=False, help='Use topk eval or not')

args = parser.parse_args()

# Logging
args = parser.parse_args()
save_dir = 'log/{}-{}-memories{}-neighborsample{}/mix-hops{}_p-hops{}_h-hops{}_dim{}_l2{}l2agg{}_epochs{}_lr{}_batch{}/'.format(args.dataset, args.kg, args.n_memory, args.neighbor_sample_size, args.n_mix_hop, args.p_hop, args.h_hop, args.dim, args.l2_weight, args.l2_agg_weight, args.n_epochs, args.lr, args.batch_size)
ensureDir(save_dir)
args.log = get_filepath(save_dir)
ensureDir('parameters/')

# Verbose
show_loss = False
show_time = False
show_topk = args.show_topk

t = time()

# Algorithm 3 â€“ Regular Training
data = load_data(args)
load_pretrain_emb = False
sw_auc = train(args, data, load_pretrain_emb, show_loss, show_topk)
del data
gc.collect()

# Algorithm 3 â€“ Stage-Wise Training
best_sw_epoch = 0
best_sw_auc = sw_auc
load_pretrain_emb = True
for sw_epoch in range(1, 6):
    
    sw_training_log = "============Stage-Wise Training Epoch {}============".format(sw_epoch)
    print(sw_training_log)
    with open(args.log, 'a') as f:
        f.write(sw_training_log)
    data = load_data(args)
    sw_auc = train(args, data, load_pretrain_emb, show_loss, show_topk)
    del data
    gc.collect()
    
    best_sw_auc, best_sw_epoch, should_stop_sw = sw_early_stopping(sw_auc, sw_epoch, best_sw_auc, best_sw_epoch)
    if should_stop_sw:
        sw_early_stopping_log = "Stage-Wise Early Stopping triggered. Best stage: stage {}. Best epoch: epoch {}.".format(best_sw_epoch, best_epoch)
        print(sw_early_stopping_log)
        with open(args.log, 'a') as f:
            f.write(sw_early_stopping_log)
	
if show_time:
    print('time used: %d s' % (time() - t))
