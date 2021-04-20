import os
import random
import numpy as np
import collections
import multiprocessing as mp
from functools import partial

def load_data(args):
    train_data, eval_data, test_data, n_users, n_items, user_history_dict = load_rating(args)
    n_entities, n_relations, n_triples, adj_entity, adj_relation, kg_head_dict = load_kg(args)
    ripple_set = get_ripple_set(args, kg_head_dict, user_history_dict)

    print(f"size of train, eval and test set: [{len(train_data), len(eval_data), len(test_data)}]")
    print(f"number of users and items: [{n_users, n_items}]")
    print(f"number of entities, relations and triples: [{n_entities, n_relations, n_triples}]")
    print("done.\n")

    return n_users, n_items, n_entities, n_relations, train_data, eval_data, test_data, adj_entity, adj_relation, ripple_set
    
import pandas as pd
def load_pre_data(args):
    path = '../../data/' + args.dataset + '/' + args.kg + '/'
    train_data = pd.read_csv(f'{path}train_pd.csv',index_col=None)
    train_data = train_data.drop(train_data.columns[0], axis=1)
    train_data = train_data[['user','item','like']].values

    eval_data = pd.read_csv(f'{path}eval_pd.csv',index_col=None)
    eval_data = eval_data.drop(eval_data.columns[0], axis=1)
    eval_data = eval_data[['user','item','like']].values

    test_data = pd.read_csv(f'{path}test_pd.csv',index_col=None)
    test_data = test_data.drop(test_data.columns[0], axis=1)
    test_data = test_data[['user','item','like']].values
    return train_data, eval_data, test_data

def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    train_file = '../../data/' + args.dataset + '/' + args.kg + '/train.txt'
    eval_file = '../../data/' + args.dataset + '/' + args.kg + '/eval.txt'
    test_file = '../../data/' + args.dataset + '/' + args.kg + '/test.txt'

    #train_data = np.loadtxt(train_file , dtype=np.int32)
    #eval_data = np.loadtxt(eval_file , dtype=np.int32)
    #test_data = np.loadtxt(test_file , dtype=np.int32)
    
    train_data, eval_data, test_data = load_pre_data(args)
    
    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for interaction in train_data:
        user = interaction[0]
        item = interaction[1]
        rating = interaction[2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)
            
    # get user and item statistics
    n_users = max(max(train_data[:, 0]), max(eval_data[:, 0]), max(test_data[:, 0])) + 1 
    n_items = max(max(train_data[:, 1]), max(eval_data[:, 1]), max(test_data[:, 1])) + 1 
    
    return train_data, eval_data, test_data, n_users, n_items, user_history_dict

def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../../data/' + args.dataset + '/' + args.kg + '/kg_final.txt'
    kg_np = np.loadtxt(kg_file, dtype=np.int64)

    # get kg statistics
    n_entities = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relations = len(set(kg_np[:, 1]))
    n_triples = len(kg_np)

    # construct adjacency matrix
    kg_head_dict = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(args, kg_head_dict, n_entities)

    return n_entities, n_relations, n_triples, adj_entity, adj_relation, kg_head_dict

def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg_head_dict = collections.defaultdict(list)
    # treat the KG as an undirected graph
    for head, relation, tail in kg_np:
        kg_head_dict[head].append((tail, relation))
        kg_head_dict[tail].append((head, relation))
    return kg_head_dict

def construct_adj(args, kg, n_entities):
    print('constructing adjacency matrix ...')
    # This is refered to as the (single-layer) receptive field S(v) in the paper
    # dim [n_entities, neighbor_sample_size]
    # each line of adj_entity stores the sampled neighbor entities for a given entity. n_entities lines.
    # each line of adj_relation stores the corresponding sampled neighbor relations. n_entities lines.
    adj_entity = np.zeros([n_entities, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([n_entities, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(n_entities):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        # Note: S(v) may contain duplicates if N(v) < K, where K is neighbor_sample_size.
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation

def get_ripple_set(args, kg_head_dict, user_history_dict):
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)
    
    global g_kg_head_dict
    g_kg_head_dict = kg_head_dict
    with mp.Pool(processes=min(mp.cpu_count(), 12)) as pool:
        job = partial(_get_ripple_set, p_hop=args.p_hop, n_memory=args.n_memory, n_neighbor=args.neighbor_sample_size)
        for user, user_ripple_set in pool.starmap(job, user_history_dict.items()):
            ripple_set[user] = np.array(user_ripple_set, dtype=np.int32)
    del g_kg_head_dict
    return ripple_set

def _get_ripple_set(user, user_history, p_hop=2, n_memory=32, n_neighbor=32):
    user_ripple_set = []
    for h in range(p_hop):
        memories_h = []
        memories_r = []
        memories_t = []

        # get neighborhood (indices of all items/entities [h+1]-hops away) from the user  
        if h == 0:
            neighbors = user_history
        else:
            neighbors = user_ripple_set[-1][2]

        # create memory set
        for entity in neighbors:
            # for tail, relation in g_kg_head_dict[entity]: # Original RippleNet-Version: Takes to long for big datasets
            for tail, relation in random.sample(g_kg_head_dict[entity], min(len(g_kg_head_dict[entity]), n_neighbor)):
                memories_h.append(entity)
                memories_r.append(relation)
                memories_t.append(tail)

        # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
        # this won't happen for h = 0, because only the items that appear in the KG have been selected
        # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
        if len(memories_h) == 0:
            user_ripple_set.append(user_ripple_set[-1])
        else:
            # sample a fixed-size memory for each user
            # note: S(u) may contain duplicates if N(u) < K, where K is n_memory.
            if len(memories_h) >= n_memory:
                sampled_indices = np.random.choice(len(memories_h), size=n_memory, replace=False)
            else:
                sampled_indices = np.random.choice(len(memories_h), size=n_memory, replace=True)
            
            memories_h = [memories_h[i] for i in sampled_indices]
            memories_r = [memories_r[i] for i in sampled_indices]
            memories_t = [memories_t[i] for i in sampled_indices]
            user_ripple_set.append((memories_h, memories_r, memories_t))
            
    return [user, user_ripple_set]

"""
def get_ripple_set(args, kg_head_dict, user_history_dict):
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in tqdm(user_history_dict):
        for h in range(args.p_hop):
            memories_h = []
            memories_r = []
            memories_t = []
            
            # get neighborhood (indices of all items/entities [h+1]-hops away) from the user  
            if h == 0:
                neighbors = user_history_dict[user]
            else:
                neighbors = ripple_set[user][-1][2]

            # create memory set
            for entity in neighbors:
                for tail, relation in kg_head_dict[entity]:
                    memories_h.append(entity)
                    memories_r.append(relation)
                    memories_t.append(tail)

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                # sample a fixed-size memory for each user
                # note: S(u) may contain duplicates if N(u) < K, where K is n_memory.
                if len(memories_h) >= args.n_memory:
                    sampled_indices = np.random.choice(len(memories_h), size=args.n_memory, replace=False)
                else:
                    sampled_indices = np.random.choice(len(memories_h), size=args.n_memory, replace=True)
            
                memories_h = [memories_h[i] for i in sampled_indices]
                memories_r = [memories_r[i] for i in sampled_indices]
                memories_t = [memories_t[i] for i in sampled_indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set
"""