import os
import numpy as np
import collections

def load_data(args):
    
    print("loading data...")
    
    train_data, eval_data, test_data, n_users, n_items = load_rating(args)
    n_entities, n_relations, n_triples, adj_entity, adj_relation = load_kg(args)
    
    print(f"size of train, eval and test set: [{len(train_data), len(eval_data), len(test_data)}]")
    print(f"number of users and items: [{n_users, n_items}]")
    print(f"number of entities, relations and triples: [{n_entities, n_relations, n_triples}]")
    print("done.\n")

    return n_users, n_items, n_entities, n_relations, train_data, eval_data, test_data, adj_entity, adj_relation

def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    train_file = '../../data/' + args.dataset + '/' + args.kg + '/train.txt'
    eval_file = '../../data/' + args.dataset + '/' + args.kg + '/eval.txt'
    test_file = '../../data/' + args.dataset + '/' + args.kg + '/test.txt'

    train_data = np.loadtxt(train_file , dtype=np.int32)
    eval_data = np.loadtxt(eval_file , dtype=np.int32)
    test_data = np.loadtxt(test_file , dtype=np.int32)
    
    # get user and item statistics
    n_users = max(max(train_data[:, 0]), max(eval_data[:, 0]), max(test_data[:, 0])) + 1 
    n_items = max(max(train_data[:, 1]), max(eval_data[:, 1]), max(test_data[:, 1])) + 1 

    return train_data, eval_data, test_data, n_users, n_items
    
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
    kg = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(args, kg, n_entities)

    return n_entities, n_relations, n_triples, adj_entity, adj_relation

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
