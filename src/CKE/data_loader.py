import os
import numpy as np
import collections

def load_data(args):
    
    print("loading data...")
    
    train_data, eval_data, test_data, n_users, n_items = load_rating(args)
    n_entities, n_relations, n_triples, kg_head_dict = load_kg(args)
    
    print(f"size of train, eval and test set: [{len(train_data), len(eval_data), len(test_data)}]")
    print(f"number of users and items: [{n_users, n_items}]")
    print(f"number of entities, relations and triples in kg: [{n_entities, n_relations, n_triples}]")
    print("done.\n")
    
    return n_users, n_items, n_entities, n_relations, n_triples, train_data, eval_data, test_data, kg_head_dict

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
    n_user = max(max(train_data[:, 0]), max(eval_data[:, 0]), max(test_data[:, 0])) + 1 
    n_item = max(max(train_data[:, 1]), max(eval_data[:, 1]), max(test_data[:, 1])) + 1 
    
    return train_data, eval_data, test_data, n_user, n_item

def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../../data/' + args.dataset + '/' + args.kg + '/kg_final.txt'
    kg_np = np.loadtxt(kg_file, dtype=np.int32)

    # get kg statistics
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    n_triples = len(kg_np)
    
    # construct knowledge graph
    kg_head_dict = construct_kg(kg_np)

    return n_entity, n_relation, n_triples, kg_head_dict

def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg_head_dict = collections.defaultdict(list)
    # treat the KG as an undirected graph
    for head, relation, tail in kg_np:
        kg_head_dict[head].append((tail, relation))
        kg_head_dict[tail].append((head, relation))
    return kg_head_dict