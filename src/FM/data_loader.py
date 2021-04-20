import os
import numpy as np
import collections
import scipy.sparse as sp

def load_data(args):
    
    print("loading data...")
    
    train_data, eval_data, test_data, n_users, n_items = load_rating(args)
    n_entities, n_relations, n_triples, kg_head_dict = load_kg(args)
    user_feature_matrix = create_user_feature_matrix(n_users)
    kg_feature_matrix = create_kg_feature_matrix(kg_head_dict, n_items, n_entities)
    
    print(f"size of train, eval and test set: [{len(train_data), len(eval_data), len(test_data)}]")
    print(f"number of users and items: [{n_users, n_items}]")
    print(f"number of entities, relations and triples in kg: [{n_entities, n_relations, n_triples}]")
    print("done.\n")
    
    return n_users, n_items, n_entities, train_data, eval_data, test_data, kg_head_dict, user_feature_matrix, kg_feature_matrix

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
    # treat the KG as a directed graph
    for head, relation, tail in kg_np:
        kg_head_dict[head].append((tail, relation))
    return kg_head_dict

def create_user_feature_matrix(n_users):
    # one-hot encoding for users.
    user_feature_matrix = sp.identity(n_users).tocsr()
    return user_feature_matrix

def create_kg_feature_matrix(kg_head_dict, n_items, n_entities):
    row = []
    col = []
    data = []

    for i_id in range(n_items):
        # one-hot encoding for items
        row.append(i_id)
        col.append(i_id)
        data.append(1)

        # multi-hot encoding for kg features of items
        if i_id not in kg_head_dict.keys(): 
            continue

        triples = kg_head_dict[i_id]
        for t_id, _ in triples:
            row.append(i_id)
            col.append(t_id)
            data.append(1.)

    kg_feature_matrix = sp.coo_matrix((data, (row, col)), shape=(n_items, n_entities)).tocsr()
    return kg_feature_matrix