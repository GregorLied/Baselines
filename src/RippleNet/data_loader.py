import collections
import os
import numpy as np

def load_data(args):
    
    print("loading data...")
    
    train_data, eval_data, test_data, n_users, n_items, user_history_dict = load_rating(args)
    n_entities, n_relations, n_triples, kg_head_dict = load_kg(args)
    ripple_set = get_ripple_set(args, kg_head_dict, user_history_dict)
    
    print(f"size of train, eval and test set: [{len(train_data), len(eval_data), len(test_data)}]")
    print(f"number of users and items: [{n_users, n_items}]")
    print(f"number of entities, relations and triples: [{n_entities, n_relations, n_triples}]")
    print("done.\n")
    
    return n_users, n_items, n_entities, n_relations, train_data, eval_data, test_data, ripple_set

def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    train_file = '../../data/' + args.dataset + '/' + args.kg + '/train.txt'
    eval_file = '../../data/' + args.dataset + '/' + args.kg + '/eval.txt'
    test_file = '../../data/' + args.dataset + '/' + args.kg + '/test.txt'

    train_data = np.loadtxt(train_file , dtype=np.int32)
    eval_data = np.loadtxt(eval_file , dtype=np.int32)
    test_data = np.loadtxt(test_file , dtype=np.int32)
    
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
    kg_np = np.loadtxt(kg_file, dtype=np.int32)

    # get kg statistics
    n_entities = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relations = len(set(kg_np[:, 1]))
    n_triples = len(kg_np)
    
    # construct knowledge graph
    kg_head_dict = construct_kg(kg_np)

    return n_entities, n_relations, n_triples, kg_head_dict

def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg

def get_ripple_set(args, kg, user_history_dict):
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(args.n_hop):
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
                for tail, relation in kg[entity]:
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
