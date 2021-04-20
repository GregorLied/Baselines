import os
import numpy as np
import collections
import scipy.sparse as sp

def load_data(args):
    
    print("loading data...")
    
    train_data, eval_data, test_data, n_users, n_items = load_rating(args)
    n_entities, n_relations, n_triples, kg_relation_dict = load_kg(args)
    
    print(f"size of train, eval and test set: [{len(train_data), len(eval_data), len(test_data)}]")
    print(f"number of users and items: [{n_users, n_items}]")
    print(f"number of entities, relations and triples in kg: [{n_entities, n_relations, n_triples}]")

    ckg_head_dict, ckg_adj_list, all_h_list, all_r_list, all_t_list, all_v_list = get_ckg(args, train_data, kg_relation_dict, n_users, n_entities, n_relations)
    n_relations = n_relations * 2 + 2

    print(f"number of entities, relations in ckg: [{n_users + n_entities, n_relations}]")
    print("done.\n")
    
    return n_users, n_items, n_entities, n_relations, train_data, eval_data, test_data, ckg_head_dict, ckg_adj_list, all_h_list, all_r_list, all_t_list, all_v_list

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
    kg_relation_dict = construct_kg(kg_np)

    return n_entity, n_relation, n_triples, kg_relation_dict

def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg_relation_dict = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg_relation_dict[relation].append((head, tail))
    return kg_relation_dict

def get_ckg(args, train_data, kg_relation_dict, n_users, n_entities, n_relations):
    print('constructing collaborative knowledge graph ...')

    # adjacency matrices of the collaborative knowledge graph
    ckg_adj_list = []
    ckg_adj_r_list = []

    # filter train_data for relevant user-item interactions
    train_data = train_data[train_data[:, 2]==1]

    # user-item-interaction part of the ckg
    adj, adj_inv = _create_adj_mat(train_data, n_users, n_entities, row_remap=0, col_remap=n_users)
    ckg_adj_list.append(adj)
    ckg_adj_r_list.append(0)
    ckg_adj_list.append(adj_inv)
    ckg_adj_r_list.append(n_relations + 1)

    # knowledge graph part of the ckg
    for relation_id in kg_relation_dict.keys():
        adj, adj_inv = _create_adj_mat(np.array(kg_relation_dict[relation_id]), n_users, n_entities, row_remap=n_users, col_remap=n_users)
        ckg_adj_list.append(adj)
        ckg_adj_r_list.append(relation_id + 1)
        ckg_adj_list.append(adj_inv)
        ckg_adj_r_list.append(n_relations + relation_id + 2)

    # perform normalization
    if args.adj_type == 'bi':
        print('perform bi-normalization')
        ckg_adj_list = [_bi_norm_lap(adj) for adj in ckg_adj_list]
    else:
        print('perform si-normalization')
        ckg_adj_list = [_si_norm_lap(adj) for adj in ckg_adj_list]

    # construct ckg_head_dict and triple lists
    ckg_head_dict = collections.defaultdict(list)
    all_h_list, all_t_list, all_r_list, all_v_list = [], [], [], []
    for idx, adj in enumerate(ckg_adj_list):

        all_h_list += list(adj.row)
        all_t_list += list(adj.col)
        all_r_list += [ckg_adj_r_list[idx]] * len(adj.row)
        all_v_list += list(adj.data)

        rows = adj.row
        cols = adj.col

        for ckg_node_id in range(len(rows)):
            head = rows[ckg_node_id]
            tail = cols[ckg_node_id]
            relation = ckg_adj_r_list[idx]
            ckg_head_dict[head].append((tail, relation))

    assert len(all_h_list) == sum([len(adj.data) for adj in ckg_adj_list])

    # resort the all_h/t/r/v_list,
    # ... since tensorflow.sparse.softmax requires indices sorted in the canonical lexicographic order
    print('reordering indices...')

    org_h_dict = dict()
    for idx, h in enumerate(all_h_list):
        if h not in org_h_dict.keys():
            org_h_dict[h] = [[],[],[]]

        org_h_dict[h][0].append(all_t_list[idx])
        org_h_dict[h][1].append(all_r_list[idx])
        org_h_dict[h][2].append(all_v_list[idx])

    sorted_h_dict = dict()
    for h in org_h_dict.keys():
        org_t_list, org_r_list, org_v_list = org_h_dict[h]
        sort_t_list = np.array(org_t_list)
        sort_order = np.argsort(sort_t_list)

        sort_t_list = _reorder_list(org_t_list, sort_order)
        sort_r_list = _reorder_list(org_r_list, sort_order)
        sort_v_list = _reorder_list(org_v_list, sort_order)

        sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]

    od = collections.OrderedDict(sorted(sorted_h_dict.items()))
    new_all_h_list, new_all_t_list, new_all_r_list, new_all_v_list = [], [], [], []

    for h, vals in od.items():
        new_all_h_list += [h] * len(vals[0])
        new_all_t_list += list(vals[0])
        new_all_r_list += list(vals[1])
        new_all_v_list += list(vals[2])

    assert sum(new_all_h_list) == sum(all_h_list)
    assert sum(new_all_t_list) == sum(all_t_list)
    assert sum(new_all_r_list) == sum(all_r_list)
    
    return ckg_head_dict, ckg_adj_list, new_all_h_list, new_all_t_list, new_all_r_list, new_all_v_list

def _create_adj_mat(np_mat, n_user, n_entity, row_remap, col_remap):
    # number of nodes in ckg
    n_all = n_user + n_entity

    # remap items / entities, to ensure that 
    # [1,...,n_user] indices in the ckg belong to the users
    # [n_user + 1,...,n_entity] indices in the ckg belong to the items / entities
    row = np_mat[:, 0] + row_remap
    col = np_mat[:, 1] + col_remap
    data = [1.] * len(row)

    # create sparse adj matrix in coo format; treat the KG as an undirected graph
    adj = sp.coo_matrix((data, (row, col)), shape=(n_all, n_all))
    adj_inv = sp.coo_matrix((data, (col, row)), shape=(n_all, n_all))

    return adj, adj_inv

def _bi_norm_lap(adj):
    # Causing RuntimeWarning: divide by zero encountered in power
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # Seems to produce empty matrices:
    # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    # So replace it with:
    bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return bi_lap.tocoo()

def _si_norm_lap(adj):
    # Causing RuntimeWarning: divide by zero encountered in power
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    return norm_adj.tocoo()

def _reorder_list(org_list, order):
    new_list = np.array(org_list)
    new_list = new_list[order]
    return new_list