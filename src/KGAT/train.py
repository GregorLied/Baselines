import tensorflow as tf
import numpy as np
import random
from time import time
from model import KGAT
from metrics import precision_at_k, recall_at_k, ndcg_at_k

def train(args, data, show_loss, show_topk):
    n_users, n_items, n_entities, n_relations = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    ckg_head_dict, ckg_adj_list, all_h_list, all_r_list, all_t_list, all_v_list = data[7], data[8], data[9], data[10], data[11], data[12]

    data_dict = dict()
    data_dict['n_users'] = n_users
    data_dict['n_items'] = n_items
    data_dict['n_entities'] = n_entities
    data_dict['n_relations'] = n_relations

    data_dict['A_in'] = sum(ckg_adj_list)
    data_dict['all_h_list'] = all_h_list
    data_dict['all_r_list'] = all_r_list
    data_dict['all_t_list'] = all_t_list
    data_dict['all_v_list'] = all_v_list

    model = KGAT(args, data_dict)

    if show_topk:
        user_list, train_user_dict, test_user_dict, item_set, k_list = topk_settings(train_data, test_data, n_items)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        best_auc = 0
        best_epoch = 0
        for epoch in range(args.n_epochs):
        
            # Training
            t0 = time()
            np.random.shuffle(train_data)

            # skip the last incomplete minibatch if its size < batch size
            start = 0
            while start + args.batch_size <= train_data.shape[0]:
                _, loss, _, _, _ = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(start, loss)
            train_phase_I_time = time() - t0

            t1 = time()
            start = 0
            while start + args.batch_size_kg <= len(all_h_list):
                kg_batch = _generate_train_A_batch(args, ckg_head_dict, n_users, n_entities)
                _, loss, _, _ = model.train_A(sess, get_kg_feed_dict(model, kg_batch))
                start += args.batch_size_kg
                if show_loss:
                    print(start, loss)
            train_phase_II_time = time() - t1

            model.update_attentive_A(sess)

            # CTR evaluation
            t2 = time()
            train_auc, train_acc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size)
            eval_auc, eval_acc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size)
            test_auc, test_acc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)
            ctr_time = time() - t2
            ctr_log = 'epoch %d [%.1f s + %.1f s + %.1f s] | train auc: %.4f  acc: %.4f  f1: %.4f | eval auc: %.4f  eval acc: %.4f  eval f1: %.4f | test auc: %.4f  test acc: %.4f  test f1: %.4f' % (epoch, train_phase_I_time, train_phase_II_time, ctr_time, train_auc, train_acc, train_f1, eval_auc, eval_acc, eval_f1, test_auc, test_acc, test_f1)
            print(ctr_log)
            with open(args.log, 'a') as f:
                f.write(ctr_log + '\n')

            # Top-K evaluation
            if show_topk:
                t3 = time()
                precision, recall, ndcg = topk_eval(
                    sess, model, user_list, train_user_dict, test_user_dict, item_set, k_list, args.batch_size)
                topk_time = time() - t3
                topk_log = 'topk eval [%.1f s] | precision: %s | recall: %s | ndcg: %s' % (topk_time, ' '.join(['%.4f' % p for p in precision]), ' '.join(['%.4f' % r for r in recall]), ' '.join(['%.4f' % n for n in ndcg]))
                print(topk_log)
                with open(args.log, 'a') as f:
                    f.write(topk_log + '\n')
            
            # Early stopping
            best_auc, best_epoch, should_stop = early_stopping(eval_auc, epoch, best_auc, best_epoch)
            if should_stop:
                early_stopping_log = "Early Stopping triggered. Best epoch: epoch %d." % best_epoch
                print(early_stopping_log)
                with open(args.log, 'a') as f:
                    f.write(early_stopping_log)
                break

def topk_settings(train_data, test_data, n_items):
    user_num = 250
    k_list = [1, 2, 5, 10, 20, 50, 100]
    train_user_dict = _get_user_record(train_data)
    test_user_dict = _get_user_record(test_data)
    user_list = list(test_user_dict.keys())
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_items)))
    return user_list, train_user_dict, test_user_dict, item_set, k_list

def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict

def get_kg_feed_dict(model, kg_batch):
    feed_dict = {model.h: kg_batch[0],
                 model.r: kg_batch[1],
                 model.pos_t: kg_batch[2],
                 model.neg_t: kg_batch[3]}
    return feed_dict

def ctr_eval(sess, model, data, batch_size):
    auc_list = []
    acc_list = []
    f1_list = []
    
    start = 0
    while start + batch_size <= data.shape[0]:
        auc, acc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)
        start += batch_size
        
    return float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list))

def topk_eval(sess, model, user_list, train_user_dict, test_user_dict, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_user_dict[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        
        hits = []
        for i in item_sorted:
            if i in test_user_dict[user]:
                hits.append(1)
            else:
                hits.append(0)

        for k in k_list:
            precision_list[k].append(precision_at_k(hits, k))
            recall_list[k].append(recall_at_k(hits, k, len(test_user_dict[user])))
            ndcg_list[k].append(ndcg_at_k(hits, k))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg

def early_stopping(cur_auc, cur_epoch, best_auc, best_epoch, stopping=10):
    if cur_auc > best_auc:
        best_auc = cur_auc
        best_epoch = cur_epoch
    if cur_epoch - best_epoch >= stopping:
        print("Early Stopping triggered. No improvements since %d epochs." % stopping)
        should_stop = True
    else:
        should_stop = False
    return best_auc, best_epoch, should_stop

def _get_user_record(data):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        rating = interaction[2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def _ckg_pos_sample(ckg_head_dict, head, num_samples = 1):
    pos_triples = ckg_head_dict[head]
    relation_samples, pos_tail_samples = [], []
    while True:
        if len(relation_samples) == num_samples: break
        index = np.random.randint(low=0, high=len(pos_triples), size=1)[0]

        pos_tail = pos_triples[index][0]
        relation = pos_triples[index][1]

        if relation not in relation_samples and pos_tail not in pos_tail_samples:
            relation_samples.append(relation)
            pos_tail_samples.append(pos_tail)

    return relation_samples, pos_tail_samples

def _ckg_neg_sample(ckg_head_dict, head, relation, n_users, n_entities, num_samples = 1):
    neg_tail_samples = []
    while True:
        if len(neg_tail_samples) == num_samples: break
        neg_tail = np.random.randint(low=0, high=n_users+n_entities, size=1)[0]
                
        if (neg_tail, relation) not in ckg_head_dict[head] and neg_tail not in neg_tail_samples:
            neg_tail_samples.append(neg_tail)

    return neg_tail_samples

def _generate_train_A_batch(args, ckg_head_dict, n_users, n_entities):

    # Get batch
    exist_heads = ckg_head_dict.keys()
    if args.batch_size_kg <= len(exist_heads):
        batch_heads = random.sample(exist_heads, args.batch_size_kg)
    else:
        batch_heads = [random.choice(exist_heads) for _ in range(args.batch_size_kg)]

    # Draw Positive and Negative Samples
    batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
    for head in batch_heads:
        relation_samples, pos_tail_samples = _ckg_pos_sample(ckg_head_dict, head)
        batch_relation += relation_samples
        batch_pos_tail += pos_tail_samples

        relation = relation_samples[0]
        batch_neg_tail += _ckg_neg_sample(ckg_head_dict, head, relation, n_users, n_entities)

    return batch_heads, batch_relation, batch_pos_tail, batch_neg_tail