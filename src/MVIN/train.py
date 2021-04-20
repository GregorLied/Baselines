import tensorflow as tf
import numpy as np
from time import time
from model import MVIN
from metrics import precision_at_k, recall_at_k, ndcg_at_k

def train(args, data, load_pretrain_emb, show_loss, show_topk):
    n_users, n_items, n_entities, n_relations = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]
    ripple_set = data[9]

    model = MVIN(args, n_users, n_entities, n_relations, adj_entity, adj_relation)
    
    if show_topk:
        user_list, train_user_dict, test_user_dict, item_set, k_list = topk_settings(train_data, test_data, n_items)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore embeddings during stage-wise training
        stage_wise_var = [v for v in tf.global_variables() if 'STWS' in v.name and 'Adam' not in v.name and 'adam' not in v.name]
        saver = tf.train.Saver(stage_wise_var)
        for variable in stage_wise_var:
            print("stage_wise variable = ", variable.name)
        if load_pretrain_emb:
            saver.restore(sess, 'parameters/params')

        best_auc = 0
        best_epoch = 0
        for epoch in range(args.n_epochs):
            
            # Training
            t0 = time()
            np.random.shuffle(train_data)
            
            # skip the last incomplete minibatch if its size < batch size
            start = 0
            while start + args.batch_size < train_data.shape[0]:
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))
            train_time = time() - t0

            # CTR evaluation
            t1 = time()
            train_auc, train_acc, train_f1 = ctr_eval(sess, args, model, train_data, ripple_set, args.batch_size)
            eval_auc, eval_acc, eval_f1 = ctr_eval(sess, args, model, eval_data, ripple_set, args.batch_size)
            test_auc, test_acc, test_f1 = ctr_eval(sess, args, model, test_data, ripple_set, args.batch_size)
            ctr_time = time() - t1
            ctr_log = 'epoch %d [%.1f s + %.1f s] | train auc: %.4f  acc: %.4f  f1: %.4f | eval auc: %.4f  eval acc: %.4f  eval f1: %.4f | test auc: %.4f  test acc: %.4f  test f1: %.4f' % (epoch, train_time, ctr_time, train_auc, train_acc, train_f1, eval_auc, eval_acc, eval_f1, test_auc, test_acc, test_f1)
            print(ctr_log)
            with open(args.log, 'a') as f:
                f.write(ctr_log + '\n')
                
            # Top-K evaluation
            if show_topk:
                t2 = time()
                precision, recall, ndcg = topk_eval(
                    sess, args, model, user_list, train_user_dict, test_user_dict, item_set, ripple_set, k_list, args.batch_size)
                topk_time = time() - t2
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

            # Save best embeddings for stage-wise training
            if epoch == best_epoch:
                saver.save(sess, 'parameters/params')

    return best_auc
            
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

def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.user_indices] = data[start:end, 0]
    feed_dict[model.item_indices] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    for i in range(args.p_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict
    
def ctr_eval(sess, args, model, data, ripple_set, batch_size):
    auc_list = []
    acc_list = []
    f1_list = []
    start = 0
    while start + batch_size <= data.shape[0]:
        auc, acc, f1 = model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list))
    
def get_feed_dict_top_k(args, model, user_list, items, labels, ripple_set):
    feed_dict = dict()
    feed_dict[model.user_items] = user_list
    feed_dict[model.item_indices] = items
    feed_dict[model.labels] = labels
    for i in range(args.p_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in user_list]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in user_list]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in user_list]
    return feed_dict
    
def topk_eval(sess, args, model, user_list, train_user_dict, test_user_dict, item_set, ripple_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_user_dict[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            user_list = [user] * batch_size
            item_list = test_item_list[start:start + batch_size]
            label_list = [1] * batch_size
            items, scores = model.get_scores(sess, get_feed_dict_top_k(args, model, user_list, item_list, label_list, ripple_set))
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            user_list = [user] * batch_size
            item_list = test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)
            label_list = [1] * batch_size
            items, scores = model.get_scores(sess, get_feed_dict_top_k(args, model, user_list, item_list, label_list, ripple_set))
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