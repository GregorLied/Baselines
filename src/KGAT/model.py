'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score, roc_auc_score

class KGAT(object):
    def __init__(self, args, data):
        self._parse_args(args, data)
        self._build_inputs()
        self._build_weights()

        # Compute Graph-based Representations of All Users & Items & KG Entities via Message-Passing Mechanism of Graph Neural Networks.
        # Optimize Recommendation (CF) Part via BPR Loss.
        self._build_model_phase_I()
        self._build_train_phase_I()

        # Compute Knowledge Graph Embeddings via TransR.
        # Optimize KGE Part via BPR Loss.
        self._build_model_phase_II()
        self._build_train_phase_II()

    def _parse_args(self, args, data):

        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.n_entities = data['n_entities']
        self.n_relations = data['n_relations']
        
        # initialize the attentive matrix A for phase I.
        self.A_in = data['A_in']
        self.all_h_list = data['all_h_list']
        self.all_r_list = data['all_r_list']
        self.all_t_list = data['all_t_list']
        self.all_v_list = data['all_v_list']

        # settings for CF part.
        self.dim = args.dim
        self.batch_size = args.batch_size

        # settings for KG part.
        self.rel_dim = args.rel_dim
        self.batch_size_kg = args.batch_size_kg

        # settings for model architecture
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.aggregator = args.aggregator
        self.l2_weight = args.l2_weight
        self.kge_weight = args.kge_weight        
        self.node_dropout = args.node_dropout
        self.mess_dropout = args.mess_dropout
        self.lr = args.lr
        self.n_fold = 200

    def _build_inputs(self):
        
        # placeholder definition for input data
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

        # for knowledge graph modeling (TransD)
        self.A_values = tf.placeholder(tf.float32, shape=[len(self.all_v_list)], name='A_values')
        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')

    def _build_weights(self):
        
        self.weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        # CF & KGE params
        self.weights['user_embed'] = tf.Variable(initializer([self.n_users, self.dim]), name='user_embed')
        self.weights['entity_embed'] = tf.Variable(initializer([self.n_entities, self.dim]), name='entity_embed')
        self.weights['relation_embed'] = tf.Variable(initializer([self.n_relations, self.rel_dim]),name='relation_embed')
        self.weights['trans_W'] = tf.Variable(initializer([self.n_relations, self.dim, self.rel_dim]))

        # GNN params
        self.weight_size_list = [self.dim] + self.weight_size
        for k in range(self.n_layers):
            self.weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            self.weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            self.weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            self.weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            self.weights['W_mlp_%d' % k] = tf.Variable(
                initializer([2 * self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            self.weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

    def _build_model_phase_I(self):
        if self.aggregator in ['bi']:
            self.ua_embeddings, self.ea_embeddings = self._create_bi_interaction_embed()

        elif self.aggregator in ['gcn']:
            self.ua_embeddings, self.ea_embeddings = self._create_gcn_embed()

        elif self.aggregator in ['graphsage']:
            self.ua_embeddings, self.ea_embeddings = self._create_graphsage_embed()
        else:
            print('please check the the aggregator argument, which should be bi, gcn, or graphsage.')
            raise NotImplementedError

        self.u_e = tf.nn.embedding_lookup(self.ua_embeddings, self.user_indices)
        self.i_e = tf.nn.embedding_lookup(self.ea_embeddings, self.item_indices)
        
        self.scores = tf.reduce_sum(tf.multiply(self.u_e, self.i_e), axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def _build_model_phase_II(self):
        self.h_e, self.r_e, self.pos_t_e, self.neg_t_e = self._get_kg_inference(self.h, self.r, self.pos_t, self.neg_t)
        self.A_kg_score = self._generate_transE_score(h=self.h, t=self.pos_t, r=self.r)
        self.A_out = self._create_attentive_A_out()

    def _build_train_phase_I(self):
        
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))
        regularizer = tf.nn.l2_loss(self.u_e) + tf.nn.l2_loss(self.i_e)
        regularizer = regularizer / self.batch_size

        self.kge_loss = tf.constant(0.0, tf.float32, [1])
        self.reg_loss = self.l2_weight * regularizer
        self.loss = self.base_loss + self.kge_loss + self.reg_loss
        
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _build_train_phase_II(self):
        def _get_kg_score(h_e, r_e, t_e):
            kg_score = tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keepdims=True)
            return kg_score

        pos_kg_score = _get_kg_score(self.h_e, self.r_e, self.pos_t_e)
        neg_kg_score = _get_kg_score(self.h_e, self.r_e, self.neg_t_e)
        
        # Using the softplus as BPR loss to avoid the nan error.
        kg_loss = tf.reduce_mean(tf.nn.softplus(-(neg_kg_score - pos_kg_score)))
        # maxi = tf.log(tf.nn.sigmoid(neg_kg_score - pos_kg_score))
        # kg_loss = tf.negative(tf.reduce_mean(maxi))

        kg_reg_loss = tf.nn.l2_loss(self.h_e) + tf.nn.l2_loss(self.r_e) + \
                      tf.nn.l2_loss(self.pos_t_e) + tf.nn.l2_loss(self.neg_t_e)
        kg_reg_loss = kg_reg_loss / self.batch_size_kg

        self.kge_loss2 = kg_loss
        self.reg_loss2 = self.kge_weight * kg_reg_loss
        self.loss2 = self.kge_loss2 + self.reg_loss2
        
        self.optimizer2 = tf.train.AdamOptimizer(self.lr).minimize(self.loss2)

    def _create_bi_interaction_embed(self):
        A = self.A_in
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        ego_embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            # A_hat_drop = tf.nn.dropout(A_hat, 1 - self.node_dropout[k], [self.n_users + self.n_items, 1])
            
            # Equation 3: Information Propagation 
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            
            # Equation 8: Bi-Interaction Aggregator
            add_embeddings = ego_embeddings + side_embeddings
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(add_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])
            # complete aggregation
            ego_embeddings = bi_embeddings + sum_embeddings
            
            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout)

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            
            # append embeddings needed for Equation 11
            all_embeddings += [norm_embeddings]

        # Equation 11: Concatinate representations from each layer into a single vector
        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _create_gcn_embed(self):
        A = self.A_in
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            # A_hat_drop = tf.nn.dropout(A_hat, 1 - self.node_dropout[k], [self.n_users + self.n_items, 1])
            
            # Equation 3: Information Propagation 
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            
            # Equation 6: GCN Aggregator
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
                
            # message dropout.
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout)
            
            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(embeddings, axis=1)

            # append embeddings, needed for equation 11
            all_embeddings += [norm_embeddings]
        
        # Equation 11: Concatinate representations from each layer into a single vector
        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _create_graphsage_embed(self):
        A = self.A_in
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        pre_embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)

        all_embeddings = [pre_embeddings]
        for k in range(self.n_layers):
            
            # Equation 3: Information Propagation 
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], pre_embeddings))
            embeddings = tf.concat(temp_embed, 0)

            # Equation 6: GraphSAGE Aggregator
            embeddings = tf.concat([pre_embeddings, embeddings], 1)
            pre_embeddings = tf.nn.relu(
                tf.matmul(embeddings, self.weights['W_mlp_%d' % k]) + self.weights['b_mlp_%d' % k])
            
            # message dropout.
            pre_embeddings = tf.nn.dropout(pre_embeddings, 1 - self.mess_dropout)

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(embeddings, axis=1)

            # append embeddings, needed for equation 11
            all_embeddings += [norm_embeddings]

        # Equation 11: Concatinate representations from each layer into a single vector
        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_entities) // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_entities
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_attentive_A_out(self):
        indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        A = tf.sparse.softmax(tf.SparseTensor(indices, self.A_values, self.A_in.shape))
        return A
        
    def _get_kg_inference(self, h, r, pos_t, neg_t):
        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        embeddings = tf.expand_dims(embeddings, 1)

        # head & tail entity embeddings: batch_size *1 * dim
        h_e = tf.nn.embedding_lookup(embeddings, h)
        pos_t_e = tf.nn.embedding_lookup(embeddings, pos_t)
        neg_t_e = tf.nn.embedding_lookup(embeddings, neg_t)

        # relation embeddings: batch_size * rel_dim
        r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)

        # relation transform weights: batch_size * rel_dim * dim
        trans_M = tf.nn.embedding_lookup(self.weights['trans_W'], r)

        # batch_size * 1 * rel_dim -> batch_size * rel_dim
        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.rel_dim])
        pos_t_e = tf.reshape(tf.matmul(pos_t_e, trans_M), [-1, self.rel_dim])
        neg_t_e = tf.reshape(tf.matmul(neg_t_e, trans_M), [-1, self.rel_dim])
        
        # Remove the l2 normalization terms (from CKE)
        # h_e = tf.math.l2_normalize(h_e, axis=1)
        # r_e = tf.math.l2_normalize(r_e, axis=1)
        # pos_t_e = tf.math.l2_normalize(pos_t_e, axis=1)
        # neg_t_e = tf.math.l2_normalize(neg_t_e, axis=1)

        return h_e, r_e, pos_t_e, neg_t_e

    def _generate_transE_score(self, h, t, r):
        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        embeddings = tf.expand_dims(embeddings, 1)

        h_e = tf.nn.embedding_lookup(embeddings, h)
        t_e = tf.nn.embedding_lookup(embeddings, t)

        # relation embeddings: batch_size * rel_dim
        r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)

        # relation transform weights: batch_size * rel_dim * dim
        trans_M = tf.nn.embedding_lookup(self.weights['trans_W'], r)

        # batch_size * 1 * rel_dim -> batch_size * rel_dim
        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.rel_dim])
        t_e = tf.reshape(tf.matmul(t_e, trans_M), [-1, self.rel_dim])

        kg_score = tf.reduce_sum(tf.multiply(t_e, tf.tanh(h_e + r_e)), 1)

        return kg_score

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss, self.base_loss, self.kge_loss, self.reg_loss], feed_dict)
        
    def train_A(self, sess, feed_dict):
        return sess.run([self.optimizer2, self.loss2, self.kge_loss2, self.reg_loss2], feed_dict)

    def update_attentive_A(self, sess):
        fold_len = len(self.all_h_list) // self.n_fold
        kg_score = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_h_list)
            else:
                end = (i_fold + 1) * fold_len

            feed_dict = {
                self.h: self.all_h_list[start:end],
                self.r: self.all_r_list[start:end],
                self.pos_t: self.all_t_list[start:end]
            }
            A_kg_score = sess.run(self.A_kg_score, feed_dict=feed_dict)
            kg_score += list(A_kg_score)

        new_A = sess.run(self.A_out, feed_dict={self.A_values: np.array(kg_score)})
        row = new_A.indices[:, 0]
        col = new_A.indices[:, 1]
        data = new_A.values
        self.A_in = sp.coo_matrix((data, (row, col)), shape=(self.n_users + self.n_entities,
                                                             self.n_users + self.n_entities))
        if self.aggregator in ['gcn']:
            self.A_in.setdiag(1.)
            
    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        f1 = f1_score(y_true=labels, y_pred=predictions)
        return auc, acc, f1
        
    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)