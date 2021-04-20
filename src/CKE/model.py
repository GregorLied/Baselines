import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

class CKE(object):
    def __init__(self, args, n_users, n_items, n_entities, n_relations):
        self._parse_args(args, n_users, n_items, n_entities, n_relations)
        self._build_inputs()
        self._build_weights()
        self._build_model()
        self._build_train()

    def _parse_args(self, args, n_users, n_items, n_entities, n_relations):

        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations

        # settings for CF part.
        self.dim = args.dim

        # settings for KG part.
        self.rel_dim = args.rel_dim

        # settings for model architecture
        self.l2_weight = args.l2_weight
        self.kge_weight = args.kge_weight        
        self.lr = args.lr

    def _build_inputs(self):
        
        # placeholder definition for input data
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

        # for knowledge graph modeling (TransD)
        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')

    def _build_weights(self):

        self.weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        # CF params
        self.weights['user_embed'] = tf.Variable(initializer([self.n_users, self.dim]), name='user_embed')
        self.weights['item_embed'] = tf.Variable(initializer([self.n_items, self.dim]), name='item_embed')

        # KG params
        self.weights['entity_embed'] = tf.Variable(initializer([self.n_entities, self.dim]), name='entity_embed')
        self.weights['relation_embed'] = tf.Variable(initializer([self.n_relations, self.rel_dim]),name='relation_embed')
        self.weights['trans_W'] = tf.Variable(initializer([self.n_relations, self.dim, self.rel_dim]))

    def _build_model(self):
        # Inference
        self.h_e, self.r_e, self.pos_t_e, self.neg_t_e = self._get_kg_inference(self.h, self.r, self.pos_t, self.neg_t)
        self.u_e, self.i_e = self._get_cf_inference()

        # Equation 8: Prediction
        self.scores = tf.reduce_sum(tf.multiply(self.u_e, self.i_e), axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def _get_kg_inference(self, h, r, pos_t, neg_t):

        # head & tail entity embeddings: batch_size_kg *1 * dim
        h_e = tf.nn.embedding_lookup(self.weights['entity_embed'], h)
        pos_t_e = tf.nn.embedding_lookup(self.weights['entity_embed'], pos_t)
        neg_t_e = tf.nn.embedding_lookup(self.weights['entity_embed'], neg_t)

        # relation embeddings: batch_size_kg * rel_dim
        r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)

        # relation transform weights: batch_size_kg * rel_dim * dim
        trans_M = tf.nn.embedding_lookup(self.weights['trans_W'], r)

        # Equation 2: batch_size_kg * 1 * rel_dim -> batch_size_kg * rel_dim
        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.rel_dim])
        pos_t_e = tf.reshape(tf.matmul(pos_t_e, trans_M), [-1, self.rel_dim])
        neg_t_e = tf.reshape(tf.matmul(neg_t_e, trans_M), [-1, self.rel_dim])
        
        # l2-normalization (setup for Equation 7)
        h_e = tf.math.l2_normalize(h_e, axis=1)
        r_e = tf.math.l2_normalize(r_e, axis=1)
        pos_t_e = tf.math.l2_normalize(pos_t_e, axis=1)
        neg_t_e = tf.math.l2_normalize(neg_t_e, axis=1)

        return h_e, r_e, pos_t_e, neg_t_e

    def _get_cf_inference(self):
        u_e = tf.nn.embedding_lookup(self.weights['user_embed'], self.user_indices)
        i_e = tf.nn.embedding_lookup(self.weights['item_embed'], self.item_indices)
        e_e = tf.reshape(tf.nn.embedding_lookup(self.weights['entity_embed'], self.item_indices), [-1, self.dim])

        # Equation 5: Compute Item Latent Vector 
        return u_e, i_e + e_e

    def _build_train(self):
        # CF Loss
        self.cf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))
        cf_ref_loss = tf.nn.l2_loss(self.u_e) + tf.nn.l2_loss(self.i_e)
        self.cf_reg_loss = self.l2_weight * cf_ref_loss

        # KG Loss
        pos_kg_score = tf.reduce_sum(tf.square((self.h_e + self.r_e - self.pos_t_e)), 1, keepdims=True)
        neg_kg_score = tf.reduce_sum(tf.square((self.h_e + self.r_e - self.neg_t_e)), 1, keepdims=True)
        
        self.kg_loss = tf.reduce_mean(tf.nn.softplus(-(neg_kg_score - pos_kg_score)))
        kg_reg_loss = tf.nn.l2_loss(self.h_e) + tf.nn.l2_loss(self.r_e) + tf.nn.l2_loss(self.pos_t_e) + tf.nn.l2_loss(self.neg_t_e)
        self.kg_reg_loss = self.kge_weight * kg_reg_loss

        # Equation 7: Total Loss
        self.loss = self.cf_loss + self.cf_reg_loss + self.kg_loss + self.kg_reg_loss
        
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)
            
    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        f1 = f1_score(y_true=labels, y_pred=predictions)
        return auc, acc, f1
        
    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)