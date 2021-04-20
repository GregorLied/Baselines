import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

class FM(object):
    def __init__(self, args, n_users, n_items, n_entities):
        self._parse_args(args, n_users, n_items, n_entities)
        self._build_inputs()
        self._build_weights()
        self._build_model()
        self._build_train()

    def _parse_args(self, args, n_users, n_items, n_entities):

        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_features = n_users + n_entities

        # settings for model architecture
        self.dim = args.dim
        self.l2_weight = args.l2_weight       
        self.lr = args.lr

    def _build_inputs(self):

        # placeholder definition for input data
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
        
        self.indices = tf.placeholder(tf.int64, shape=[None, 2], name='indices')
        self.values = tf.placeholder(tf.float32, shape=[None], name='values')
        self.shape = tf.placeholder(tf.int64, shape=[2], name='shape')

    def _build_weights(self):

        self.weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        self.weights['var_linear'] = tf.Variable(initializer([self.n_features, 1]), name='var_linear')
        self.weights['var_factor'] = tf.Variable(initializer([self.n_features, self.dim]), name='var_factor')

    def _build_model(self):

        self.scores = self._get_fm_prediction(tf.SparseTensor(self.indices, self.values, self.shape))
        self.scores_normalized = tf.sigmoid(self.scores)

    def _get_fm_prediction(self, features):
        
        # Equation 5: Make prediction
        # Linear terms: batch_size * 1
        term0 = tf.sparse_tensor_dense_matmul(features, self.weights['var_linear'])

        # Interaction terms w.r.t. first sum then square: batch_size * dim.
        # e.g., sum_{k from 1 to K}{(v1k+v2k)**2}
        term1 = tf.square(tf.sparse_tensor_dense_matmul(features, self.weights['var_factor']))

        # Interaction terms w.r.t. first square then sum: batch_size * dim.
        #   e.g., sum_{k from 1 to K}{v1k**2 + v2k**2}
        term2 = tf.sparse_tensor_dense_matmul(tf.square(features), tf.square(self.weights['var_factor']))

        pred = term0 + 0.5 * tf.reduce_sum(term1 - term2, 1, keepdims=True)

        return pred

    def _build_train(self):
        
        print("-----")
        print(self.labels.shape)
        print(self.scores.shape)
        
        self.cf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(self.labels, 1), logits=self.scores))
        cf_ref_loss = 2 * tf.nn.l2_loss(tf.constant(1., tf.float32, [self.dim, 1]))
        self.cf_reg_loss = self.l2_weight * cf_ref_loss

        # Equation 6: Total Loss
        self.loss = self.cf_loss + self.cf_reg_loss
        
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