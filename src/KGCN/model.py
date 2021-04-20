import tensorflow as tf
import numpy as np
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score

class KGCN(object):
    def __init__(self, args, n_users, n_entities, n_relations, adj_entity, adj_relation):
        self._parse_args(args, adj_entity, adj_relation, n_users, n_entities, n_relations)
        self._build_inputs()
        self._build_weights()
        self._build_model()
        self._build_train()

    def _parse_args(self, args, adj_entity, adj_relation, n_users, n_entities, n_relations):
        # [n_entities, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.n_neighbor = args.neighbor_sample_size
        self.n_iter = args.n_iter
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)
        self.lr = args.lr
        self.batch_size = args.batch_size

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
        
    def _build_weights(self):
        
        self.weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        self.weights['user_embed'] = tf.Variable(initializer([self.n_users, self.dim]), name='user_embed')
        self.weights['entity_embed'] = tf.Variable(initializer([self.n_entities, self.dim]), name='entity_embed')
        self.weights['relation_embed'] = tf.Variable(initializer([self.n_relations, self.dim]),name='relation_embed')

    def _build_model(self):

        # [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(self.weights['user_embed'], self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items 
        # dimensions of the list-elements in entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        # therefore, the list elements in entities contain each batch_size rows with the following row-wise information in it:
        # {[itemindex], [inidices of 1-hop neighbors], [inidices of 2-hop neighbors], ..., [inidices of n_iter-hop neighbors]}
        # each row-wise information corresponds to the neighbors of the itemindex in this row
        entities, relations = self.get_neighbors(self.item_indices)

        # [batch_size, dim]
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

        # Equation 7: [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        # Takes item_indices of current batch as seeds
        # Reshape item_indices-seeds from [batch_size,] to [batch_size,1]
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.weights['entity_embed'], i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.weights['relation_embed'], i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, activation=tf.nn.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            # This Information Propagation Scheme is programmed very efficiently:
            # In the first layer the embeddings of all nodes n_hops away from the items are being updated
            # In the second layer the embeddings of only the nodes n_hops - 1 away from the items are being updated
            # ...
            # In the n_iter layer the embeddings of only the items are being updated
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                # [batch_size, -1, dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings)
                entity_vectors_next_iter.append(vector)
            # Update entity_vectors
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = tf.nn.l2_loss(self.weights['user_embed']) + tf.nn.l2_loss(
            self.weights['entity_embed']) + tf.nn.l2_loss(self.weights['relation_embed'])
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

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
