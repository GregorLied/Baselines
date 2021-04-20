import tensorflow as tf
from abc import abstractmethod

class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, activation, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(name)
        self.name = name
        self.dropout = dropout
        self.activation = activation
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass      

class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., activation=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, activation, name)

        with tf.variable_scope(self.name+'_weights'):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        with tf.variable_scope(self.name+'_bias'):
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

        with tf.variable_scope(self.name+'_urh_weights'):
            self.urh_weights = tf.get_variable(
                shape=[3 * self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        with tf.variable_scope(self.name+'_urh_bias'):
            self.urh_bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), name='bias')

    # Section 4.2 â€“ User-Oriented Entity Projection â€“ Equation 4 / Equation 5
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        # return output
        return self.activation(output)

    # Section 4.1 â€“ User-Oriented Relation Attention
    def _mix_neighbor_vectors(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):

        # [batch_size, 1, 1, dim]
        user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])
        user_embeddings = tf.tile(user_embeddings, multiples=[1, neighbor_relations.get_shape()[1], neighbor_relations.get_shape()[2], 1])

        # [batch_size, -1, 1, dim]
        self_vectors = tf.expand_dims(self_vectors, axis=2)
        self_vectors = tf.tile(self_vectors, multiples=[1, 1, neighbor_relations.get_shape()[2], 1])

        # [batch_size, -1, -1, dim * 4]
        urh_matrix = tf.concat([user_embeddings, neighbor_relations, self_vectors], -1)
        
        # Equation 3: [-1, 1]
        user_oriented_scores = tf.matmul(tf.reshape(urh_matrix,[-1, 3 * self.dim]), self.urh_weights)
        user_oriented_scores = tf.reshape(user_oriented_scores,[neighbor_vectors.get_shape()[0],neighbor_vectors.get_shape()[1],neighbor_vectors.get_shape()[2]])

        # Equation 2: [batch_size, -1, dim]
        user_oriented_scores_normalized = tf.nn.softmax(user_oriented_scores)
        
        # [batch_size,-1, n_memory, 1]
        user_oriented_scores_expanded = tf.expand_dims(user_oriented_scores_normalized, axis= -1)

        # Equation 1: [batch_size, -1, dim]
        neighbors_aggregated = tf.reduce_mean(user_oriented_scores_expanded * neighbor_vectors, axis=2)

        return neighbors_aggregated