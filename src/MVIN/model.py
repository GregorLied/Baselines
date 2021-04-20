import tensorflow as tf
from aggregators import SumAggregator
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

class MVIN(object):
    def __init__(self, args, n_users, n_entities, n_relations, adj_entity, adj_relation):
        self._parse_args(args, n_users, n_entities, n_relations, adj_entity, adj_relation)
        self._build_inputs()
        self._build_model()
        self._build_train()

    def _parse_args(self, args, n_users, n_entities, n_relations, adj_entity, adj_relation):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.dataset = args.dataset
        
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        
        self.n_memory = args.n_memory
        self.n_neighbor = args.neighbor_sample_size
        self.p_hop = args.p_hop
        self.n_mix_hop = args.n_mix_hop
        self.h_hop = args.h_hop
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.l2_agg_weight = args.l2_agg_weight
        self.lr = args.lr
        self.batch_size = args.batch_size

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

        self.memories_h = []
        self.memories_r = []
        self.memories_t = []
        for hop in range(self.p_hop):
            self.memories_h.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_" + str(hop)))
            self.memories_r.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_" + str(hop)))
            self.memories_t.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_" + str(hop)))

    def _build_model(self):

        with tf.variable_scope("user_emb_matrix_STWS"):
            self.user_emb_matrix = tf.get_variable(name='user_emb_matrix_STWS', 
                                                   shape=[self.n_users, self.dim], 
                                                   initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("entity_emb_matrix_STWS"):
            self.entity_emb_matrix = tf.get_variable(name='entity_emb_matrix_STWS', 
                                                     shape=[self.n_entities, self.dim], 
                                                     initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("relation_emb_matrix_STWS"):
            self.relation_emb_matrix = tf.get_variable(name='relation_emb_matrix_STWS', 
                                                       shape=[self.n_relations,self.dim],
                                                       initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("relation_emb_KGE_matrix_STWS"):
            self.relation_emb_KGE_matrix = tf.get_variable(name='relation_emb_KGE_matrix_STWS', 
                                                           shape=[self.n_relations,self.dim, self.dim], 
                                                           initializer=tf.contrib.layers.xavier_initializer())

        #===============================================
        #=================Layer Mixing==================
        #===============================================

        # Used in Algorithm 2 – Line 2
        self.transfer_matrix_list = []
        self.transfer_matrix_bias = []
        for n in range(self.n_mix_hop*self.h_hop+1):
            with tf.variable_scope("transfer_agg_matrix"+str(n)):
                self.transform_matrix = tf.get_variable(name='transfer_agg_matrix'+str(n), 
                                                   shape=[self.dim, self.dim], 
                                                   dtype=tf.float32,
                                                   initializer=tf.contrib.layers.xavier_initializer())
                self.transform_bias = tf.get_variable(name='transfer_agg_bias'+str(n), 
                                                 shape=[self.dim],
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())
                self.transfer_matrix_bias.append(self.transform_bias)
                self.transfer_matrix_list.append(self.transform_matrix)

        # Used in Algorithm 2 – Line 9
        self.enti_transfer_matrix_list = []
        self.enti_transfer_bias_list = []
        for n in range(self.n_mix_hop):
            with tf.variable_scope("enti_mlp_matrix"+str(n)):
                self.enti_transfer_matrix = tf.get_variable(name='transfer_matrix'+str(n),
                                                            shape=[self.dim * (self.h_hop+1), self.dim], 
                                                            initializer=tf.contrib.layers.xavier_initializer())
                self.enti_transfer_bias = tf.get_variable(name='transfer_bias'+str(n),
                                                          shape=[self.dim], 
                                                          initializer=tf.contrib.layers.xavier_initializer())
                self.enti_transfer_matrix_list.append(self.enti_transfer_matrix)
                self.enti_transfer_bias_list.append(self.enti_transfer_bias)

        #===============================================
        #====KG-Enhanced User Representation Weights====
        #===============================================
        # Used in Equation 9
        with tf.variable_scope("h_emb_item_mlp_matrix"):
            self.h_emb_item_mlp_matrix = tf.get_variable(name='h_emb_item_mlp_matrix',
                                                         shape=[self.dim * 2, 1], 
                                                         initializer=tf.contrib.layers.xavier_initializer())
            self.h_emb_item_mlp_bias = tf.get_variable(name='h_emb_item_mlp_bias',
                                                       shape=[1], 
                                                       initializer=tf.contrib.layers.xavier_initializer())

        # Used in Algorithm 1 – Line 6 / Equation 13
        with tf.variable_scope("user_mlp_matrix"):
            self.user_mlp_matrix = tf.get_variable(name='user_mlp_matrix',
                                                   shape=[self.dim * (self.p_hop+1), self.dim], 
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.user_mlp_bias = tf.get_variable(name='user_mlp_bias',
                                                 shape=[self.dim], 
                                                 initializer=tf.contrib.layers.xavier_initializer())

        # Copy of RippleNet
        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.p_hop):
            # [batch size, n_memory, dim]
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))
            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_KGE_matrix, self.memories_r[i]))
            # [batch size, n_memory, dim]
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))

        # Slightly changes to RippleNet
        user_embeddings, transfer_o = self._key_addressing()

        # Copy of KGCN (Only change is the introduction of self.n_mix_hop in range())
        # [batch_size, dim]
        entities, relations = self.get_neighbors(self.item_indices)

        # Slightly changes to KGCN (Main Innovation is the idea of Wide Layer Mixing)
        item_embeddings, self.aggregators = self.aggregate(entities, relations, transfer_o)

        # Algorithm 3 – Line 9
        self.scores = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def _key_addressing(self):
        def soft_attention_h_set():
            # [batch_size, dim]
            user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
            # [batch_size, 1, dim]
            item = tf.expand_dims(user_embeddings, axis=1)
            # [batch_size, n_memory, dim]
            item = tf.tile(item, [1, self.h_emb_list[0].shape[1], 1])
            # [batch_size, n_memory, 2 * dim]
            h_emb_item = tf.concat([self.h_emb_list[0],item], 2)
            # [-1 , dim * 2]
            h_emb_item = tf.reshape(h_emb_item,[-1,self.dim * 2])

            # Equation 9:
            # [-1]
            probs = tf.squeeze(tf.matmul(h_emb_item, self.h_emb_item_mlp_matrix), axis=-1) + self.h_emb_item_mlp_bias
            # [batch_size, n_memory]
            probs = tf.reshape(probs,[-1,self.h_emb_list[0].shape[1]])
            probs_normalized = tf.nn.softmax(probs)
            # [batch_size, n_memory,1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # Equation 8: [batch_size, dim] 
            user_h_set = tf.reduce_sum(self.h_emb_list[0] * probs_expanded, axis=1)
            return user_h_set

        o_list = []
        transfer_o = []

        # Algorithm 1 – Line 4 (Newly introduced in MVIN)
        user_h_set = soft_attention_h_set()
        o_list.append(user_h_set)

        # Algorithm 1 – Line 2/3 (Introduced in RippleNet)
        item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.item_indices)
        for hop in range(self.p_hop):
            
            # Equation 11:
            # [batch_size, n_memory, dim, 1]
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)
            # [batch_size, n_memory, dim]
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)
            # [batch_size, n_memory, dim]
            v = tf.expand_dims(item_embeddings, axis=2)
            # [batch_size, n_memory]
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)
            # [batch_size, n_memory]
            probs_normalized = tf.nn.softmax(probs) 
            # [batch_size, n_memory, 1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)
            
            # Equation 10: [batch_size, dim]
            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)
            
            # However: In contrast to RippleNet no update_item_embedding at this position
            
            o_list.append(o)

        # Algorithm 1 – Line 5 / Equation 12 – Concatenate all user preference responses u_o^p
        o_list = tf.concat(o_list, -1)

        # Algorithm 1 – Line 6 / Equation 13 – Generate preference embedding u_o for user u
        user_embeddings = tf.matmul(tf.reshape(o_list,[-1,self.dim * (self.p_hop+1)]), self.user_mlp_matrix) + self.user_mlp_bias

        transfer_o.append(user_embeddings)

        return user_embeddings, transfer_o

    def get_neighbors(self, seeds):
        # Takes item_indices of current batch as seeds
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        n = self.n_neighbor
        for i in range(self.n_mix_hop*self.h_hop):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, n])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, n])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
            n *= self.n_neighbor
        return entities, relations

    # Algorithm 3 - Line 8 / Algorithm 2 / Section 4.2 – Entity-Entity-Representation
    def aggregate(self, entities, relations, transfer_o):

        user_query = transfer_o[0]
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        # Algorithm 2 – Line 2 (Newly introduced in MVIN)
        for index in range(len(transfer_o)):
            transfer_o[index] = tf.expand_dims(transfer_o[index], axis=1)
        for index in range(len(transfer_o)):
            for e_i in range(len(entity_vectors)):
                # [b,1,dim]
                n_entities = entity_vectors[e_i] + transfer_o[index]
                # [-1,dim]
                n_entities = tf.matmul(tf.reshape(n_entities, [-1,self.dim]), self.transfer_matrix_list[e_i]) + self.transfer_matrix_bias[e_i]
                # [b,n,dim]
                entity_vectors[e_i] = tf.reshape(n_entities, [self.batch_size, entity_vectors[e_i].shape[1],self.dim])
                # [b,?*n,dim]
                transfer_o[index] = tf.tile(transfer_o[index],[1,self.n_neighbor,1])

        # Algorithm 2 – Line 3/4/5/6/7/8/9 (Applying Wide Layer Mixing to KGCN Embeddings)
        for n in range(self.n_mix_hop):
            mix_hop_tmp = []
            mix_hop_tmp.append(entity_vectors)
            
            # Line 7 / 8
            for i in range(self.h_hop):
                aggregator = SumAggregator(self.batch_size, self.dim, name = str(i)+'_'+str(n))
                aggregators.append(aggregator)
                
                entity_vectors_next_iter = []
                for hop in range(self.h_hop*self.n_mix_hop-(self.h_hop*n+i)):
                    shape = [self.batch_size, entity_vectors[hop].shape[1], self.n_neighbor, self.dim]
                    vector = aggregator(self_vectors=entity_vectors[hop],
                                        neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                        neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                        user_embeddings=user_query)
                    entity_vectors_next_iter.append(vector)
                entity_vectors = entity_vectors_next_iter
                mix_hop_tmp.append(entity_vectors)

            # Line 9
            entity_vectors = []
            for mip_hop in zip(*mix_hop_tmp):
                mip_hop = tf.concat(mip_hop, -1)
                mip_hop = tf.matmul(tf.reshape(mip_hop,[-1,self.dim * (self.h_hop+1)]), self.enti_transfer_matrix_list[n]) + self.enti_transfer_bias_list[n]
                mip_hop = tf.reshape(mip_hop,[self.batch_size,-1,self.dim]) 
                entity_vectors.append(mip_hop)
                if len(entity_vectors) == (self.n_mix_hop-(n+1))*self.h_hop+1:  break

        mix_hop_res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return mix_hop_res, aggregators

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = 0
        for hop in range(self.p_hop):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))

        self.l2_loss += tf.nn.l2_loss(self.relation_emb_matrix) 
        
        self.l2_loss += tf.nn.l2_loss(self.user_mlp_matrix) + tf.nn.l2_loss(self.user_mlp_bias)
        self.l2_loss += tf.nn.l2_loss(self.transform_matrix) + tf.nn.l2_loss(self.transform_bias)

        for n in range(self.h_hop+1):
            self.l2_loss += tf.nn.l2_loss(self.transfer_matrix_list[n]) + tf.nn.l2_loss(self.transfer_matrix_bias[n])

        self.l2_loss += tf.nn.l2_loss(self.h_emb_item_mlp_matrix) +  tf.nn.l2_loss(self.h_emb_item_mlp_bias)

        self.l2_agg_loss = 0
        self.l2_agg_loss += tf.nn.l2_loss(self.user_emb_matrix)
        for aggregator in self.aggregators:
            self.l2_agg_loss += tf.nn.l2_loss(aggregator.weights)
            self.l2_agg_loss += tf.nn.l2_loss(aggregator.urh_weights)

        for n in range(self.n_mix_hop):
            self.l2_agg_loss += tf.nn.l2_loss(self.enti_transfer_matrix_list[n]) + tf.nn.l2_loss(self.enti_transfer_bias_list[n])

        self.loss = self.base_loss + self.l2_weight * self.l2_loss + self.l2_agg_weight * self.l2_agg_loss 

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