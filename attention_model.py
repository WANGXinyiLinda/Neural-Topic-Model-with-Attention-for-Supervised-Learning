#!/usr/bin/python
#coding:utf-8
import os
import tensorflow as tf
import time
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from topic_model import VariationalTopicModel  

class Vanilla_attention_model(object):

    def __init__(self, num_classes=20, pretrained_embed=None, embedding_size=100, 
                hidden_size=64, dropout_keep_proba=0.8, max_word_num=200, train_embed=True):

        self.num_classes = int(num_classes)
        self.embedding_size = int(embedding_size)
        self.pretrained_embed = pretrained_embed # [vocab_size, embedding_size]
        self.hidden_size = int(hidden_size)
        self.dropout_keep_proba = dropout_keep_proba
        self.max_word_num = int(max_word_num)
        self.train_embed = train_embed

        with tf.variable_scope('placeholder'):
            self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.max_word_num], name='input_x_rnn')
            if self.num_classes > 0:
                self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='input_y_label')
            else:
                self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, ], name='input_y_label')
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        with tf.variable_scope("word_embedding"):
            word_embedding_valid = tf.Variable(initial_value=self.pretrained_embed, trainable=self.train_embed, dtype=tf.float32)
            word_embedding_pad = tf.constant(value=0, dtype=tf.float32, shape=[1, self.embedding_size])
            self.word_embedding_mat = tf.concat([word_embedding_pad, word_embedding_valid], axis = 0)
            #shape: [batch_size, max_word_num, embedding_size]
            self.embedded_input = tf.nn.embedding_lookup(self.word_embedding_mat, self.input_x)

        with tf.variable_scope("doc2vec"):
            # doc_encoded: [batch_size, max_word_num, hidden_size*2]
            doc_encoded = self.BidirectionalGRUEncoder(self.embedded_input, self.hidden_size, name='bi-gru')
            print("bi-GRU out shape: ", doc_encoded.shape)
            # doc_vec: [batch_size, hidden_size*2]
            doc_vec, self.weights = self.AttentionLayer(doc_encoded, self.hidden_size, name='attention')
            print("attention out shape: ", doc_vec.shape)
            doc_vec_dropped = layers.dropout(doc_vec, keep_prob=self.dropout_keep_proba, is_training=self.is_training)
            if self.num_classes > 0:
                out = layers.fully_connected(inputs=doc_vec_dropped, num_outputs=self.num_classes, activation_fn=None)
            else:
                out = layers.fully_connected(inputs=doc_vec_dropped, num_outputs=1, activation_fn=None)
            print("logit shape: ", out.shape)
        
        if self.num_classes > 0:
            with tf.variable_scope('cross_entro_loss'):
                # cross-entropy loss
                self.cross_entro = tf.losses.softmax_cross_entropy(onehot_labels=self.input_y, logits=out, reduction=tf.losses.Reduction.MEAN)
        else:
            with tf.variable_scope('mse_loss'):
                # mse loss
                self.mse = tf.losses.mean_squared_error(labels=self.input_y, predictions=tf.squeeze(out), reduction=tf.losses.Reduction.MEAN)
        
        self.predict = tf.argmax(out, axis=1, name='predict')

        if self.num_classes > 0:
            with tf.variable_scope('accuracy'):
                self.label = tf.argmax(self.input_y, axis=1, name='label')
                self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.label), tf.float32))
    
    # add new tensors for training and setup summary and savers
    def train_settings(self, out_dir, lr, sess):
        if self.num_classes > 0:
            self.loss = self.cross_entro
        else:
            self.loss = self.mse
        timestamp = str(int(time.time()))
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, out_dir, timestamp))
        print("Model Writing to {}\n".format(self.out_dir))
        self.global_step = tf.Variable(0, trainable=False)
        
        optimizer = tf.train.AdamOptimizer(lr)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        grads_and_vars = tuple(zip(grads, tvars))
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        loss_summary = tf.summary.scalar('total_loss', self.loss)
        if self.num_classes > 0:
            acc_summary = tf.summary.scalar('accuracy', self.acc)
        else:
            mse_summary = tf.summary.scalar('mse', self.loss)

        self.train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        self.dev_summary_op = tf.summary.merge_all()
        dev_summary_dir = os.path.join(self.out_dir, "summaries", "dev")
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.recent_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        sess.run(tf.global_variables_initializer())
    
    # return the length of each sequence
    def length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length
    
    def BidirectionalGRUEncoder(self, inputs, units, name):
        #inputs shape: [batch_size, max_time, voc_size]
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(units)
            GRU_cell_bw = rnn.GRUCell(units)
            # fw_outputs, bw_outputs size: [batch_size, max_time, hidden_size]
            # time_major=False,
            # if time_major = True, tensor shape: `[max_time, batch_size, depth]`.
            # if time_major = False, tensor shape`[batch_size, max_time, depth]`.
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                    cell_bw=GRU_cell_bw,
                                                                                    inputs=inputs,
                                                                                    sequence_length=self.length(inputs),
                                                                                    dtype=tf.float32)
            #outputs size [batch_size, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs
    
    def AttentionLayer(self, values, units, name):
        # values: [batch_size, max_time, encoder_size=hidden_size*2]
        with tf.variable_scope(name):
            # u_context: [units,]
            u_context = tf.Variable(tf.truncated_normal([units]), name='u_context')
            # h1: [batch_size, max_time, units]
            self.h1 = layers.fully_connected(values, units, activation_fn=tf.nn.tanh)
            # shape [batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(self.h1, u_context), axis=-1, keep_dims=True), dim=1)
            # atten_out shape: (batch_size, encoder_size)
            atten_out = tf.reduce_sum(tf.multiply(values, alpha), axis=1)

            return atten_out, alpha
    

class Topical_attention_model(Vanilla_attention_model):
    # override
    def __init__(self, reduced_vocab_size=2000, num_topic=50, num_classes=20, pretrained_embed=None, 
                 embedding_size=100, RNN_hidden_size=64, topic_hidden_size=64, dropout_keep_proba=0.8, 
                 max_word_num=500, threshold=0, train_embed=True):
    
        with tf.variable_scope('variational_topic_model'):
            self.vtm = VariationalTopicModel(reduced_vocab_size, topic_hidden_size, 
                                            num_topic, embedding_size, dropout_keep_proba)
        with tf.variable_scope('clf_model'):
            if threshold > 0:
                self.threshold = tf.constant(threshold, dtype=tf.float32)
            else:
                self.threshold = None
            Vanilla_attention_model.__init__(self, num_classes, pretrained_embed, embedding_size, 
                                            RNN_hidden_size, dropout_keep_proba, max_word_num, train_embed)

    # override
    def AttentionLayer(self, values, units, name):
        # values: [batch_size, max_time, encoder_size=hidden_size*2]
        # self.topic: [batch_size, num_topic]
        # self.topic_embed: [num_topic, embedding_size]
        if self.threshold != None:
            w = self.vtm.topic-self.threshold
        else:
            w = self.vtm.topic
        with tf.variable_scope(name):
            topic_embed_unstack = tf.unstack(self.vtm.topic_embed) # a list of topic vectors
            topic_atten_weights = []
            # h1: [batch_size, max_time, units]
            self.h1 = layers.fully_connected(values, self.vtm.embedding_size, activation_fn=tf.nn.tanh)
            for i in range(self.vtm.num_topic):
                query = topic_embed_unstack[i]
                # multiplitive attention
                # shape [batch_size, max_time, 1]
                score = tf.reduce_sum(tf.multiply(self.h1, query), axis=-1, keep_dims=True)
                # attention_weights shape == (batch_size, max_length, 1)
                attention_weights = tf.nn.softmax(score, axis=1)
                topic_atten_weights.append(attention_weights)
            
            # topic_atten shape: (batch_size, max_length, 1) 
            topic_atten = tf.matmul(tf.concat(topic_atten_weights, -1), tf.expand_dims(w, -1))
            # atten_out shape after sum == (batch_size, encoder_size)
            atten_out = tf.reduce_sum(tf.multiply(topic_atten, values), axis=1)

            return atten_out, topic_atten_weights

    # overrider
    def train_settings(self, out_dir, lr, sess, pretrain_epoch=0, ckpt_name=None):
        timestamp = str(int(time.time()))
        if self.K>0:
            self.out_dir = os.path.abspath(os.path.join(os.path.curdir, out_dir)+str(self.K))
        else:
            self.out_dir = os.path.abspath(os.path.join(os.path.curdir, out_dir))
        print("Model Writing to {}\n".format(self.out_dir))
        self.global_step = tf.Variable(0, trainable=False)
        
        optimizer = tf.train.AdamOptimizer(lr)

        var_loss = tf.reduce_mean(self.vtm.var_loss)
        self.inference_loss = tf.reduce_mean(self.vtm.kl_divergence)
        self.generative_loss = tf.reduce_mean(self.vtm.likelihood)

        if self.num_classes > 0:
            total_loss = self.cross_entro + var_loss
            self.clf_loss = self.cross_entro
        else:
            total_loss = self.mse + var_loss
            self.clf_loss = self.mse
        
        tvars = tf.trainable_variables()
        grads = tf.gradients(total_loss, tvars)
        grads_and_vars = tuple(zip(grads, tvars))
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        clf_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'clf_model')
        self.train_clf_op = optimizer.minimize(self.clf_loss, global_step=self.global_step, var_list=clf_tvars)

        vtm_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'variational_topic_model')
        self.train_vtm_op = optimizer.minimize(var_loss, global_step=self.global_step, var_list=vtm_tvars)

        clf_loss_sum = tf.summary.scalar('clf_loss', self.clf_loss)
        if self.num_classes > 0:
            acc_summary = tf.summary.scalar('accuracy', self.acc)
        else:
            mse_summary = tf.summary.scalar('mse', self.clf_loss)
        inference_loss_sum = tf.summary.scalar("KL_div", self.inference_loss)
        generative_loss_sum = tf.summary.scalar("likelihood", self.generative_loss)
        loss_sum = tf.summary.scalar("total_loss", total_loss)
        perp_sum = tf.summary.scalar("perplexity_per_batch", self.vtm.perp)
        if self.threshold != None:
            threshold_sum = tf.summary.scalar("threshold", self.threshold)

        train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
        dev_summary_dir = os.path.join(self.out_dir, "summaries", "dev")

        self.train_summary_op = tf.summary.merge_all()
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        self.dev_perp = tf.placeholder(tf.float32, [])
        dev_perp_sum = tf.summary.scalar("dev_perp_per_epoch", self.dev_perp)
        if self.num_classes > 0:
            self.dev_acc = tf.placeholder(tf.float32, [])
            dev_acc_sum = tf.summary.scalar("dev_acc_per_epoch", self.dev_acc)
            self.dev_summary_op = tf.summary.merge([dev_perp_sum, dev_acc_sum])
        else:
            self.dev_mse = tf.placeholder(tf.float32, [])
            dev_mse_sum = tf.summary.scalar("dev_mse_per_epoch", self.dev_mse)
            self.dev_summary_op = tf.summary.merge([dev_perp_sum, dev_acc_sum])
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.best_acc_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_perp_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.recent_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        self.current_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        if pretrain_epoch>0:
            self.current_saver.restore(sess, os.path.join(self.checkpoint_dir, ckpt_name))
        else:
            sess.run(tf.global_variables_initializer())