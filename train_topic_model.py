#!/usr/bin/python
#coding:utf-8
from __future__ import division
from topic_model import VariationalTopicModel
import tensorflow as tf
import time
import numpy as np
import os
from utilities import *
from sklearn.utils import shuffle

# load my data
wv_matrix, vocab_dict, old_vocab = read_vector("data/preprocessed/embedding/20newsVec.txt")
old_vocab_size = len(old_vocab)
print("vocab size: ", old_vocab_size)
vocab_size=2000
print("reduced vocab size: ", vocab_size)
vocab = []
with open("data/topic_model_vocab.txt") as r_f:
    for line in r_f:
        vocab.append(line.strip())

_, train_y, train_x_bow, num_train_docs = read_topical_atten_data("data/preprocessed/train-processed.tab", vocab_dict, vocab)
_, test_y, test_x_bow, num_dev_docs = read_topical_atten_data("data/preprocessed/test-processed.tab", vocab_dict, vocab)

# hyperparameters
tf.flags.DEFINE_integer("num_topics", 50, "number of topics")
tf.flags.DEFINE_integer("embedding_size", 100, "Dimensionality of word embedding")
tf.flags.DEFINE_integer("hidden_size", 64, "Dimensionality of hidden layer")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 400, "Number of training epochs")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
tf.flags.DEFINE_integer("vocab_size", 2000, "vocabulary size")

FLAGS = tf.flags.FLAGS

# python train_topic_model.py --num_topics 50

def train_step(sess, model, x_batch):
    feed_dict = {
        model.x: x_batch,
        model.is_training: True
    }
    _, step, summaries, likelihood_cost, kl_cost, var_cost, perpl = sess.run([model.train_op, model.global_step, model.train_summary_op, model.generative_loss, model.inference_loss, model.variational_loss, model.perp], feed_dict)

    print("train: step {}, likelihood loss {:g}, kl_div loss {:g}, variantional loss {:g}, perplexity {:g}".format(
            step, likelihood_cost, kl_cost, var_cost, perpl))
    model.train_summary_writer.add_summary(summaries, step)

def dev_step(sess, model, x, perp_record, epoch, mode="batch"):
    if mode == "whole":
        feed_dict = {
            model.x: x,
            model.is_training:False
        }
        summaries, cost, perp = sess.run([model.dev_summary_op, model.generative_loss, model.perp], feed_dict)
    elif mode=="batch":
        perp_list= []
        cost_list = []
        num_batch = len(x)//FLAGS.batch_size
        for i in range(num_batch):
            feed_dict = {
                model.x: x[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size],
                model.is_training:False
            }
            cost, perp = sess.run([model.generative_loss, model.perp], feed_dict)
            perp_list.append(perp)
        perp = np.mean(perp_list)
        cost = np.mean(cost_list)
        summaries = tf.Summary()
        summaries.value.add(tag="dev_perp_per_epoch", simple_value=perp)
        summaries.value.add(tag="dev_likelihood_per_epoch", simple_value=cost)
    model.dev_summary_writer.add_summary(summaries, epoch)
    print("** dev **: epoch {}, likelihood loss {:g}, perplexity {:g} ".format(epoch, cost, perp))
    if perp < perp_record:
        perp_record = perp
        model.best_saver.save(sess, model.checkpoint_dir + '/best-model-prep={:g}-epoch{}.ckpt'.format(perp_record, epoch))
        print("new best dev perplexity: ", perp_record)
    else:
        print("the best dev perplexity: ", perp_record)
    if epoch % 100 == 0:
        model.recent_saver.save(sess, model.checkpoint_dir + '/model-prep={:g}-epoch{}.ckpt'.format(perp, epoch))
    return perp_record

def main(conti=False):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True # allocate only as much GPU memory based on runtime allocations
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.Session(config=config) as sess:
        with tf.variable_scope('variational_topic_model'):
            model = VariationalTopicModel(vocab_size=FLAGS.vocab_size,
                            latent_dim = FLAGS.hidden_size,
                            num_topic = FLAGS.num_topics,
                            embedding_size=FLAGS.embedding_size,
                            dropout_keep_proba=0.5
                            )
        if conti:
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_NTM_new", ''))            
        else:
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_NTM_new/{}".format(FLAGS.num_topics), timestamp)) 
        
        model.train_settings(out_dir, FLAGS.learning_rate, sess)
        
        if conti:
            model.recent_saver.restore(sess, model.checkpoint_dir + '')
            print("Model restored.")

        perp_record = np.inf

        for epoch in range(1, FLAGS.num_epochs+1):
            X_train = shuffle(train_x_bow)
            X_test = shuffle(test_x_bow)
            print('current epoch %s' % (epoch))
            for i in range(train_x_bow.shape[0]//FLAGS.batch_size):
                x_batch = X_train[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
                train_step(sess, model, x_batch)
            perp_record = dev_step(sess, model, X_test, perp_record, epoch, "batch")

if __name__ == '__main__':
    main(conti=False)