#!/usr/bin/python
#coding:utf-8
from __future__ import division
from attention_model import Topical_attention_model
import tensorflow as tf
import time
import numpy as np
import random
import os
from utilities import *
from sklearn.utils import shuffle

# load vocab
wv_matrix, vocab_dict, vocab = read_vector("data/preprocessed/embedding/20newsVec.txt")
vocab_size = len(vocab)
print("original vocab size: ", vocab_size)

new_vocab = []
with open("data/topic_model_vocab.txt") as r_f:
    for line in r_f:
        new_vocab.append(line.strip())

train_x_rnn, train_y, train_x_bow, num_train_docs = read_topical_atten_data("data/preprocessed/train-processed.tab", vocab_dict, new_vocab)
test_x_rnn, test_y, test_x_bow, num_test_docs = read_topical_atten_data("data/preprocessed/test-processed.tab", vocab_dict, new_vocab)

# hyperparameters
tf.flags.DEFINE_integer("vocab_size", 2000, "vocabulary size for nueral topic model")
tf.flags.DEFINE_integer("num_classes", num_classes, "number of classes")
tf.flags.DEFINE_integer("embedding_size", EMBEDDING_SIZE, "Dimensionality of word embedding (default: 100)")
tf.flags.DEFINE_integer("RNN_hidden_size", 64, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("topic_hidden_size", 64, "Dimensionality of GRU hidden layer (default: 64)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 50)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
tf.flags.DEFINE_integer("word_num", MAX_NUM_WORD, "the max number of words in a document")

tf.flags.DEFINE_integer("num_topics", 50, "number of topics for nueral topic model")

tf.flags.DEFINE_integer("pretrain_epoch", 0, "for loading pretrain weights and continue training")
tf.flags.DEFINE_string("ckpt_name", '', "checkpoint name")
tf.flags.DEFINE_string("timestamp", '', "only useful for continue training multiple runs")

tf.flags.DEFINE_boolean("train_embed", True, "whether make word embeddings trainable or not. Default not trainable.")

tf.flags.DEFINE_float("threshold", 0.1, "threshold value")

FLAGS = tf.flags.FLAGS

def train_step(sess, model, x_rnn_batch, x_bow_batch, y_batch, mode="train_all"):
    feed_dict = {
        model.input_x: x_rnn_batch,
        model.vtm.x: x_bow_batch,
        model.input_y: y_batch,
        model.is_training: True,
        model.vtm.is_training: True
    }
    if mode=="train_clf":
        train_op = model.train_clf_op
    elif mode=="train_vtm":
        train_op = model.train_vtm_op
    else:
        train_op = model.train_op
    if FLAGS.threshold > 0:
        _, step, summaries, cost, acc, likelihood, kl, perp, thr = sess.run(
        [train_op, model.global_step, model.train_summary_op, model.clf_loss, model.acc, model.generative_loss, model.inference_loss, model.vtm.perp, tf.reduce_mean(model.threshold)], feed_dict)
        print("step {}, cross-entro {:.2f}, acc {:.4f}; likelihood loss {:.2f}, kl_div loss {:.2f}, perplexity {:.2f}, threshold {:.4f}".format(step, cost, acc, likelihood, kl, perp, thr))
    else:
        _, step, summaries, cost, acc, likelihood, kl, perp = sess.run(
            [train_op, model.global_step, model.train_summary_op, model.clf_loss, model.acc, model.generative_loss, model.inference_loss, model.vtm.perp], feed_dict)
        print("step {}, cross-entro {:.2f}, acc {:.4f}; likelihood loss {:.2f}, kl_div loss {:.2f}, perplexity {:.2f}".format(step, cost, acc, likelihood, kl, perp))
    model.train_summary_writer.add_summary(summaries, step)
    return step

def dev_step(sess, model, x_rnn, x_bow, y, acc_record, perp_record, epoch, mode="batch"):
    if mode == "whole":
        feed_dict = {
            model.input_x: x_rnn,
            model.vtm.x: x_bow,
            model.input_y: y,
            model.is_training:False,
            model.vtm.is_training: False
        }
        step, summaries, perp, acc = sess.run([model.global_step, model.dev_summary_op, model.vtm.perp, model.acc], feed_dict)
    elif mode=="batch":
        perp_list= []
        acc_list = []
        num_batch = len(y)//FLAGS.batch_size
        for i in range(num_batch):
            feed_dict = {
                model.input_x: x_rnn[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size],
                model.vtm.x: x_bow[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size],
                model.input_y: y[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size],
                model.is_training:False,
                model.vtm.is_training: False
            }
            perp, acc = sess.run([model.vtm.perp, model.acc], feed_dict)
            perp_list.append(perp)
            acc_list.append(acc)
        perp = np.mean(perp_list)
        acc = np.mean(acc_list)
        summaries = sess.run(model.dev_summary_op, feed_dict={model.dev_perp: perp, model.dev_acc: acc})
    print("+++++++++++++dev+++++++++++++: epoch {}, acc {:.4f}; perplexity {:.2f} ".format(epoch, acc, perp))
    model.dev_summary_writer.add_summary(summaries, epoch)

    if acc > acc_record:
        acc_record = acc
        print("new best acc: ", acc_record)
        model.best_acc_saver.save(sess, model.checkpoint_dir + '/best-acc-model-acc={:.4f}-perp={:.2f}-epoch{}.ckpt'.format(acc_record, perp, epoch))
    else:
        print("the best dev acc: ", acc_record)
    if perp < perp_record:
        perp_record = perp
        print("new best perplexity: ", perp_record)
        model.best_perp_saver.save(sess, model.checkpoint_dir + '/best-perp-model-acc={:.4f}-perp={:.2f}-epoch{}.ckpt'.format(acc, perp_record, epoch))
    else:
        print("the best dev perplexity: ", perp_record)

    model.current_saver.save(sess, model.checkpoint_dir + '/cur-model-acc={:.4f}-prep={:.2f}-epoch{}.ckpt'.format(acc, perp, epoch))
    if epoch % 100 == 0:
        model.recent_saver.save(sess, model.checkpoint_dir + '/model-acc={:.4f}-prep={:.2f}-epoch{}.ckpt'.format(acc, perp, epoch))
    return acc_record, perp_record

def train_topical(acc_record, perp_record):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True # allocate only as much GPU memory based on runtime allocations

    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        model = Topical_attention_model(
                        reduced_vocab_size=FLAGS.vocab_size,
                        num_topic=FLAGS.num_topics,
                        num_classes=FLAGS.num_classes,
                        pretrained_embed=wv_matrix,
                        embedding_size=FLAGS.embedding_size,
                        RNN_hidden_size=FLAGS.RNN_hidden_size,
                        topic_hidden_size=FLAGS.topic_hidden_size,
                        dropout_keep_proba=0.8,
                        max_word_num=FLAGS.word_num,
                        threshold=FLAGS.threshold, 
                        train_embed = FLAGS.train_embed
                        )
        if FLAGS.train_embed:
            out_dir = 'topical_runs_threshold_new'
        else:
            out_dir = 'topical_runs_threshold_no_embed'
        out_dir += "/topics={}_threshold={}_batch={}_".format(FLAGS.num_topics, FLAGS.threshold, FLAGS.batch_size)

        model.train_settings(out_dir, FLAGS.learning_rate, sess, FLAGS.pretrain_epoch, FLAGS.ckpt_name)

        for epoch in range(FLAGS.pretrain_epoch+1, FLAGS.num_epochs+1):
            X_rnn_train, X_bow_train, Y_train = shuffle(train_x_rnn, train_x_bow, train_y)
            X_rnn_test, X_bow_test, Y_test = shuffle(test_x_rnn, test_x_bow, test_y)
            print('current epoch %s' % (epoch))
            mode = "train_all"
            print("train mode: ", mode)
            for i in range(train_y.shape[0]//FLAGS.batch_size):
                x_rnn_batch = X_rnn_train[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
                x_bow_batch = X_bow_train[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
                y_batch = Y_train[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
                step = train_step(sess, model, x_rnn_batch, x_bow_batch, y_batch, mode)
            acc_record, perp_record = dev_step(sess, model, X_rnn_test, X_bow_test, Y_test, acc_record, perp_record, epoch, "batch")

if __name__ == '__main__':
    acc_record = 0.0
    perp_record = np.Infinity
    train_topical(acc_record, perp_record)