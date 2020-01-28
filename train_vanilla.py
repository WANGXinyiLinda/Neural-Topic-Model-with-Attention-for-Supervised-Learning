#!/usr/bin/python
#coding:utf-8
from __future__ import division
from attention_model import Vanilla_attention_model
import tensorflow as tf
import numpy as np
import os
from utilities import *
from sklearn.utils import shuffle

# load data
wv_matrix, vocab_dict, vocab = read_vector("data/preprocessed/embedding/20newsVec.txt")
vocab_size = len(vocab)
print("vocab size: ", vocab_size)

new_vocab = []
with open("data/topic_model_vocab.txt") as r_f:
    for line in r_f:
        new_vocab.append(line.strip())

train_x_rnn, train_y, _, num_train_docs = read_topical_atten_data("data/preprocessed/train-processed.tab", vocab_dict, new_vocab)
test_x_rnn, test_y, _, num_dev_docs = read_topical_atten_data("data/preprocessed/test-processed.tab", vocab_dict, new_vocab)

# hyperparameters
tf.flags.DEFINE_integer("num_classes", num_classes, "number of classes")
tf.flags.DEFINE_integer("embedding_size", EMBEDDING_SIZE, "Dimensionality of word embedding (default: 100)")
tf.flags.DEFINE_integer("RNN_hidden_size", 64, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("num_checkpoin", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
tf.flags.DEFINE_integer("word_num", MAX_NUM_WORD, "the max number of words in a document")
tf.flags.DEFINE_float("train_dev_split", 0.8, "train dev split (default 9:1)")

FLAGS = tf.flags.FLAGS

def train_step(sess, model, x_batch, y_batch):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.is_training: True
    }
    _, step, summaries, cost, acc = sess.run([model.train_op, model.global_step, model.train_summary_op, model.loss, model.acc], feed_dict)
    print("step {}, loss: {:g}, acc: {:g}".format(step, cost, acc))
    model.train_summary_writer.add_summary(summaries, step)
    return step

def dev_step(sess, model, x, y, acc_record, epoch):
    feed_dict = {
        model.input_x: x,
        model.input_y: y,
        model.is_training:False
    }
    summaries, cost, acc = sess.run([model.dev_summary_op, model.loss, model.acc], feed_dict)
    model.dev_summary_writer.add_summary(summaries, epoch)
    print("++++++++++++++dev+++++++++++++: epcoh {}, loss: {:g}, acc; {:g} ".format(epoch, cost, acc))
    if acc > acc_record:
        acc_record = acc
        print("new best acc: ", acc_record)
        model.best_saver.save(sess, model.checkpoint_dir + '/best-model-acc={:g}-epoch{}.ckpt'.format(acc_record, epoch))
    else:
        print("the best dev acc: ", acc_record)
    if epoch % 10 == 0:
        model.recent_saver.save(sess, model.checkpoint_dir + '/model-acc={:g}-epoch{}.ckpt'.format(acc, epoch))
    return acc_record

def train_vanilla(acc_record):
    with tf.Session() as sess:
        model = Vanilla_attention_model(
                        num_classes=FLAGS.num_classes,
                        pretrained_embed=wv_matrix,
                        embedding_size=FLAGS.embedding_size,
                        hidden_size=FLAGS.RNN_hidden_size,
                        dropout_keep_proba=0.8,
                        max_word_num=FLAGS.word_num,
                        )
        model.train_settings("vanilla_runs/", FLAGS.learning_rate, sess)

        for epoch in range(1, FLAGS.num_epochs+1):
            _X, _Y = shuffle(train_x_rnn, train_y)
            print('current epoch %s' % (epoch))
            for i in range(train_x_rnn.shape[0]//FLAGS.batch_size):
                x_batch = _X[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
                y_batch = _Y[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
                step = train_step(sess, model, x_batch, y_batch)
            acc_record = dev_step(sess, model, test_x_rnn, test_y, acc_record, epoch)

if __name__ == '__main__':
    acc_record = 0.0
    train_vanilla(acc_record)