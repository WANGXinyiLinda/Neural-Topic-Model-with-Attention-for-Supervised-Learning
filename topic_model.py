import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import os

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

class VariationalTopicModel(object):
    """
    Neural Variational Model for Document Modeling. 
    Reference: https://github.com/pbhatia243/neural_topic_models
    Not exactly the same as GSM. Added batch normalization and dropout.
    """
    def __init__(self, vocab_size, latent_dim, num_topic, embedding_size, dropout_keep_proba=0.8):
        
        self.num_topic = num_topic
        self.embedding_size = embedding_size

        with tf.variable_scope("placeholder"):
            self.x = tf.placeholder(tf.float32, [None, vocab_size], name="input_x")
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        with tf.variable_scope("encoder"):
            self.hidden_encoder = layers.fully_connected(self.x, latent_dim, activation_fn=tf.nn.relu, scope="lambda1")
            self.pi = layers.fully_connected(self.hidden_encoder, latent_dim, activation_fn=tf.nn.relu, scope="pi")
            self.pi_normalized = layers.batch_norm(self.pi, is_training=self.is_training) # really important!!!
            self.pi_dropped = layers.dropout(self.pi_normalized, keep_prob=dropout_keep_proba, is_training=self.is_training)

            # Mean Encoder
            self.mu_encoder = layers.fully_connected(self.pi_dropped, latent_dim, activation_fn=None, scope="mu_encoder")

            # Sigma Encoder
            self.logvar_encoder = layers.fully_connected(self.pi_dropped, latent_dim, activation_fn=None, scope="logvar_encoder")

            # Sample epsilon
            self.epsilon = tf.random_normal((tf.shape(self.logvar_encoder)), 0, 1, name='epsilon')

            self.std_dev = tf.sqrt(tf.exp(self.logvar_encoder))
            self.h = self.mu_encoder + self.std_dev * self.epsilon

        with tf.variable_scope("decoder"):
            # topic: doc-topic distribution. shape: [batch_size, num_topic]
            self.topic = layers.fully_connected(self.h, self.num_topic, activation_fn=tf.nn.softmax, scope="topic")

            self.W = tf.Variable(xavier_init(vocab_size, self.embedding_size), dtype=tf.float32, name="word_embed") 
            self.W_dropped = layers.dropout(self.W, keep_prob=dropout_keep_proba, is_training=self.is_training)
            self.topic_embed = tf.Variable(xavier_init(self.num_topic, self.embedding_size), name="topic_embed")

            # beta: topic-word distribution. shape: [num_topic, vocab_size]
            self.beta = tf.nn.softmax(tf.matmul(self.topic_embed, self.W_dropped, transpose_b=True))
            # p(x|topic). shape: [batch_size, vocab_size]
            self.p_x = tf.matmul(self.topic, self.beta)
            
        # Calculate loss
        with tf.variable_scope("var_loss"):
            # (negative) KL Divergence Loss
            self.kl_divergence = -0.5*tf.reduce_sum(1.0 + self.logvar_encoder - tf.square(self.mu_encoder) - tf.exp(self.logvar_encoder), 1)
            # (negative) Log likelihood
            self.likelihood = -tf.reduce_sum(tf.multiply(tf.log(self.p_x + 1e-10), self.x), 1)
            # variational loss = -ELBO (evidence lower bound)
            self.var_loss = self.likelihood + self.kl_divergence

        # calculate the perplexity of the topic model using likelihood
        with tf.variable_scope("perplexity"):
            # num_words: [batch_size, ]
            num_words = tf.reduce_sum(self.x, axis=-1)
            # prep: scalar
            self.perp = tf.exp(tf.reduce_mean(tf.truediv(self.likelihood, num_words)))

    # add new tensors for training and setup summary and savers
    def train_settings(self, out_dir, lr, sess):

        print("Model Writing to {}\n".format(out_dir))

        optimizer = tf.train.AdamOptimizer(lr)

        self.inference_loss = tf.reduce_mean(self.kl_divergence)
        self.generative_loss = tf.reduce_mean(self.likelihood)
        self.variational_loss = tf.reduce_mean(self.var_loss)

        inference_loss_sum = tf.summary.scalar("KL_div", self.inference_loss)
        generative_loss_sum = tf.summary.scalar("likelihood", self.generative_loss)
        variational_loss_sum = tf.summary.scalar("variational_loss", self.variational_loss)
        perp_sum = tf.summary.scalar("perplexity_per_batch", self.perp)
        
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")

        self.checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.global_step = tf.Variable(0, trainable=False)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.variational_loss, tvars)
        grads_and_vars = tuple(zip(grads, tvars))
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        self.train_summary_op = tf.summary.merge_all()
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        self.dev_summary_op = tf.summary.merge([generative_loss_sum, perp_sum])
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.recent_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        sess.run(tf.global_variables_initializer())