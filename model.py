#-*- coding: utf-8 -*-

import os
import hgtk
import numpy as np

# Define Network
rnn_size = 1024
num_layers = 2
grad_clip = 5.

MODEL_PATH = 'models'
LOGS_PATH = 'logs'


class KoreanCharacterRNN:

    def __init__(self, tf, data_loader, start_learning_rate=0.002, decay_step=50, decay_rate=0.97):

        self.tf = tf
        self.data = data_loader

        self.batch_size = data_loader.batch_size
        self.seq_length = data_loader.seq_length

        # Select RNN Cell
        unitcell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([unitcell] * num_layers)

        # Set paths to the graph
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.seq_length], name='input_data')
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.seq_length], name='targets')
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        # Set Network
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, self.data.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [self.data.vocab_size])

            #with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.data.vocab_size, rnn_size])
            inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), self.seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # # Loop function for seq2seq
        # def loop(prev, _):
        #     prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
        #     prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
        #     return tf.nn.embedding_lookup(embedding, prev_symbol)

        # Output of RNN
        outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(
            inputs, self.initial_state, self.cell, loop_function=None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # Next word probability
        self.probs = tf.nn.softmax(logits)

        # Define LOSS
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],  # Input
                                                                  [tf.reshape(self.targets, [-1])],  # Target
                                                                  [tf.ones([self.batch_size * self.seq_length])],  # Weight
                                                                  self.data.vocab_size)

        # Define Optimizer
        self.cost = tf.reduce_sum(loss) / self.batch_size / self.seq_length
        tf.summary.scalar("cost", self.cost)

        self.final_state = last_state
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)

        # for weight decay
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                                   decay_step, decay_rate, staircase=False)
        tf.summary.scalar("learning_rate", learning_rate)

        self.optm = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, tvars))

        self.merged = tf.summary.merge_all()

        # make directories
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)

        if not os.path.exists(LOGS_PATH):
            os.mkdir(LOGS_PATH)

        # init session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # saver
        self.saver = tf.train.Saver(tf.global_variables())
        self.summary_writer = tf.summary.FileWriter(LOGS_PATH, graph=self.sess.graph)

    def restore(self):
        latest_checkpoint = self.tf.train.latest_checkpoint(MODEL_PATH)

        if not latest_checkpoint:
            print("Trained network not found on ", MODEL_PATH)
            return False
        
        print("Restore network from ", latest_checkpoint)
        self.saver.restore(self.sess, latest_checkpoint)
        return True

    def save(self, step):
        checkpoint_path = os.path.join(MODEL_PATH, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=step)
        print("model saved to {}".format(checkpoint_path))
        
    def get_state(self):
        return self.sess.run(self.initial_state)

    def train(self, x, y, last_state, step):
        train_loss, state, _, summary = self.sess.run(
            [self.cost, self.final_state, self.optm, self.merged], {
                self.input_data: x,
                self.targets: y,
                self.initial_state: last_state
            })
        self.summary_writer.add_summary(summary, step)
        return train_loss, state

    # Sampling function
    def sample(self, num=200, prime='오늘은'):
        state = self.sess.run(self.cell.zero_state(1, self.tf.float32))
        prime = list(hgtk.text.decompose(prime))

        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = self.data.vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = self.sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return int(np.searchsorted(t, np.random.rand(1) * s))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = self.data.vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [_probsval, state] = self.sess.run([self.probs, self.final_state], feed)
            p = _probsval[0]

            # sample = int(np.random.choice(len(p), p=p))
            sample = weighted_pick(p)
            pred = self.data.chars[sample]
            ret += pred
            char = pred

        return ret
