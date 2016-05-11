# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

import collections
import argparse
import time
import os
import codecs

from six.moves import cPickle


# In[2]:

# This is for loading TEXT!
class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding="utf8") as f:
            data = f.read()

        #data = list(data)

        print "Counting.."
        counter = collections.Counter(data)
        print "Sorting.."
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        print "Making dictionary.."
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        print "Saving dictionary.."
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


# In[3]:

# Load text
batch_size  = 50
seq_length  = 50
data_dir    = "data/han1"
#data_dir    = "data/han2"
#data_dir    = "data/han3"
#data_dir    = "data/han4"
#data_dir    = "data/han5"
save_dir    = "save"
data_loader = TextLoader(data_dir, batch_size, seq_length)

args = 0

with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
    cPickle.dump(args, f)
with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'wb') as f:
    cPickle.dump((data_loader.chars, data_loader.vocab), f)


print ("Done")


# In[4]:

data_loader.vocab_size
data_loader.vocab
data_loader.chars

print ( "type of 'data_loader' is %s, length is %d" % (type(data_loader.vocab), len(data_loader.vocab)) )
print ( "\n" )
print ("data_loader.vocab looks like \n%s. " % (data_loader.vocab))

print ( "\n" )
print ( "type of 'data_loader.chars' is %s, length is %d" % (type(data_loader.chars), len(data_loader.chars)) )
print ( "\n" )
print ("data_loader.chars looks like \n%s. " % (data_loader.chars,))


# In[ ]:

rnn_size   = 1024
num_layers = 2
grad_clip  = 5.

vocab_size = data_loader.vocab_size

# Select RNN Cell
unitcell = rnn_cell.BasicLSTMCell(rnn_size)
cell = rnn_cell.MultiRNNCell([unitcell] * num_layers)

# Set paths to the graph
input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
targets    = tf.placeholder(tf.int32, [batch_size, seq_length])
initial_state = cell.zero_state(batch_size, tf.float32)

# Set Network
with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
        inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, input_data))
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
# Loop function for seq2seq
def loop(prev, _):
    prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
    return tf.nn.embedding_lookup(embedding, prev_symbol)
# Output of RNN
outputs, last_state = seq2seq.rnn_decoder(inputs, initial_state, cell, loop_function=None, scope='rnnlm')
output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
# Next word probability
probs = tf.nn.softmax(logits)
# Define LOSS
loss = seq2seq.sequence_loss_by_example([logits], # Input
    [tf.reshape(targets, [-1])], # Target
    [tf.ones([batch_size * seq_length])], # Weight
    vocab_size)
# Define Optimizer
cost = tf.reduce_sum(loss) / batch_size / seq_length
final_state = last_state
lr = tf.Variable(0.0, trainable=False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
_optm = tf.train.AdamOptimizer(lr)
optm = _optm.apply_gradients(zip(grads, tvars))

print ("Network Ready")


# In[ ]:

# Train the model!
num_epochs    = 50
save_every    = 1000
learning_rate = 0.002
decay_rate    = 0.97


sess = tf.Session()
sess.run(tf.initialize_all_variables())
summary_writer = tf.train.SummaryWriter('/tmp/tf_logs/char_rnn', graph=sess.graph)
saver = tf.train.Saver(tf.all_variables())
for e in range(num_epochs): # for all epochs

    # Learning rate scheduling
    sess.run(tf.assign(lr, learning_rate * (decay_rate ** e)))

    data_loader.reset_batch_pointer()
    state = sess.run(initial_state)
    for b in range(data_loader.num_batches):
        start = time.time()
        x, y = data_loader.next_batch()
        feed = {input_data: x, targets: y, initial_state: state}
        # Train!
        train_loss, state, _ = sess.run([cost, final_state, optm], feed)
        end = time.time()

        if b % 100 == 0:
            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"                 .format(e * data_loader.num_batches + b,
                        num_epochs * data_loader.num_batches,
                        e, train_loss, end - start))

        if (e * data_loader.num_batches + b) % save_every == 0:
            checkpoint_path = os.path.join(save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
            print("model saved to {}".format(checkpoint_path))



# In[ ]:



