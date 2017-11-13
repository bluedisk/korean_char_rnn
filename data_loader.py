#-*- coding: utf-8 -*-

import os
import zipfile
import collections

import hgtk
import pickle
import numpy as np

DATA_PATH = 'data'

ZIPPED_INPUT_FILE = os.path.join(DATA_PATH, 'bible.zip')

INPUT_FILE = os.path.join(DATA_PATH, 'bible.txt')
VOCAB_FILE = os.path.join(DATA_PATH, 'vocab.pkl')
TENSOR_FILE = os.path.join(DATA_PATH, 'data.npy')


# This is for loading TEXT!
class TextLoader:
    def __init__(self, batch_size=50, seq_length=50):
        self.chars = None
        self.vocab = None
        self.vocab_size = 0
        self.tensor = None
        self.num_batches = 0
        self.pointer = 0
        self.x_batches = None
        self.y_batches = None

        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = INPUT_FILE
        vocab_file = VOCAB_FILE
        tensor_file = TENSOR_FILE

        # make directories
        if not os.path.exists(DATA_PATH):
            os.mkdir(DATA_PATH)

        # unzip text data
        if not os.path.exists(INPUT_FILE):
            with zipfile.ZipFile(ZIPPED_INPUT_FILE, "r") as zip_ref:
                zip_ref.extractall(DATA_PATH)

        # read file
        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)

        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):

        with open(input_file, "r") as f:
            data = f.read()

        data = hgtk.text.decompose(data)
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        with open(vocab_file, 'wb') as f:
            pickle.dump(self.chars, f)

        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = pickle.load(f)
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

    def summary(self):
        print("type of 'data_loader' is %s, length is %d" % (type(self.vocab), len(self.vocab)))
        print("\n")
        print("data_loader.vocab looks like \n%s. " % self.vocab)

        print("\n")
        print("type of 'data_loader.chars' is %s, length is %d" % (type(self.chars), len(self.chars)))
        print("\n")
        print("data_loader.chars looks like \n%s. " % self.chars)
