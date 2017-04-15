import argparse
import reader
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.layers import Embedding
import json

import numpy as np
import random
import sys
import os

import shutil
import requests



parser = argparse.ArgumentParser()
parser.add_argument("--test", help="path of test.txt file")
args = parser.parse_args()

f=open('word-ix.txt', 'r')
word_to_ix=json.loads(f.read())
f.close()

model = Sequential()
model.add(Embedding(10000, output_dim=64))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(10000))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

if args.test:
    test_given=reader._file_to_word_ids(args.test, word_to_ix)
    maxlen = 15
    step = 5
    sentences=[]
    next_words=[]
    for i in range(0,len(test_given)-maxlen, 1):
        sentences2 = (test_given[i: i + maxlen])
        sentences.append(sentences2)
        next_words.append((test_given[i + maxlen]))

    X_test = np.zeros((len(sentences), maxlen))
    y_test = np.zeros((len(sentences)), dtype=np.int32)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            X_test[i, t] = word
        y_test[i]=next_words[i]

    if os.path.isfile('weights'):
        model.load_weights('weights')
    else:
        print ("Downloading Model ......")
        url = 'https://raw.githubusercontent.com/shivam207/deep_learning/master/prog_asg3/weights/model_cnn.h5'
        response = requests.get(url, stream=True)
        with open('model_cnn.h5', 'w') as out_file:
            shutil.copyfileobj(response.raw, out_file)
    del response
    preds = model.predict(X_test, verbose=0)
    loss = 0
    for i, pred in enumerate(preds):
        loss += -pred[y_test[i]]*np.log(pred[y_test[i]])
    print loss

else:
    train_data, valid_data, test_data, vocabulary, word_to_ix = reader.ptb_raw_data("data")
    words = set(train_data)
    words.union(valid_data)
    words.union(test_data)
    maxlen = 15
    step = 5
    print("maxlen:",maxlen,"step:", step)
    sentences = []
    next_words = []
    sentences2=[]
    for i in range(0,len(train_data)-maxlen, step):
        sentences2 = (train_data[i: i + maxlen])
        sentences.append(sentences2)
        next_words.append((train_data[i + maxlen]))
    print('nb sequences(length of sentences):', len(sentences))
    print("length of next_word",len(next_words))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen))
    y = np.zeros((len(sentences), len(words)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            #print(i,t,word)
            X[i, t] = word
        y[i, next_words[i]] = 1

    for iteration in range(1, 300):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=200 ,nb_epoch=2)
        model.save_weights('weights',overwrite=True)


