from typing import re

import numpy as np
import scipy
from sklearn import linear_model
from sklearn import metrics
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack

from Tokenizer import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers, losses
from keras.models import Sequential
from keras import metrics
from keras import regularizers

import tensorflow as tf

t = Tokenizer()

###################################################################
#TRAINING DATA
###################################################################
char_label = t.read_in('e04-data/train')#training data

#fill char2id
for c in char_label:
    if c[0] not in t.char2id: #if the char is not in the char2id add it
        t.char2id.append(c[0])
t.char2id.append(t.unseen_char)

#fill label2id
for l in char_label:
    if l[1] not in t.label2id: #if the label is not in the label2id add it
        t.label2id.append(l[1])

#get onehots for all training characters
chars = []
for tpl in char_label:
    chars.append(tpl[0])
X_train = t.one_hot(chars) #onehots of chars

#get classes for training characters
labels = []
for tpl in char_label:
    labels.append(tpl[1])

#make the onehot method more general so that it can be used for the labels aswell
idx = [] #list of characters into list of indecies
for l in labels:
    if l in t.label2id:
        idx.append(t.label2id.index(l))
y_train = np.eye(len(t.label2id))[idx]

###################################################################
#TEST DATA
###################################################################
test_data = t.read_in('e04-data/test')#test data
chars_test = []
for tpl in test_data: #get the unseen characters for prediction
    chars_test.append(tpl[0])
labels_test = []
for tpl in test_data:
    labels_test.append(tpl[1])

#prepare test data
X_predict = t.one_hot(chars_test) #onehot representation of the unseen data
X_predict = t.context_window(X_predict).toarray()

#make the onehot method more general so that it can be used for the labels aswell
idx = [] #list of characters into list of indecies
for l in labels_test:
    if l in t.label2id:
        idx.append(t.label2id.index(l))
y_correct = np.eye(len(t.label2id))[idx]  #the correct target classes


#fill onehot_label2id for turning onehots back to numbers
for o in y_train:
    if o.tolist() not in t.onehot_label2id:
        t.onehot_label2id.append(o.tolist())

"""
NN with a single layer; uses +-2 context window of onehots
"""
X_train  = t.context_window(X_train).toarray()

single_layer_model = Sequential()
single_layer_model.add(Dense(100, input_dim=X_train.shape[1], activation='relu', activity_regularizer=regularizers.l2(0.01)))
single_layer_model.add(Dropout(1))
single_layer_model.add(Dense(4, activation='softmax', activity_regularizer=regularizers.l2(0.01)))
single_layer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
single_layer_model.fit(X_train, y_train, epochs=5,validation_data=(X_predict, y_correct))
loss = single_layer_model.evaluate(X_predict, y_correct)
print()
print('Loss on test-data',loss)

y_predicted = single_layer_model.predict(X_predict)
y_predicted = np.argmax(y_predicted, axis=1)#get the indecies of the highest values

#tokenize some input
#test1 = 'From the AP comes this story : President Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area.'
#test2 = 'As such, the only hypothesis that needs to be specified in this test and which embodies the counter-claim is referred to as the null hypothesis (that is, the hypothesis to be nullified). A result is said to be statistically significant if it allows us to reject the null hypothesis. That is, as per the reductio ad absurdum reasoning, the statistically significant result should be highly improbable if the null hypothesis is assumed to be true. The rejection of the null hypothesis implies that the correct hypothesis lies in the logical complement of the null hypothesis. However, unless there is a single alternative to the null hypothesis, the rejection of null hypothesis does not tell us which of the alternatives might be the correct one.'
#tokenized = t.tokenizeNN(y_predicted, test1)
#print(tokenized)




