import numpy as np
import scipy
from sklearn import linear_model
from sklearn import metrics
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack

from Tokenizer import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers, losses
from keras.models import Sequential

t = Tokenizer()
char_label = t.read_in('e04-data/train')

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
y_train = t.labels_2_numbers(labels)


"""
NN with a single layer; uses +-2 context window of onehots
"""
X_train  = t.context_window(X_train)

single_layer_model = Sequential()
single_layer_model.add(Dense(4, input_dim=490))
print('stage 1')
single_layer_model.compile(optimizer='adam', loss='mean_squared_error')
print('stage 2')
single_layer_model.fit(X_train, y_train, epochs=5)

print('JUHU')




