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


#for turning onehots back to numbers
onehot_label2id = []
for o in y_train: #some magic is going on here
    if o.tolist() not in onehot_label2id:
        onehot_label2id.append(o.tolist())

"""
NN with a single layer; uses +-2 context window of onehots
"""
X_train  = t.context_window(X_train).toarray()

single_layer_model = Sequential()
single_layer_model.add(Dense(4, input_dim=X_train.shape[1], activation='softmax', bias_regularizer=regularizers.l2(0.01)))
single_layer_model.add(Dense(4, input_dim=X_train.shape[1], activation='softmax', bias_regularizer=regularizers.l2(0.01)))
single_layer_model.compile(optimizer='adam', loss='categorical_crossentropy')
single_layer_model.fit(X_train, y_train, epochs=10)
loss = single_layer_model.evaluate(X_predict, y_correct)
print()
print(loss)

y_predicted = single_layer_model.predict(X_predict)
y_predicted = np.argmax(y_predicted, axis=1)#get the indecies of the highest values

z_mtrx = np.zeros((len(y_predicted),4))#zero matrix
for row, idx in zip(z_mtrx, y_predicted):#fille the zero matrix
    row[idx] = 1

#get pure numbers
labels_num = []
for row in z_mtrx:
    if row.tolist() in onehot_label2id:
        labels_num.append(onehot_label2id.index(row.tolist()))
print(labels_num)

#tokenize some input
test1 = 'From the AP comes this story : President Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area.'

characters = list(test1)
print(characters)
#get the character labels
labels_ch = []
for lbl in labels_num:
    labels_ch.append(t.label2id[lbl])
print(labels_ch)
sentences = []
sent = []
token = ""
for c, l in zip(characters, labels_ch):
    if l == 'I':
        token += c
    elif l == 'T' or l == 'S':
        if len(token) > 0:
            sent.append(token)  # add token to the sentence
        token = c  # reset token and add first char of the next token
        if l == 'S' and len(sent) > 0:
            sentences.append(sent)
            sent = []
# at the end add the last token and sent to its higher structure
if len(token) > 0:
    sent.append(token)
if len(sent) > 0:
    sentences.append(sent)

print(sentences)





