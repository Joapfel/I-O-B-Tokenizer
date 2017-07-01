import numpy as np
from sklearn import linear_model
from sklearn import metrics

class Tokenizer(object):
    __slots__ = 'char2id', 'label2id', 'unseen_char'

    def __init__(self):
        self.char2id = [] #where id == index
        self.label2id = [] #where id == index
        self.unseen_char = '</unseen>'

    def read_in(self, path):
        if not isinstance(path,str):
            raise TypeError('Expected argument for the path is string!')
        chars_labels = []
        with open(path, 'r') as f:
            for line in f:
                line = line.split()
                if len(line) == 2:
                    chars_labels.append((line[0], line[1])) #add tuples of the form (char, label) to the list
                elif len(line) == 1:
                    chars_labels.append(('<wb>', line[0])) #<wb> == word boundary
        return chars_labels

    def one_hot(self, characters): #arg = list of characters
        if not isinstance(characters, list):
            raise TypeError('Expected argument is a list (of characters).')
        idx = [] #list of characters into list of indecies
        for c in characters:
            if c in self.char2id:
                idx.append(self.char2id.index(c))
            else:
                idx.append(self.char2id.index(self.unseen_char))
        return np.eye(len(self.char2id))[idx]

    def labels_2_numbers(self, labels):
        idx = []
        for l in labels:
            idx.append(self.label2id.index(l))
        return np.array(idx)


"""
Simple logistic classifier (maxent) that predicts the class of the character only from its
one-hot representation
"""
##########################################################################
#TRAIN
##########################################################################
t = Tokenizer()
char_label = t.read_in('e04-data/train') #list of touples (char, label)

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

#train simple logistic classifier
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train)

##########################################################################
#PREDICT
##########################################################################

#read in test data
test_data = t.read_in('e04-data/test') #list of touples (char, label)
chars_test = []
for tpl in test_data: #get the unseen characters for prediction
    chars_test.append(tpl[0])
labels_test = []
for tpl in test_data:
    labels_test.append(tpl[1])

#prepare test data
X_predict = t.one_hot(chars_test) #onehot representation of the unseen data
y_correct = t.labels_2_numbers(labels_test) #the correct target classes
y_predicted = logreg.predict(X_predict) #the predicted classes

print('RESULTS WIHTOUT CONTEXT WINDOW: ')
print()

accuracy = metrics.accuracy_score(y_correct, y_predicted)
print('accuracy: ', accuracy)
print()

precision = metrics.precision_score(y_correct, y_predicted, average='macro')
print('precision: ', precision)
print()

recall = metrics.recall_score(y_correct, y_predicted, average='macro')
print('recall: ', recall)
print()

f1_score = metrics.f1_score(y_correct, y_predicted, average='macro')
print('f1_score: ', f1_score)
print()





