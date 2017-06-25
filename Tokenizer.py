import numpy as np
from sklearn import linear_model
from sklearn import metrics

class Tokenizer(object):
    __slots__ = 'char2id', 'label2id', 'logreg', 'chars_labels_train', 'X_onehot', 'y', 'chars_labels_test',  'chars_test', 'labels_test'

    def __init__(self):
        self.char2id = [] #where id == index
        self.label2id = [] #where id == index
        self.logreg = linear_model.LogisticRegression()
        self.chars_labels_train = []
        self.chars_labels_test = []
        self.chars_test = [] #all characters from the test set (extracted from chars_labels_test)
        self.labels_test = [] #the true labels from the data (extracted from chars_labels_test)

    def set_X_onehot(self):
        x_tmp = []
        for tmp in self.chars_labels_train:
            x_tmp.append(tmp[0])
        self.X_onehot = self.one_hot(x_tmp)

    def set_y(self):
        y_tmp = []
        for tmp in self.chars_labels_train:
            y_tmp.append(tmp[1])
        self.y = self.labels_2_numbers(y_tmp)

    def read_in_train(self, path):
        if not isinstance(path, str):
            raise TypeError('Expected argument for the path is string!')
        with open(path, 'r') as f:
            for line in f:
                line = line.split()
                if len(line) == 2:
                    self.chars_labels_train.append((line[0], line[1])) #add tuples of the form (char, label) to the list
                    #fill char2id for one hots
                    if line[0] not in self.char2id:
                        self.char2id.append(line[0])
                    #fill label2id to get target classes
                    if line[1] not in self.label2id:
                        self.label2id.append(line[1])
                elif len(line) == 1:
                    self.chars_labels_train.append(('</w>', line[0]))
                    if '</w>' not in self.char2id:
                        self.char2id.append('</w>')
                    #fill label2id
                    if line[0] not in self.label2id:
                        self.label2id.append(line[0])
        self.set_X_onehot()
        self.set_y()

    def read_in_test(self, path):
        if not isinstance(path, str):
            raise TypeError('Expected argument for the path is string!')
        with open(path, 'r') as f:
            for line in f:
                line = line.split()
                if len(line) == 2:
                    self.chars_labels_test.append((line[0], line[1]))
                elif len(line) == 1:
                    self.chars_labels_test.append(('</w>', line[0]))
        #set chars_test var
        for tpl in self.chars_labels_test:
            self.chars_test.append(tpl[0])
        #set labels_test
        for tpl in self.chars_labels_test:
            self.labels_test.append(tpl[1])

    def one_hot(self, characters): #arg = list of characters
        if not isinstance(characters, list):
            raise TypeError('Expected argument is a list (of characters).')
        idx = []
        for c in characters:
            if c in self.char2id:
                idx.append(self.char2id.index(c))
            else:
                print()
        return np.eye(len(self.char2id))[idx]

    def labels_2_numbers(self, labels):
        idx = []
        for l in labels:
            idx.append(self.label2id.index(l))
        return np.array(idx)

    def train_simple(self):
        self.logreg.fit(self.X_onehot, self.y)


t = Tokenizer()
t.read_in_train('e04-data/train')
t.read_in_test('e04-data/test')
t.train_simple()

print(t.label2id)
print(t.chars_test)

X_predict = t.one_hot(t.chars_test)
predicted_classes = t.logreg.predict(X_predict)

#for item in zip(predicted_classes, t.chars_labels_test):
#    print(item)

print(len(t.labels_test))
print(len(predicted_classes))
accuracy = metrics.accuracy_score(t.labels_test, predicted_classes)
print(accuracy)







