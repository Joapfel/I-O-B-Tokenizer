import numpy as np
import scipy
from keras.models import Sequential
from sklearn import linear_model
from sklearn import metrics
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack

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
                    chars_labels.append((' ', line[0])) # whitespace
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

    def context_window(self, m):
        # add padding to the training data
        padding = np.zeros((2, m.shape[1]))
        m_padding = np.vstack((padding, m))  # add two zero rows add the beginning
        m_padding = np.vstack((m_padding, padding))  # add two zero rows add the end

        m_window = csr_matrix(m_padding[:5].flatten())
        for i in range(1, len(m_padding) - 4):
            m_window = vstack((m_window, csr_matrix(m_padding[i:i + 5].flatten())))
        return m_window

    def tokenize(self, s, lc): #lc is the classifier
        if not isinstance(s, str):
            raise TypeError('input s needs to be a string')
        characters = list(s)
        characters_onehot = self.one_hot(characters)
        characters_context = self.context_window(characters_onehot).toarray()
        labels_num = lc.predict(characters_context)
        labels_ch = []
        for lbl in labels_num:
            labels_ch.append(self.label2id[lbl])
        sentences = []
        sent = []
        token = ""
        for c, l in zip(characters, labels_ch):
            if l == 'I':
                token += c
            elif l == 'T' or l == 'S':
                if len(token) > 0:
                    sent.append(token) #add token to the sentence
                token = c #reset token and add first char of the next token
                if l == 'S' and len(sent) > 0:
                    sentences.append(sent)
                    sent = []
        #at the end add the last token and sent to its higher structure
        if len(token) > 0:
            sent.append(token)
        if len(sent) > 0:
            sentences.append(sent)
        return sentences



if '__main__' == __name__:

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
    logreg.fit(csr_matrix(X_train), y_train)

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
    y_predicted = logreg.predict(csr_matrix(X_predict)) #the predicted classes

    print('##############################################')
    print('RESULTS WITHOUT CONTEXT WINDOW: ')

    accuracy = metrics.accuracy_score(y_correct, y_predicted)
    print('accuracy: ', accuracy)

    precision = metrics.precision_score(y_correct, y_predicted, average='macro')
    print('precision: ', precision)

    recall = metrics.recall_score(y_correct, y_predicted, average='macro')
    print('recall: ', recall)

    f1_score = metrics.f1_score(y_correct, y_predicted, average='macro')
    print('f1_score: ', f1_score)
    print('##############################################')
    print()


    """
    logistic classifier (maxent) that predicts the class of the character using onehot representation
    with +-2 window
    """
    ##########################################################################
    #TRAIN
    ##########################################################################

    logreg_2 = linear_model.LogisticRegression()
    logreg_2.fit(t.context_window(X_train), y_train)


    ##########################################################################
    #PREDICT
    ##########################################################################

    y_predicted_2 = logreg_2.predict(t.context_window(X_predict))

    print('##############################################')
    print('RESULTS WITH +-2 CONTEXT WINDOW: ')

    accuracy = metrics.accuracy_score(y_correct, y_predicted_2)
    print('accuracy: ', accuracy)

    precision = metrics.precision_score(y_correct, y_predicted_2, average='macro')
    print('precision: ', precision)

    recall = metrics.recall_score(y_correct, y_predicted_2, average='macro')
    print('recall: ', recall)

    f1_score = metrics.f1_score(y_correct, y_predicted_2, average='macro')
    print('f1_score: ', f1_score)
    print('##############################################')
    print()


    """
    confusion matrix for the simple classifier results vs the classifier with context window
    """
    print('confusion matrix for simple classifier')
    print(metrics.confusion_matrix(y_correct, y_predicted))
    print()
    print('confusion matrix for classifier with +-2 context window')
    print(metrics.confusion_matrix(y_correct, y_predicted_2))


    """
    use the trained model to tokenize some input
    """
    print()
    test1 = 'From the AP comes this story : President Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area.'
    tok = t.tokenize(test1, logreg_2)
    print(tok)

