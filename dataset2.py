# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:17:08 2017

@author: shen
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
import gensim
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import np_utils
import tensorflow as tf
#set up dataset folder 
dataset_folder = 'C:/ProgramData/Anaconda3/'

#data preprocessing
data = pd.read_csv(dataset_folder+'train.csv') # read data.csv
sentences_pre = []

for sentence in data['text']:
    sentences_pre.append(gensim.parsing.preprocess_string(sentence)) # delete <function strip_tags>, <function strip_punctuation>, <function strip_multiple_whitespaces>, <function strip_numeric>, <function remove_stopwords>, <function strip_short>, <function stem_text>
print("finish parsing sentences...")
dct = gensim.corpora.Dictionary(sentences_pre) # detect the list and build a vector with length of list dimensions
dct.filter_extremes(no_below=20, no_above=0.3) # no_below: word appearance under 20 docs, no_above : doc appearance above 30% of all corpus

# X
bow_len = len(dct.token2id.keys())  # dictionary.token2id : assign a unique integer id to all words appearing in the corpus with the gensim.corpora.dictionary.Dictionary
# bow_len means the number of words in the bag of words
def bow2vec(bow): # bow: (word id 0~2362, word appear times in doc)
    output = np.zeros(bow_len) # output is a row of zero(length bow)
    for word in bow:
        output[word[0]] = word[1] # change the specific position to times
    return output # output: nparray of ( 0,0,0,0,0, word appearance times,0,0,0)
#doing doc to bow : check if (word in doc) in dct or not. yes: give id , no: ignored
X = np.array([bow2vec(dct.doc2bow(sentence)) for sentence in sentences_pre])# sentence : list of words ; doc2bow : return (id,appear times) ; bow2vec:
# X = a np array contains every (word id , word appear times in doc) np arrays 

# Y

le = preprocessing.LabelEncoder()
le.fit(data['author']) # encode 3 author name to classes_: 0, 1, 2 
Y = le.transform(data['author']) # transform the data column 'author' with type of classes (all the data of that column)
Y = np_utils.to_categorical(Y) # turn Y into a one-hot type even if Y originally values from 0~2 (now is a vector with length 3)

print("finish bow and label encoding one-hot method...")

# train ANN

model_sk = MLPClassifier()
model_sk.fit(X,Y)

print("training by crossvalidation...")
scores = cross_val_score(model_sk, X, Y, cv=10) # return array of float ; Array of scores of the estimator for each run of the cross validation.
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    

'''
# preprocess
train.drop("author", 1, inplace=True)
target_vars = ["EAP", "HPL", "MWS"]
train.head(2)
sentences = train['text'] #use gensim to preprocess

#author = train['author'] # one hot  label

# training 

## fit x(data),y(answer) 
'''