# Created by Liqi Jiang
# BNB_sentiment

import sys
import csv
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


def preprocess(sentence):
    processed_sentence = []
    url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
    illegal_pattern = r'[^a-zA-Z0-9#@_$%\s]'
    for line in sentence:
        no_url_line = re.sub(url_pattern, ' ', line)
        legal_line = re.sub(illegal_pattern, '', no_url_line)
        processed_sentence.append(legal_line)
    return processed_sentence


# read file, generate X_train, y_train, X_test, test_id
# training.tsv
training_file = open(sys.argv[1], 'r')
csv_reader_train = csv.reader(training_file, delimiter='\t')
X_train = []
y_train = []
for line in csv_reader_train:
    # print(line)
    X_train.append(line[1])
    y_train.append(line[2])
# print(X_train)
# print(y_train)
# test.tsv
test_file = open(sys.argv[2], 'r')
csv_reader_test = csv.reader(test_file, delimiter='\t')
test_id = []
X_test = []
y_test = []
for line in csv_reader_test:
    test_id.append(line[0])
    X_test.append(line[1])
    y_test.append(line[2])

# data preprocessing
X_train = np.array(preprocess(X_train))
X_test = np.array(preprocess(X_test))
y_train = np.array(y_train)
y_test = np.array(y_test)


# create count vectorizer and fit it with training data
count = CountVectorizer(token_pattern='[a-zA-Z0-9#@_$%]{2,}', lowercase=False)
X_train_bag_of_words = count.fit_transform(X_train)

# transform the test data into bag of words creaed with fit_transform
X_test_bag_of_words = count.transform(X_test)

# print("----mnb")
clf = MultinomialNB()
model = clf.fit(X_train_bag_of_words, y_train)
predicted_y = model.predict(X_test_bag_of_words)
# predict_and_test(model, X_test_bag_of_words)
for i in range(len(test_id)):
    # print(test_id[i])
    print(str(test_id[i]), predicted_y[i])