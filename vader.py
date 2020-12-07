import sys
import csv
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


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


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# analyse with VADER
analyser = SentimentIntensityAnalyzer()
predicted_y = []
for text in X_test:
    score = analyser.polarity_scores(text)
    if score['compound'] >= 0.05:
        predicted_y.append('positive')
    elif score['compound'] <= -0.05:
        predicted_y.append('negative')
    else:
        predicted_y.append('neutral')


print(classification_report(y_test, predicted_y))