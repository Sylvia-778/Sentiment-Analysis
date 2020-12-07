import sys
import csv
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

'''
def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    words = sentence.split(' ')
    filtered_words = [w for w in words if w not in stop_words]
    remove_stop_sentence = ' '.join(w for w in filtered_words)
    return remove_stop_sentence


def stemming_words(sentence):
    ps = PorterStemmer()
    words = sentence.split(' ')
    stem_words = [ps.stem(w) for w in words]
    stemmed_sentence = ' '.join(w for w in stem_words)
    return stemmed_sentence
'''

def preprocess(sentence):
    processed_sentence = []
    url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
    illegal_pattern = r'[^a-zA-Z0-9#@_$%\s]'
    for line in sentence:
        no_url_line = re.sub(url_pattern, ' ', line)
        legal_line = re.sub(illegal_pattern, '', no_url_line)
        # removed_line = remove_stopwords(legal_line)
        # stemmed_line = stemming_words(removed_line)
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


# create count vectorizer and fit it with training data
count = CountVectorizer(token_pattern='[a-zA-Z0-9#@_$%]{2,}', max_features=1000, lowercase=True)
X_train_bag_of_words = count.fit_transform(X_train)

# transform the test data into bag of words creaed with fit_transform
X_test_bag_of_words = count.transform(X_test)

# if random_state id not set. the feaures are randomised, therefore tree may be different each time
# print("----dt")
clf = tree.DecisionTreeClassifier(min_samples_leaf=int(0.01*len(X_train)), criterion='entropy', random_state=0)
model = clf.fit(X_train_bag_of_words, y_train)
predicted_y = model.predict(X_test_bag_of_words)
print(classification_report(y_test, predicted_y, zero_division=0))