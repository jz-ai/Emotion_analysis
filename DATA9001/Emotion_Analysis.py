# Load libraries
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,classification_report
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from nltk.stem import PorterStemmer
import re

""" instances is a list of tweets """
def stem_instances(instances):
    stemmer = PorterStemmer()
    #Porter词干提取
    stemmed_instances = []
    for inst in instances:
        words = re.split('\W+', inst)
        for word in words:
            stemmed_word = stemmer.stem(word)
            inst = inst.replace(word, stemmed_word)
            # print(word, stemmed_word)
        stemmed_instances.append(inst)
    return stemmed_instances

def predict_and_test(model, X_test_bag_of_words):
    num_dec_point = 3
    predicted_y = model.predict(X_test_bag_of_words)
    # print(y_test, predicted_y)
    # print(model.predict_proba(X_test_bag_of_words))
    a_mic = accuracy_score(y_test, predicted_y)
    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(y_test,
                        predicted_y,
                        average='micro',
                        warn_for=())
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_test,
                        predicted_y,
                        average='macro',
                        warn_for=())
    print('micro acc,prec,rec,f1: ',round(a_mic,num_dec_point), round(p_mic,num_dec_point), round(r_mic,num_dec_point), round(f1_mic,num_dec_point),sep="\t")
    print('macro prec,rec,f1: ',round(p_mac,num_dec_point), round(r_mac,num_dec_point), round(f1_mac,num_dec_point),sep="\t")
    print(classification_report(y_test, predicted_y,zero_division=0))

# Read data
data = pd.read_csv("assign3_tweets.tsv", sep='\t')
# stemmed_data = stem_instances(data["text"])
stemmed_data=data["text"]
text_data = np.array(stemmed_data)
X = text_data

# Create target vector
y = data["sentiment"]

# split into train and test
X_train = X[:4000]
X_test = X[4000:]
y_train = y[:4000]
y_test = y[4000:]

# create count vectorizer and fit it with training data
count = CountVectorizer(lowercase=False, stop_words='english',
                                     token_pattern='[a-zA-Z0-9@]{2,}') # this will keep @ character
X_train_bag_of_words = count.fit_transform(X_train)

# transform the test data into bag of words creaed with fit_transform
X_test_bag_of_words = count.transform(X_test)

# print("----NB")
# clf = BernoulliNB()
# model = clf.fit(X_train_bag_of_words, y_train)
# predict_and_test(model, X_test_bag_of_words)

# print("----KNN")
# for k in range(1,11):
#     print("This is k:",k)
#     clf = KNeighborsClassifier(n_neighbors=k)
#     model = clf.fit(X_train_bag_of_words, y_train)
#     predict_and_test(model, X_test_bag_of_words)

# if random_state id not set. the feaures are randomised, therefore tree may be different each time
print("----DT")
for tree_deep in range(5,16):
     print("This is tree_deepth",tree_deep)
     clf = tree.DecisionTreeClassifier(max_depth=tree_deep,criterion='entropy',random_state=0)
     model = clf.fit(X_train_bag_of_words, y_train)
     predict_and_test(model, X_test_bag_of_words)




