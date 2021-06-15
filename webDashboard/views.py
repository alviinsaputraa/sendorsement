import json

from django.http import HttpResponse
from django.shortcuts import render

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

import pandas as pd
import itertools
import pdb
import re
import string
from os import getcwd

import fileinput
import csv
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from collections import Counter


#method view
def index(request):
    dfLabel = pd.read_excel('labelling.xls')

    # # Tokenization, Stopwords removal, Stemming

    # # Pre-processing

    # In[19]:

    # In[20]:

    # #load dataset
    url = "labelling.xls"  # input path where you save the csv file containing the dataset along with the file name .csv
    names = ['tweet']
    dataset = pd.read_excel(url)
    df = pd.DataFrame()

    # print(dataset['review'].head(5))

    def preprocess_text(text):
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
        text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
        text = re.sub(' +', ' ', text)
        return text.strip()

    dataset['tweet'] = [preprocess_text(t) for t in dataset['tweet']]

    # case folding - lower case
    dataset['tweet'] = dataset['tweet'].apply(lambda x: x.lower())
    dataset['tweet'] = dataset['tweet'].str.replace('[^\w\s]', '')

    # print(dataset['review'].head(5))

    StemFactory = StemmerFactory()
    stemmer = StemFactory.create_stemmer()
    StopFactory = StopWordRemoverFactory()
    stopwords = StopFactory.get_stop_words()
    stopwords.remove('ok')
    # print(stopwords)

    dataset['tweet'] = dataset['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
    # print(dataset['review'].head(25))

    dataset['tweet'] = dataset['tweet'].apply(lambda x: " ".join(stemmer.stem(x) for x in x.split()))
    print(dataset['tweet'].head(25))

    freq = pd.Series(' '.join(dataset['tweet']).split()).value_counts()[:10]
    print(freq)

    # ====================
    df = pd.DataFrame(dataset['tweet'])  # , dataset['class'])

    # df.to_csv('result.csv', mode="a", header=False) #input path and file name where you WANT to save the stemming results in the csv file

    # In[21]:

    dfLabel[:14]

    # # Modelling

    # In[22]:

    from collections import Counter
    import csv

    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.naive_bayes import BernoulliNB

    # In[84]:

    # random urutan dan split ke data training dan test
    from sklearn.model_selection import train_test_split

    dfLabel['label'] = dfLabel['label'].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(df['tweet'], dfLabel['label'], test_size=0.1, random_state=207)

    # rs NB = 165 / 149 / 207
    # rs KNN = 307

    print("Data training:")
    print(len(X_train))
    print(Counter(y_train))
    print("=================================\n")

    print("Data testing:")
    print(len(X_test))
    print(Counter(y_test))
    print("=================================\n")

    # In[85]:

    # spot check algorithms
    print("Spot Check Algorithms")
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('NB', MultinomialNB))
    models.append(('SVM', SVC(gamma='auto')))
    print("=================================\n")

    # In[67]:

    # transform ke tfidf dan train dengan naive bayes
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.metrics import roc_curve
    from sklearn.metrics import plot_confusion_matrix
    import matplotlib.pyplot as plt

    print("Make Predictions on Validation Dataset\n\n")

    print("Naive Bayes")
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('NB', MultinomialNB())])

    nb.fit(X_train, y_train)
    nb.score(X_train, y_train)
    y_predictions = nb.predict(X_test)
    pred_acc = accuracy_score(y_test, y_predictions)
    print('Accuration Score = ', pred_acc)
    # print(confusion_matrix(y_test, y_predictions))
#    print(classification_report(y_test, y_predictions))
#    print("=================================\n")
#    plt.subplots(figsize=(10, 10))
#    sns.heatmap(confusion_matrix(y_test, y_predictions), annot=True, cmap=plt.cm.Blues)

    # In[86]:

    # AUC-ROC
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from matplotlib import pyplot
    from sklearn.multiclass import OneVsRestClassifier

    y_pred_proba = nb.predict_proba(X_test)
    pred_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

    # In[26]:

    print("Data Predict:")
    print(Counter(y_predictions))
    print("Data Real:")
    print(Counter(y_test))
    print("=================================\n")

    # In[27]:

#    print("KNeighborsClassifier")
#    knn = Pipeline([('vect', CountVectorizer()),
#                    ('tfidf', TfidfTransformer()),
#                    ('KNN', KNeighborsClassifier())])
#
#    knn.fit(X_train, y_train)
#    predictions = knn.predict(X_test)
#    print(accuracy_score(y_test, predictions))
#    print(confusion_matrix(y_test, predictions))
#    print(classification_report(y_test, predictions))
#    print("=================================\n")

    # In[28]:

#    print("Data Predict:")
#    print(Counter(predictions))
#    print("Data Real:")
#    print(Counter(y_test))
#    print("=================================\n")

    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('NB', MultinomialNB())])

    nb.fit(X_train, y_train)
    nb.score(X_train, y_train)
    predictions = nb.predict(df['tweet'])

    dfAll = pd.DataFrame()
    dfAll['tweet'] = df['tweet']
    dfAll['label'] = dfLabel['label']
    dfAll['predict'] = predictions
    dfAll['date'] = dfLabel['date']

    colTweet = dfAll['tweet'].values.tolist()
    colDate = dfAll['date'].values.tolist()
    colLabel = dfAll['label'].values.tolist()
    colPredict = dfAll['predict'].values.tolist()

    dfgroup = dfAll.groupby(dfAll.date.dt.month_name())['predict'].agg('count')
    dfGroup = pd.DataFrame()
    dfGroup['month'] = dfgroup.index
    dfGroup['count'] = dfgroup.values



    dfpos = dfAll[dfAll['predict'] == 'positive']
    dfpos = dfpos.groupby(dfpos.date.dt.month_name())['predict'].agg('count')
    dfPos = pd.DataFrame()
    dfPos['month'] = dfpos.index
    dfPos['count'] = dfpos.values

    dfneg = dfAll[dfAll['predict'] == 'negative']
    dfneg = dfneg.groupby(dfneg.date.dt.month_name())['predict'].agg('count')
    dfNeg = pd.DataFrame()
    dfNeg['month'] = dfneg.index
    dfNeg['count'] = dfneg.values

    dfnet = dfAll[dfAll['predict'] == 'netral']
    dfnet = dfnet.groupby(dfnet.date.dt.month_name())['predict'].agg('count')
    dfNet = pd.DataFrame()
    dfNet['month'] = dfnet.index
    dfNet['count'] = dfnet.values


    dfGroup['countPos'] = dfPos['count']
    dfGroup['countNet'] = dfNet['count']
    dfGroup.loc[dfGroup.month == 'November', 'countNeg'] = dfneg.values
    dfGroup['countNeg'] = dfGroup['countNeg'].fillna(0)
    dfGroup['month'] = pd.to_datetime(dfGroup['month'], format='%B')
    dfGroup = dfGroup.sort_values(by='month')

    import calendar

    dfGroup['month'] = dfGroup['month'].dt.month
    dfGroup['month'] = dfGroup['month'].apply(lambda x: calendar.month_name[x])


    dfPosCount = dfGroup['countPos'].values.tolist()
    dfNetCount = dfGroup['countNet'].values.tolist()
    dfNegCount = dfGroup['countNeg'].values.tolist()
    dfGroupMonth = dfGroup['month'].values.tolist()
    dfGroupCount = dfGroup['count'].values.tolist()

    dfgroupPie = dfAll.groupby(dfAll.predict)['predict'].agg('count')
    dfGroupPie = pd.DataFrame()
    dfGroupPie['class'] = dfgroupPie.index
    dfGroupPie['count'] = dfgroupPie.values

    dfPiePos = dfGroupPie['count'][2]
    dfPieNeg = dfGroupPie['count'][0]
    dfPieNet = dfGroupPie['count'][1]

    return render(request, 'index.html', {
                                            'colTweet' : json.dumps(colTweet),
                                            'colLabel' : json.dumps(colLabel),
                                            'colDate' : json.dumps(colDate),
                                            'colPredic' : json.dumps(colPredict),
                                            'dfNegCount': json.dumps(dfNegCount),
                                            'dfPosCount': json.dumps(dfPosCount),
                                            'dfNetCount': json.dumps(dfNetCount),
                                            'dfGroupMonth': json.dumps(dfGroupMonth),
                                            'dfGroupCount': json.dumps(dfGroupCount),
                                            'dfPiePos' : dfPiePos,
                                            'dfPieNeg' : dfPieNeg,
                                            'dfPieNet' : dfPieNet,
                                            'acc' : pred_acc,
                                            'auc' : pred_auc })

