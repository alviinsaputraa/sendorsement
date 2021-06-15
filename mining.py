#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install Sastrawi


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

import itertools
import pdb
import re
import string
from os import getcwd

nltk.download('stopwords')


# In[3]:


df1 = pd.read_json('D:\Alvin\Thesis\JSON\sprite_bintangemon_tweet.json', lines = True)
df2 = pd.read_json('D:\Alvin\Thesis\JSON\sprite_bintangemon_waterlymon.json', lines = True)
df3 = pd.read_json('D:\Alvin\Thesis\JSON\sprite_emon_tweet.json', lines = True)
df4 = pd.read_json('D:\Alvin\Thesis\JSON\sprite_waterlemon_tweet.json', lines = True)
df5 = pd.read_json('D:\Alvin\Thesis\JSON\sprite_waterlymon_hashtag.json', lines = True)
df6 = pd.read_json('D:\Alvin\Thesis\JSON\sprite_waterlymon_tweet.json', lines = True)
df7 = pd.read_json('D:\Alvin\Thesis\JSON\water_emon_tweet.json', lines = True)
df8 = pd.read_json('D:\Alvin\Thesis\JSON\wateremon_tweet.json', lines = True)
df9 = pd.read_json('D:\Alvin\Thesis\JSON\waterlymon_tweet.json', lines = True)


# In[4]:


df = df1.append(df2, ignore_index=True)
df = df.append(df3, ignore_index=True)
df = df.append(df4, ignore_index=True)
df = df.append(df5, ignore_index=True)
df = df.append(df6, ignore_index=True)
df = df.append(df7, ignore_index=True)
df = df.append(df8, ignore_index=True)
df = df.append(df9, ignore_index=True)


# In[5]:


df.head()


# # Cleansing

# In[6]:


df = df.drop_duplicates(subset=['id'])
df = df[df['language']=='in']
df.shape


# In[7]:


df


# In[8]:


print("sprite bintangemon tweet = ", df1.shape)
print("bintangemon waterlymon = ", df2.shape)
print("sprite emon = ", df3.shape)
print("sprite waterlemon = ", df4.shape)
print("#SpriteWaterlymon = ", df5.shape)
print("sprite waterlymon = ", df6.shape)
print("water emon = ", df7.shape)
print("wateremon = ", df8.shape)
print("waterlymon = ", df9.shape)


# In[9]:


df = df[df['date']>'2020-01-01']


# In[10]:


df = df.drop(['conversation_id', 'created_at', 'time', 'timezone',
       'user_id', 'username', 'name', 'place', 'language', 'mentions',
       'urls', 'photos', 'replies_count', 'retweets_count', 'likes_count',
       'hashtags', 'cashtags', 'link', 'retweet', 'quote_url', 'video',
       'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
       'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src',
       'trans_dest'], axis=1)


# In[11]:


#df.to_csv('dataClean.csv')


# In[12]:


df


# # Labelled Data

# In[13]:


dfLabel = pd.read_excel('labelling.xlsx')
dfLabel.head()


# In[14]:


dfLabel['label'].value_counts()


# In[15]:


y_value_counts = dfLabel['label'].value_counts()

print("Negative tweets  = ", y_value_counts[2], "with percentage ", (y_value_counts[2]*100)/(y_value_counts[0]+y_value_counts[1]+y_value_counts[2]),'%')
print("Positive tweets  = ", y_value_counts[1], "with percentage ", (y_value_counts[1]*100)/(y_value_counts[0]+y_value_counts[1]+y_value_counts[2]),'%')
print("Netral tweets  = ", y_value_counts[0], "with percentage ", (y_value_counts[0]*100)/(y_value_counts[0]+y_value_counts[1]+y_value_counts[2]),'%')


# In[16]:


sns.countplot(x = 'label', data = dfLabel)


# In[17]:


temp = dfLabel.groupby('label').count()['tweet'].reset_index().sort_values(by='tweet',ascending=False)
temp.style.background_gradient(cmap='Blues')


# In[18]:


from plotly import graph_objs as go

fig = go.Figure(go.Funnelarea(
    text = temp.label,
    values = temp.tweet,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()


# # Tokenization, Stopwords removal, Stemming

# # Pre-processing

# In[19]:


import fileinput
import csv
from pandas.plotting import scatter_matrix
from pandas import DataFrame
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


# In[20]:


# #load dataset
url = "labelling.xlsx" #input path where you save the csv file containing the dataset along with the file name .csv
names = ['tweet']
dataset = pd.read_excel(url)
df = pd.DataFrame()
#print(dataset['review'].head(5))


def preprocess_text(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +',' ', text)
    return text.strip()
dataset['tweet'] = [preprocess_text(t) for t in dataset['tweet']]

#case folding - lower case
dataset['tweet'] = dataset['tweet'].apply(lambda x: x.lower())
dataset['tweet'] = dataset['tweet'].str.replace('[^\w\s]','')

#print(dataset['review'].head(5))

StemFactory = StemmerFactory()
stemmer = StemFactory.create_stemmer()
StopFactory = StopWordRemoverFactory()
stopwords = StopFactory.get_stop_words()
stopwords.remove('ok')
# print(stopwords)

dataset['tweet'] = dataset['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
#print(dataset['review'].head(25))

dataset['tweet'] = dataset['tweet'].apply(lambda x: " ".join(stemmer.stem(x) for x in x.split()))
print(dataset['tweet'].head(25))

freq = pd.Series(' '.join(dataset['tweet']).split()).value_counts()[:10]
print(freq)

# ====================
df = pd.DataFrame(dataset['tweet'])#, dataset['class'])

#df.to_csv('result.csv', mode="a", header=False) #input path and file name where you WANT to save the stemming results in the csv file


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

X_train, X_test, y_train, y_test = train_test_split(df['tweet'], dfLabel['label'], test_size=0.2, random_state=207)

#rs NB = 165 / 149 / 207
#rs KNN = 307

print("Data training:")
print(len(X_train))
print(Counter(y_train))
print("=================================\n")

print("Data testing:")
print(len(X_test))
print(Counter(y_test))
print("=================================\n")


# In[85]:


#spot check algorithms
print("Spot Check Algorithms")
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('NB',MultinomialNB))
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
print('Accuration Score = ',pred_acc)
#print(confusion_matrix(y_test, y_predictions))
print(classification_report(y_test, y_predictions))
print("=================================\n")
plt.subplots(figsize=(10,10))
sns.heatmap(confusion_matrix(y_test, y_predictions), annot=True, cmap = plt.cm.Blues)


# In[86]:


#AUC-ROC
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


print("KNeighborsClassifier")
knn = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('KNN', KNeighborsClassifier() )])

knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("=================================\n")


# In[28]:


print("Data Predict:")
print(Counter(predictions))
print("Data Real:")
print(Counter(y_test))
print("=================================\n")

