# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import spacy


filepath = 'E:/research/machine learning/Kaggle/Quora Duplicate Questions/'

df_train = pd.read_csv(filepath + 'train.csv')
df_test = pd.read_csv(filepath + 'test.csv')

df_train.question1.fillna('', inplace=True)
df_train.question2.fillna('', inplace=True)

df_test.question1.fillna('', inplace=True)
df_test.question2.fillna('', inplace=True)

#Inspect data
print('Total number of question pairs for training: {}'.format(len(df_train)))
print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))
qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
print('Total number of questions in the training data: {}'.format(len(
    np.unique(qids))))
print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

plt.figure(figsize=(12, 5))
plt.hist(qids.value_counts(), bins=50)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
print()


#Feature generator functions
nlp = spacy.load('en')

def f_jacc_sim_lemma(q1, q2):
    s_lemma1, s_lemma2 = set([tok.lemma_ for tok in nlp(q1)]), set([tok.lemma_ for tok in nlp(q2)])
    return len(s_lemma1.intersection(s_lemma2))/len(s_lemma1.union(s_lemma2))

from difflib import SequenceMatcher

def f_seq_match_ratio(q1, q2):
    return SequenceMatcher(None, q1, q2).ratio()

def gen_feat_set(df):
    if True: #make it False to override this feature
        print('Generating feature: jacc_sim_lemma...')
        df['f_jacc_sim_lemma'] = df.apply(lambda r: f_jacc_sim_lemma(r.question1, r.question2), axis=1)

    if True: #make it False to override this feature
        print('Generating feature: seq_match_ratio...')
        df['f_seq_match_ratio'] = df.apply(lambda r: f_seq_match_ratio(r.question1.lower(), r.question2.lower()), axis=1)

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    
    return df

#Generate feature sets for the training and test data sets and save feature-expanded data sets 

from time import clock

start_time = clock()
gen_feat_set(df_train)
print('Elapsed time in sec: %f' % (clock()-start_time))

df_train.to_pickle(filepath + 'df_train.pkl')

start_time = clock()
gen_feat_set(df_test)
print('Elapsed time in sec: %f' % (clock()-start_time))

df_test.to_pickle(filepath + 'df_test.pkl')

#Visualize feature discriminatory power
def plot_feat_disc_chart(df, feat):
    plt.figure(figsize=(12, 4))
    plt.hist(df[feat][df['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
    plt.hist(df[feat][df['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
    plt.legend()
    plt.title(feat, fontsize=12)
    return


for feat in df_train.columns:
    if feat[:2]=='f_':
        plot_feat_disc_chart(df_train, feat)



#Generate X and y sets for training and validation  -could also normalize X via Scaler

#from sklearn.preprocessing import StandardScaler

X = np.array(df_train[[c for c in df_train.columns if c[:2]=='f_']])
y = np.array(df_train[['is_duplicate']]).ravel()

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1969)


#Run classifier 

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import log_loss

models = []
models.append(('Logistic Regression',   LogisticRegression()))
#models.append(('Nearest Neighbors',    KNeighborsClassifier(3)))
#models.append(('Linear SVM',    SVC(kernel="linear", C=0.025)))
#models.append(('RBF SVM',    SVC(gamma=2, C=1)))
#models.append(('Gaussian Process',    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)))
#models.append(('Decision Tree',    DecisionTreeClassifier(max_depth=5)))
models.append(('Random Forest',         RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))
#models.append(('Neural Net',    MLPClassifier(alpha=1)))
#models.append(('AdaBoost',    AdaBoostClassifier()))
#models.append(('Naive Bayes',    GaussianNB()))
#models.append(('QDA',    QuadraticDiscriminantAnalysis()))


results = []

for name, clf in models:
    clf.fit(X_train, y_train)
    score_t = log_loss(y_train, clf.predict_proba(X_train)[:,1])
    score_v = log_loss(y_val, clf.predict_proba(X_val)[:,1])
    results.append((name, score_t, score_v))
    print('%s:\t%f %f' % (name, score_t, score_v))


#Run baseline classifier

name = 'Baseline'
p = y_train.mean() # Our predicted probability
score_t = log_loss(y_train, np.zeros_like(y_train) + p)
score_v = log_loss(y_val, np.zeros_like(y_val) + p)
results.append((name, score_t, score_v))
print('%s:\t%f %f' % (name, score_t, score_v))    


#Generate submission

y_hat = clf.predict_proba(np.array(df_test[[c for c in df_test.columns if c[:2]=='f_']]))[:,1]
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = y_hat
sub.to_csv(filepath + 'qdq_0.csv', index=False)
sub.head()





