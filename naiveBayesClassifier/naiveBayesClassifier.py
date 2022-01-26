# Naive_Bayes => GaussianNB,MultinomialNB,BernoulliNB
# GaussianNB used for normal distribution
# MultinomialNB used for discrete counts for eg: text classification problem instead of word occuring it count the words occured in the document
# BernoulliNB used for  binary classification problem using stop words, back of words
# Naive_Bayes is used for spam detection, hand written /character recognition, weather forecast, face detection, news article categorization


import math
import pandas as pd
from seaborn.distributions import kdeplot

from sklearn import datasets
from sklearn import metrics

#wine datasets
wine = datasets.load_wine()
# print(wine.keys())
# print(wine.data)
# print(wine.feature_names)
# print(wine.target_names)

X = pd.DataFrame(wine.data,columns=wine.feature_names)
# print(X)
y = wine.target
# print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

nb = GaussianNB()
nb.fit(X_train,y_train)

y_pred = nb.predict(X_test)
# print(y_pred)


ac = accuracy_score(y_test,y_pred)
# print(ac)
cm = confusion_matrix(y_test,y_pred)
# print(cm)

#diabetes datasets
diabetes = pd.read_csv('/mnt/A4B278D5B278AD84/Python+ml Crash Course/pythonPracticeQuest/Algorithm/mlAglo/naiveBayesClassifier/diabetes.csv')
# print(list(diabetes))

X = diabetes.drop('Outcome',axis=1)
# print(X)

y = diabetes['Outcome']
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)


nbc = GaussianNB()
nbc.fit(X_train, y_train)

diabetes_pred = nbc.predict(X_test)
# print(diabetes_pred)

diabetes_ac = accuracy_score(y_test, diabetes_pred)
# print(diabetes_ac)

diabetes_cf = confusion_matrix(y_test, diabetes_pred)
# print(diabetes_cf)

diabetes_cr = classification_report(y_test, diabetes_pred)
# print(diabetes_cr)


titanic_train = pd.read_csv('/mnt/A4B278D5B278AD84/Python+ml Crash Course/pythonPracticeQuest/Algorithm/mlAglo/naiveBayesClassifier/titanic/train.csv')
# print(titanic_train.info())
# print(titanic_train)

import seaborn as sns
import matplotlib.pyplot as plt 

# sns.heatmap(titanic_train.isnull())
# plt.show()

titanic_train.drop(['PassengerId', 'Name', 'SibSp','Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# print(titanic_train)
X = titanic_train.drop('Survived', axis=1)
# print(X)
y = titanic_train['Survived']
# print(y)

# one hot encoding for sex columns
sex = titanic_train[['Sex']]
# print(sex)
sex = pd.get_dummies(sex)
# print(sex)

X = pd.concat([X,sex], axis=1)
X.drop('Sex', axis=1, inplace=True)
# print(X)

# print(X.isna().any())
# print(X.columns[X.isna().any()]) #return na columns

#handle nan value with mean
# print(X['Age'][:20])
X['Age'] = X['Age'].fillna(int(X['Age'].mean()))
# print(X['Age'][:20])
# print(X.isna().any())
# print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

nb_titanic = GaussianNB()
nb_titanic.fit(X_train, y_train)

train_score = nb_titanic.score(X_train,y_train)
# print(train_score)
test_score = nb_titanic.score(X_test,y_test)
# print(test_score)

titanic_pred = nb_titanic.predict(X_test)
# print(titanic_pred)
metrics_r2_score = metrics.r2_score(y_test,titanic_pred)
# print(metrics_r2_score)
titanic_pred_prob = nb_titanic.predict_proba(X_test)
# print(titanic_pred[:10])
# print(titanic_pred_prob[:10]) #says the person survived

titanic_ac = accuracy_score(y_test, titanic_pred)
titanic_cm = confusion_matrix(y_test, titanic_pred)
titanic_cr = classification_report(y_test, titanic_pred)
# print(titanic_ac)
# print(titanic_cm)
# print(titanic_cr)

test_data = pd.read_csv('/mnt/A4B278D5B278AD84/Python+ml Crash Course/pythonPracticeQuest/Algorithm/mlAglo/naiveBayesClassifier/titanic/test.csv')
# print(test_data.columns)
test_data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis=1, inplace=True)

# print(test_data.isna().any())
test_data['Age'] = test_data['Age'].fillna(test_data.Age.mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data.Fare.mean())
# print(test_data.isna().any())



#------NOT CORRECT----------
# import pickle

# file = open('titanic.pkl','wb')
# pickle.dump(nb_titanic, file)


# model = open('titanic.pkl','rb')
# nbc_titanic = pickle.load(model)

# test_data_pred = nbc_titanic.predict(test_data)
# print(test_data_pred)