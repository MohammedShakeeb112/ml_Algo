import pandas as pd
from sklearn import metrics

df = pd.read_csv('/mnt/A4B278D5B278AD84/Python+ml Crash Course/pythonPracticeQuest/Algorithm/mlAglo/naiveBayesClassifier/spam.csv')
df.dropna(axis=0,inplace=True)
# print(df)
# print(df['Category'].value_counts())
# print(df.isna().any()) #0 missing values
# print(df.groupby('Category').describe())

# df.replace({'ham':0,'spam':1},inplace=True)
# print(df)
# df['spam'] = list(map(lambda x: 0 if x!='spam' else 1, df['Category']))
# df['spam'] = df['Category'].apply(lambda x: 0 if x!='spam' else 1)
df['spam'] = df.loc[:,'Category'].map(lambda x: 0 if x!='spam' else 1)
# print(df)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.3, random_state=42)
# print(X_train[:10])
# print(X_train.values[:10])
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
# print(vectorizer)
X_train_count = vectorizer.fit_transform(X_train.values)
# print(X_train_count)
# print(X_train_count.toarray())

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train_count,y_train)


emails = [
    'hey mohan, can we go to paris, because i got a ticket',
    'free upto 60% off on lewis buy now, exclusive offer just for you',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!',
    'exclusive offer just for you to win a chance to visit paris'

]

emails_count = vectorizer.transform(emails)
# print(emails_count)
pred = mnb.predict(emails_count)
# print(pred)

X_test_count = vectorizer.transform(X_test)
# print(mnb.score(X_test_count,y_test))

# PIPELINE
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train,y_train)
# print(clf.fit(X_train,y_train))
y_pred = clf.predict(X_test)
# print(y_pred)
# print(clf.score(X_test,y_test))
# print(metrics.r2_score(y_test,y_pred))
# print(clf.predict(emails))