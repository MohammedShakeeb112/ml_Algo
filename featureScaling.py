from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = sns.load_dataset('titanic')
# print(df.head())

df2 = df[['survived','pclass','age','parch']]
# print(df2)

# print(df2.mean())
# sns.heatmap(df2.isnull(),cmap='viridis',yticklabels=False)
df3 = df2.fillna(df2.mean())
# print(df3)
# sns.heatmap(df3.isnull(),cmap='viridis', yticklabels=False)

X = df3.drop('survived', axis=1)
# print(X)

y = df3['survived']
# print(y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# Standard Scalar => mean=0, variance=1
sc = StandardScaler()
sc.fit(X_train)

#mean
# print(f'mean : {sc.mean_}')
#standard deviation
# print(f'standard deviation : {sc.scale_}')
# print(X_train.describe())

X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

# print(X_train_sc)

X_train_sc = pd.DataFrame(X_train_sc, columns=['pclass','age','parch'])
# print(X_train_sc)
X_test_sc = pd.DataFrame(X_test_sc, columns=['pclass','age','parch'])
# print(X_test_sc)
# print(X_train_sc.describe().round(2))

# MinMaxScaler => min=0, max=1
mms = MinMaxScaler()
mms.fit(X_train)
X_train_mms = mms.transform(X_train)
# print(X_train_mms)
X_test_mms = mms.transform(X_test)

X_train_mms = pd.DataFrame(X_train_mms, columns=['pclass','age','parch'])
# print(X_train_mms)
X_test_mms = pd.DataFrame(X_test_mms, columns=['pclass','age','parch'])
# print(X_test_mms)
# print(X_train_mms.describe().round(2))


#before and after scaling the feature data wont get changes  in the ditribution
sns.pairplot(X_train)
sns.pairplot(X_train_sc)
sns.pairplot(X_train_mms)
plt.show()

