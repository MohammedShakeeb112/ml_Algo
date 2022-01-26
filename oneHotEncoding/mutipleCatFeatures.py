from datetime import date
import pandas as pd
import numpy as np


#mercedes-bens dataset
data = pd.read_csv("/mnt/A4B278D5B278AD84/Python+ml Crash Course/pythonPracticeQuest/Algorithm/mlAglo/oneHotEncoding/train.csv",encoding='utf-8', usecols=['X1','X2','X3','X4','X5','X6'])
# print(data.shape) #shape=> (4209,6) => 6 feature in which each feature has multiple categorical value

# for col in data.columns:
#     print(col,len(data[col].unique()),'Unique labels')

# print(pd.get_dummies(data, drop_first=True).shape) #shape=> (4209,117) => 117 feature created using dummpymethod which curse of dimensionality
#to avoid cure of dimensionality=>  get topmost frequent of labels of the variable

# print(data.X2.value_counts().sort_values(ascending=False).head(10)) #most frequent 20 categorical variable repeated 
#get top 10 categorical feature variable form X2 column then do OneHotEncoding remaining as 0
# top_10 = ['X2_{}'.format(i) for i in data['X2'].value_counts().sort_values(ascending=False).head(10).index]
# print(top_10)

#variable categorical values in X2 feature into binary variable
# for label in top_10:
#     data[label] = np.where(data['X2']==label,1,0)  

# print(data[['X2']+top_10].head(20))

def top_x(data, variable):
    top_10 = [i for i in data[variable].value_counts().sort_values(ascending=False).head(10).index]
    # for i in data[variable].value_counts().sort_values(ascending=False).head(10).index:
    #     top_10.append(i)
    return top_10


def oneHotEncoding(df, variable, top_x):
    for label in top_x:
        df[variable+'_'+label] = np.where(df[variable]==label, 1,0)

    return df

data = pd.read_csv("/mnt/A4B278D5B278AD84/Python+ml Crash Course/pythonPracticeQuest/Algorithm/mlAglo/oneHotEncoding/train.csv", usecols=['X1','X2','X3','X4','X5','X6'])
# print(data)
# top_10 = [i for i in data['X1'].value_counts().sort_values(ascending=False).head(10).index]
# oneHotEncoding(data,'X1',top_10 )

# print(top_x(data,'X2'))
# print(oneHotEncoding(data,'X2',top_x(data,'X2')))


oneHotEncoding(data,'X1',top_x(data,'X1'))
oneHotEncoding(data,'X2',top_x(data,'X2'))
oneHotEncoding(data,'X3',top_x(data,'X3'))
oneHotEncoding(data,'X4',top_x(data,'X4'))
oneHotEncoding(data,'X5',top_x(data,'X5'))
oneHotEncoding(data,'X6',top_x(data,'X6'))

data.drop(['X1','X2','X3','X4','X5','X6'], axis=1, inplace=True)
print(data)
