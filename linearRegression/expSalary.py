import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

data = {
    'experience':[np.NaN, np.nan, 'five', 'two', 'seven', 'three', 'ten', 'eleven'],
    'test_score(out of 10)':[8, 8, 6, 10, 9, 7, np.nan, 7],
    'interview_score(out of 10)':[9, 6, 7, 10, 6, 10, 7, 8],
    'salary($)':[50000, 45000, 60000, 65000, 70000, 62000, 72000, 80000]
}
# print(data)

df = pd.DataFrame(data)
# print(df)
df['experience'] = df['experience'].fillna('zero')
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean()).astype(int)
df.replace({'zero':0,'two':2,'three':3,'five':5,'seven':7,'ten':10,'eleven':11}, inplace=True)
# print(df)

reg_model = LinearRegression()
reg_model.fit(df.drop('salary($)',axis=1), df['salary($)'])
print(reg_model.coef_)
print(reg_model.intercept_)

test = reg_model.predict([[2,9,6]]).astype(int)
print(test)
print(test[0]==(reg_model.coef_[0]*2+reg_model.coef_[1]*9+reg_model.coef_[2]*6+reg_model.intercept_).astype(int))

test=[[12,10,10]]
print(reg_model.predict(test).astype(int)[0]==(reg_model.coef_[0]*test[0][0]+reg_model.coef_[1]*test[0][1]+reg_model.coef_[2]*test[0][2]+reg_model.intercept_).astype(int))
print(reg_model.predict(test).astype(int))