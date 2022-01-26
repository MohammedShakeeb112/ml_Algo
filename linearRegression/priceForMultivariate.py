import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

data = {
    'area': [2600, 3000, 3200, 3600, 4000, 4100],
    'bedrooms': [3, 4, np.NaN, 3, 5, 6],
    'age': [20, 15, 18, 30, 8, 8],
    'price': [550000, 565000, 610000, 595000, 760000, 810000]	
}

df = pd.DataFrame(data)
#handling missing values with mean
# df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].mean())
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
# print(df)
X = df.drop('price', axis=1)
# print(X)
y = df['price']
# print(y)
# print(df['bedrooms'].median()) #midpoint/2
# print(df['bedrooms'].mode()) #maximum repeating number
# price = m1*area + m2*bedrooms + m3*age + b
reg_model = LinearRegression()

sc = MinMaxScaler()
sc.fit_transform(X)
reg_model.fit(X, y)
# print(reg_model.coef_)
# print(reg_model.intercept_)
third_index = reg_model.coef_[0]*X['area'][3]+reg_model.coef_[1]*X['bedrooms'][3]+reg_model.coef_[2]*X['age'][3]+reg_model.intercept_
# print(third_index)

test = {
    'area':[3600, 4000, 4100],
    'bedrooms':[3, 5, 6],
    'age':[30, 8, 8]
}
test = pd.DataFrame(test)
sc.transform(test)
# print(test)
pred = reg_model.predict(test).astype(int)
# print(pred)
testpred = reg_model.predict([[3600,3,30]])
print(testpred)