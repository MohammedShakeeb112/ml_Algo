import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
df = {
    'area': [2600, 3000, 3200, 3600, 4000],
    'price': [550000, 565000, 610000, 680000, 725000]
}

df = pd.DataFrame(df)
# print(df)
plt.figure(figsize=(10,6))
plt.scatter(df['area'], df['price'], marker='+', c='r')
plt.xlabel('Area in sq ft')
plt.ylabel('Price in Rupees')

reg_model = LinearRegression()
reg_model.fit(df[['area']], df['price'])

#y = mx + b => y-->predicting price, m-->slope of the line, x-->area, b-->intercept on y axis  
# print(reg_model.coef_) #slope
# print(reg_model.intercept_) #intercept
# y = reg_model.coef_*4000 + reg_model.intercept_
# print(y)
test = {
    'area': [2600, 3000, 3200, 3600, 4000]
}
area={
    'area': [1000,1500,2300,3540,4120,4560,5490,3460,4750,2300,9000,8600,7100]
}
test = pd.DataFrame(test)
ytest_pred = reg_model.predict(test)
print(ytest_pred)
area = pd.DataFrame(area)
y_pred = reg_model.predict(area).astype(int)
print(y_pred)

plt.figure(figsize=(10,5))
plt.scatter(test['area'],ytest_pred, c='b', edgecolors='k')
plt.plot()
mae = mean_absolute_error(df['price'],ytest_pred)
print(mae)
mse = mean_squared_error(df['price'], ytest_pred)
print(mse)
rmse = np.sqrt(mean_squared_error(df['price'],ytest_pred))
print(rmse)


plt.figure(figsize=(10,5))
plt.scatter(df['area'], df['price'], c='r')
plt.plot(df['area'], ytest_pred, c='b')
plt.show()