# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import math 

# def gradient_descent(x, y):
#     n = len(x)
#     iteration = 1000
#     slope = intercept = 0
#     learning_rate = 0.0001
#     cost_prev = 0
#     for i in range(iteration):
#         y_pred = slope * x + intercept
#         derivative_of_slope = (-2/n) * sum(x * (y - y_pred))
#         derivative_of_intercept = (-2/n) * sum(y - y_pred)
#         slope -= learning_rate * derivative_of_slope
#         intercept -= learning_rate * derivative_of_intercept
#         cost_function = (1/n) * sum([val for val in (y - y_pred)])
#         if math.isclose(cost_function, cost_prev, rel_tol=1e-20):
#             break
#         cost_prev = cost_function
#         # print(f'slope {slope}, intercept {intercept}, cost_function {cost_function}, iteration {i}')
#     return slope, intercept


# def predict_by_sklearn(df):
#     linear_reg = LinearRegression()
#     linear_reg.fit(df[['math']], df['cs'])
#     return linear_reg.coef_[0], linear_reg.intercept_


# if __name__ == '__main__':
#     df = pd.read_csv('test.csv')
#     df.dropna(axis=0, inplace=True)
#     x = df['math'].values.astype(int)
#     y = df['cs'].values.astype(int)
#     # print(x)
#     # print(y)
#     m_gd, b_gd = gradient_descent(x,y)
#     print(f'slope {m_gd}, intercept {b_gd}')
#     m_sklearn, b_sklearn = predict_by_sklearn(df)
#     print(f'slope {m_sklearn}, intercept {b_sklearn}')


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

def predict_using_sklean():
    df = pd.read_csv("test.csv")
    df.dropna(axis=0,inplace=True)
    r = LinearRegression()
    r.fit(df[['math']], df['cs'])
    return r.coef_, r.intercept_

def gradient_descent(x,y):
    m_curr = 0
    b_curr = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002

    cost_previous = 0
    
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n)*sum([value**2 for value in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print(f'cost {cost}, cost_prev {cost_previous}')
        if math.isclose(cost, cost_previous, rel_tol=1e-1):
            break
        cost_previous = cost
        # print ("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost, i))

    return m_curr, b_curr

if __name__ == "__main__":
    df = pd.read_csv("test.csv")
    df.dropna(axis=0,inplace=True)
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x,y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = predict_using_sklean()
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))