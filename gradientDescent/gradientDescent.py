import numpy as np
import matplotlib.pyplot as plt

x = list(range(1,10))
x = np.array(x)
y = []
for i in x:
    y.append(62*i+35)
y = np.array(y)
print(x, y)

def gradient_descent(x, y):
    slope = intercept = 0
    learning_rate = 0.01 
    iteration = 10000
    n = len(x)
    plt.figure(figsize=(10,5))
    plt.scatter(x, y, c='r', marker='*', linewidths=3)
    plt.show()
    for i in range(iteration):
        y_pred = slope * x + intercept
        plt.figure(figsize=(10,5))
        plt.plot(x, y_pred, c='g')
        plt.show()
        cost_function = (1/n) * sum([val for val in (y - y_pred)])
        derivative_of_slope = (-2/n) * sum(x * (y - y_pred))
        derivative_of_intercept = (-2/n) * sum(y - y_pred)
        slope -= learning_rate * derivative_of_slope
        intercept -= learning_rate * derivative_of_intercept
        print(f'slope {slope}, intercept {intercept}, cost_function {cost_function}, iteration {i}')
    
gradient_descent(x, y)