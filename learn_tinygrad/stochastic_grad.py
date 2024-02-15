#!/usr/bin/env python3

# https://chat.openai.com/share/f7806681-bdc3-44d2-81fb-d7b7dbe77d6b
# https://arxiv.org/pdf/1709.07615.pdf
# Ethan Bleier

import numpy as np

X = np.array([1, 2, 3, 4, 5])  # Input features
y = np.array([2, 4, 6, 8, 10]) # Target values

learning_rate = 0.01
epochs = 1000
theta = np.random.randn(2) 

for epoch in range(epochs):
    permutation = np.random.permutation(len(X))
    X_shuffled = X[permutation]
    y_shuffled = y[permutation]
    
    for i in range(len(X)):
        y_pred = theta[0] * X_shuffled[i] + theta[1]
        
        grad_theta0 = -2 * X_shuffled[i] * (y_shuffled[i] - y_pred)
        grad_theta1 = -2 * (y_shuffled[i] - y_pred)
        
        theta[0] -= learning_rate * grad_theta0
        theta[1] -= learning_rate * grad_theta1

    mse = np.mean((y - (theta[0] * X + theta[1])) ** 2)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Mean Squared Error = {mse}')

print(f'Optimal Parameters: Slope = {theta[0]}, Intercept = {theta[1]}')
