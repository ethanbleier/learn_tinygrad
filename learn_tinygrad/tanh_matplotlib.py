#!/usr/bin/env python3

# https://www.askpython.com/python/examples/tanh-activation-function 

from math import exp
import matplotlib.pyplot as plt 
 
#defining the sigmoid function
def sigmoid(x):
    return 1/(1+exp(-x))
 
#defining the tanh function using the relation
def tanh(x):
    return 2*sigmoid(2*x)-1
 
#input to the tanh function
input = []
for x in range(-5, 5):
    input.append(x)
     
#output of the tanh function
output = []
for ip in input:
    output.append(tanh(ip))
     
#plotting the graph for tanh function
plt.plot(input, output)
plt.grid()
#adding labels to the axes
plt.title("tanh activation function")
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.show()
