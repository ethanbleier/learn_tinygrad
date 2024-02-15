#!/usr/bin/env python3

import math

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

x = 1.5
print("tanh({}) = {}".format(x, tanh(x)))
