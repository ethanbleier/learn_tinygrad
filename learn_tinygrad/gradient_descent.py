#!/usr/bin/env python3

from tinygrad.tensor import Tensor
import numpy as np
import matplotlib.pyplot as plt
from tinygrad.nn.optim import Adam

def f(x):
	if not isinstance(x, Tensor): x = Tensor(x)
	return (x ** 2 + x ** 3 - 5 * x).cos() ** 3 + 1.5

data = np.float32(np.linspace(-1.5, 1.5, 150))
plt.plot(data, f(data).numpy())
# plt.show()

x = Tensor([-1.3])
plt.plot(x.numpy(), f(x).numpy(), 'go')

adam = Adam([x], lr = 0.001)
y_exp = Tensor([0])


for _ in range(29):
	# 
	# Current problem:
	# Cant make >30 steps without crashing
	#
	# line 24, in <module>
	#     adam.step()
	#   File "/Users/ethanbleier/tinygrad/tinygrad/nn/optim.py", line 72, in step
	#     self.realize([self.t] + self.m + self.v)
	#
	# 

	loss = -(y_exp ** 2 - f(x) ** 2).mean()
	loss.backward()
	adam.step()
	print(f"Loss: {loss.numpy()}")
	if loss.numpy() < 0.3: break

print(f"Starting value of x: -1.3\nFinal value of x: {x.numpy()}")
dist = -1.3 - x.numpy()
print(f"distance: {dist}")
plt.plot(x.numpy(), f(x).numpy(), 'ro')
plt.show()