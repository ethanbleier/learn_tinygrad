from tinygrad.tensor import Tensor
import numpy as np
import matplotlib.pyplot as plt

def f(x):
	if not isinstance(x, Tensor):
		x = Tensor(x)
	return (x **2 + x**3 - 5 * x).cos() ** 3 + 1.5

data = np.float32(np.linspace(-1.5, 1.5, 150))
plt.plot(data, f(data).numpy())

x = Tensor([-1.3])
plt.plot(x.numpy(), f(x).numpy(), 'go')
plt.show()