# learn_tinygrad
history of learning neural nets and Tinygrad
---
### Feb 11, 2024
Started repo after watching this [video](https://www.youtube.com/watch?v=0ncx4H0YmK0&ab_channel=CircuitChronicles)

Wrote gradient descent from scratch.

TODO: Fix bug in gradient_descent.py where we cannot loop more than 29 times (yes, 29) without getting 

`	 line 24, in <module> adam.step() ~/tinygrad/tinygrad/nn/optim.py", line 72, in step
	     self.realize([self.t] + self.m + self.v)
 `
...
