from tinygrad import tensor, nn
import tqdm

# https://www.youtube.com/watch?v=0ncx4H0YmK0&ab_channel=CircuitChronicles
class Net:
	def __init__(self):
		self.l1 = nn.Linear(2, 12)
		self.l2 = nn.Linear(12, 1)

	def __call__(self, x):
		x = self.l1(x)
		x = tensor.Tensor.sigmoid(x)
		x = self.l2(x)
		return x

net = Net()
optim = nn.optim.SGD(nn.state.get_parameters(net), lr=0.001)
X = tensor.Tensor([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
y = tensor.Tensor([1, 0, 0, 1]).reshape(4, 1)

print(net(X).numpy())

for i in tqdm.tqdm(range(2000)):
	for j in range(4):
		y_predicted = net(X[j])
		loss = ((y_predicted - y[j]) ** 2).mean()
		loss.backward()
		optim.step()

print(net(X).numpy())