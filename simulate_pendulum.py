import torch
from nn_controller import NN_IBP
import matplotlib.pyplot as plt

from dynamics import pendulum_f, pendulum_control_matrix

model = NN_IBP(input_dim=2, hidden_dims=[64,64], trainable_NCM=False)

model.load_state_dict(torch.load('inverted_pendulum_new.pth', weights_only=True))
model.eval()

T = 500000
dt = 0.001

xs = torch.zeros((2, T))
x = xs[:,0] + torch.tensor([0.2, 0.0])
zero = torch.zeros_like(x)

print(model(x) -model(zero))
print(pendulum_f(x))
for i in range(T):
    x += dt * (pendulum_f(x).squeeze() + pendulum_control_matrix() @ (model(x) - model(zero))) 
    xs[:,i] = x

print(xs[:,-1])

plt.plot(xs[0,:].detach().numpy())
plt.show()