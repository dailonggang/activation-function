import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.linspace(-6, 6, 100)
relu = nn.ReLU() # 从nn模块中调入Tanh()层，在nn.functional中有对应的函数
y_relu = relu(x)

# Visualize
plt.figure(figsize=(6, 6))
plt.plot(x.data.numpy(), y_relu.data.numpy(), "o-")
plt.title("ReLU")
plt.grid()
plt.show()
