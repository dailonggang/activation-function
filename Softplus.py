import torch
import torch.nn as nn
import matplotlib.pyplot as plt
x = torch.linspace(-6, 6, 100)
softplus = nn.Softplus() # 从nn模块中调入Tanh()层，在nn.functional中有对应的函数
y_softplus = softplus(x)

# Visualize
plt.figure(figsize=(6, 6))
plt.plot(x.data.numpy(), y_softplus.data.numpy(), "o-")
plt.title("Softplus")
plt.grid()
plt.show()
