import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.linspace(-6, 6, 100)
tanh = nn.Tanh() # 从nn模块中调入Tanh()层，在nn.functional中有对应的函数
y_Tanh = tanh(x)

# Visualize
plt.figure(figsize=(6, 6))
plt.plot(x.data.numpy(), y_Tanh.data.numpy(), "o-")
plt.title("Tanh")
plt.grid()
plt.show()
