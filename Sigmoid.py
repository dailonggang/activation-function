import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.linspace(-6, 6, 100)  # -6到6线性等分100点
sigmoid = nn.Sigmoid()
y_sigmoid = sigmoid(x)

plt.figure(figsize=(6, 6), facecolor='blue', frameon=True)
plt.plot(x, y_sigmoid, "o-")  # o-格式控制字符串
plt.title("Sigmoid")
plt.grid()
plt.show()
