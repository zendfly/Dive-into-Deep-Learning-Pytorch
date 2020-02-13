# 感知机

本质是一个全连接的神经网络，并加入了激活函数进行非线性变换。感知机包含输入层、隐藏层和输出层。

以单隐层感知机为例，各层大小：

输入层$I:X\subseteq R^{n\times d}$ ，$n$为大小，$d$为个数；

隐藏层$H:H\subseteq R^{n \times h}$，$n$为大小，$h$为个数；

输出层$O:O\subseteq R^{1 \times q}$，q为个数。

各层的权重和偏执：

隐藏层，权重$W_h\subseteq R^{n \times s} $，偏执$b_h \subseteq R^{1 \times s}$‘，$s$为个数；

输出层，权重$W_o \subseteq R^{h \times q}$，偏执$b_o \subseteq R^{1 \times q}$，$q$为个数。

各层的计算：

隐藏层：
$$
H = XW_h + b_h
$$
输出层：
$$
O=HW_o + b_o
$$
两式连起来，可得：
$$
O = (XW_h+b_h)W_0+b_0=XW_hW_0+b_hW_0+b_0
$$
其中，$XW_hW_0$为输出层的权重，$b_hW_0+b_0$为输出层的偏执。

# 激活函数

## ReLu

公式：
$$
Relu(x) = max(x,0)
$$
可以看出，ReLu将负数转换成0。

## Sigmod

公式：
$$
Sigmod(x)=\frac{1}{1+e^{-x}}=\frac{1}{1+exp(-x)}
$$
Sigmod函数，将元素映射到一个[0,1]区间，$x$越小，结果越接近0，反之越接近1。

## 激活函数选择

ReLu是一种通用的激活函数，但只能用于隐藏层中。

Sigmod函数，在进行分类任务时，效果较好。由于梯度消失的问题，通常需要和其它激活函数组合使用。



##  code_pytorch

该代码是在kesci（科赛）提供的镜像(d2l-pytorch)环境下运行。

### 手动实现感知机(perceptron)

导入相关model

```python
import torch
import numpy as np
import sys
sys.path.append("/home/kesci/input")
import d2lzh1981 as d2l
print(torch.__version__)
```

导入相关数据集

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,root='/home/kesci/input/FashionMNIST2065')
```

定义模型参数

```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 隐藏层权重和偏差
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
# 输出层权重和偏差
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)	#pytorch中tensor的一个自带的属性，用于求导定义
```

定义激活函数

```python
def relu(X)：
	return troch.max(input=x,other=torch.tensor(0.0))
```

定义网络

```python
def net(X):
    X = X.view((-1, num_inputs))	#对输入进行转换，view(),将一个多行的tensor拼接成一行
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2
```

定义损失

```python
loss = torch.nn.CrossEntropyLoss()	#使用 pytorch自带的交叉熵损失函数
```

训练

```python
num_epochs, lr = 5, 100.0
# def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
#               params=None, lr=None, optimizer=None):
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
#         for X, y in train_iter:
#             y_hat = net(X)
#             l = loss(y_hat, y).sum()
#             
#             # 梯度清零
#             if optimizer is not None:
#                 optimizer.zero_grad()
#             elif params is not None and params[0].grad is not None:
#                 for param in params:
#                     param.grad.data.zero_()
#            
#             l.backward()
#             if optimizer is None:
#                 d2l.sgd(params, lr, batch_size)
#             else:
#                 optimizer.step()  # “softmax回归的简洁实现”一节将用到
#             
#             
#             train_l_sum += l.item()
#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
#             n += y.shape[0]
#         test_acc = evaluate_accuracy(test_iter, net)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
#               % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
```

多层感知机—pytorch实现

```python
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("/home/kesci/input")
import d2lzh1981 as d2l

print(torch.__version__)
```

初始化模型和各个参数

```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256
    
net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
        )
    
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)
```

训练

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,root='/home/kesci/input/FashionMNIST2065')
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
```

