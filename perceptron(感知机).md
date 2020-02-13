# 感知机

本质是一个全连接的神经网络，并加入了激活函数进行非线性变换。感知机包含输入层、隐藏层和输出层。

以单隐层感知机为例，各层大小：

输入层$I:X\subseteq R^{n\times d}$ ，$n$为大小，$d$为个数；

隐藏层$H:H\subseteq R^{n \times h}$，$n$为大小，$h$为个数；

输出层$O:O\subseteq R^{1 \times q}$，q为个数。

各层的权重和偏执：

隐藏层，权重$W_h\subseteq R^{n \times s} $，偏执$b_h \subseteq R^{1 \times s}$‘，$s$为个数；

输出层，权重$W_o \subseteq R^{n \times q}$，偏执$b_o \subseteq R^{1 \times q}$，$q$为个数。

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