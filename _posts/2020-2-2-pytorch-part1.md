---
layout:     post                    # 使用的布局（不需要改）
title:      pytorch笔记part1               # 标题 
subtitle:   第2-3章 #副标题
date:       2020-02-02              # 时间
author:     serenity                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Python
    - deep learning
    - pytorch
---

# 第二章
## tensor
*初始化一个tensor的例子(随机初始化)*  
x = torch.rand(5, 3)
print(x)  



*tensor的inplace加法的例子*  
*adds x to y*
y.add_(x)
print(y)  

> 注意,PyTorch操作inplace版本都有后缀_, 例如x.copy_(y), x.t_() 



*给出一个tensor索引的例子*    

```python
y = x[0, :] # y是x的第一行
y += 1
print(y)
print(x[0, :]) # 源tensor也被改了
```

> 需要注意的是：**索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。**



*给出一个使用view()来改变形状的例子*

```python
x=tensor.rand(5,3)
y = x.view(15)
z = x.view(-1, 5)  # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())
```

```
输出：torch.Size([5, 3]) torch.Size([15]) torch.Size([3, 5])
```

> 注意**`view()`返回的新`Tensor`与源`Tensor`虽然可能有不同的`size`，但是是共享`data`的，也即更改其中的一个，另外一个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)**如果需要一个不共用data的副本，可以先`clone`一个副本再使用`view`



*item()函数*  

item函数可以将一个**标量**tensor转化为一个number  



*一个广播机制的例子*  

两个形状不同的tensor运算时可能会触发[^广播机制]  

```python
# x是1*2的tensor;y是3*1的
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)

输出:
tensor([[1, 2]])
tensor([[1],
        [2],
        [3]])
tensor([[2, 3],
        [3, 4],
        [4, 5]])
```

[^广播机制]:先适当复制元素使这两个`Tensor`形状相同后再按元素运算  



*tensor和numpy相互转换*  

> 用`numpy()`和`from_numpy()`将`Tensor`和NumPy中的数组相互转换。但是需要注意的一点是： **这两个函数所产生的的`Tensor`和NumPy中的数组共享相同的内存（所以他们之间的转换很快），改变其中一个时另一个也会改变！！！**

```python
a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

输出
tensor([1., 1., 1., 1., 1.]) [1. 1. 1. 1. 1.]
tensor([2., 2., 2., 2., 2.]) [2. 2. 2. 2. 2.]
tensor([3., 3., 3., 3., 3.]) [3. 3. 3. 3. 3.]

```



## 自动求梯度

*一个tensor求梯度的例子*  

```python
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
z = y * y * 3
out = z.mean() #mean函数求平均值
out.backward() # 等价于 out.backward(torch.tensor(1.))
print(x.grad)

输出:
    tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```



*再给出一个不是标量的例子*  

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v) #需要传入一个和z同形的权重向量
print(x.grad)

输出:
    tensor([2.0000, 0.2000, 0.0200, 0.0020])
```





# 第三章



## 线性回归

> 线性回归的输出都是连续值,这与softmax回归不同。因此softmax回归适用于分类问题。



*一个预测房价的线性回归例子*  

线性回归模型房屋价格预测的表达式为: 
$$
\hat{y}^{(i)} = x_1^{(i)} w_1 + x_2^{(i)} w_2 + b
$$
损失函数为:
$$
\ell(w_1, w_2, b) =\frac{1}{n} \sum_{i=1}^n \ell^{(i)}(w_1, w_2, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right)^2
$$
采用的优化算法为小批量随机梯度下降:
$$
\begin{aligned}
w_1 &\leftarrow w_1 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{ \partial \ell^{(i)}(w_1, w_2, b)  }{\partial w_1} = w_1 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}x_1^{(i)} \left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right),\\
w_2 &\leftarrow w_2 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{ \partial \ell^{(i)}(w_1, w_2, b)  }{\partial w_2} = w_2 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}x_2^{(i)} \left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right),\\
b &\leftarrow b -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{ \partial \ell^{(i)}(w_1, w_2, b)  }{\partial b} = b -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}\left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right).
\end{aligned}
$$
下面给出pytorch版本的代码实现  

```python
#生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

#读取数据
import torch.utils.data as Data
batch_size = 10
dataset = Data.TensorDataset(features, labels)	# 将训练数据的特征和标签组合
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)	# 随机读取小批量

#定义模型
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )
# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......
# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])

#初始化模型参数
from torch.nn import init
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

#定义损失函数
loss = nn.MSELoss()

#定义优化算法
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

#训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

#输出结果
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
```



### *一个softmax图像分类的例子*  

softmax回归与线性回归一样将输入特征和权重做线性叠加。softmax回归的输出是多个，而非一个。softmax回归的输出值个数等于标签里的类别数。  

以一个2*2像素，表示三种动物的图片集为例，因为有4个特征，3个输出，所以权重包含12个标量带下标的$w$）、偏差包含3个标量（带下标的$b$）
$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{21} + x_3 w_{31} + x_4 w_{41} + b_1,\\
o_2 &= x_1 w_{12} + x_2 w_{22} + x_3 w_{32} + x_4 w_{42} + b_2,\\
o_3 &= x_1 w_{13} + x_2 w_{23} + x_3 w_{33} + x_4 w_{43} + b_3.
\end{aligned}
$$
softmax运算将输出值变换成正数且和为1的概率分布。
$$
\hat{y}_1, \hat{y}_2, \hat{y}_3 = \text{softmax}(o_1, o_2, o_3)
$$

其中exp是以e为底的指数函数$e^x$

$$
\hat{y}_1 = \frac{ \exp(o_1)}{\sum_{i=1}^3 \exp(o_i)},\quad
\hat{y}_2 = \frac{ \exp(o_2)}{\sum_{i=1}^3 \exp(o_i)},\quad
\hat{y}_3 = \frac{ \exp(o_3)}{\sum_{i=1}^3 \exp(o_i)}.
$$


***单个样本的矢量计算表达式***  
$$
\begin{aligned}
\boldsymbol{o}^{(i)} &= \boldsymbol{x}^{(i)} \boldsymbol{W} + \boldsymbol{b},\\
\boldsymbol{\hat{y}}^{(i)} &= \text{softmax}(\boldsymbol{o}^{(i)}).
\end{aligned}
$$
其中
$$
\boldsymbol{W} = 
\begin{bmatrix}
    w_{11} & w_{12} & w_{13} \\
    w_{21} & w_{22} & w_{23} \\
    w_{31} & w_{32} & w_{33} \\
    w_{41} & w_{42} & w_{43}
\end{bmatrix},\quad
\boldsymbol{b} = 
\begin{bmatrix}
    b_1 & b_2 & b_3
\end{bmatrix},
$$

$$
\boldsymbol{x}^{(i)} = \begin{bmatrix}x_1^{(i)} & x_2^{(i)} & x_3^{(i)} & x_4^{(i)}\end{bmatrix},\boldsymbol{o}^{(i)} = \begin{bmatrix}o_1^{(i)} & o_2^{(i)} & o_3^{(i)}\end{bmatrix}
$$

对于多个样本（比如n个样本），可以增加$x^{(i)}$的行数，使之成为n\*4的矩阵。这样最后的输出结果会改变为一个n\*3的矩阵



**损失函数**采用**交叉熵**，即: 

对于样本$i$，我们构造向量$ \boldsymbol{y}^{(i)}\in \mathbb{R}^{q} $ ，使其第$y^{(i)}$（样本$i$类别的离散数值）个元素为1，其余为0。交叉熵的公式如下:
$$
H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = -\sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)},
$$

其中带下标的$y_j^{(i)}$是向量$\boldsymbol y^{(i)}$中非0即1的元素，需要注意将它与样本$i$类别的离散数值，即不带下标的$y^{(i)}$区分。在上式中，我们知道向量$ \boldsymbol y^{(i)}$中只有第$y^{(i)} $个元素$y^{(i)}_{y^{(i)}}$为1，其余全为0,于是  

$$H(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}) = -\log \hat y_{y^{(i)}}^{(i)}$$


假设训练数据集的样本数为$n$，交叉熵损失函数定义为
$$\ell(\boldsymbol{\Theta}) = \frac{1}{n} \sum_{i=1}^n H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ),$$

其中$\boldsymbol{\Theta}$代表模型参数。同样地，如果每个样本只有一个标签，那么交叉熵损失可以简写成$\ell(\boldsymbol{\Theta}) = -(1/n)  \sum_{i=1}^n \log \hat y_{y^{(i)}}^{(i)}$。从另一个角度来看，我们知道最小化$\ell(\boldsymbol{\Theta})$等价于最大化$\exp(-n\ell(\boldsymbol{\Theta}))=\prod_{i=1}^n \hat y_{y^{(i)}}^{(i)}$，即最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率。  



*pytorch实现softmax回归*  



### *多层感知机*








