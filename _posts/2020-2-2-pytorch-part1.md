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

