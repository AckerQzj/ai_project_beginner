## python基础

**列表 字典 切片** **None**

```
列表list

字典dict	哈希表	字典中的key和value都可以是任意类型

迭代list时得到的是list中的key值

dist[key] = value;

切片
```

**列表删除：**

```
list.remove()	按值删除

del list[index]	按索引删除
```

**列表新增元素：**

```
list.append()	尾插元素

list.extend()	尾插列表形成新的列表	eg:list.extend([8,9,0])
```

**循环后置**

```
在循环之前，对循环元素i进行操作

list = [i for i in range(0,10)]
```

**运算**

```
/	普通除法

//	取整除法

a ** b	乘方
```

**函数**

```
def fun():

​	...

函数不指定返回值，会显示None
```

**类（深度学习的模型）**

```
class ClassName():

​	...

​	def __init__(self,name,age):

​		self.name = name

​		self.age = age
```

**继承**

```
class ExtendName(ClassName):

​	重新初始化

​	def __init__():

​		...

​		super()...

用super来进行初始化
```

**numpy**

```
numpy用于处理矩阵操作

tensor：简单理解为矩阵
```



```
import numpy as np

list = [
	[1,2,3,4,5],
	[6,7,8,9,10],
	[11,12,13,14,15]
]

array = np.array(list) // 把list转化为了矩阵


# 矩阵操作

# 矩阵合并 
# create a matrix same as array
array2 = np.array(list)
array3 = np.concatenate((array,array2)) 注意参数格式 纵向合并

array3 = np.concatenate((array,array2),axis = 1)
# axis 水平向下为第0维，水平向右为1，默认向下合并


# 切片 矩阵切片能切到矩阵中间的方框
list[1:3]	list[:]
array[1:3] 切矩阵1，2行
array[1:3,1:3] 切矩阵1，2行，1，2列
: 表示所有行列
idx = [1,3]
array[:,idx] 也能起到切片效果

```

**Tensor张量**

```
import torch

list = [
	[1,2,3,4,5],
	[6,7,8,9,10],
	[11,12,13,14,15]
]

# 矩阵
array = np.array(list)
# tensor
tensor1 = torch.tensor(list)

输出区别：tensor终端输出时显示tensor

相当于把一个矩阵放在一张张量网上，能求梯度

x = torch.tensor(3.0) # 创建张量
x.requires_grad_(True) # x在计算的时候需要计算梯度
y = x ** 2# y == 9 
y.backward() #2x = 6

y2 = x ** 2
y2.backward()
print(y2) # 9
print(x.grad()) # 12	因为张量网中记录原来计算的梯度，原梯度为6保存至x上，新计算的梯度仍为6，所以新的梯度为6+6

解决方案：
x.grad = torch.tensor(0.0)
或:x = x.detach() # x不能再计算梯度

```

```
# 创建张量

tensor1 = torch.ones((100,4)) # 一百行 四列的张量
tensor2 = torch.zeros((10,4)) # 十行 全是0
tensor3 = torch.normal(0,0.01,(3,10,4)) # 正态分布

# 张量求和

sum1 = torch.sum(tensor1)
sum1 = torch.sum(tensor1,dim = 0) # 竖着加
sum1 = torch.sum(tensor1.dim = 1) #横着加
sum1 = torch.sum(tensor1.dim = 1，keepdim = True) #横着加，形状不变

# 张量形状

tensor1.shape
```

**引用**

```
from mytensor import tensor1

from myclass import superman
```

