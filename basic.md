# pytorch 入门指南

## 1. pytorch 概述
pytorch是facebook 开发的torch（Lua语言）的python版本，于2017年引爆学术界
官方宣传pytorch侧重两类用户：numpy的gpu版、深度学习研究平台
pytorch使用动态图机制，相比于tensorflow最开始的静态图，更为灵活
当前pytorch支持的系统包括：win，linux，macos

## 2. pytorch基本库
常用的pytorch基本库主要包括：
 - torch： 内含一些常用方法，与numpy比较像
 - torch.Tensor：内含一些操作tensor的方法，可通过tensor.xx()进行调用
 - torch.nn：内含一些常用模型，如rnn，cnn等
 - torch.nn.functional：内含一些常用方法，如sigmoid，softmax等
 - torch.optim：内含一些优化算法，如sgd，adam等
 - torch.utils.data：内含一些数据迭代方法

## 3. 基本操作
### a. tensor操作
```
# 初始化空向量
torch.empty(3,4)

# 随机初始化数组
torch.rand(4,3)

# 初始化零向量
torch.zeros(4,3, dtype=torch.int)

# 从数据构建数组
x = torch.tensor([3,4],dtype=torch.float)
x = torch.IntTensor([3,4])

# 获取tensor的尺寸，元组
x.shape
x.size()

# _在方法中的意义：表示对自身的改变
x = torch.ones(3,4) 
# 以下三个式子 含义相同
x = x + x
x = torch.add(x, x)
x.add_(x)

# 索引,像操作numpy一样
x[:,1]

# 改变形状
x.view(-1)
x.view(4,3)

# 如果只包含一个元素值，获取
x = torch.randn(1)
x.item()

# 增加一维
input = torch.randn(32, 32)
input = input.unsqueeze(0)
input.size()

# tensor的data还是tensor，但是requires_grad=False
x.data.requires_grad

# 改变类型
x.type(torch.LongTensor)
```

### b. numpy 与 tensor的转换
```
# 转换, 共享内存
a= numpy.array([1,2,3])
a = torch.from_numpy(a)
a.numpy()
```

### c. 调用gpu
```
# gpu是否可用
torch.cuda.is_available()
# 调用设备
device = torch.device('cpu') # cuda or cpu
a = torch.tensor([1,2,3], device='cuda')  # 直接在gpu上创建
a = a.to(device) # 上传
a = a.to('cpu') # 上传, cpu or cuda
a = a.cuda()  # 上传cuda
```

### d. 梯度
 - .requires_grad ，决定是否可微（梯度）
 - .backward(), 计算梯度；如果单独一个值则不需指定参数，否则需传入权重（尺寸与tensor的size同）
 - .grad, 用于存储梯度累计值。 只有tensor有梯度值，计算节点没有
 - .detach(), 相当于新建了一个变量，历史的计算图无效
 - with torch.no_grad():, 评估模型时可用到，不计算梯度
 - .grad_fn, 节点是如何产生的；用户创造的tensor([1,2,3]).grad_fn 为None
 - .data(), tensor值，requires_grad=False
```
# 创建可微的tensor
x = torch.ones(2,3,requires_grad=True)

# 改变可微性
x.requires_grad_(False)

# 获得梯度值
x = torch.ones(2, 2, requires_grad=True)
y = x +2
z = y * y *3
out = torch.sum(z)
out.backward()
x.grad

# 无梯度， 报错
with torch.no_grad():
    x = torch.ones(2, 2, requires_grad=True)
    y = x +2
    z = y * y *3
    out = torch.sum(z)
    out.backward()
    x.grad
```

### e. 定义模型
两种定义方式
 - class定义
 - Sequential定义
```
# 通过class定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 下面通过实例变量的形式声明模型内需要学习的参数
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10,20)

    def forward(self, x):
        # 下面定义计算图
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
net = Net()

# 通过Sequential定义
net = Sequential(
    nn.Linear(5, 10),
    nn.Relu(),
    nn.Linear(10, 20)
)
```

### f. 模型参数操作
```
# 获取模型参数
net.parameters() #可用for 迭代

# 模型内参数梯度清零
net.zero_grad()
```

### g. 定义损失函数
```
loss = nn.CrossEntropyLoss()
```

### h. 定义优化算子
```
optimizer = optim.SGD(net.parameters(), lr=0.01)
```

### i. 训练
```
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() 
```

### j. 测试
```
# 测试
with torch.no_grad():
    output = net(input)
```

### k. 保存与载入
```
# 模型
torch.save(net, file)
net = torch.load(file)

# 参数
torch.save(model.state_dict(), file)
net = Model()
net.load_state_dict(file)
```

## 4. 一个完整的机器学习流程
 1. 数据
    1. 载入数据
    2. 数据处理
    3. 构建迭代器
 2. 模型
    1. loss
    2. optimizer
    3. 新建/载入模型
        - 新建
        - 载入
           - 直接载入模型
           - 载入参数
               1. 新建模型
               2. 载入模型参数（对于adam等优化器，其参数也需载入)
 3. 训练
    1. batch训练

    ```
for i, batch in enumerate(dataloader):
         x_batch, y_batch = batch
         outputs = net(x_batch)
         loss = criterion(output, target)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
    ```

    2. 每隔一段时间，打印验证集loss
    3. 每隔一段时间，存储模型

 4. 测试
    1. 载入测试数据
    2. 数据处理
    3. 构建迭代器（可选）
    4. 放入模型，输出结果
    5. 计算accuracy





















