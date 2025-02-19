# 作业题目：实现一个多层感知机模型，使用不同的激活函数进行训练，并绘制每种激活函数的损失变化曲线。

import torch
import numpy as np
import matplotlib.pyplot as plt

# 读取压缩包中的数据，使用 np.loadtxt，分隔符为逗号
xy = np.loadtxt('../dataset/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])  # 输入数据
y_data = torch.from_numpy(xy[:, [-1]])   # 目标输出数据

# 定义多层感知机模型
class Model(torch.nn.Module):
    def __init__(self, activation_fn=torch.nn.Sigmoid()):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 第一层线性变换
        self.linear2 = torch.nn.Linear(6, 4)  # 第二层线性变换
        self.linear3 = torch.nn.Linear(4, 1)  # 第三层线性变换
        self.activation_fn = activation_fn    # 激活函数

    def forward(self, x):
        x = self.activation_fn(self.linear1(x))  # 第一层激活
        x = self.activation_fn(self.linear2(x))  # 第二层激活
        x = torch.sigmoid(self.linear3(x))       # 输出层使用sigmoid激活
        return x

# 定义不同的激活函数
activation_fns = {
    'Sigmoid': torch.nn.Sigmoid(),
    'ReLU': torch.nn.ReLU(),
    'Tanh': torch.nn.Tanh(),
    'Softplus': torch.nn.Softplus(),
}

#二元分类问题使用二元交叉熵损失作为损失函数
criterion = torch.nn.BCELoss(reduction='mean')  # 定义损失函数为二元交叉熵损失

# 初始化每种激活函数的损失值列表
loss_values = {k: [] for k in activation_fns.keys()}

# 训练循环
for activation_name, activation_fn in activation_fns.items():
    model = Model(activation_fn=activation_fn)  # 创建模型实例
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 定义优化器

    for epoch in range(100):
        y_pred = model(x_data)  # 前向预测
        loss = criterion(y_pred, y_data)  # 计算损失
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        loss_values[activation_name].append(loss.item())  # 记录损失值

# 绘制损失变化曲线
plt.figure(figsize=(10, 5))

for activation_name, losses in loss_values.items():
    plt.plot(losses, label=activation_name)

plt.xlabel('Epoch')  # x轴标签
plt.ylabel('Loss')   # y轴标签
plt.title('Loss by Activation Function')  # 图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表
