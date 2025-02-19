# 任务说明：一个线性回归模型，使用不同的优化器（如SGD、Adam等）进行训练，并绘制每种算法的损失变化曲线。

import torch
import matplotlib.pyplot as plt

# step 1: 准备数据集
x_data = torch.Tensor([[1.0], [2.0], [3.0]])  # 输入数据
y_data = torch.Tensor([[2.0], [4.0], [6.0]])  # 目标输出数据

# step 2: 设计模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1, bias=True)  # 定义线性层，输入和输出均为1维
    def forward(self, x):
        y_pred = self.linear(x)  # 前向传播，计算预测值
        return y_pred

# 创建不同优化器的模型实例
models = {
    'SGD': LinearModel(),
    'Adam': LinearModel(),
    'Adagrad': LinearModel(),
    'Adamax': LinearModel(),
    'ASGD': LinearModel(),
    'RMSprop': LinearModel(),
    'Rprop': LinearModel(),
}

# step 3: 定义损失函数和优化器
criterion = torch.nn.MSELoss(size_average=False)  # 均方误差损失函数
optimizer = {
    'SGD': torch.optim.SGD(models['SGD'].parameters(), lr=0.01),
    'Adam': torch.optim.Adam(models['Adam'].parameters(), lr=0.01),
    'Adagrad': torch.optim.Adagrad(models['Adagrad'].parameters(), lr=0.01),
    'Adamax': torch.optim.Adamax(models['Adamax'].parameters(), lr=0.01),
    'ASGD': torch.optim.ASGD(models['ASGD'].parameters(), lr=0.01),
    'RMSprop': torch.optim.RMSprop(models['RMSprop'].parameters(), lr=0.01),
    'Rprop': torch.optim.Rprop(models['RMSprop'].parameters(), lr=0.01),
}

# 初始化每种优化器的损失值列表
loss_values = {k: [] for k in optimizer.keys()}

# step 4: 训练循环
# 加入测试集 x_test = [4.0]
for opt_name, optimizer in optimizer.items():
    model = models[opt_name]  # 获取当前优化器对应的模型
    for epoch in range(100):
        y_pred = model(x_data)              # 前向预测
        loss = criterion(y_pred, y_data)    # 计算损失
        optimizer.zero_grad()               # 清零梯度
        loss.backward()                     # 反向传播
        optimizer.step()                    # 更新参数

        loss_values[opt_name].append(loss.item())  # 记录损失值

    # 测试模型，输入一个新的样本 x_test
    x_test = torch.Tensor([[4.0]])
    # 通过模型计算预测值
    y_test = model(x_test)
    # 打印预测结果
    print(opt_name , ": y_pred = ", y_test.data)

# 绘制损失变化曲线
plt.figure(figsize=(10, 5))
for opt_name, losses in loss_values.items():
    plt.plot(losses, label=opt_name)

plt.xlabel("Epoch")  # x轴标签
plt.ylabel("Loss")   # y轴标签
plt.legend()        # 显示图例
plt.title("Loss by Optimization Algorithm")  # 图表标题
plt.show()         # 显示图表
