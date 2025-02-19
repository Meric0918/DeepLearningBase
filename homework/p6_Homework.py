# 作业题目：实现一个逻辑回归模型，使用不同的优化器进行训练，并绘制每种算法的损失变化曲线。

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 输入数据
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# 目标输出数据
y_data = torch.Tensor([[0], [0], [1]])

# 定义逻辑回归模型
class Logistic_Regression_Model(torch.nn.Module):
    def __init__(self):
        super(Logistic_Regression_Model, self).__init__()
        # 定义线性层，输入和输出均为1维
        self.linear = torch.nn.Linear(1, 1, bias=True)

    def forward(self, x):
        # 前向传播，使用sigmoid激活函数
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
    
# 创建不同优化器的模型实例
models = {
    'SGD': Logistic_Regression_Model(),
    'Adam': Logistic_Regression_Model(),
    'Adagrad': Logistic_Regression_Model(),
    'Adamax': Logistic_Regression_Model(),
    'ASGD': Logistic_Regression_Model(),
    'RMSprop': Logistic_Regression_Model(),
    'Rprop': Logistic_Regression_Model(),
}

# 定义损失函数为二元交叉熵损失
criterion = torch.nn.BCELoss(size_average=False)
# 定义不同的优化器
optimizers = {
    'SGD': torch.optim.SGD(models['SGD'].parameters(), lr=0.01),
    'Adam': torch.optim.Adam(models['Adam'].parameters(), lr=0.01),
    'Adagrad': torch.optim.Adagrad(models['Adagrad'].parameters(), lr=0.01),
    'Adamax': torch.optim.Adamax(models['Adamax'].parameters(), lr=0.01),
    'ASGD': torch.optim.ASGD(models['ASGD'].parameters(), lr=0.01),
    'RMSprop': torch.optim.RMSprop(models['RMSprop'].parameters(), lr=0.01),
    'Rprop': torch.optim.Rprop(models['Rprop'].parameters(), lr=0.01),
}

# 初始化每种优化器的损失值列表
loss_values = {k: [] for k in optimizers.keys()}

# 训练循环
for opt_name, optimizer in optimizers.items():
    model = models[opt_name]
    for epoch in range(1000):
        # 前向预测
        y_pred = model(x_data)
        # 计算损失
        loss = criterion(y_pred, y_data)
        # 清零梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # 记录损失值
        loss_values[opt_name].append(loss.item())

# 绘制损失变化曲线
plt.figure(figsize=(10, 5))

for opt_name, losses in loss_values.items():
    plt.plot(losses, label=opt_name)

plt.xlabel('Epoch')  # x轴标签
plt.ylabel("Loss")   # y轴标签
plt.legend()        # 显示图例
plt.title("Loss by Optimization Algorithm")  # 图表标题
plt.show()         # 显示图表
