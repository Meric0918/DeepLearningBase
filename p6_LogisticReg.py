import torch
# import torch.nn.functional as F

'''
# @Collector and Speaker : little snow
# @Owner : 考研数学计算机之路 2024-2025 项目班

# 代码重点解读：
    逻辑斯蒂回归和上一节线性模型相比，在线性模型的后面，添加了激活函数(非线性变换)，见 line:30
     逻辑斯蒂回归用于处理分类问题
'''


# 准备数据集
# x_data 是输入特征，包含三个样本，每个样本是一个一维特征
# y_data 是目标输出，表示对应的标签，0表示负类，1表示正类
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

# 设计逻辑回归模型，继承自 torch.nn.Module
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        # 定义一个线性层，输入特征维度为1，输出特征维度也为1
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 前向传播函数
        # 通过线性层计算预测值，并应用sigmoid激活函数
        # sigmoid函数将线性输出转换为概率值，范围在0到1之间
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

# 实例化逻辑回归模型
model = LogisticRegressionModel()

# 定义损失函数为二元交叉熵损失
# reduction='sum'表示返回所有样本的总损失，而不是平均损失
criterion = torch.nn.BCELoss(reduction='sum')

# 定义优化器为随机梯度下降（SGD），学习率为0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练循环，进行前向传播、反向传播和参数更新
for epoch in range(1000):
    # 前向传播，计算预测值
    y_pred = model(x_data)
    # 计算损失
    loss = criterion(y_pred, y_data)
    # 打印当前迭代次数和损失值
    print(epoch, loss.item())

    # 清零梯度
    optimizer.zero_grad()
    # 反向传播，计算梯度
    loss.backward()
    # 更新模型参数
    optimizer.step()

# 打印训练后的模型参数（权重和偏置）
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# 测试模型，输入一个新的样本 x_test
x_test = torch.Tensor([[4.0]])
# 通过模型计算预测值
y_test = model(x_test)
# 打印预测结果
print('y_pred = ', y_test.data)
