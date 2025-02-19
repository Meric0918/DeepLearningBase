# 作业题目：实现一个简单的线性回归模型，使用PyTorch进行训练
# 该模型通过最小化损失函数来学习权重和偏置

import numpy as np
import matplotlib.pyplot as plt
import torch

x_data = [1.0, 2.0, 3.0]  # 输入数据
y_data = [2.0, 4.0, 6.0]  # 目标输出数据

w1 = torch.Tensor([1.0])  # 初始权值w1
w1.requires_grad = True  # 计算梯度，默认是不计算的
w2 = torch.Tensor([1.0])  # 初始权值w2
w2.requires_grad = True
b = torch.Tensor([1.0])   # 初始偏置b
b.requires_grad = True

def forward(x):
    # 前向传播函数，计算预测值
    return w1 * x**2 + w2 * x + b

def loss(x, y):
    # 计算损失函数，构建计算图
    y_pred = forward(x)
    return (y_pred - y) ** 2

before_training = forward(4)

for epoch in range(2000):
    l = loss(1, 2)  # 为了在for循环之前定义l,以便之后的输出，无实际意义
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # 计算当前的损失
        l.backward()  # 反向传播，计算梯度
        # print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        # 更新权重和偏置
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data
        # 清零梯度
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print('Epoch:', epoch, l.item())

print('Predict (before training)', 4, before_training.item())
print('Predict (after training)', 4, forward(4).item())
