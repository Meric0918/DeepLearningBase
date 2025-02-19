import torch

'''
# @Collector and Speaker : little snow
# @Owner : 考研数学计算机之路 2024-2025 项目班

# 相关前备知识：
    Tensor和tensor、torch.Tensor和torch.tensor的区别：
        https://blog.csdn.net/tfcy694/article/details/85338745
        torch.Tensor(1) 与 torch.Tensor([1]) 标量1是作为size传入的，向量1是作为value传入的
        
# 代码重要解读：
    本算法的反向传播过程主要通过调用 l.backward() 方法实现。
    该方法执行后，w.grad 会从 None 转变为 Tensor 类型，而w.grad.data 的值将用于更新 w.data。
    通过 l.backward()，计算图中所有需要计算梯度的部分都会被求出，梯度会被存储在相应的参数中。
    计算完成后，计算图会被释放，以优化内存使用。
    需要注意的是，直接访问 tensor 的 data 不会创建计算图。
'''

# 定义输入数据，x_data为自变量，y_data为因变量
x_data = [1.0, 2.0, 3.0]  # 自变量数据
y_data = [2.0, 4.0, 6.0]  # 因变量数据

# 初始化权重w，使用PyTorch的Tensor，并设置requires_grad为True以便进行梯度计算
w = torch.Tensor([1.0])  # 权重初始化为1.0
w.requires_grad = True  # 允许对权重进行梯度计算

# 定义前向传播函数
def forward(x):
    # 计算预测值，返回输入x与权重w的乘积
    return x * w  # 预测值为输入x与权重w的乘积

# 定义损失函数
def loss(x, y):
    # 计算当前输入x的预测值
    y_pred = forward(x)  # 调用前向传播函数获取预测值
    # 返回预测值与真实值之间的平方差
    return (y_pred - y) ** 2  # 返回均方误差

# 打印训练前的预测结果，预测输入4的输出
print("predict (before training)", 4, forward(4).item())  # 预测输入4的输出

# 进行100个epoch的训练
for epoch in range(100):
    # 遍历每一对输入输出数据
    for x, y in zip(x_data, y_data):
        # 计算当前输入的损失值
        l = loss(x, y)  # 计算损失
        l.backward()  # 反向传播，计算梯度
        # 打印当前输入、真实输出和计算得到的梯度
        print('\tgrad:', x, y, w.grad.item())  # 打印当前的梯度值

        # 更新权重w，使用学习率0.01
        w.data = w.data - 0.01 * w.grad.data  # 更新权重

        # 清零梯度，以便下一次迭代使用
        w.grad.data.zero_()  # 清空梯度

    # 打印当前epoch的进度和损失值
    print("progress:", epoch, l.item())  # 打印当前epoch的损失值

# 打印训练后的预测结果，预测输入4的输出
print("predict (after training)", 4, forward(4).item())  # 预测输入4的输出
