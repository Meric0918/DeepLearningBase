'''
# @Collector and Speaker : little snow
# @Owner : 考研数学计算机之路 2024-2025 项目班
'''

# 定义输入数据，x_data为自变量，y_data为因变量
from matplotlib import pyplot as plt

x_data = [1.0, 2.0, 3.0]  # 自变量数据
y_data = [2.0, 4.0, 6.0]  # 因变量数据
w = 1.0  # 初始化权重w

# 定义前向传播函数
def forward(x):
    # 计算预测值，返回输入x与权重w的乘积
    return x * w

# 定义损失函数
def cost(xs, ys):
    cost = 0  # 初始化损失值
    # 遍历每一对输入输出数据
    for x, y in zip(xs, ys):
        y_pred = forward(x)  # 计算当前输入的预测值
        cost += (y_pred - y) ** 2  # 累加预测值与真实值的平方差
    return cost / len(xs)  # 返回平均损失

# 定义梯度计算函数
def gradient(xs, ys):
    grad = 0  # 初始化梯度值
    # 遍历每一对输入输出数据
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)  # 计算当前输入的梯度
    return grad / len(xs)  # 返回平均梯度

epoch_list = []
loss_list = []

# 打印训练前的预测结果
print('Predict (before training)', 4, forward(4))  # 预测输入4的输出
# 进行100个epoch的训练
for epoch in range(100):
    cost_val = cost(x_data, y_data)  # 计算当前的损失值
    grad_val = gradient(x_data, y_data)  # 计算当前的梯度值
    w -= 0.01 * grad_val  # 更新权重w
    epoch_list.append(epoch)
    loss_list.append(cost_val)
    # 打印当前epoch的权重和损失值
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)

# 打印训练后的预测结果
print('Predict (after training)', 4, forward(4))  # 预测输入4的输出

# 绘制训练次数与均方误差的关系图
plt.plot(epoch_list, loss_list)  # 绘制权重与均方误差的曲线
plt.ylabel('loss')  # 设置y轴标签
plt.xlabel('epoch')  # 设置x轴标签
plt.show()  # 显示图形
