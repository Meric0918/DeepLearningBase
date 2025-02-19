# 导入必要的库
import numpy as np  # 用于数值计算的库
import matplotlib.pyplot as plt  # 用于绘图的库

'''
# @Collector and Speaker : little snow
# @Owner : 考研数学计算机之路 2024-2025 项目班


# 相关前备知识：
    np.arange的用法 : https://blog.csdn.net/qq_45154565/article/details/115690186
    python中zip()用法： https://blog.csdn.net/csdn15698845876/article/details/73411541
'''


# 定义输入数据
x_data = [1.0, 2.0, 3.0]  # 自变量数据
y_data = [2.0, 4.0, 6.0]  # 因变量数据

# 定义前向传播函数
def forward(x):
    # 计算预测值，这里使用全局变量w
    return x * w  # 返回输入x与权重w的乘积

# 定义损失函数
def loss(x, y):
    # 计算预测值
    y_pred = forward(x)  # 调用前向传播函数获取预测值
    # 返回均方误差
    return (y_pred - y) ** 2  # 返回预测值与真实值的平方差

# 穷举法来寻找最佳的权重w
w_list = []  # 存储不同权重的列表
mse_list = []  # 存储对应的均方误差列表

# 在0到4之间以0.1为步长遍历所有可能的权重w
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)  # 打印当前的权重值
    l_sum = 0  # 初始化损失总和
    # 遍历每一对输入输出数据
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)  # 计算当前输入的预测值
        loss_val = loss(x_val, y_val)  # 计算当前输入的损失值
        l_sum += loss_val  # 累加损失值
        # 打印当前输入、真实输出、预测输出和损失值
        print('\t', x_val, y_val, y_pred_val, loss_val)
    # 打印当前权重下的均方误差
    print('MSE=', l_sum / 3)  # 计算并打印均方误差
    w_list.append(w)  # 将当前权重添加到权重列表中
    mse_list.append(l_sum / 3)  # 将当前均方误差添加到均方误差列表中

# 绘制权重与均方误差的关系图
plt.plot(w_list, mse_list)  # 绘制权重与均方误差的曲线
plt.ylabel('Loss')  # 设置y轴标签
plt.xlabel('w')  # 设置x轴标签
plt.show()  # 显示图形
