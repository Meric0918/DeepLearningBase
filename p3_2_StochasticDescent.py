'''
# @Collector and Speaker : little snow
# @Owner : 考研数学计算机之路 2024-2025 项目班


相关前备知识：
    注意：损失函数由cost()更改为loss()
代码重要解读：
    随机梯度: 即每次拿一个训练数据来训练，然后更新梯度参数。
    当前 p3_2 随机梯度算法中梯度总共更新100(epoch)x3 = 300次。
    而   p3_1 梯度下降法中梯度总共更新100(epoch)次。
'''

# 定义自变量数据，x_data是输入特征
x_data = [1.0, 2.0, 3.0]
# 定义因变量数据，y_data是目标输出
y_data = [2.0, 4.0, 6.0]

# 初始化权重w，初始值为1.0
w = 1.0

# 定义前向传播函数
def forward(x):
    # 计算预测值，返回输入x与权重w的乘积
    return x * w

# 定义损失函数
def loss(x, y):
    # 计算当前输入x的预测值
    y_pred = forward(x)
    # 返回预测值与真实值之间的平方差
    return (y_pred - y) ** 2

# 定义梯度计算函数
def gradient(x, y):
    # 计算当前输入x的梯度，使用损失函数对权重w的导数
    return 2 * x * (x * w - y)

# 打印训练前的预测结果，预测输入4的输出
print('Predict (before training)', 4, forward(4))

# 进行100个epoch的训练
for epoch in range(100):
    # 遍历每一对输入输出数据
    for x, y in zip(x_data, y_data):
        # 计算当前输入的梯度
        grad = gradient(x, y)
        # 更新权重w，使用学习率0.01
        w = w - 0.01 * grad
        # 打印当前输入、真实输出和计算得到的梯度
        print("\tgrad: ", x, y, grad)
        # 计算当前输入的损失值
        l = loss(x, y)

    # 打印当前epoch的进度、权重和损失值
    print("progress:", epoch, "w=", w, "loss=", l)

# 打印训练后的预测结果，预测输入4的输出
print('Predict (after training)', 4, forward(4))
