import torch

'''
# @Collector and Speaker : little snow
# @Owner : 考研数学计算机之路 2024-2025 项目班

# 代码重点解读：
    代码行 line:36、line:58、line:78 中的 (x) 隐藏着Pytorch的关键知识：
    torch.nn.Module 类的 __call__ 方法被重载了。具体来说，当你调用模型实例（例如 model(x_data)）时，
    实际上是调用了 torch.nn.Module 中的 __call__ 方法，而这个方法会自动处理一些额外的功能，比如：
    1. 前向传播：__call__ 方法实际会调用 model.forward 方法，这就是为什么你可以直接用 model(x) 来进行预测。
    2. 注册钩子：__call__ 方法还会处理一些与模型相关的钩子（hooks），这些钩子可以在前向传播或反向传播时执行自定义操作。
    3. 输入验证：它会验证输入的形状和类型，以确保它们符合模型的要求。
    因此，model(x_data) 实际上是调用了 __call__ 方法，进而调用了 model.forward 方法，返回预测值 y_pred。
    
    详情见：https://blog.csdn.net/xxboy61/article/details/88101192
'''

# 定义输入数据 x_data 和目标输出数据 y_data
# x_data 是一个包含三个样本的张量，每个样本是一个一维特征
# y_data 是对应的目标输出，表示线性关系 y = 2 * x
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# 定义线性模型类，继承自 torch.nn.Module
class LinearModel(torch.nn.Module):
    def __init__(self):
        # 调用父类构造函数
        super(LinearModel, self).__init__()
        # 定义一个线性层，输入特征维度为1，输出特征维度也为1
        self.linear = torch.nn.Linear(1, 1)

    # 前向传播函数
    def forward(self, x):
        # 通过线性层计算预测值
        y_pred = self.linear(x)
        return y_pred

# 实例化线性模型
model = LinearModel()

# 定义损失函数为均方误差损失
# size_average=False 表示返回所有样本的总损失，而不是平均损失
criterion = torch.nn.MSELoss(size_average=False)

# 定义优化器为随机梯度下降（SGD），学习率为0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 进行1000次训练迭代
for epoch in range(1000):
    '''每一次epoch的训练过程，总结就是
        1、前向传播，求y hat （输入的预测值）
        2、根据y_hat和y_label(y_data)计算loss
        3、反向传播 backward (计算梯度)
        4、根据梯度，更新参数
    '''
    # 通过模型计算预测值
    y_pred = model(x_data)
    # 计算损失
    loss = criterion(y_pred, y_data)
    # 打印当前迭代次数和损失值
    print(epoch, loss.item())

    # 清零梯度
    optimizer.zero_grad()
    # 反向传播计算梯度
    loss.backward()
    # 更新模型参数
    optimizer.step()

# 打印训练后的模型参数（权重和偏置）
print("w = ", model.linear.weight.item())
print("b =", model.linear.bias.item())

# 测试模型，输入一个新的样本 x_test
x_test = torch.Tensor([[4.0]])
# 通过模型计算预测值
y_test = model(x_test)
# 打印预测结果
print("y_pred = ", y_test.data)
