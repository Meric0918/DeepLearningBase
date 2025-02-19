import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

'''
# @Collector and Speaker : little snow
# @Owner : 考研数学计算机之路 2024-2025 LLM微调实战项目

# 前备知识：
    with torch.no_grad()使用： https://zhuanlan.zhihu.com/p/673420509
    
# 代码重点解读：
    不要忘了将数据集 diabetes.csv 跟脚本放在同一级目录，line:24 控制读取数据集
    DataLoader对数据集先打乱(shuffle)，然后划分成mini_batch，增强训练稳定性和模型泛化性
    训练中每一轮epoch中迭代次数 = len / batch_size
    minibatch体现在 line:101 inputs, labels = data中：
        inputs的shape是[32,8],labels 的shape是[32,1]
'''

# 读取原始数据，并划分训练集和测试集
# 使用numpy的loadtxt函数从'diabetes.csv'文件中加载数据，数据以逗号为分隔符，数据类型为32位浮点数
raw_data = np.loadtxt('dataset/diabetes.csv', delimiter=',', dtype=np.float32)

# 将数据的所有列（除了最后一列）作为特征X
X = raw_data[:, :-1]
# 将数据的最后一列作为目标输出y
y = raw_data[:, [-1]]

# 使用train_test_split函数将数据集划分为训练集和测试集，测试集占30%
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3)

# 将测试集转换为PyTorch张量
Xtest = torch.from_numpy(Xtest)
Ytest = torch.from_numpy(Ytest)

# 将训练数据集进行批量处理
# 准备数据集类，继承自torch.utils.data.Dataset
class DiabetesDataset(Dataset):
    '''
    继承DataSet的类需要重写__init__，__getitem__，__len__魔法函数
        这些函数的任务是：加载数据集，获取数据索引，获取数据总量。
    '''
    def __init__(self, data, label):
        # 初始化数据集，记录数据的行数
        self.len = data.shape[0]  # shape(多少行，多少列)
        # 将输入数据和标签转换为PyTorch张量
        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(label)

    def __getitem__(self, index):
        # 根据索引返回输入数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回数据集的总行数
        return self.len

# 实例化训练数据集
train_dataset = DiabetesDataset(Xtrain, Ytrain)

# 创建数据加载器，设置批量大小为32，shuffle=True表示在每个epoch开始时打乱数据
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)  # num_workers 多线程

# 设计模型类，继承自torch.nn.Module
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义四个线性层，分别为 8 -> 6, 6 -> 4, 4 -> 2, 2 -> 1 的变换
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        # 定义Sigmoid激活函数
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # 前向传播过程，依次通过每一层并应用Sigmoid激活函数
        x = self.sigmoid(self.linear1(x))  # 第一层输出
        x = self.sigmoid(self.linear2(x))  # 第二层输出
        x = self.sigmoid(self.linear3(x))  # 第三层输出  
        x = self.sigmoid(self.linear4(x))  # 第四层输出，即预测值 y_hat
        return x  # 返回最终输出

# 实例化模型
model = Model()

# 构建损失函数和优化器
# 使用二元交叉熵损失函数，reduction='mean'表示返回平均损失
criterion = torch.nn.BCELoss(reduction='mean')
# 使用随机梯度下降（SGD）作为优化器，学习率为0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练循环，进行前向传播、反向传播和参数更新
def train(epoch):
    train_loss = 0.0  # 初始化训练损失
    count = 0  # 初始化计数器
    # 遍历训练数据加载器
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data  # 获取输入数据和标签
        y_pred = model(inputs)  # 前向传播，计算预测值

        loss = criterion(y_pred, labels)  # 计算损失

        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        train_loss += loss.item()  # 累加损失
        count = i  # 更新计数器

    # 每2000个epoch打印一次训练损失
    if epoch % 2000 == 1999:
        print("train loss:", train_loss/count, end=',')

# 测试函数
def test():
    with torch.no_grad():  # 在测试时不计算梯度
        y_pred = model(Xtest)  # 通过模型计算预测值
        # 将预测值转换为标签，阈值为0.5
        y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
        # 计算准确率
        acc = torch.eq(y_pred_label, Ytest).sum().item() / Ytest.size(0)
        print("test acc:", acc)  # 打印测试准确率

# 主程序入口
if __name__ == '__main__':
    # 进行50000个epoch的训练
    for epoch in range(50000):
        train(epoch)  # 训练模型
        # 每2000个epoch进行一次测试
        if epoch % 2000 == 1999:
            test()
