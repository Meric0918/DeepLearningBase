import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

'''
# @Collector and Speaker : little snow
# @Owner : 考研数学计算机之路 2024-2025 LLM微调实战项目

# 前备知识：
    torch.max使用： https://zhuanlan.zhihu.com/p/583117614

# 代码重点解读：
    这次我们调的库不再是torch.utils.data 而是 from torchvision import datasets，该datasets里面init，getitem,len魔法函数已实现。
    此次我们无需先准备好数据集，datasets.MNIST中download=True 参数指示程序在本地找不到数据集时自动下载它。
'''

# 设置每个批次的大小为64
batch_size = 64

# 定义数据预处理的转换操作
# 将图像转换为张量，并进行标准化处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize((0.1307,), (0.3081,))  # 对图像进行标准化，均值为0.1307，标准差为0.3081
])

# 加载MNIST训练数据集
# root参数指定数据集存储的路径，train=True表示加载训练集，download=True表示如果数据集不存在则自动下载
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)

# 创建训练数据的DataLoader，shuffle=True表示每个epoch打乱数据，batch_size指定每个批次的大小
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# 加载MNIST测试数据集
# train=False表示加载测试集
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)

# 创建测试数据的DataLoader，shuffle=False表示不打乱数据
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# 定义神经网络模型类
class Net(torch.nn.Module):
    def __init__(self):
        # 调用父类构造函数
        super(Net, self).__init__()
        # 定义五个全连接层，输入层784个神经元，输出层10个神经元（对应数字0-9）
        self.l1 = torch.nn.Linear(784, 512)  # 第一层：784 -> 512
        self.l2 = torch.nn.Linear(512, 256)  # 第二层：512 -> 256
        self.l3 = torch.nn.Linear(256, 128)  # 第三层：256 -> 128
        self.l4 = torch.nn.Linear(128, 64)   # 第四层：128 -> 64
        self.l5 = torch.nn.Linear(64, 10)    # 输出层：64 -> 10

    # 定义前向传播函数
    def forward(self, x):
        # 将输入数据展平为一维向量
        x = x.view(-1, 784)
        # 依次通过每一层，并应用ReLU激活函数
        x = F.relu(self.l1(x))  # 第一层
        x = F.relu(self.l2(x))  # 第二层
        x = F.relu(self.l3(x))  # 第三层
        x = F.relu(self.l4(x))  # 第四层
        return self.l5(x)  # 返回输出层的结果

# 实例化神经网络模型
model = Net()

# 定义损失函数为交叉熵损失
criterion = torch.nn.CrossEntropyLoss()

# 定义优化器为随机梯度下降（SGD），学习率为0.01，动量为0.5
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 定义训练函数
def train(epoch):
    running_loss = 0.0  # 初始化当前epoch的损失
    # 遍历训练数据集
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data  # 获取输入数据和目标标签
        optimizer.zero_grad()  # 清零梯度

        outputs = model(inputs)  # 前向传播，计算输出
        loss = criterion(outputs, target)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数

        running_loss += loss.item()  # 累加当前批次的损失
        # 每300个批次打印一次损失
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0  # 重置损失

# 定义测试函数
def test():
    correct = 0  # 初始化正确预测的数量
    total = 0  # 初始化总样本数量
    with torch.no_grad():  # 在测试时不计算梯度
        # 遍历测试数据集
        for data in test_loader:
            images, labels = data  # 获取测试图像和标签
            outputs = model(images)  # 前向传播，计算输出
            _, predicted = torch.max(outputs.data, dim=1)  # 获取预测结果
            total += labels.size(0)  # 更新总样本数量
            correct += (predicted == labels).sum().item()  # 统计正确预测的数量

    # 打印测试集上的准确率
    print('accuracy on test set: %d %% ' % (100*correct/total))

# 主程序入口
if __name__ == '__main__':
    # 进行10个epoch的训练和测试
    for epoch in range(10):
        train(epoch)  # 训练模型
        test()  # 测试模型
