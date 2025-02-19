import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler        # pip install scikit-learn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

'''
# （选做作业题）这是Kaggle Otto集团产品分类挑战赛，旨在通过机器学习模型对Otto集团的产品进行分类。
'''

class OttoDataset(Dataset):
    def __init__(self, feature_filepath, label_filepath=None, mode='train', scaler=None):
        super(OttoDataset, self).__init__()

        # 将数据集加载到pandas数据框中。
        data = pd.read_csv(feature_filepath)

        if mode == 'train':
            # 提取类标签的数字部分，转换为整数，并转换为零基索引。
            self.labels = torch.tensor(data.iloc[:, -1].apply(lambda x: int(x.split('_')[-1]) - 1).values, dtype=torch.long)
            
            # 初始化StandardScaler。
            # StandardScaler将通过减去均值并除以标准差来标准化特征（即数据集的每一列）。
            # 这将特征列中心化到均值为0，标准差为1。
            self.scaler = StandardScaler()

            # 选择除'id'和'target'之外的所有列作为特征。
            # 然后应用scaler进行标准化。
            features = data.iloc[:, 1:-1].values
            self.features = torch.tensor(self.scaler.fit_transform(features), dtype=torch.float32)

        elif mode == 'test':
            features = data.iloc[:, 1:].values

            # 对测试集特征应用与训练集相同的缩放。使用self.scaler.transform
            self.scaler = scaler if scaler is not None else StandardScaler()
            self.features = torch.tensor(self.scaler.transform(features), dtype=torch.float32)
            
            if label_filepath is not None:
                label_data = pd.read_csv(label_filepath)
                # 假设'id'之后的第一列是独热编码的类标签，
                # 找到每行中最大值的索引，这对应于预测的类。
                self.labels = torch.tensor(label_data.iloc[:, 1:].values.argmax(axis=1), dtype=torch.long)

            else:
                self.labels = None

        # 如果未指定'train'或'test'模式，则引发错误。
        else:
            raise ValueError("模式必须是'train'或'test'")
        
        # 存储数据集的长度。
        self.len = len(self.features)

    def __len__(self):
        # 当调用len(dataset)时，返回数据集的长度。
        return self.len
    
    def __getitem__(self, index):
        # 此方法检索指定索引的特征和标签。
        return self.features[index], self.labels[index] if self.labels is not None else -1
    

class FullyConnectedModel(torch.nn.Module):
    def __init__(self, input_features, output_classes):
        super(FullyConnectedModel, self).__init__()
        
        # 定义网络层
        self.fc1 = torch.nn.Linear(input_features, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, output_classes)

        # 可以选择增加更多的层

        # 定义dropout层，可以减少过拟合
        self.dropout = torch.nn.Dropout(p=0.3)

        # 定义batchnorm层，帮助稳定学习过程
        self.batchnorm1 = torch.nn.BatchNorm1d(128)
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        self.batchnorm3 = torch.nn.BatchNorm1d(32)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
    

def train(epoch, train_loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 300 == 0:
            print('Epoch:[{}/{}], Loss:{:.4f}'.format(epoch, batch_idx, running_loss/300))

    # 计算平均损失
    average_loss = running_loss / len(train_loader)
    return average_loss


def test(test_loader, model):
    model.eval()
    correct = 0.0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * (correct / total)
    print("测试数据的准确率为 {:.2f}".format(accuracy))
    return accuracy



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 准备数据集
    train_dataset = OttoDataset(feature_filepath='../dataset/p9_homework_Otto/train.csv', mode='train')
    scaler = train_dataset.scaler
    test_dataset = OttoDataset(feature_filepath='../dataset/p9_homework_Otto/test.csv', label_filepath='../dataset/p9_homework_Otto/otto_correct_submission.csv', mode='test', scaler=scaler)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 设计模型
    model = FullyConnectedModel(input_features=93, output_classes=9).to(device)

    # 构建损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    
    # 训练和测试
    train_losses = []
    test_accuracies = []

    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train(epoch, train_loader, model, criterion, optimizer)
        train_losses.append(train_loss)
        
        if epoch % 2 == 0 or epoch == num_epochs-1:
            test_accuracy = test(test_loader, model)
            test_accuracies.append(test_accuracy)


        # 更新学习率
        scheduler.step()
        
    # 保存模型参数以备将来使用
    # torch.save(model.state_dict(), 'model/09_kaggle_OttoDataset_model.pth')

    # 可视化
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(0, 101, 2), test_accuracies, label='测试准确率')  # 调整x轴以适应测试准确率
    plt.title('测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')  
    plt.legend()

    plt.savefig('picture/p9_training_and_accuracy_plot.png')  # 保存图片到picture路径下
    plt.show()
