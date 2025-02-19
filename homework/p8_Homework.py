import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

'''
# （选做作业题）这是Kaggle 泰坦尼克号机器学习竞赛:
# 要求是使用机器学习创建一个模型，预测哪些乘客在泰坦尼克号沉船中幸存下来。
'''

class TitanicDataset(Dataset):
    def __init__(self, filepath, scaler=None, is_train=True):
        super(TitanicDataset, self).__init__()

        # 初始化函数，读取 CSV 文件
        self.dataframe = pd.read_csv(filepath)
        self.scaler = scaler
        # 调用预处理函数来处理 DataFrame
        self.preprocess(self.dataframe, is_train)

    def preprocess(self, df, is_train):
        # 移除不需要的类别
        df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        
        # 处理缺失值
        df['Age'].fillna(df['Age'].mean(), inplace=True)                # Age 缺失的值用平均值来填充
        df['Fare'].fillna(df['Fare'].mean(), inplace=True)              # Fare 缺失的值用平均值来填充
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)   # Embarked 缺失的值用众值来填充

        # 使用 LabelEncoder 来转换性别和登船口为数值形式
        # LabelEncoder 适用于将文本标签转换为一个范围从 0 到 n_classes-1 的数值。这种方法适用于转换具有顺序性的分类特征。例如“低”，“中”，“高”。
        label_encoder = LabelEncoder()
        df['Sex'] = label_encoder.fit_transform(df['Sex'])
        df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

        # 与 LabelEncoder 不同，One-Hot 编码 创建了一个二进制列来表示每个类别，没有数值的大小意义。当分类特征的不同类别之间没有顺序或等级的概念时，通常使用独热编码。
        # 注意：要使用 One-Hot的话，input_features=10
        # df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

        if is_train:
            # 如果是训练集，创建新的 StandardScaler，并进行 fit_transform, 来标准化 'Age' 和 'Fare' 列的数值
            # 如果特征的数值范围差异很大，那么算法可能会因为较大范围的特征而受到偏向，导致模型性能不佳。
            self.scaler = StandardScaler()
            df[['Age', 'Fare']] = self.scaler.fit_transform(df[['Age', 'Fare']])

            # 如果是训练数据，将 'Survived' 列作为标签
            self.labels = df['Survived'].values
            self.features = df.drop('Survived', axis=1).values

        else:
            # 如果是测试集，使用传入的 scaler 进行 transform
            df[['Age', 'Fare']] = self.scaler.transform(df[['Age', 'Fare']])

            # 对于测试数据，可能没有 'Survived' 列，因此特征就是整个 DataFrame
            self.features = df.values
            self.labels = None      # 标签设置为 None


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # 获取单个样本，包括特征和标签（如果有的话）
        # 如果有标签，同时返回特征和标签
        if self.labels is not None:
            return torch.tensor(self.features[index], dtype=torch.float), torch.tensor(self.labels[index], dtype=torch.float)
        # 对于没有标签的测试数据，返回一个占位符张量，例如大小为 1 的零张量
        else:
            return torch.tensor(self.features[index], dtype=torch.float), torch.zeros(1, dtype=torch.float)


# 自定义 二分类模型
class BinaryClassificationModel(torch.nn.Module):
    def __init__(self, input_features):
        super(BinaryClassificationModel, self).__init__()
        self.linear1 = torch.nn.Linear(input_features, 64)
        self.linear2 = torch.nn.Linear(64, 64)
        self.linear3 = torch.nn.Linear(64, 1)       

        # 定义 dropout 层，可以减少过拟合
        self.dropout = torch.nn.Dropout(p=0.1)

        # 定义 batchnorm层，帮助稳定学习过程
        self.batchnorm1 = torch.nn.BatchNorm1d(64)
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))     # 第一层激活函数为 ReLU
        x = self.batchnorm1(x)          # 应用 batch normalization
        x = self.dropout(x)             # 应用 dropout

        x = F.relu(self.linear2(x))     # 第二层激活函数为 ReLU
        x = self.batchnorm2(x)          # 应用 batch normalization
        x = self.dropout(x)             # 应用 dropout

        x = self.linear3(x)             # 输出层
        return torch.sigmoid(x)         # 应用 sigmoid 激活函数

# 训练过程
def train(models, train_loader, criterion, optimizers, num_epochs):
    epoch_losses = {k: [] for k in optimizers.keys()}

    print('start training')

    for optim_name, optimizer in optimizers.items():
        model = models[optim_name]
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()                       # 梯度清零
                outputs = model(inputs)                     # 前向传播
                loss = criterion(outputs.squeeze(), labels) # 使用 squeeze 调整输出形状
                loss.backward()                             # 反向传播
                optimizer.step()                            # 更新权重
                # 乘以 inputs.size(0) 的目的是为了累积整个批次的总损失，而不仅仅是单个数据点的平均损失。
                # 调用 loss = criterion(outputs, labels) 时，计算的是当前批次中所有样本的平均损失。
                # 为了得到整个训练集上的总损失，我们需要将每个批次的平均损失乘以该批次中的样本数（inputs.size(0)）。
                # 这样做可以确保每个样本，无论它们属于哪个批次，对总损失的贡献都是平等的。
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}')
            epoch_losses[optim_name].append(epoch_loss)
    return epoch_losses

# 测试
def test(model, test_loader, optimizers):
    results = {}
    for optim_name, _ in optimizers.items():
        model = models[optim_name]
        model.eval()
        
        predictions = []
        with torch.no_grad():   # 不计算梯度，减少计算和内存消耗
            for inputs, _ in test_loader:
                outputs = model(inputs)
                # test没有标签，只输出结果
                predicted = (outputs > 0.5).float().squeeze()
                predictions.extend(predicted.tolist())  # 使用 extend 和 tolist 将 predicted 中的每个元素添加到 predictions
        print("Predict result: ", predictions)
        results[optim_name] = predictions
    return  results


# 加载数据
# 训练数据集，没有传入 scaler，因此会创建一个新的
train_dataset = TitanicDataset('../dataset/titanic/train.csv', scaler=None, is_train=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)

# 测试数据集，传入从训练数据集得到的 scaler
test_dataset = TitanicDataset('../dataset/titanic/test.csv', scaler=train_dataset.scaler, is_train=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)

# 实例化模型，输入特征数量为10: Pclass Name Sex Age SibSp Parch Ticket Fare Cabin Embarked
# 但是注意，预处理之后，只采用了7个: Pclass Sex Age SibSp Parch Fare Embarked
models = {
    'Adam': BinaryClassificationModel(input_features=7),
    'SGD': BinaryClassificationModel(input_features=7),
    }

# 定义损失函数，优化器
criterion = torch.nn.BCELoss(reduction='mean')
optimizers = {
    'Adam': torch.optim.Adam(models['Adam'].parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001),
    'SGD': torch.optim.SGD(models['SGD'].parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)
}


# 训练模型
num_epochs = 100
losses = train(models, train_loader, criterion, optimizers, num_epochs)

# 测试模型
# 已知test的结果保存在 gender_submission.csv 文件中，获取准确的 labels 和 predicted 结果算精度
labels_path = '../dataset/titanic/gender_submission.csv'
data_frame = pd.read_csv(labels_path)
data_frame.drop(['PassengerId'], axis=1, inplace=True)
labels = data_frame['Survived'].values
print('Test Dataset 正确结果: ', labels)

# 模型预测结果
results = test(models, test_loader, optimizers)
print('Test Dataset 预测结果: ', results)

# 精度计算
for optimizer_name, predicted in results.items():
    accuracy = 100 * (predicted == labels).sum() / len(predicted)
    print(f'Accuracy for {optimizer_name}: {accuracy:.2f}%')


plt.figure(figsize=(10, 5))
for optim_name, losses in losses.items():
    plt.plot(losses, label=optim_name)
    final_accuracy = 100 * (results[optim_name] == labels).sum() / len(results[optim_name])
    plt.annotate(f'Final Acc: {final_accuracy:.2f}%', xy=(num_epochs - 1, losses[-1]), xytext=(-40, 10), textcoords='offset points', fontsize=10)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Titanic Dataset training Loss Curve')
plt.legend()
plt.savefig('p8_titanic_training_loss_curve.png')  # 保存图片
plt.show()
