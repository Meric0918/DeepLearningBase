# Numpy
import numpy
# For plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

'''
有w，b两个参数，穷举最小值
这里我们通过穷举法来寻找最优的权重w和偏置b，使得模型的预测值与真实值之间的均方误差最小。
'''

# 输入数据
x_data = [1.0, 2.0, 3.0]  # 自变量
y_data = [2.0, 4.0, 6.0]  # 因变量

# 前向传播函数，计算预测值
def forward(w: numpy.ndarray, b: numpy.ndarray, x: float) -> numpy.ndarray:
    return w * x + b  # 线性模型的预测公式

# 损失函数，计算均方误差
def loss(y_hat: numpy.ndarray, y: float) -> numpy.ndarray:
    return (y_hat - y) ** 2  # 预测值与真实值之间的平方差

# 定义权重和偏置的取值范围
w_cor = numpy.arange(0.0, 4.0, 0.1)  # 权重w的取值范围
b_cor = numpy.arange(-2.0, 2.1, 0.1)  # 偏置b的取值范围

# 输出穷举的权重和偏置的数量
print(f'穷举的权重数量: {len(w_cor)}')  # 打印权重的数量
print(f'穷举的偏置数量: {len(b_cor)}')  # 打印偏置的数量

# 此处直接使用矩阵进行计算
w, b = numpy.meshgrid(w_cor, b_cor)  # 创建权重和偏置的网格
mse = numpy.zeros(w.shape)  # 初始化均方误差矩阵

# 计算每对权重和偏置下的均方误差
for x, y in zip(x_data, y_data):
    _y = forward(w, b, x)  # 计算预测值
    print(len(_y))
    mse += loss(_y, y)  # 累加损失
mse /= len(x_data)  # 计算平均损失
print()

# h = plt.contourf(w, b, mse)  # 可选：绘制等高线图

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', auto_add_to_figure=False)
fig.add_axes(ax)
plt.xlabel(r'w', fontsize=20, color='cyan')  # x轴标签
plt.ylabel(r'b', fontsize=20, color='cyan')  # y轴标签
# 绘制均方误差的三维曲面图
ax.plot_surface(w, b, mse, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()  # 显示图形

'''
得到的三维图形展示了不同权重w和偏置b组合下的均方误差（MSE）。
图中的每一个点代表一个(w, b)组合对应的MSE值，颜色的深浅表示误差的大小。
通过观察图形，我们可以找到使得MSE最小的w和b值，图中最小点不是一个点，是很多最优点组成的线。
'''
