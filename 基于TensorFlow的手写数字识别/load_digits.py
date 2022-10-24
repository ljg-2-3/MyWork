import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 设置超参数
learn_rate = 0.005
batch_size = 256
epoch = 500

# 1.读取数据
# 导入手写数字的数据集
digits = load_digits()  # 加载数据集
# 划分数据集为训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42)  # 切分数据

# 标签编码，变成二进制形式
train_y = tf.one_hot(train_y, depth=10)
test_y = tf.one_hot(test_y, depth=10)

# 3.设计神经网络模型
model = tf.keras.models.Sequential()
# 输入层
model.add(tf.keras.layers.Input(shape=train_x.shape[1:]))
# 全连接层，输出最后一维维度为300，激活函数为relu
model.add(tf.keras.layers.Dense(300, activation='relu'))
# 让神经元以一定的概率停止工作，提高模型的泛化能力，防止过拟合
model.add(tf.keras.layers.Dropout(0.2))
# 全连接层，激活函数为relu
model.add(tf.keras.layers.Dense(100, activation='relu'))
# 让神经元以一定的概率停止工作，提高模型的泛化能力，防止过拟合
model.add(tf.keras.layers.Dropout(0.3))
# 全连接层，激活函数为softmax
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# 打印神经网络结构，统计参数数目
model.summary()

# 4.模型超参数设定
# 使用model.compile()方法来配置训练方法
model.compile(
    optimizer=tf.keras.optimizers.SGD(0.005),  # 优化器
    loss=tf.keras.losses.categorical_crossentropy,  # 损失函数
    metrics='accuracy')  # 准确率

# 5.模型训练
model.fit(
    x=train_x,  # 图片数据
    y=train_y,  # 标签
    batch_size=batch_size,  # 指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
    epochs=epoch,  # 迭代次数epochs
    validation_data=(test_x, test_y)  # 训练集预测精度
)

# 保存模型
model.save('./my_model-1.h5')
