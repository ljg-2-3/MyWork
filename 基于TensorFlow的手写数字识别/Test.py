import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# 对测试数据进行数据处理
test_x = test_x.reshape(10000, 28 * 28) / 255.0
test_y = tf.one_hot(test_y, 10)

# 加载训练好的模型
model = tf.keras.models.load_model('model/my_model.h5')


# 模型准确率
j = 0
predict = model.predict(test_x[:10000])
for i in range(10000):
    p = test_x[i].reshape(28, 28)
    if tf.argmax(test_y[i]) == tf.argmax(predict[i]):
        j += 1
print("模型准确率为: ", j / 10000)


# 随机抽取一张图片查看识别结果
# predict = model.predict(test_x[:10000])
# 随机产生一个1到60000之间的整数，作为样本的索引值
num = np.random.randint(1, 10000)
p = test_x[num].reshape(28, 28)
plt.imshow(p, cmap='gray')
plt.axis('off')
plt.title("Real result: " + str(tf.argmax(test_y[num]).numpy()) + '\n'
          + "Prediction result: " + str(tf.argmax(predict[num]).numpy()))
plt.show()


# 查看预测错误的图片和预测结果
a = []
# predict = model.predict(test_x[:10000])
for i in range(10000):
    p = test_x[i].reshape(28, 28)
    if tf.argmax(test_y[i]) != tf.argmax(predict[i]):
        a.append(i)

for i in range(5):
    n = np.random.randint(0, len(a))
    p = test_x[a[n]].reshape(28, 28)
    plt.imshow(p, cmap='gray')
    plt.axis('off')
    plt.title("id: " + str(a[n]) + '\n'
              + "Real result: " + str(tf.argmax(test_y[a[n]]).numpy()) + '\n'
              + "Prediction result: " + str(tf.argmax(predict[a[n]]).numpy()))
    plt.show()
