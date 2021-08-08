import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import settings
#导入数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 归一化处理
train_images, test_images = train_images / 255.0, test_images / 255.0

#将标签进行one_hot编码
train_labels=tf.one_hot(train_labels,depth=10)
train_labels=tf.squeeze(train_labels,axis=1)
test_labels=tf.one_hot(test_labels,depth=10)

#设置训练集大小
train_images=train_images[:settings.size]
train_labels=train_labels[:settings.size]

#设置测试集大小
test_images=test_images[:settings.size1]
test_labels=test_labels[:settings.size1]

def plt_show(num):
    """

    :param num: 表示第几张图片
    :return:
    """
    plt.imshow(train_images[num-1])
    plt.show()

#测试集构建
dataest = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataest = dataest.shuffle(buffer_size=settings.buffer_size).prefetch(tf.data.experimental.AUTOTUNE).repeat(settings.repeat).batch(settings.batch_size)
print(dataest)
#训练集构建
test_dataest = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataest = test_dataest.shuffle(buffer_size=settings.buffer_size1).prefetch(tf.data.experimental.AUTOTUNE).repeat(settings.repeat1).batch(settings.buffer_size1)

