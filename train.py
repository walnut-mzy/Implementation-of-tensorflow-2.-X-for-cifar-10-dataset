import tensorflow as tf
from tensorflow import optimizers,metrics,losses
import settings
import numpy as np
import os
import utils1
from tqdm import tqdm
from model import model
# 使用Adam优化器
optimizer = tf.keras.optimizers.Adam(settings.LEARNING_RATE)
#使用交叉熵函数
loss=tf.keras.losses.categorical_crossentropy
acc_meter = metrics.CategoricalAccuracy()
acc_meter1=metrics.CategoricalAccuracy()

# 使用tf.function加速训练
@tf.function
def train_one_step(x,y):
    """
    一次迭代过程
    """
    # 求loss
    with tf.GradientTape() as tape:
        predictions = model(x)
        acc_meter.update_state(y_true=y, y_pred=predictions)
        loss1 = loss(y_true=y,y_pred=predictions)
    # 求梯度
    grad = tape.gradient(loss1,  model.trainable_variables)
    # 梯度下降，更新噪声图片
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return loss1
@tf.function
def test_acc():
    for x1, y1 in test_dataset:
        pred = model(x1)  # 前向计算
        acc_meter1.update_state(y_true=y1, y_pred=pred)  # 更新准确率统计
    print("测试集正确率为：",acc_meter1.result())
    acc_meter1.reset_states()
if __name__ == '__main__':
    dataset=utils1.dataest
    test_dataset=utils1.test_dataest
    # 创建保存生成模型的文件夹
    if not os.path.exists(settings.OUTPUT_DIR):
        os.mkdir(settings.OUTPUT_DIR)

    # 共训练settings.EPOCHS个epochs
    for epoch in range(settings.EPOCHS):
        # 使用tqdm提示训练进度
        with tqdm(total=settings.size/settings.buffer_size, desc='Epoch {}/{}'.format(epoch, settings.EPOCHS)) as pbar:
            # 每个epoch训练settings.STEPS_PER_EPOCH次
            for x,y in dataset:
                loss2=train_one_step(x,y)
                pbar.set_postfix(loss= '%.4f' % float(loss2[len(loss2)-1]),acc=float(acc_meter.result()))
                pbar.update(1)
            test_acc()
            # 每个epoch保存一次图片
    model.save(settings.OUTPUT_DIR+"/"+"model.h5")