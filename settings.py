#设置训练集参数(size<50000)
#buffer_size大小
buffer_size=1000
#数据集信息重复多少次
repeat=10
#batch大小
batch_size=500
#数据集大小
size=20000

#设置测试集参数（size1<10000）
buffer_size1=1000
repeat1=10
batch_size1=500
size1=1000

#图片信息
height=32
width=32
channel=3

LEARNING_RATE=0.0005

OUTPUT_DIR="./model"

#train
EPOCHS=10