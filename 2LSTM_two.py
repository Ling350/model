            ####LSTM预测例子时，每天的数据是一个二维向量####
import tensorflow as tf
import numpy as np
import pandas as pd
import sys

roundT=1000   #训练轮次数
learnRateT=0.1#学习率

#命令行参数
argt=sys.argv[1:]
print('argt:%s'%argt)

for v in argt:
    if v.startswith('-round='):
        roundT=int(v[len('-round='):])
    if v.startswith('-learnrate='):
        learnRateT=float(v[len('-learnrate='):])

fileData=pd.read_csv('exchangeData2.txt',dtype=np.float32,header=None)
wholeData=np.reshape(fileData.as_matrix(),(-1,2))#2列

print('wholeData:%s'%wholeData)#获取整个数据

cellCount=3#三个时刻为一批
unitCount=5 #每个结构元有5个神经元节点

testData=wholeData[-cellCount-1:-1]
print('testData:%s\n'%testData)   #二维数组

rowCount=wholeData.shape[0]-cellCount#行-cellCount，29
print('rowCount:%d\n'%rowCount)

xData=[wholeData[i:i+cellCount]for i in range(rowCount)]
yTrainData=[wholeData[i+cellCount][0]for i in range(rowCount)]

print('xData:%s'%xData)
print('yTrainData:%s'%yTrainData)

x=tf.placeholder(shape=[cellCount,2],dtype=tf.float32)
ytrain=tf.placeholder(dtype=tf.float32)

cellT=tf.nn.rnn_cell.BasicLSTMCell(unitCount)#定义一个LSTM结构单元，并指定神经元节点数量

initState=cellT.zero_state(1,dtype=tf.float32)

h,finalState=tf.nn.dynamic_rnn(cellT,tf.reshape(x,[1,cellCount,2]),initial_state=initState,dtype=tf.float32)

hr=tf.reshape(h,[cellCount,unitCount])

w2=tf.Variable(tf.random_normal([unitCount,1]),dtype=tf.float32)

b2=tf.Variable(tf.random_normal([1]),dtype=tf.float32)

y=tf.reduce_sum(tf.matmul(hr,w2)+b2)#行维度求和

loss=tf.reduce_mean(tf.square(y-ytrain))

optimizer=tf.train.RMSPropOptimizer(learnRateT)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for i in range(roundT):
    lossSum=0.0
    for j in range(rowCount):
        result = sess.run([train, x, ytrain, y, h, finalState, loss], feed_dict={x: xData[j], ytrain: yTrainData[j]})

        lossSum = lossSum + float(result[len(result) - 1])
        if j == (rowCount - 1):
            print('i:%d,x:%s,yTrain:%s,y:%s,h:%s,finalState:%s,loss:%s,avgLoss:%10.10f\n' % (
            i, result[1], result[2], result[3], result[4], result[5], result[6], (lossSum / rowCount)))

result = sess.run([x, y], feed_dict={x: testData})

print('x:%s,y:%s\n' % (result[0], result[1]))



