                   ###LSTM 预测例子###

import tensorflow as tf
import numpy as np
import pandas as pd
import sys

roundT=100#训练轮次数
learnRateT=0.001#学习率

##命令行参数
argt=sys.argv[1:]
print('argt:%s'%argt)

for v in argt:
    if v.startswith('-round='):
        roundT=int(v[len('-round'):])
    if v.startswith('-learnrate='):
        learnRateT=float(v[len('-learnrate='):])


fileData=pd.read_csv('exchangeData.txt',dtype=np.float32,header=None)
wholeData=np.reshape(fileData.as_matrix(),(-1))#因为read_csv函数读取的数据用as_matrix函数转换后为二维数组，因此调用numpy的
#reshape函数把它转换为一维数组。“-1”表示把以前的二维矩阵“拉平”

print('wholeData:%s'%wholeData)

cellCount=3#代表LSTM层中结构元的数量，因为输入数据准备按3天为1批
unitCount=5#每个结构元中的神经元节点数量

testData=wholeData[-cellCount:]#取wholeData中的最后3项数据，并形成一个一维数组，作为测试数据
print('testData:%s\n'%testData)

rowCount=wholeData.shape[0] - cellCount #数据行的总数-cellCount值
print('rowCount:%d\n'%rowCount)

xData=[wholeData[i:i+cellCount] for i in range(rowCount)]#一个二维数组，每行有三项，[n,3]的二维数组，n为总的训练数据条数
yTrainData=[wholeData[i+cellCount] for i in range(rowCount)]#一维数组

print('xData:%s\n'%xData)
print('yTrainData:%s\n'%yTrainData)

x=tf.placeholder(shape=[cellCount],dtype=tf.float32)
yTrain=tf.placeholder(dtype=tf.float32)

cellT=tf.nn.rnn_cell.BasicLSTMCell(unitCount)#定义一个LSTM结构元，并指定其中的神经元节点数为unitCount个

initState=cellT.zero_state(1,dtype=tf.float32)#指定结构元的初始状态为零状态，第一个参数1是代表批次，一批只处理一组数据

#h==>>[1,cellCount,unitCount]
h,finalState=tf.nn.dynamic_rnn(cellT,tf.reshape(x,[1,cellCount,1]),initial_state=initState,dtype=tf.float32)

hr=tf.reshape(h,[cellCount,unitCount])#3*5

w2=tf.Variable(tf.random_normal([unitCount,1]),dtype=tf.float32)#5*1
b2=tf.Variable(0.0,dtype=tf.float32)

y=tf.reduce_sum(tf.matmul(hr,w2)+b2)

loss=tf.abs(y-yTrain)

optimizer=tf.train.RMSPropOptimizer(learnRateT)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for i in range(roundT):
    lossSum=0.0
    for j in range(rowCount):
        result=sess.run([train,x,yTrain,y,h,finalState,loss],feed_dict={x:xData[j],yTrain:yTrainData[j]})

        lossSum=lossSum+float(result[len(result)-1])
        if j==(rowCount-1):
            print('i:%d,x:%s,yTrain:%s,y:%s,h:%s,finalState:%s,loss:%s,avgLoss:%10.10f\n'%(i,result[1],result[2],result[3],result[4],result[5],result[6],(lossSum/rowCount)))


result=sess.run([x,y],feed_dict={x:testData})

print('x:%s,y:%s\n'%(result[0],result[1]))





