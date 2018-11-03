from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
%matplotlib inline
# --------------------------------------------------
# 加载游戏环境
# 你可以自行调整游戏难度（网格大小），小的网格可以使网络训练更快，大的网格可以提升游戏难度
from gridworld import gameEnv
env = gameEnv(partial=False, size=5)
# --------------------------------------------------
# 实现网络
class Qnetwork():
    def __init__(self, h_size):
        # 网络接收到游戏传递出的一帧图像并将之转化为数组
        # 之后调整大小并通过四个卷积层
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[7,7], stride=[1,1], padding='VALID', biases_initializer=None)
        # 取得最后一层卷积层的输出进行拆分，分别计算价值与决策
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        #参考API介绍 可以知道 这里沿着shape里面的channel进行被分为了2部分
        #原先的channel或者说卷积核个数为512 这里分为两部分也就是h_size//2
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2, env.actions]))
        self.VW = tf.Variable(xavier_init([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        
        # 分开再合并 Q=V+1/num_A*sum(A) 综合得到最终的Q值
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)
        
        # 将目标Q值和预测Q值作差平方和作为损失值
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
# --------------------------------------------------
# 历程重现
# 这个类赋予了网络存储、重采样来进行训练的能力
class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
            #清0前面的buffer_size个元素
        self.buffer.extend(experience)
    
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])
         #采样size个，并变形成我们需要的shape
# --------------------------------------------------
# 用于处理游戏返回帧的函数
def processState(states):
    return np.reshape(states, [21168])
# --------------------------------------------------
# 利用主网络参数更新目标网络
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0: total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
# --------------------------------------------------
batch_size = 32 #每次训练使用多少训练记录
update_freq = 4 # 多久执行一次训练操作
y = .99 # Q 值的折算因子
startE = 1 # 随机行动的初始概率
endE = 0.1 # 随机行动的最低概率
annealing_steps = 10000. # startE衰减至endE所需的步骤数
num_episodes = 10000 # 网络在游戏环境下训练的episodes数
pre_train_steps = 10000 # 训练开始前允许的随机行动次数
max_epLength = 50 # episode的最大允许值
load_model = False # 是否载入保存的模型
path = "./dqn" # 我们模型的保存路径
h_size = 512 # 最后一个卷积层的尺寸
tau = 0.001 # 目标网络更新至主网络的速率
# --------------------------------------------------
tf.reset_default_graph()
#基于DDQN的想法 于是创建两个NN：一个主网络用于选择动作，另一个目标网络用于生成目标Q值
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)

myBuffer = experience_buffer()

# 设置随机决策的衰减速率
e = startE
stepDrop = (startE - endE)/annealing_steps

#创建每个episode中包含所有收益和操作记录的列表
jList = []
rList = []
total_steps = 0

#创建用于保存模型的目录
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        # 初始化环境
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        # Q网络
        while j < max_epLength: # 如果agent移动了超过200次还没有接触任何方块，停止本次训练
            j+=1
            # 根据Q网络和贪心法则选取行动（有随机行动的可能性）
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,4)
            else:
                a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
            s1,r,d = env.step(a)
            s1 = processState(s1)
            total_steps += 1#总体的step数目 不管episode
            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) # 保存训练记录至缓冲器
            
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size) # 从记录中随机获取训练批次数据
                    # 使用 Double-DQN 更新目标Q值

                    #下面的是基于主NN的next state的动作选择
                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    #对应的target网络里面next state的Q情况
                    Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    end_multiplier = -(trainBatch[:,4] - 1)
                    #依据主NN的动作选取情况 从target NN里面选出来Q值
                    doubleQ = Q2[range(batch_size),Q1]
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    # 利用目标值更新网络
                    _ = sess.run(mainQN.updateModel, \
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                    
                    updateTarget(targetOps,sess) # 更新目标网络至主网络 这里更新方法也还是指数衰减平均
            rAll += r
            s = s1
            
            if d == True:

                break
        
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        # 周期性保存训练结果
        if i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.ckpt')
            print("Saved Model")
        if len(rList) % 10 == 0:
            print(total_steps,np.mean(rList[-10:]), e)
    saver.save(sess,path+'/model-'+str(i)+'.ckpt')
print("平均得分: " + str(sum(rList)/num_episodes))
# --------------------------------------------------
rMat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)
# --------------------------------------------------