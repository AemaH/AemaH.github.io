import gym
import numpy as np
import random
import tensorflow as tf
#import tensorflow.contrib.silm as slim
import matplotlib.pyplot as plt
import scipy.misc
import os


slim = tf.contrib.slim

"""
本来想写成tensorboard的形式的 相关的参数和variable_scope也已经设置好了，也只差
"""

#加载环境 一个gridworld环境
from gridworld import gameEnv
env=gameEnv(partial=False,size=5)

class Qnetwork():
    def __init__(self,h_size):
        #接受进去的是[21168,]的向量
        self.scalarInput=tf.placeholder(shape=[None,21168],dtype=tf.float32)
        self.imageIn=tf.reshape(self.scalarInput,shape=[-1,84,84,3])
        with tf.name_scope("conv_layers"):
            self.conv1= slim.conv2d( \
                inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
            self.conv2 = slim.conv2d( \
                inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
            self.conv3 = slim.conv2d( \
                inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
            self.conv4 = slim.conv2d( \
                inputs=self.conv3,num_outputs=h_size,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)
            conv_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'Conv')
        tf.summary.histogram('conv_weights_1', conv_vars[0])
        tf.summary.histogram('conv_weights_2', conv_vars[1])
        tf.summary.histogram('conv_weights_3', conv_vars[2])
        tf.summary.histogram('conv_weights_4', conv_vars[3])
        flatten_layer=slim.flatten(self.conv4)
        with tf.name_scope("fcn"):
            self.Qout=slim.fully_connected(inputs=flatten_layer,num_outputs=env.actions
                                     ,activation_fn=None)
            fcn_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES,'fully_connected')
        tf.summary.histogram("fcn_weight",fcn_vars[0])
        
        self.predict=tf.argmax(self.Qout,1)
        #self.predict_value=tf.reduce_max(self.Qout,1)
        #利用目标和预测Q值之间的平方和来得到损失。
        self.targetQ= tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,env.actions,dtype=tf.float32)
        
        self.Q=tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error= tf.square(self.targetQ - self.Q)
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(self.td_error)
            tf.summary.scalar("loss",self.loss)
        with tf.name_scope("train"):
            self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self.trainer.minimize(self.loss)

def processState(states):
    return np.reshape(states,[21168])
def updateTargetGraph(tfVars,tau):
# 利用主网络参数更新目标网络
    total_vars=len(tfVars)
    op_holder=[]
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign(
            (var.value()*tau)+((1-tau)*tfVars[idx+total_vars//2].value())))
        return op_holder
def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

class experience_buffer():
    def __init__(self,buffer_size=50000):
        self.buffer=[]
        self.buffer_size=buffer_size
        
    def add(self,experience):
        if len(self.buffer)+len(experience)>=self.buffer_size:
            self.buffer[0:(len(self.buffer)+len(experience))-self.buffer_size]=[]
            #清0前面的buffer_size个元素
        self.buffer.extend(experience)
    
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
    #采样size个，并变形成我们需要的shape
tf.reset_default_graph()

#基于DQN的想法 于是创建两个NN一个主网络用于选择动作，另一个目标网络用于计算target Q
mainQN=Qnetwork(h_size)
targetQN=Qnetwork(h_size)
init=tf.global_variables_initializer()
saver=tf.train.Saver()

trainable_variable=tf.trainable_variables()
targetOps=updateTargetGraph(trainable_variable,tau)#用来基于主网络的更新target网络
myBuffer=experience_buffer()

#Set the rate of random action decrease. 设置随机动作的概率减少
e = startE
stepDrop = (startE - endE)/annealing_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

#设置一个用于保存模型的路径
if not os.path.exists(path):
    os.makedirs(path)
    
with tf.Session() as sess:
    #用于tensorboard的这部分代码 个人实验的时候没加上 这是后来又写的 没实践
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    
    #后添加代码

    sess.run(init)
    if load_model==True:
        print("读取在{}保存的模型").format(path)
        ckpt=tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        
    for i in range(num_episodes):
        episodeBuffer=experience_buffer()
        
        #reset环境 并获取初始的observation
        s=env.reset()
        s=processState(s)
        d=False
        rAll=0
        j=0
        
        while j<max_epLength:
        #每一个episode 都有着最长的step限制
            j+=1
            #当满足条件时候 随机选取action 或者 依照最大Q值的选取action
            if np.random.rand(1)<e or total_steps<pre_train_steps:
                a=np.random.randint(0,4)
            else:
                a=sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
            s1,r,d=env.step(a)
            s1=processState(s1)
            total_steps+=1#总体的step数目 不管episode
            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5]))#当前的episode的存储经验
            
            if total_steps>pre_train_steps:
                #当step已经进行足够的数目后 e开始减少
                if e>endE:
                    e-=stepDrop
                if total_steps % (update_freq) == 0:
                    trainBatch=myBuffer.sample(batch_size) #从总体的经验里面抽取

                    #获取target的Q值 首先里面的next_state的Q值来自于 targetnet
                    next_Q=sess.run(targetQN.Qout
                                    ,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    max_next_Q=tf.reduce_max(next_Q,axis=1)
                    
                    targetQ=trainBatch[:,2]+(gamma*max_next_Q)
                    
                       
                
                    #更新主网络
                    _ = sess.run(mainQN.updateModel, \
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0])
                                   ,mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                    #进而更新target网络
                    updateTarget(targetOps,sess) 
                    #注：未正面加了下面两行代码后的可用性
                    rs=sess.run(merged,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0])
                                   ,mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                    writer.add_summary(rs,it)
            rAll+=r
            s=s1
            if d==True:
                break
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        
        #定期保存模型
        if i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.ckpt')
            print("Saved Model")
            
        if len(rList) % 10 == 0:
            print(total_steps,np.mean(rList[-10:]), e)
    saver.save(sess,path+'/model-'+str(i)+'.ckpt')
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")            