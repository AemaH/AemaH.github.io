from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline

env = gym.make('FrozenLake-v0')

#设置网络层相关的信息
input_state=tf.placeholder(shape=[1,16],dtype=tf.float32)
with tf.name_scope("Q_net_variable"):
    W=tf.Variable(tf.random_uniform([16,4],0,0.01))#最小值为0 max为0.01
    tf.summary.histogram("w",W)
Q_out=tf.matmul(input_state, W)
predict_action=tf.argmax(Q_out, axis=1)

target_Q=tf.placeholder(tf.float32,shape=[1,4])
with tf.name_scope("loss"):
    loss=tf.reduce_sum(tf.square(target_Q-Q_out))
    tf.summary.scalar("loss",loss)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
trainer=optimizer.minimize(loss)

initer=tf.global_variables_initializer()

gamma=0.00
epsilon=0.1
num_episodes=200
#create lists to contain total rewards and steps per episode
jList = []#每个episode内部帧数
rList = []#总回报

with tf.Session() as sess:
    merged=tf.summary.merge_all()
    writer=tf.summary.FileWriter("logs/",sess.graph)
    sess.run(initer)
    for i in range(num_episodes):
        s=env.reset()
        r_ALL=0
        is_terminal=False
        j=0
        while j<99:
            j+=1
            a,Q_about_s=sess.run([predict_action,Q_out],
                                 feed_dict={input_state : np.identity(16)[s:s+1]})
            if np.random.rand(1) < epsilon:
                a[0]=env.action_space.sample()
            next_s,r,is_terminal,_=env.step(a[0])
            #下面写成a[0]是为了提出来其中的value 不然依旧为array([value])
            
            Q_about_next=sess.run(Q_out,
                                    feed_dict={input_state:np.identity(16)[next_s:next_s+1]})
            max_Q_next=np.max(Q_about_next)
            targetQ=Q_about_s#在此基础上修改
            targetQ[0][a[0]]=r+gamma*max_Q_next
            
            #开始优化 获得新的W
            _,W1=sess.run([trainer,W],feed_dict={input_state:np.identity(16)[s:s+1] , target_Q:targetQ})

            r_ALL+=r
            s=next_s
            if is_terminal ==True:
                #Reduce chance of random action as we train the model.
                epsilon=1.0/((i/50)+10)
                rs=sess.run(merged,feed_dict={input_state : np.identity(16)[s:s+1],target_Q:targetQ})
                writer.add_summary(rs,i)
                break
        jList.append(j)
        rList.append(r_ALL)
        
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")