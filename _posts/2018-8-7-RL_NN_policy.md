---
layout:     post                    # 使用的布局（不需要改）
title:      Reinforcement Learning               # 标题 
subtitle:   值函数和策略梯度 #副标题
date:       2018-07-31              # 时间
author:     ERAF                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 强化学习
---

在上一个章节，完成了对于强化学习整体结构的描述，对于常见的强化学习的分类也有所了解，然后在最后描述了关于qtable的实现方法；

在qtable的里面，可以看到 这里我们并没有强制的对于全部的state-action 包含的对应的值函数进行遍历 确定数值，而是单纯的进行了`num_episodes`次的训练；原因主要包括：

1.  毕竟本质上 还是进行实现一个解决实际问题的算法，而真实的实际问题 大多都是当做model-free的环境来进行的，那么在某些状态下的转移概率是不明确的，那么又如何使用 值迭代 的方法来进行更新整个table呢；

2.  其次 既然都说是qtable了，那么对于值函数确定的方法采取肯定的是TD方法啊；那么又怎么会用动态规划的值迭代方法呢，至于蒙特卡洛的方法 需要知道全部的状态情况的时候 才能确定每个状态进而进行选择 有些本末倒置 同时效率也很低，这也是另外的一种方法了 和这里的qlearning不符；

    >   顺带这里谈一下关于有关确定值函数描述的几种方法：动态规划的值迭代、蒙特卡洛方法、TD方法；蒙特卡洛上面也说一些，其对于值函数的定义还是值函数最原始的定义，于是需要知道全部的回报进行积累才确定；而无论是 值迭代 还是TD方法 中计算某状态「当s为$s_{t}$」值函数的时候，从其定义公式$E_{\pi}[R_{t+1}+\gamma v_{\pi}(s_{t+1})]$ 显然知道 计算这个值函数 需要知道所有后续状态以确定当前状态的值函数，但不同之处在于 动态规划的方法的依托是model-based，也就是说能够自我推导后续全部状态；而TD借助实验才能得到后续状态 ，所以 这里我们需要设置多次episode 以遍历尽量多的状态；

    ### Qnet

    当然上面尽管说了几种方法，但依旧还是用于评估值函数，紧接着才能使用这样的值函数来改善当前策略；但既然是值函数 那么肯定也还是能使用函数进行逼近的，而下面要描述的qnet方法 也就是建立在这样的基础上；

    换句话说，我们只是找到一个线性亦或者非线性函数来描述值函数罢了，qnet也就是这里面使用了神经网络来描述非线性函数的方法罢了；「神经网络的万能近似性」

    既然借助一个函数来描述了，那么现在对于值函数的更新 其实也就变成了一个对于函数参数的优化过程；翻看前面值函数更新的公式，比如这个TD的$Q(S,A) \leftarrow Q(S,A)+\alpha[R+\gamma max_{a}Q(S',a)-Q(S,A)]$ 我们可以看到 其实值函数还是有着自己的target值 并朝着这个方向进行更新，这是什么？不就是监督学习的label嘛，既然这样 诸如梯度下降等方法 也可以用于进行更新优化；

    >   具体的来说，之前的Q值就是一个表，借助qlearning这样的TD方法 直接更新值就好，现在对于Q值的得到是一个函数「NN」了，那么我们就优化更新他就好 然后需要的时候 直接用这个函数输入input 得到的结果就是Q值；

    这样 又带来一个新的问题就是，优化的方向是什么？换句话说就是目标函数是什么？这里直接采取欧氏距离来描述和目标的函数的差距来优化；

    对于算法的实现，整体结构 和之前还是一样的：

    区别在于 这里我们不必强制使用numpy或者dict来建立形象的table，只是建立了一个神经网络来拟合函数用于生成Q值，可以看到 其实这样就不要全部把Q值都存储下来，解决了维数灾难等问题；

    结构如下：

    ```python
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            # 初始化环境，得到第一个状态观测值
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            # Q网络
            while j < 99:
                j += 1
                # 根据Q网络和贪心算法(有随机行动的可能)选定当前的动作
                a, allQ = sess.run([predict, Qout], feed_dict={inputs1:np.identity(16)[s:s+1]})
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                # 获取新的状态值和奖励值
                s1, r, d, _ = env.step(a[0])
                # 通过将新的状态值传入网络获取Q'值
                Q1 = sess.run(Qout, feed_dict={inputs1:np.identity(16)[s1:s1+1]})
                # 获取最大的Q值并选定我们的动作
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + y*maxQ1
                # 用目标Q值和预测Q值训练网络
                _, W1 = sess.run([updateModel, W], feed_dict={inputs1:np.identity(16)[s:s+1], nextQ:targetQ})
                rAll += r
                s = s1
                if d == True:
                    # 随着训练的进行不断减小随机行动的可能性
                    e = 1./((i/50) + 10)
                    break
            jList.append(j)
            rList.append(rAll)
    print("Percent of succesful episodes: " + str(sum(rList)/num_episodes))
    # 网络性能统计
    plt.plot(rList)
    plt.plot(jList)
    ```

    如果和之前的qtable的实现一起连起来看，我们可以看到 其中的整体框架是不变的，比如强化学习算法整体需要进行的步骤：reset environment、获取动作a和a‘、获取next state和Q来构建target value「在qtable中 并没有直白的给出target value，而是隐藏在更新公式之中」；

    不同之处也就是上面说过的 不是直白的更新Q表，而是使用预测值和目标值来训练Q值生成网络

    

    其他的就是一些相关的设置问题了：

    ```python
    import gym
    import numpy as np
    import random
    import tensorflow as tf
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    # 加载实验环境
    env = gym.make('FrozenLake-v0')
    # Q网络解法
    tf.reset_default_graph()
    # 建立用于选择行为的网络的前向传播部分
    inputs1 = tf.placeholder(shape=[1,16], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
    Qout = tf.matmul(inputs1, W)
    predict = tf.argmax(Qout, 1)
    # 计算预期Q值和目标Q值的差值平方和（损失值）
    nextQ = tf.placeholder(shape=[1,4], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)
    # 训练网络
    init = tf.initialize_all_variables()
    # 设置超参数
    y = .99
    e = 0.1
    num_episodes = 2000 # 为了快速设置为2000，实验调为20000时可以达到0.6的成功率
    # 创建episodes中包含当前奖励值和步骤的列表
    jList = []
    rList = []
    ```

    然后是整体实现的地址：[这里](https://github.com/AemaH/AemaH.github.io/blob/master/source_code/qnet.py)

    

    ### Policy Network

    在之前A3C的文章中，提到过关于策略梯度的方法；而这种方法 也正是下面要说的策略搜索的一种方法；

    首先 先回顾一下关于值函数的方法，总结性的来说就是：值函数近似的方法是通过描述构建起来一个评估函数：值函数「评估策略的好坏的」，进而优化这个值函数 使之达到最优，当其最优的时候，作为最能描述策略好坏的值函数，那么此时 采用**贪婪策略「确定性策略」或者ε-greedy「随机策略」** 在某状态 选取对应的action，也就是当前状态所选的最优动作， 整体来说 也就是最优策略了；

    于是也能看到，对于值函数方法中，努力优化的是值函数，以此作为评判标准来评价其对应的策略的好坏；但不得不说 作为值函数方法本身 作为一个评价函数需要有着确定化的输入「状态」和输出「动作 或者 Q」，当面对诸如 连续动作的时候并不能有着很好的表现；

    于是引入了直接的策略搜索的方法；我们先说一下关于两者实现的区别：值函数的方法提过很多遍了，是需要先构建好值函数 然后优化好值函数，才能得到策略；而策略搜索的方法不同 则是使用一个参数化的线性或者非线性函数用以描述策略 然后寻找最优的策略，是强化学习的目标：累计回报的期望 最大；

    于是 策略搜索 顾名思义的也正是在策略空间直接进行搜索，不再借助值函数那样的间接完成对于最优策略的确定；于是也带来了收敛性快这样的好处；当然策略梯度的方法 也只是策略搜索中 无模型的策略搜索中的随机策略方法中的一种，其他这里不再赘述；

    这里 我们从似然函数的角度简单推导一下策略梯度，也就是借助诸如SGD需要求解的梯度；

    >   上面 我们说过无论是基于值函数还是基于策略搜索的方法，本质上的目的也都是强化学习的目的：让累计折扣回报最大化；对于一个马尔科夫过程 所产生的一组状态-动作序列 我们用$\tau $表示$s_0,a_0,s_1,a_1...s_H,a_H$
    >
    >   那么对于这么一组状态-动作对 或者说这一条轨迹的回报我们用 $R(\tau)=\sum^{H}_{t=0}R(s_t,a_t)$，而该轨迹出现的概率我们使用$P(\tau;\theta)$表示，那么显而易见的从强化学习的目标：最大化回报 这个目的出发，可以写作 $max_{\theta}U(\theta)=max_{\theta}E(\sum^{H}_{t=0}R(s_t,a_t))=max_{\theta}\sum_{\tau}P(\tau;\theta)R(\tau)$ 「从回报期望的形式 到写作与概率相乘的形式，之前似然函数的部分说过」
    >
    >   进而为了优化这个目标函数，常见的如GD 找到那个最快下降的方向，于是对其求导有：
    >
    >   $\nabla _{\theta } U( \theta ) =\nabla _{\theta }\sum\limits _{\tau } P( \tau ;\theta ) R( \tau ) $
    >
    >   $=\sum\limits _{\tau } \nabla _{\theta } P( \tau ;\theta ) R( \tau ) $
    >
    >   $=\sum\limits _{\tau }\frac{P( \tau ;\theta )}{P( \tau ;\theta )} \nabla _{\theta } P( \tau ;\theta ) R( \tau ) $
    >
    >   $=\sum\limits _{\tau } P( \tau ;\theta )\frac{\nabla _{\theta } P( \tau ;\theta ) R( \tau )}{P( \tau ;\theta )} $
    >
    >    $=\sum\limits _{\tau } P( \tau ;\theta ) R( \tau ) \nabla _{\theta } logP( \tau ;\theta ) $
    >
    >   于是 梯度的计算转换为求解$ R( \tau ) \nabla _{\theta } logP( \tau ;\theta ) $的期望，此时可以利用蒙特卡洛法近似「经验平均」估算，即根据当前策略π采样得到m条轨迹 ，利用m条轨迹的经验平均逼近策略梯度 于是有：
    >
    >   $\nabla _{\theta } U( \theta ) \approx \frac{1}{m}\sum\limits ^{m}_{i=0} R( \tau ) \nabla _{\theta } logP( \tau ;\theta ) $
    >

    如此 我们就得到策略梯度的推导结果；如果直观的理解上面的公式的话，$\nabla _{\theta } logP( \tau ;\theta )$就是轨迹随着参数θ变化最陡的方向，而$R( \tau ) $代指了步长 影响了某轨迹出现的概率；进一步的 由似然函数 损失函数中的log部分可以推导为只包含动作策略的$log\pi_{\theta}(a|s)$ 「转移概率无θ可删去；具体步骤可以参考RL :introduction」，因包含策略的表示，根据需要有着多种方式「具体参考RL:introduction」本文我们仅通过实例来进行说明 所构建出来的损失函数；

    

    #### Cart-Pole

    OpenAI gym包含了一系列强化学习问题所需的环境，本文也正是利用其中的一个经典案例：Cart-Pole（[查看相关文档](https://gym.openai.com/docs/)）。在这个案例中，我们希望agent在学习之后可以使木杆平衡尽可能长的时间不倒下。和双臂赌博机不同，这个任务需要额外考虑以下两点：

    -   *观测值* —— agent需要直到当前木杆的位置以及它的平衡角。为了得到它，我们的agent在每次输出行动的概率分布时都将进行观测并使用它。
    -   *延时收益* —— 保持木杆平衡的时间尽可能长意味着当前的行动对目前和将来都是有利的。为了达成这一功能，我们将设计一个函数，使收益值按照设计的权重分配在过去的一系列行动上。

    同时在神经网络的实现过程中，也不存在熟悉的监督学习中的 y label. 取而代之的是我们选的 action. 看到了log 下意识的也想到了使用交叉熵来表示损失函数；我们都知道对于 cross-entropy 的分类误差. 分类问题中的标签是真实 x 对应的 y ，但这是强化学习啊，哪来的label，不过幸好 参考上面对于整个策略梯度的式子的解释 我们知道：「$\nabla _{\theta } logP( \tau ;\theta )$就是轨迹随着参数θ变化最陡的方向，而$R( \tau ) $代指了步长 影响了某轨迹出现的概率；」我们都知道 基于强化学习建立的标准 肯定是希望容易获得高回报但不容易被选中的动作方向上的梯度 ；换句话说 那些高回报的动作乘上了advantage也正是这里的label；

    于是一方面这里的损失函数 我们可用直接使用cross-entropy，写作

    ```python
    loglik=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score,labels=1-input_y) 
    #注意这里的score和后面我们得到的不一样
    loss=-tf.reduce_mean(loglik*advantages)
    ```

    >   不过这里其实又提到另外一种写法`loglik=tf.log(input_y*(input_y-probability)+(1-input_y)*(input_y+probability))`,这里顺嘴提一下；首先这里的input_y并不是指的是action，而是1-action；参考代码也知道`y=1 if action==0 else 0`，代指了action的相反的选择；而 这里依旧有`probability = tf.nn.sigmoid(score)`；所以 对于上面的理解：而对于action来说 只是单纯的0或者1的选择，那么 当action为1的时候 input_y为0 此时输出的对应probability，于是有 `1*log(probability)`；而action为0的时候 此时的输出概率为probability，那么相反的为1的概率为1-probability，于是写作`1*log(1-probability)`；因而本质上是一样，当然如果action有更多种选择的时候 就不能这样写了 还是老老实实写成交叉熵的形式吧；
    >
    >   

以下为全部代码：

```python
%matplotlib inline
import numpy as np
from matplotlib import animation
from IPython.display import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import math

import gym
env = gym.make("CartPole-v0")

# 尝试随机行动来测试环境
env.reset()
reward_sum = 0
num_games = 10
num_game = 0
while num_game < num_games:
    env.render()
    observation, reward, done, _ = env.step(env.action_space.sample())
    reward_sum += reward
    if done:
        print("本episode的收益：{}".format(reward_sum))
        reward_sum = 0
        num_game += 1
        env.reset()

# 初始化agent的神经网络
# 我们使用基于策略梯度的神经网络来接受观测值并传递给隐藏层来产生选择各个行为（左移/右移）的概率分布
# 神经网络超参数
hidden_layer_neurons = 13
batch_size = 50
learning_rate = 1e-2
gamma = .99
dimen = 4

tf.reset_default_graph()

# 定义输入占位符
observations = tf.placeholder(tf.float32, [None, dimen], name="input_x")

# 第一个权重层
W1 = tf.get_variable("W1", shape=[dimen, hidden_layer_neurons], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))

# 第二个权重层
W2 = tf.get_variable("W2", shape=[hidden_layer_neurons, 1], initializer=tf.contrib.layers.xavier_initializer())
output = tf.nn.sigmoid(tf.matmul(layer1, W2))

# 定义网络用于学习的计算图组件
trainable_vars = [W1, W2]
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# 损失函数
with tf.name_scope("loss"):
    loglik=tf.log(input_y*(input_y-probability)+(1-input_y)*(input_y+probability)) 
    
    loss=-tf.reduce_mean(loglik*advantages)
    tf.summary.scalar("loss",loss)

# 梯度
new_grads = tf.gradients(loss, trainable_vars)
W1_grad = tf.placeholder(tf.float32, name="batch_grad1")
W2_grad = tf.placeholder(tf.float32, name="batch_grad2")

# 学习
batch_grad = [W1_grad, W2_grad]
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
update_grads = adam.apply_gradients(zip(batch_grad, [W1, W2]))

def discount_rewards(r, gamma=0.99):
    """
    输入一维的收益数组，输出折算后的收益值，例：f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801]
    """
    return np.array([val * (gamma ** i) for i, val in enumerate(r)])


reward_sum = 0
init = tf.global_variables_initializer()

# 定义观测值，输出值，收益值的占位符
xs = np.empty(0).reshape(0, dimen)
ys = np.empty(0).reshape(0, 1)
rewards = np.empty(0).reshape(0, 1)

# 初始化环境
sess = tf.Session()
rendering = False
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)
observation = env.reset()

# 梯度的占位符
gradients = np.array([np.zeros(var.get_shape()) for var in trainable_vars])

num_episodes = 10000
num_episode = 0

while num_episode < num_episodes:
    # 将观测值作为该批次的输入
    x = np.reshape(observation, [1, dimen])
    
    # 运行神经网络来决定输出
    tf_prob = sess.run(output, feed_dict={observations: x})
    
    # 基于我们的网络来决定输出，允许一定的随机性
    y = 0 if tf_prob > np.random.uniform() else 1
    
    # 将观测值和输出值追加至列表中以供学习
    xs = np.vstack([xs, x])
    ys = np.vstack([ys, y])
    
    # 获取行动的结果
    observation, reward, done, _ = env.step(y)
    reward_sum += reward
    tf.summary.scalar("reward_sum",reward_sum/episode_number)
    rewards = np.vstack([rewards, reward])
    
    if done:
        # 标准化收益值
        discounted_rewards = discount_rewards(rewards, gamma)
        discounted_rewards -= discounted_rewards.mean()
        discounted_rewards /= discounted_rewards.std()
        
        # 根据实时得到的梯度调整梯度
        gradients += np.array(sess.run(new_grads, feed_dict={observations: xs, input_y: ys, advantages: discounted_rewards}))
        
        # 重置游戏变量
        xs = np.empty(0).reshape(0, dimen)
        ys = np.empty(0).reshape(0, 1)
        rewards = np.empty(0).reshape(0, 1)
        
        # 一个batch运行结束
        if num_episode % batch_size == 0:
            # 更新梯度
            sess.run(update_grads, feed_dict={W1_grad: gradients[0], W2_grad: gradients[1]})
            # 重置梯度
            gradients *= 0
            # 输出本轮运行状态
            print("episode = {} 时的平均收益：{}".format(num_episode, reward_sum / batch_size))
            
            if reward_sum / batch_size > 150:
                print("问题在episode = {} 时解决！".format(num_episode))
                break
            reward_sum = 0
            rs=sess.run(merged,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            writer.add_summary(rs,episode_number)
        num_episode += 1
        observation = env.reset()

# 去除随机决策，测试agent的性能
observation = env.reset()
observation
reward_sum = 0

while True:
    env.render()
    
    x = np.reshape(observation, [1, dimen])
    y = sess.run(output, feed_dict={observations: x})
    y = 0 if y > 0.5 else 1
    observation, reward, done, _ = env.step(y)
    reward_sum += reward
    if done:
        print("最终分数: {}".format(reward_sum))
        break
```

