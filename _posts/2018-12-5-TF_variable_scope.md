---
layout:     post                    # 使用的布局（不需要改）
title:      tensorflow              # 标题 
subtitle:   有关variable和scope的一点用法的笔记 #副标题
date:       2018-12-05              # 时间
author:     ERAF                      # 作者
header-img: img/nogizaka_6th.jpg   #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 深度学习
---

## 首先是name_scope和variable_scope的用处区别

对于前者的`tf.name_scope`，作用更多的还是一个编码习惯和tensorboard里面展示单元的作用，如下：
```python
with tf.name_scope('data'):
    iterator = dataset.make_initializable_iterator()
    center_words, target_words = iterator.get_next()
with tf.name_scope('embed'):
    embed_matrix = tf.get_variable('embed_matrix', 
                                    shape=[VOCAB_SIZE, EMBED_SIZE], ...)
    embed = tf.nn.embedding_lookup(embed_matrix, center_words)
with tf.name_scope('loss'):
    nce_weight = tf.get_variable('nce_weight', shape=[VOCAB_SIZE, EMBED_SIZE], ...)
    nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, …)
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
```
这样带来的好处一方面在看代码的时候，也起到了注释的作用 更为清晰和明了，此外对于输出的tensor的名字，也更容易分辨是哪部分的：
```python
with tf.name_scope("first"):
    a=tf.Variable(1,name="a")
a
>>> <tf.Variable 'first_1/a:0' shape=() dtype=int32_ref>
```
同时如果使用tensorboard，也可以在一张图中展示出来，更为明白和直观，上面那一组代码 如果用tensorboard展示就有着下图效果：![](https://ws1.sinaimg.cn/large/005A8OOUly1fxv0n7agy4j30ks0as40r.jpg)

以上是关于`tf.name_scope`，可以看到这里其实更多还是一种习惯的问题，就实际的效果还是很少的，但相比而言`tf.variable_scope`也就有着更为有用的效果：起方便variable的共享，这里其实也涉及了一个包括对于`tf.Variable`和`tf.get_variable`的使用问题，首先如下一个建立双层NN结构的函数：
```python
def two_hidden_layers(x):
    w1 = tf.Variable(tf.random_normal([100, 50]), name='h1_weights')
    b1 = tf.Variable(tf.zeros([50]), name='h1_biases')
    h1 = tf.matmul(x, w1) + b1

    w2 = tf.Variable(tf.random_normal([50, 10]), name='h2_weights')
    b2 = tf.Variable(tf.zeros([10]), name='2_biases')
    logits = tf.matmul(h1, w2) + b2
    return logits
logits1 = two_hidden_layers(x1)
logits2 = two_hidden_layers(x2)
```
如果我们在tensorboard里面来看的话，可以发现![](https://ws1.sinaimg.cn/large/005A8OOUly1fxv1f9ioy4j30h20axgny.jpg)
会出现两组参数，而实际上 我们只是希望还是这个网络结构和参数 只是不同的输入就是了，而对于其解决就是使用的`tf.get_variable`配合上`tf.variable_scope`对于`reuse`参数的设置；
当然不能直接只是僵硬的吧`tf.Variable`换成`tf.get_variable`，不然的话 上面两次对于`two_hidden_layers`的调用就会导致错误提示：`ValueError: Variable h1_weights already exists, disallowed. Did you mean to set reuse=True in VarScope?`
正确的使用方法应该如下将variable放在一个scope，然后对于该scope设置好reuse允许 如下：
```python
def two_hidden_layers(x):
    assert x.shape.as_list() == [200, 100]
    w1 = tf.get_variable("h1_weights", [100, 50], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("h1_biases", [50], initializer=tf.constant_initializer(0.0))
    h1 = tf.matmul(x, w1) + b1
    assert h1.shape.as_list() == [200, 50]  
    w2 = tf.get_variable("h2_weights", [50, 10], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable("h2_biases", [10], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(h1, w2) + b2
    return logits
with tf.variable_scope('two_layers') as scope:
    logits1 = two_hidden_layers(x1)
    scope.reuse_variables()
    logits2 = two_hidden_layers(x2)

```
![](https://ws1.sinaimg.cn/large/005A8OOUly1fxv1k4zdt5j30ki0b90ub.jpg)
可以看到这时候就有着相同的变量参数 只是输入发生了改变罢了；
进一步的上面的代码还是有点累赘，我们可以复用`tf.variable_scope`这种方法，对于每层、每个双层都有着使用，于是有：
```python
def fully_connected(x, output_dim, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable("weights", [x.shape[1], output_dim], initializer=tf.random_normal_initializer())
        b = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b

def two_hidden_layers(x):
    h1 = fully_connected(x, 50, 'h1')
    h2 = fully_connected(h1, 10, 'h2')

with tf.variable_scope('two_layers') as scope:
    logits1 = two_hidden_layers(x1)
    logits2 = two_hidden_layers(x2)


```
得到![](https://ws1.sinaimg.cn/large/005A8OOUly1fxv1motypkj30eq0b2js4.jpg)

## tf.Varibale和tf.get_variable
既然上面提到了关于tf.Varibale和tf.get_variable，这里也一起拿来看看说一下；
既然说到variable，那么首先在于tensorflow里面 首先其还是作为tensor-like object的一种，然后创建的方法就是如上基于`tf.Varibale`或者`tf.get_variable`；如果对于前者的话，创建的是会 除了类似tensor中设置：dtype, shape,name之外，最主要的是需要进行初始化，也就是说只有在初始化之后才会有具体的值；比如shape的设置就是基于这一个initial_value来得到的；
```python
#通过查询API 可以知道tf.constant这样的tensor建立 只是一个op 而tf.Variable() 是一个类，初始化的对象有多个op
var_obj = tf.Variable(
    initial_value, 
    dtype=None, 
    name=None, 
    trainable=True,
    collections=None,
    validate_shape=True
)
# 初始化参数
initial_value：可由 Python 内置数据类型提供，也可由常量 Tensor 的内置 op 来快速构建，但所有这些 op 都需要提供 shape；「tensor或者可以转化为tensor的python目标」
trainable：指明了该变量是否可训练
collections: 指明该变量存放的位置
validate_shape:  If False, allows the variable to be initialized with a value of unknown shape. If True, the default, the shape of initial_value must be known.
 #返回值：
变量实例对象(Tensor-like)
```
然后是`tf.get_variable`
```python
# 基于参数获得一个已存的变量 或者 创建一个全新的
tf.get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    trainable=True,
    regularizer=None,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None
)
# 初始化参数
name: 变量的名字
shape: 变量的shape
dtype: 变量的类型 
initializer: 初始化的方法
trainable: 是否是可训练的，如果是True 则添加到graph里面，才能在后面进行训练If True also add the variable to the graph collection tf.GraphKeys.TRAINABLE_VARIABLES.
regularizer: 正则化的设置 将创建一个变量存储在graph里面方便后面建立loss的时候的正则项的创建
    A (Tensor -> Tensor or None) function; the result of applying it on a newly created variable will be added to the collection tf.GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
collections: List of graph collections keys to add the Variable to. Defaults to [GraphKeys.GLOBAL_VARIABLES] (see tf.Variable). 

# Note:
>>> name 参数必须要指定，如果仅给出 shape 参数而未指定 initializer，那么它的值将由 tf.glorot_uniform_initializer 随机产生，数据类型为tf.float32; 
>>> 另外，initializer 可以为一个constant，这种情况下，变量的值和形状即为此张量的值和形状(就不必指定shape 了)。「包括诸如list或者array等形式」
>>> 此函数经常和 tf.variable_scope() 一起使用，产生共享变量
```
### 两者的区别；
1. 第一点在于两者的初始化方式，可以看到在tf.Variable 里面需要具体的指定initial_value，同时 该variable的shape也是根据该initial_value 的数据来的，不能具体的说明； 而tf.get_variable() 则不用指定具体的值 只需要说明清楚初始化的方法 当然 这个时候就需要说明shape了；**总之 基本上tf.Varibale()的初始化 你需要建立设置起来一个具体的值了。而tf.get_variable()只需要指明一种初始化的方法 同时需要在后面说明shape；**
当然也不是绝对的，其实也就是上面所提到的`initializer`和`initial_value`用法的区别，这一点在后面会单独的介绍；
2. 第二点就是在于主要的区别共享变量；
众所周知，tensorflow中每个tensor会在graph里面建立一个节点，variable也不例外，而建立的方式 也就是我们这些`tf.constant`、`tf.Variable`等操作，这里其实带来一个问题，如果我们输入了两个完全一样的代码来创造变量会怎么样？
```python
>>> tf.constant([1,2],name="c")
<tf.Tensor 'c:0' shape=(2,) dtype=int32>
>>> tf.constant([1,2],name="c")
<tf.Tensor 'c_1:0' shape=(2,) dtype=int32>
```


可以看到 其会直接再创建一个新的变量；那么 `tf.Variable`和`tf.get_variable`会有什么区别呢？
```python
>>> tf.Variable(tf.constant([1,2]),name="a")
<tf.Variable 'a:0' shape=(2,) dtype=int32_ref>
>>> tf.Variable(tf.constant([1,2]),name="a")
<tf.Variable 'a_1:0' shape=(2,) dtype=int32_ref>

>>> tf.get_variable(name="b",initializer=tf.constant([1,2]))
<tf.Variable 'b:0' shape=(2,) dtype=int32_ref>
>>> tf.get_variable(name="b",initializer=tf.constant([1,2]))
ValueError: Variable b already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
```

当tf.Variable 'b:0'这个变量已经存在时候 再次同样命名就会出错 ；因为`tf.get_variable()` 会检查当前命名空间下是否存在同样name的变量，可以方便共享变量。而`tf.Variable` 每次都会新建一个变量。 

为了共享变量需要搭配``tf.variable_scope() ``和其中的reuse属性：
```python
with tf.variable_scope("scope1"):
    w1 = tf.get_variable("w1", shape=[])
    w2 = tf.Variable(0.0, name="w2")
with tf.variable_scope("scope1", reuse=True):
    w1_p = tf.get_variable("w1", shape=[])
    w2_p = tf.Variable(1.0, name="w2")

print(w1 is w1_p, w2 is w2_p)
#输出
#True  False
print(w1,w1_p)
#<tf.Variable 'scope1/w1:0' shape=() dtype=float32_ref> 
#<tf.Variable 'scope1/w1:0' shape=() dtype=float32_ref>
print(w2,w2_p)
#<tf.Variable 'scope1/w2:0' shape=() dtype=float32_ref> 
#<tf.Variable 'scope1_1/w2:0' shape=() dtype=float32_ref>
```
可以看到这个时候 针对这个`scope1`这个variable_scope，基于tf.Variable的创建了两个变量；而基于tf.get_variable的则是同一个；
### 关于reuse的一点用法
这里顺带说一点东西，尽管借助于对`reuse`参数的设置或者类似的`scope.reuse_variables()`完成对于参数的贡献，但很显然有很多矛盾的场景 不是单纯的都需要复用或者不需要复用就能解决的；
比如如下这样一种情况：
```python
import tensorflow as tf
def test(mode):
    w = tf.get_variable(name=mode+"w", shape=[1,2])
    u = tf.get_variable(name="u", shape=[1,2])
    return w, u
with tf.variable_scope("test") as scope:
    w1, u1 = test("mode1")
    #scope.reuse_variables()
    w2, u2 = test("mode2")
```
可以看到上面场景中，w参数不需要复用，而u需要复用，这样的话，这里分别展示展示：不复用、使用`reuse=True`复用、和`scope.reuse_variables()`复用的方式,结果如下：
```python
# 关于 不复用时候
ValueError: Variable test/u already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
# 使用`reuse=True`复用
ValueError: Variable test/mode1w does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?

#和`scope.reuse_variables()`复用的方式
ValueError: Variable test/mode2w does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?

with tf.variable_scope("test") as scope:
    scope.reuse_variables()
    w1, u1 = test("mode1")
    w2, u2 = test("mode2")
ValueError: Variable test/mode1w does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?
```
哦哦 这里两种reuse设置的方法并没有区别蛤，只是之前写的时候 这里的`scope.reuse_variables()`放在了中间才导致现实有点区别，可以看到[`tf.variable_scope`的源码](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/variable_scope.py)中关于`reuse_variables`的源码 其实也只是将`self._reuse`改为True罢了；
```python
  def reuse_variables(self):
    """Reuse variables in this scope."""
    self._reuse = True
```
可以说两者是没区别的；
如果是在tf.variable_scope里面直接设置为True，那么表示着当前的scope里面只会在之前的代码里面找有没有一样的代码 比如：
```python
with tf.variable_scope("scope1"):
    w1 = tf.get_variable("w1", shape=[])
    #w2 = tf.Variable(0.0, name="w2")
    w2 = tf.get_variable("w2", shape=[])
with tf.variable_scope("scope1", reuse=True):
    w1_p = tf.get_variable("w1", shape=[])
    #w2_p = tf.Variable(1.0, name="w2")
    w2_p = tf.get_variable("w2", shape=[])
```
如上就不会报错，而你如果将后一个scope里面改为`tf.get_variable("w3", shape=[])`就会报和之前一样的错误，提示找不到一样的东西，因而报错，也就是说`reuse=True`的方法只适用于变量全部已经存在的时候，只是拿来共享使用；
> 这里顺带说一句蛤，如果你在之前一个使用的是`tf.Variable`建立的variable在后面一个使用`tf.get_variable`也是无法使用的；

那么 **针对这一有些参数是共享，有些参数是不需要共享该怎么办呢？**这时候 tensorflow提供另一种开启共享的方法；
```python
def test(mode):
    w = tf.get_variable(name=mode+"w", shape=[1,2])
    u = tf.get_variable(name="u", shape=[1,2])
    return w, u

with tf.variable_scope("test", reuse=tf.AUTO_REUSE) as scope:
    w1, u1 = test("mode1")
    w2, u2 = test("mode2")
```
这里只是加了一个参数 ``reuse=tf.AUTO_REUSE``，但正如名字所示，这是一种自动共享的机制，当系统检测到我们用了一个之前已经定义的变量时，就开启共享，否则就重新创建变量。这几乎是「万金油」式的写法;

### 对于使用tf.get_variable参数共享的机制
首先我们要知道 当一开始我们单纯的调用`tf.get_variable(name, shape, dtype, initializer)` 时，tensorflow会判断是否需要共享变量；

翻看[`tf.get_variable`函数的源码](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/variable_scope.py#L1184)，恩 只是一个`return get_variable_scope().get_variable`，无关紧要；再往里面看，其实是来自于`VariableScope`类的函数，再对应看其中的`get_variable`函数，其实是来自于[_VariableStore类的`get_variable`函数](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/variable_scope.py#L226),更本质的说是其本质的一句：`return self._get_single_variable(...)`，而对于该`_get_single_variable`函数 其中会先判断该variable是否在`self._var`这样一个变量名的字典中，然后再判断reuse如果是false就直接报错；['tf.get_variable'的源码](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/variable_scope.py#L732)，
> 写到这个地方的时候 一开始的时候其实是直接发现的_VariableStore类的`get_variable`函数，然后就忽然想到对于scope会怎样呢？然后发现关于'tf.get_variable'的函数内并未注意这些，只是name，而没有对于name对应的加上"scope_name/"的操作，但在其所属的类中有一个`open_variable_scope`等操作；然后才发现 [variable_scope.py](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/variable_scope.py)这个文件里面有无数个`get_variable`，
>如果只是这个针对的`_VariableStore`这个类里面的`get_variable`，其是主要更为本质的操作，name什么的添加是交给调用该基函数的 其他函数所做的，果然，如果是有着scope的情况 所使用的`get_variable`就是来自于`VariableScope`里面的了[VariableScope的get_variable](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/variable_scope.py#L1068)有着添加名字；

总之 tensorflow会先基于name判断该variable是否在`self._var`这样一个变量名的字典中「for循环」，然后再判断reuse如果是false就直接报错「if判断」 抛出 `ValueError` 异常说明该变量已经存在；若reuse为True，则在其中寻找对应name的variable；
依次遍历现有的`self._var`内的变量，判断完成后「for结束」；
此外 还有种情况就是不存在已有的variable，但reuse为True，则再次抛出判断`if reuse is True:`，显然表示着没有可以复用的变量，直接抛出错误`ValueError("Variable %s does not exist`

换句话说 上面相当于两个前提要求：是否已经存在变量 和 是否reuse为True；
如果是已存在变量 reuse为false就直接抛出 `ValueError` 异常说明该变量已经存在；若reuse为True，则在其中寻找对应name的variable「顺带试图寻找一个不存在的变量还是一样的和下面不存在变量一样的错误」
```python
with tf.variable_scope("scope1"):
    w1 = tf.get_variable("w1", shape=[])
    #w2 = tf.Variable(0.0, name="w2")
    w2 = tf.get_variable("w2", shape=[])
with tf.variable_scope("scope1", reuse=True):
    w1_p = tf.get_variable("w1", shape=[])
    #w2_p = tf.Variable(1.0, name="w2")
    w2_p = tf.get_variable("w3", shape=[])

ValueError: Variable scope1/w3 does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?
```
若不存在变量，reuse为true，直接抛出错误`ValueError("Variable %s does not exist`；reuse为false就根据 initializer 创建新的变量；
> **注：这里对于`self._var`没太细看，因而对于上面所述的第一个要求「是否存在变量」，意思是否应该改为：是否存在一样name的名字，我感觉后者应该更符合意思，因为没太细看所以还是不太确定**

### 关于variable赋值的方法
其实不谈tensorflow，只是说日常数学中的variable这种东西，其是不停改变的 但是 其总归也还是有个初值的，这也正是tensorflow中 无论是`tf.Variable`还是`tf.get_variable`都有着`initializer`或者`initial_value`，用来给他们初始化设置一个初始值来用，这里就顺带来谈一下tensor的初始化；
**variable的初始化**
说是tensor的初始化，其实对于`constant`来说，在一开始定义`tf.constant`的时候 已经初始化过了，同时这里的值并不能发生更改「也就是说针对这个tensor」
而对于variable来说，设置的时候`tf.Variable()`中写出`initial_value`、`initializer`并不代表该variable已经被初始化「即使他是个list、array」，要初始化它，就需要一波特殊的操作,`tf.global_variables_initializer()` 是并行的初始化全部变量，如何有选择地初始化部分变量呢？使用 `tf.initialize_variables()`「在2017-3-2之前是这个函数，在之后被`tf.variables_initializer`替代」，比如要初始化v_6, v_7, v_8三个变量；或者说初始化某个变量a就是，直接针对性的`a.initializer.run()`「无论是`tf.Variable`还是`tf.get_variable`，里面是不是initializer这个形参，都可以直接使用 这样的方法初始化其中的初始化器」：
```python
# 变量使用前一定要初始化
init = tf.global_variables_initializer() # 初始化全部变量
sess.run(init)

# 使用变量的 initializer 属性初始化
sess.run(v.initializer)

#初始化单独的几个
init_new_vars_op = tf.variables_initializer([v_6, v_7, v_8])
sess.run(init_new_vars_op)

#初始化v_6
v_6.initializer.run()#其中的初始化器初始化
```
**用另一个变量的初始化值给当前变量初始化** ：

-   由于tf.global_variables_initializer()是并行地初始化所有变量，所以直接使用另一个变量的初始化值来初始化当前变量会报错(因为你用另一个变量的值时，它没有被初始化)
-   在这种情况下需要使用另一个变量的initialized_value()属性。你可以直接把已初始化的值作为新变量的初始值，或者把它当做tensor计算得到一个值赋予新变量。
```python
# Create a variable with a random value.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
# Create another variable with the same value as 'weights'.
w2 = tf.Variable(weights.initialized_value(), name="w2")
# Create another variable with twice the value of 'weights'
w_twice = tf.Variable(weights.initialized_value() * 0.2, name="w_twice")
```

以上也都是建立在已经在`tf.Variable`里面设置好`initial_value`、或者在`tf.get_variable`里面设置好`initializer`，这样设置好相关的初始化的方法，才能给后面的初始化变量的方法给予机会。

而初始化的设置本身，根据是在`tf.Variable`还是在`tf.get_variable`是有所不一样的，毕竟其中的形参就不同：
>initial_value：可由 Python 内置数据类型提供，也可由常量 Tensor 的内置 op 来快速构建，但所有这些 op 都需要提供 shape；
>initializer：可以直接使用和initial_value一样的包含shape的constant，但此时就不能再指明shape了；此外还有一系列的独有的初始化的 给initializer使用的方法；见下面的代码

对于下面这些初始化方法，参考initializer本身的方法 也可以用于生成tensor，比如：
```python
init=tf.ones_initializer()
tensor_A=init([3])
#<tf.Tensor 'ones:0' shape=(3,) dtype=float32>
tf.constant_initializer(
    value=0, 
    dtype=dtypes.float32, 
    verify_shape=False
)
#生成一个常量的初始化器
#由此衍生的两个
I、tf.zeros_initializer()
II、tf.ones_initializer()
#############################################
value=np.array([[1,2,3],[2,3,4]])
init=tf.constant_initializer(value)
with tf.Session() as sess:
    x=tf.get_variable(initializer=init,shape=[6,],name="x")
    x.initializer.run()#其中的初始化器初始化 也可以用tf.variables_initializer([x]).run()
    print(sess.run(x))
    #[1. 2. 3. 2. 3. 4.]

# 生成截断正态分布的随机数，方差一般选0.01等比较小的数
tf.truncated_normal_initializer(
    mean=0.0,
    stddev=1.0,
    seed=None,
    dtype=tf.float32
)

# 生成标准正态分布的随机数，方差一般选0.01等比较小的数
tf.random_normal_initializer(
    mean=0.0,
    stddev=1.0,
    seed=None,
    dtype=tf.float32
)    

# 生成均匀分布的随机数
tf.random_uniform_initializer(
    minval=0,
    maxval=None,
    seed=None,
    dtype=tf.float32
)

# 和均匀分布差不多，只是这个初始化方法不需要指定最小最大值，是通过计算出来的
# 它的分布区间为[-max_val, max_val]
tf.uniform_unit_scaling_initializer(
    factor=1.0,
    seed=None,
    dtype=tf.float32
)
max_val = math.sqrt(3 / input_size) * self.factor
# input size is obtained by multiplying W's all dimensions but the last one
# for a linear layer factor is 1.0, relu: ~1.43, tanh: ~1.15

tf.variance_scaling_initializer(
    scale=1.0,
    mode='fan_in',
    distribution='normal',
    seed=None,
    dtype=tf.float32
)
# 初始化参数
scale: Scaling factor (positive float).缩放尺度
mode: One of "fan_in", "fan_out", "fan_avg".用于计算标准差stddev的值
distribution: Random distribution to use. One of "normal", "uniform".分布类型
# 1、当 distribution="normal" 的时候：
生成 truncated normal distribution（截断正态分布）的随机数，其中mean = 0, stddev = sqrt(scale / n)，
n 的计算与 mode 参数有关：
    如果mode = "fan_in"， n 为输入单元的结点数         
    如果mode = "fan_out"，n 为输出单元的结点数
    如果mode = "fan_avg",n 为输入和输出单元结点数的平均值
# 2、当distribution="uniform”的时候：
生成均匀分布的随机数，假设分布区间为[-limit, limit]，则limit = sqrt(3 * scale / n)

# 又称 Xavier uniform initializer
tf.glorot_uniform_initializer(
    seed=None,
    dtype=tf.float32
)
#为了使得在经过多层网络后，信号不被过分放大或过分减弱，我们尽可能保持每个神经元的输入和输出的方差一致! 从数学角度来讲，就是让权重满足均值为 0，方差为 2/(fan_in+fan_out)，随机分布的形式可以为均匀分布或者高斯分布。
#「有一个均匀分布来初始化数据，假设均匀分布的区间是[-limit, limit],则limit=sqrt(6 / (fan_in + fan_out))其中的fan_in和fan_out分别表示输入单元的结点数和输出单元的结点数。」
# It draws samples from a uniform distribution within [a=-limit, b=limit] 
limit： sqrt(6 / (fan_in + fan_out)) 
fan_in：the number of input units in the weight tensor 
fan_out：the number of output units in the weight tensor
mean = (b + a) / 2
stddev = (b - a)**2 /12

# 又称 Xavier normal initializer
tf.glorot_normal_initializer(
    seed=None,
    dtype=tf.float32
)
# It draws samples from a truncated normal distribution centered on 0 with 
# stddev = sqrt(2 / (fan_in + fan_out)) 
#「假设均匀分布的区间是[-limit, limit],则limit=sqrt(6 / (fan_in + fan_out))其中的fan_in和fan_out分别表示输入单元的结点数和输出单元的结点数。」
fan_in：the number of input units in the weight tensor 
fan_out：the number of output units in the weight tensor

```



**assign的用法**
可能很好奇 毕竟讲到现在都是说的是variable的初始化方法，那么这里说是variable赋值的方法是为什么呢？其实这里还提到了一种有关`assign`的用法；
我们都知道 在使用包含variable的时候，session里面首先要做的还是一步相关变量的`variable.initializer`，完成初始化之后，才能`sess.run()`输出当前的的值；
但借助于assign，倒是可以初始化都不用，直接输出一个值 emmm 好吧 这不是重点，重点更多的还是在于对于a其中的值可以完成更改和赋值：
```python
>>> a=tf.Variable([3,4])
>>> assign_a=tf.assign(a,[1,2])
>>> with tf.Session() as sess:
...     sess.run(a)
tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value Variable_1

>>> with tf.Session() as sess:
...     sess.run(a.initializer)
...     sess.run(assign_a)
...     print(a,sess.run(a))
array([1, 2])
<tf.Variable 'Variable_1:0' shape=(2,) dtype=int32_ref> [1 2]

```
可以看到这个时候的variable内部的值发生了变化，好吧 也可以当做初始化更改其中值的一种方法（？）当然`tf.constant`是没有这种用法的，毕竟本质上还是属于variable的一个方法，` 'Tensor' object has no attribute 'assign'`
