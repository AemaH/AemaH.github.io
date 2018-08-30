---
layout:     post                    # 使用的布局（不需要改）
title:      如何写blog文章               # 标题 
subtitle:    #副标题
date:       2018-06-26              # 时间
author:     ERAF                      # 作者
header-img: img/from_.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 生活

---

## 标题格式

每一篇文章文件命名采用的是2017-02-04-Hello-2017.md时间+标题的形式，空格用-替换连接。
文件的格式是 .md 的 MarkDown 文件。

在开头 我们需要加上这样一个结构「不要忘记上下面的---」：

**注意**：关于在blog中显示出来的标题是上面的---之间的title；而关于issue中建立的 和评论有关的是这个md文件的名字；

关于文件的名字 为了issue的建立考虑，还是直接用英文吧；

## 关于其中的公式
测试： 关于`￥`的使用:$e=mc^2$
而关于￥￥的使用:$$e=mc^2$$；
$$
\begin{aligned} \dot{x} &= \sigma(y-x) \\ 
\dot{y} &= \rho x - y - xz \\ 
\dot{z} &= -\beta z + xy \end{aligned} 
$$

## 关于其中的开头要加的格式
```markdown
---
layout:     post                    # 使用的布局（不需要改）
title:      My First Post               # 标题 
subtitle:   Hello World, Hello Blog #副标题
date:       2017-02-06              # 时间
author:     BY                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 生活
---

```
后面就直接加上那些markdown就好了；

## 关于其中添加图片
在gitpages里面插入图片是一件麻烦的事情，本地的话 最好还是改掉使用相对地址的毛病，为了本地阅读和同步考虑 这里决定统统换为图床；

## 关于其中使用 竖杠
注意 若想在blog里面使用![](https://ws1.sinaimg.cn/large/005A8OOUly1fuqvvo3xvwj302x01l742.jpg)
只能写作$p(x\|y)$ 虽然本地看会不一样；

$$
\begin{aligned}
H(P)&=H(Y|X)\\
&=\sum_xP(x)H(Y|X=x)\\
&=-\sum_xP(x)\sum_yP(y|x)log(P(y|x))\\
&=-\sum_{x,y}P(x)P(y|x)log(P(y|x))\\
\end{aligned}
$$
