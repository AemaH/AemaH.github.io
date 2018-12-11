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


## 添加统计字数的功能
[参考网站](https://extingstudio.com/2017/09/jekyll_tutorials/)以及其网站的源码
主要步骤大致是对于博客中的\_layouts\post.html，找到其中的`<header class="intro-header" >`这样的`header`中添加`{{ page.content | strip_html | strip_newlines | remove: " " | size }}`语句用于统计字数，同样的 添加`{{ page.content | strip_html | strip_newlines | remove: " " | size | divided_by: 350 | plus: 1 }}`来统计大致阅读时间；  
如下：本博客原先的关于博客header为：
```html
<header class="intro-header" >
    <div class="header-mask"></div>
    <div class="container">
        <div class="row">
            <div class="col-lg-8 col-lg-offset-2 col-md-10 col-md-offset-1">
                <div class="post-heading">
                    <div class="tags">
                        {% for tag in page.tags %}
                        <a class="tag" href="{{ site.baseurl }}/tags/#{{ tag }}" title="{{ tag }}">{{ tag }}</a>
                        {% endfor %}
                    </div>
                    <h1>{{ page.title }}</h1>
                    {% comment %}
                        always create a h2 for keeping the margin , Hux
                    {% endcomment %}
                    {% comment %} if page.subtitle {% endcomment %}
                    <h2 class="subheading">{{ page.subtitle }}</h2>
                    {% comment %} endif {% endcomment %}
                    <span class="meta"> {% if page.author %}{{ page.author }}{% else %}{{ site.title }}{% endif %} Posted by {% if page.author %}{{ page.author }}{% else %}{{ site.title }}{% endif %} on {{ page.date | date: "%B %-d, %Y" }}</span>
                </div>
            </div>
        </div>
    </div>
</header>
```

现在变成了
```html
<header class="intro-header" >
    <div class="header-mask"></div>
    <div class="container">
        <div class="row">
            <div class="col-lg-8 col-lg-offset-2 col-md-10 col-md-offset-1">
                <div class="post-heading">
                    <div class="tags">
                        {% for tag in page.tags %}
                        <a class="tag" href="{{ site.baseurl }}/tags/#{{ tag }}" title="{{ tag }}">{{ tag }}</a>
                        {% endfor %}
                    </div>
                    <h1>{{ page.title }}</h1>
                    {% comment %}
                        always create a h2 for keeping the margin , Hux
                    {% endcomment %}
                    {% comment %} if page.subtitle {% endcomment %}
                    <h2 class="subheading">{{ page.subtitle }}</h2>
                    {% comment %} endif {% endcomment %}
                    <span class="meta"> {% if page.author %}{{ page.author }}{% else %}{{ site.title }}{% endif %} post almost {{ page.content | strip_html | strip_newlines | remove: " " | size }} words on {{ page.date | date: "%B %-d, %Y" }}</span>
                </div>
            </div>
        </div>
    </div>
</header>
```

至于显示区别 之前的是：Posted by 作者名 on 月日年；现在变成了：作者名 post almost 字的个数 words on 月日年；

## 在jekyll下使用Latex
找了一圈才发现写在了自己的知乎里面，可以参考其中第六个问题：https://zhuanlan.zhihu.com/p/34979398

