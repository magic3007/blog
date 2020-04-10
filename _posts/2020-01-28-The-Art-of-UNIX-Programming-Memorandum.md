---
layout: article 
title: The Art of UNIX Programming
Tag: [UNIX]
---



[TOC]


列举几个例子加以理解.

# 设计原则

## Modularity 模块性

> 如何能够真正实现 "不会直接调用其他模块的实现码" ?

实现模块化与解耦合是实际上不是一件那么容易的事情. 在面向对象编程中, 就目前的经验来看, 我会把API分成两大类:

- 一类是提供给他人使用, 不会对自身的状态产生影响. 这类比较简单, 直接提供接口即可.
- 另外一类是上希望对方在特定的地方调用, 或者是会对自身状态的产业影响的. 这些需要自己把API<u>注册</u>到对方的函数表上, 对方不知道这个函数是什么, 只需要知道在特定的事件调用的即可, 表现在汇编语言上即为间接跳转, 另一方面也体现着 把知识叠入数据的*表示原则*.

> 最佳模块大小是多少?

400-800物理行, 过多过少均不宜.

> 常见的违反紧凑性和正交性的例子有哪些?

- API入口点(如函数参数)超过了七个(*The Magical Number Seven, Plus or Minus Two: Some Limits on Our Capacity for Processing Information*)

- 目标是从某一(源)格式到另一个(目标)格式进行数据读取和解析, 却想当然得认为是从磁盘读取(为什么不能是标准输入, 网络套接字或者其他来源的数据流呢?)

> 如何提高设计的紧凑性和正交性?

- ~~入禅门: "教外别传, 不立文字", 依赖将导致痛苦~~
- 围绕"解决一个定义明确的问题"的强核心<u>算法</u>组织设计, 避免人为的假设(抽象, 抽象, 还是抽象!)

> 如何看待UNIX的薄胶合层原则与OO(面向对象)的过度封装?

- UNIX的模块化强调薄胶合层原则, 即硬件和程序顶层对象之间的抽象层越少越好
- 需要警惕OO显示出来的某种使程序员过度封装的倾向.

## Textuality 文本化

### 数据文件格式

总结几种常见的<u>数据文件格式</u>, 我们可以从上面得到一些关于对设计*UNIX文件格式的约定*的启示(详细条款见书).

| Format                          | Memo                                                         | Example                                                      |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DSV(Delimiter-Separated Values) | 每个记录一行, 字段用冒号隔开, 如文件`/etc/passwd`<br />适合传统UNIX工具, 如`grep`, `sed`, `awk`, `cut`<br />与Microsoft中常用的DSV类似, 但是仅仅用反斜杠`/`进行转义, 比DSV通过把整个字段用双引号包围更简单 |                                                              |
| XML                             | 类似于HTML, 利用尖括号标签和`&`记号<br />适合递归结构, 但是不适合传统UNIX工具<br />我们有两个工具解决这个问题: `xmltk`: 提供类似于`grep`等的面向流工具过滤XML文档; `PYX`: XML文档的面向行表示 |                                                              |
| Windows INI                     | 不是UNIX自带的, 但是在Windows的影响下UNIX开始支持.<br />可读性好, 但是与XML一样, 与`grep`等配合不好. | [INI file From Wikipedia](https://en.wikipedia.org/wiki/INI_file) |
| Record-Jar                      | 主要结合了cookie-jar格式(使用节格式,  用`%%`作为记录分隔符)和RFC 822格式(使用了`关键字:值`的方式进行记录) |                                                              |
| PNG                             | 像素数据用二进制保存, 但是由一系列自描述字节块(chunk)组成, 且包含版本号, 可拓展性好.<br />`SNG`格式是PNG的纯文本表示, 可与PNG无损转换, 利于用户编辑. |                                                              |

### 应用协议格式

- 经典的互联网元格式是文本格式
- 目前HTTP有作为通用应用协议的趋势, 其请求部分采用类似RFC822/MIME的格式
- 应用协议的一个发展趋势是在MIME中使用XML格式来架构请求和有效数据载荷.

## Transparency 透明性

区分两个概念:

- 透明性: 容易理解程序产生的效果, 但是对如何实现的不甚了解.
- 可显性: 容易理解程序的逻辑, 即理解代码的准入门槛较低. 提高可显性的例子是在程序执行过程中打印各个步骤的log. 

几个关于透明性和可显性的思考问题:

- 程序递归层次不要小于等于3
- API需要正交(抽象, 以算法为中心)
- API避免过多的magic flag
- 分层打印程序行为的log(info, trace, debug等)

## Minilanguages 微型语言

> 什么是微型语言?

微型语言提出基于长期观察得到的结论: 程序员每百行代码的出错率和所使用的语言是无关的. 模块化是一个方向, 但是在模块化编程的过程中, 人们发现在特定领域中总是有一些十分通用的功能模块，并且可以利用配置文件将这些功能模块组织起来去完成这个领域的多种不同的功能. 面对这种状况, 人们设计了一种具备了逻辑控制能力的“配置文件”，这就是微型语言, 如

- 文本匹配的正则表达式(如拓展的`perl`正则表达式)

- 编写shell程序的一些实用工具(`awk`、`sed`,`dc`,`bc`)

- 软件开发工具(`make`, `lex`, `yacc`等)

  

一些常用的实例如下:

- `m4`: 用于描述文本转换的宏处理程序, 与C预编译器执行的任务类似(但是谨慎使用宏拓展)
-  `Glade`: 图形界面创建工具 glade->XML->C/C++, python,或perl代码
-  `troff`: 排版格式器, 是格式工具套件(Documenter's Workbench或DWB)的核心, 其他组件在其中常作为`troff`的后处理器(如当代打印机可接受格式`PostScript`)或者预处理器(如用于制表的`tbl`, 用于排版数学公式的`eqn`和绘图的`pic`, 提供测绘图功能的`grap`, 在实际中他们通常通过管道进行连接组成一个协作系统. 基于此的`GPU plotutils`是一个有用的图像工具链).
-  `awk`: awk程序包含模式(正则表达式)/行为对, 执行时按行过滤文件. `awk`的设计之初注意针对报表, 但是其依靠的模式驱动框架的模式阻止了其通用性, 现代逐渐被`perl`取代.
-  `dc` & `bc` : `dc`逆波兰表达式计算器, 中值表达式计算器`bc`, 无限精度, 图灵完备, 只需作为从进程, 即可获得这种计算能力. 如可以用`perl`与逆波兰表达式计算器`dc`通过管道组合得到RSA公钥算法.
-  `JavaScript`: 最初不可修改磁盘, 但是后来对其的约束变少, 可通过DOM(文件对象模型)与环境交互, 但是可能被滥用(如浏览器弹出广告).



## Generation 生成

两个比较关键的概念是**数据驱动编程**和**代码生成**.



## Configuration

**configuration location**

- `etc` -> system environment variables -> dot files under `$HOME` -> user environment variables -> command-line options
- the latter covers the former



**command-line option style**

- UNIX Style: `-ab`=`-ba` `-j8`(whether is delimited by space is optional)

- GNU Style:  `--arch amd64` `--arch=amd64`

  

**common meanings of command-line options**(`-a` - `-z`):

| option | common meanings                           |
| ------ | ----------------------------------------- |
| -a     | all; append                               |
| -b     | buffer, block(`du`, `df`); batch          |
| -c     | command(`sh`); check                      |
| -d     | debug; directory; delete                  |
| -D     | define(define macro in `gcc`)             |
| -h     | header; help                              |
| -i     | interactive; initialize                   |
| -I     | include(`gcc`)                            |
| -k     | keep                                      |
| -l     | list; load(`gcc`)                         |
| -m     | message                                   |
| -n     | number(with parameters)                   |
| -o     | output(with parameters)                   |
| -p     | port; protocol                            |
| -q     | quite                                     |
| -r/-R  | recurse; reverse                          |
| -s     | silent; size                              |
| -t     | tag(with parameters)                      |
| -u     | user                                      |
| -v     | verbose; version                          |
| -V     | version                                   |
| -w     | width; warning                            |
| -x     | debug(similar to `-d`); extract           |
| -y     | yes(allow potential destructive behavior) |
| -z     | compress(`tar`, `zip`)                    |



## Language

| Language                   | Pros                                                         | Cons                                     | Portability                                                  |
| -------------------------- | ------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------------ |
| C                          | efficiency; portability                                      | resource/memory management               | poor support for IPC, threads and GUI                        |
| C++                        | OO; generic programming; STL; opensource library available online | complicated OO; cons of C are not solved | ! no C++ compiler completely achieve C++99 ISO standard nowadays |
| shell                      | small script                                                 |                                          | pure shell script is portable; however, there are always 3rd commands or filters in shell scripts |
| perl                       | enhanced shell; regex expression                             |                                          | not bad except for some plug-ins from CPAN                   |
| tcl(tool command language) |                                                              |                                          |                                                              |
| Python                     | legible                                                      | poor efficiency                          | brilliant except for the the updating gap between Python2 and Python3 |
| Java                       | OO; JVM                                                      |                                          | brilliant                                                    |



## Open Source

*Ensure the archived file is always extracted to a new directory instead of current directory.*

A trick in `makefile` to compress all the files under a directory.

```makefile
foobar-$(VERS).tar.gz:
	@find $(SRC) -type f | sed s:^:foobar-$(VERS)/: >MANIFEST
	@(cd ..; ln -s foobar foobar-$(VERS))
	(cd ..; tar -czvf foobar/foobar-$(VERS).tar.gz `cat foobar/MANIFEST`)
	@(cd ..; rm foobar-$(VERS))
```

See more info in: http://en.tldp.org/HOWTO/Software-Release-Practice-HOWTO/distpractice.html



# Reference

[The Art of Unix Programming](http://www.catb.org/~esr/writings/taoup/html/)

