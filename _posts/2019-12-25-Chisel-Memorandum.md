---
layout: article 
title: Chisel Memorandum 
---

最近看了一下[**Chisel Bootcamp**](https://mybinder.org/v2/gh/freechipsproject/chisel-bootcamp/master)，这里记录一下心得体会.

[TOC]

## Scala & Chisel Primer

Chisel是一种基于Scala的高层次硬件描述语言. 而Scala的设计哲学即为集成面向对象编程和函数式编程, 非常适合用来作为硬件描述语言. Scala 运行在Java虚拟机上, 并兼容现有的Java程序. 作为基础我们必须先了解一些Scala的一些语法以及编程特性. 但是需要时刻记住的是, Chisel毕竟是一门硬件描述语言, 我们需要时刻注意`Scala Object`与`Chisel Object`的区别与联系.

Scala一门面向对象语言, 可以认为Scala程序是`Object`对象的集合. 顺便复习一下下面几个概念的关系:

- Object对象: 对象是类的实例
- Class类: 类是对象的抽象
- Method方法: 一个类可以包括多个方法
- Field字段: 每个对象都有其唯一的实例化的变量集合, 即字段. 

### Naming Specifications

- Scala是大小敏感的
- Class类名要求首字母大写, Method要求首字母小写
- 程序文件名建议等于类名, 并以`.scala`作为拓展名
- 入口函数定义为`def main(args: Array[String])`

### Package definitions & Importing Packages

Scala中定义包有两种方法, 一种和Java类似, 在文件头用`package <package name>`定义, 这种方式只能在一个文件里面定义一个类; 另一种类似于C#, 加上大括号curly braces用`package <package name> {...}`定义, 这种方式允许在一个文件内定义多个类.如

```scala
// Method one
package com.runoob
class HelloWorld

// Method two
package com.runoob {
  class HelloWorld 
}
```

`import`与java类似, 甚至可以引用java的包, 注意通配符是下划线 `_`.

```scala
import java.awt.Color  // 引入Color
import java.awt._  // 引入包内所有成员
```

### Data Types

Scala的常用数据类型见[此](https://www.tutorialspoint.com/scala/scala_quick_guide.htm). 这里记录几个特殊的:

- `Unit`: 表示无值, 功能类似于Java里面`void`.

- `Nothing`: Scala中所有类的子类

- `Any`:  Scala中所有类的基类

- `AnyRef`: Scala中所有引用类reference class的基类 

注意在Scala中数据类型也都是对象, Scala并没有类似于java的原生类型, 因此在Scala中对一个基本数字变量可以调用方法.

### Variables & Constants

在Scala中变量variable和常量constant分别用`var`和`val`指定, 同时可指定数据类型.

```scala
<val|var> <VariableName> [: DataType] [= Initial Value]
```

需要注意的是, `Chisel Oject`描述的是一个一个硬件实体, 即使这个硬件存的值会变, 但是这个硬件本身不会变, 故需要用常量`val`定义.

```scala
// Chisel Code: Declare a new module definition
class Passthrough extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(4.W))
    val out = Output(UInt(4.W))
  })
  io.out := io.in
}
```

我们可以看到, 一个`Chisel Object`实际是一个继承一个被称为`Module`的`Scala object`, 其定义了`reset`, `clock`等基本硬件单元的接线.

### Conditionals

Scala的条件变量与C/C++, Java等类似, 同时`if`语句也有类似于C/C++中三目运算符的返回值, 其返回值为其所选择的分支的最后一行语句的返回值.

```scala
val alphabet = "abcdefghijklmnopqrstuvwxyz"
val likelyCharactersSet = if (alphabet.length == 26)
    "english"
else 
    "not english"
println(likelyCharactersSet)
```

### Loop

类似于java, Scala的循环语句有`while`, `do while` 和 `for`, 其中只有`for`与Java略有不同, 可以设置步长, 通过分号设置多个循环变量(实际是设置多重循环), 循环集合List, 还能在for循环中通过if语句进行过滤(用分号可以设置多个循环条件).

```scala
for (var x <- a to b) // [a,b]

for (var x <- a until b) // [a,b)

for (var x <- a to b; var y <- a to b)

for (var x <- list)

for (var x <- list
     if condition1; is condition2 ...)
```

### `yield`  in `for` statement

类似于Python, 我们还能通过`yield`关键字保存循环变量的值.

```scala
val numList = List(1,2,3,4,5,6,7,8,9,10);

var retVal = for{ a <- numList 
                if a != 3; if a < 8
              }yield a

for( a <- retVal){
 println( "Value of a: " + a );
}
```

### Methods(Functions)

Scala的函数Functions的定义形式如下. 注意几个要点

- Scala是强调类型的, 不支持隐式类型转换, 函数返回类型最好还是写一下
- Scala是面向对象的, 每个函数都是一个`Scala Object`
- 如果没有写等号和函数体, 被视为声明; 如果在其他地方没有补充函数体, 则该函数会被声明为抽象类
- Scala支持默认参数, 类似于C/C++,  无默认值的参数在前，默认参数在后.
- Scala指定传参的时候乱序指定参数名

```scala
def <function name>(<parameter name> : <type> [ = <InitialValue> ] [, ...]): [return type] = { 
	... 
}
```
如果函数没有返回值, 可以返回`Unit`, 类似于Java的`void`:

```scala
def printMe( ) : Unit = {
    println("Hello, Scala!")
}
```

类似地, Scala的函数支持内嵌定义(作用域为大括号curly braces内).

```scala
def asciiTriangle(rows: Int) {
    
    // This is cute: multiplying "X" makes a string with many copies of "X"
    def printRow(columns: Int): Unit = println("X" * columns)
    
    if(rows > 0) {
        printRow(rows)
        asciiTriangle(rows - 1) // Here is the recursive call
    }
}

// printRow(1) // This would not work, since we're calling printRow outside its scope
asciiTriangle(6)
```

作为函数式编程设计语言, Scala有很多编程特性值得我们学习一下.

#### Call-by-name && Call-by-value

Scala在解析函数参数的时候有两个方式, 分别是call-by-name传值调用和call-by-value传名调用, 两者的区别如下:

- 传值调用(call-by-value): 先计算参数表达式的值，再应用到函数内部

- 传名调用(call-by-name): 将未计算的参数表达式直接应用到函数内部

其中传名调用的写法为

```scala
def <MethodName>( <ParameterName> => <DataType>)
```

从下面这个例子我们可以看到这两者的区别:

```scala
def main(args: Array[String]) {
        delayed(time());
}

def time() = {
  System.nanoTime
}

def delayed( t: => Long ) = {
  println("parameter 1： " + t)
  println("parameter 2:  " + t)
}
```

#### Variable Parameters

Scala允许指定最后一个参数是可重复的, 在最后一个参数类型后加一个星号`*`即可写法为:

```scala
def <MethodName>(<ParameterName>:<DataType>*) = {
    ...
}
```

#### Anonymous Methods

在函数式编程中, 匿名函数Anonymous Functions也是经常用到的. 在Scala中, 其形式定义如下.

```scala
(<parameter name>: <type>[, ...]) => ... or {...} // returned value is given by the last line of the function body
```

#### Higher-Order Function

之前说过, Scala中每一个函数都是一个对象`Object`, 可以为其他函数参数, 也可以作为函数返回值. 特别地, 一个函数的类型为`(parameter list) => <return type>`.

```scala
def main(args: Array[String]) {

      println( apply( layout, 10) )

   }

def apply(f: Int => String, v: Int) = f(v)

def layout[A](x: A) = "[" + x.toString() + "]" // method template
```

#### Partial Methods

实际上还是利用Scala中每一个函数都是一个对象`Object`, 先固定函数的部分参数, 构造出一个新的函数.

```scala
def main(args: Array[String]) {
      val date = new Date
      val logWithDateBound = log(date, _ : String)

      logWithDateBound("message1" )
      Thread.sleep(1000)
      logWithDateBound("message2" )
      Thread.sleep(1000)
      logWithDateBound("message3" )
   }

   def log(date: Date, message: String)  = {
     println(date + "----" + message)
   }
```

#### Function Currying

函数柯里化(Currying)指的是将原来接受两个参数的函数变成新的接受一个参数的函数的过程, 新的函数返回一个以原有第二个参数为参数的函数.

实际上还是利用Scala中函数是对象的特性以及匿名函数实现,如:

```scala
def add(x:Int,y:Int)=x+y

def add(x:Int)=(y:Int)=>x+y // function currying


val result = add(1) 
val sum = result(2)
println(sum) // 3
```

#### Closure

类似Python, Scala也可以定义闭包, 一个函数的返回值依赖于一个外部的自由变量.

```scala
def main(args: Array[String]) {  
      println( "muliplier(1) value = " +  multiplier(1) )  
      println( "muliplier(2) value = " +  multiplier(2) )  
   }  
var factor = 3  
val multiplier = (i:Int) => i * factor
```

### Template

函数模板的一个例子如下

```scala
def <MethodName>[T](<parameter name> : T [, ...]): [return type] = { 
	... 
}
```

### Class, Object & Trait

- Scala的类通过`new`创建对象
- Scala通过`extend`关键字继承类, 重新非抽象字段需要使用`overwrite`, 同时Scala不支持多重继承
- singleton使用`object`; 当单例对象与某个类共享同一个名称时，他被称作是这个类的伴生对象companion object, 类和它的伴生对象可以互相访问其私有成员 
- Scala中接口用特征trait

## 模块参数化

