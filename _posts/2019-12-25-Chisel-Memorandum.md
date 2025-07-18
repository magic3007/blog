---
layout: article
title: Chisel Memorandum
---

最近看了一下[**Chisel Bootcamp**](https://mybinder.org/v2/gh/freechipsproject/chisel-bootcamp/master)，这里记录一下心得体会.

[TOC]

## Scala Primer

Chisel是一种基于Scala的高层次硬件描述语言. 而Scala的设计哲学即为集成面向对象编程和函数式编程, 非常适合用来作为硬件描述语言. Scala 运行在Java虚拟机上, 并兼容现有的Java程序. 作为基础我们必须先了解一些Scala的一些语法以及编程特性.

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

值匹配使用`match/case`, 通配符是`-`:

```scala
// y is an integer variable defined somewhere else in the code
val y = 7
/// ...
val x = y match {
  case 0 => "zero" // One common syntax, preferred if fits in one line
  case 1 =>        // Another common syntax, preferred if does not fit in one line.
      "one"        // Note the code block continues until the next case
  case 2 => {      // Another syntax, but curly braces are not required
      "two"
  }
  case _ => "many" // _ is a wildcard that matches all values
}
println("y is " + x)
```

Scala还支持多重值匹配:

```scala
def animalType(biggerThanBreadBox: Boolean, meanAsCanBe: Boolean): String = {
  (biggerThanBreadBox, meanAsCanBe) match {
    case (true, true) => "wolverine"
    case (true, false) => "elephant"
    case (false, true) => "shrew"
    case (false, false) => "puppy"
  }
}
println(animalType(true, true))
```

Scala对于类型匹配和多重类型匹配, 分别需要使用`s: <Type>` 和 `_: <Type>`:

```scala
val sequence = Seq("a", 1, 0.0)
sequence.foreach { x =>
  x match {
    case s: String => println(s"$x is a String")
    case s: Int    => println(s"$x is an Int")
    case s: Double => println(s"$x is a Double")
    case _ => println(s"$x is an unknown type!")
  }
}
```

```scala
val sequence = Seq("a", 1, 0.0)
sequence.foreach { x =>
  x match {
    case _: Int | _: Double => println(s"$x is a number!")
    case _ => println(s"$x is an unknown type!")
  }
}
```

但是注意Scala的类型匹配可能有多态导致的问题:

```scala
val sequence = Seq(Seq("a"), Seq(1), Seq(0.0))
sequence.foreach { x =>
  x match {
    case s: Seq[String] => println(s"$x is a String")
    case s: Seq[Int]    => println(s"$x is an Int")
    case s: Seq[Double] => println(s"$x is a Double")
  }
}
/*
List(a) is a String
List(1) is a String
List(0.0) is a String
*/
```

### Loop

类似于java, Scala的循环语句有`while`, `do while` 和 `for`, 其中只有`for`与Java略有不同, 可以设置步长, 通过分号设置多个循环变量(实际是设置多重循环), 循环集合List, 还能在for循环中通过if语句进行过滤(用分号可以设置多个循环条件).

```scala
for (var x <- a to b) // [a,b]

for (var x <- a until b) // [a,b)

for (var x <- a to b by 2) // [a,b], step = 2

for (var x <- a to b; var y <- a to b)

for (var x <- list)

for (var x <- list
     if condition1; is condition2 ...)
```

### *yield*  in for statement

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

在Scala中, 函数的类型通过`(<para. list>) => T`指定. 在函数式编程中, 我们往往只关心返回值的类型, 也可以用`=> T`指定, 如`List`的`fill`函数定义为:

```scala
def fill[A](n: Int)(elem: => A): LazyList[A]
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

### Object Oriented Programming

先来看一个简单的定义class的例子是:

```scala
class MyBundle extends Bundle {
	val a = Bool ()
	val b = UInt (32. W )
}
```

Scala中关于面向对象编程有几个特殊概念需要注意, 分别是abstract class抽象类, trait特征(相当于java的接口, 与抽象类类似, 但是没有构造参数, 可被多重继承), object单例对象, companion object伴生对象和companion class伴生类, case class样例类 以及anonymous class匿名类.

Scala的类通过`new`创建对象, 或者后面我们可以看到也可以通过伴生对象的来创建.

Scala通过`extend`关键字继承类, 重新定义非抽象字段需要使用`overwrite`, 同时Scala不支持多重继承类, 但是trait是一个例外, 一个class可以继承多个trait. trait的多重继承如下

```scala
class MyClass extends HasTrait1 with HasTrait2 with HasTrait3
```

Scala中singleton使用单例对象object, 单例对象object不需要被new. 当单例对象与某个类共享同一个名称时，他被称作是这个类的**伴生对象**(companion object); 反过来这个类被称为这个对象的companion class**伴生类**. 类和它的伴生对象可以互相访问其私有成员. 区分companion object及其companion class的重要依据是名字前是否有new

```scala
object Lion {
    def roar(): Unit = println("I'M AN OBJECT!")
}
class Lion {
    def roar(): Unit = println("I'M A CLASS!")
}
new Lion().roar()
Lion.roar()
/*
I'M A CLASS!
I'M AN OBJECT!
*/
```

伴生对象和伴生类在Scala中非常重要. 之前说过, Scala中的`Int`, `List`等不是原生数据类型, 是一个对象. 其实这个说法不同准确, 实际上Scala中`UInt`, `List`等, 既是一个类名, 也是一个(单例)对象名, 它们两个互为伴生类和伴生对象. 伴生对象的用处在于:

1. 定义伴生类中常量
2. 为伴生类定义多个构造函数(通过`apply`statement), 同时允许在调用类构造函数前后执行某些操作

> #### *Apply*
>
> 这里必须讨论一下关键字`apply`, 这在Chisel的源码的经常用到. 在Scala中, 函数也是对象, 对于函数来说`apply`方法意味着调用函数本身, 即`fun.apply([parameters list]) = fun([parameters list])`
>
> ```scala
> val f = (x : Int) => x + 1
> f.apply(3) // 4
> ```
>
> `apply`关键字另外一个作用是在单例对象中用作设计模式中的**工厂模式**. 之前说过, Scala中的`List`既是一个类名, 也是一个单例对象名. 举个例子, 我们可以对`Object List` 使用`List.apply(1, 2, 3)`或者直接`List(1, 2, 3)`创建一个`Class List`的对象实例.

case class样本类在Scala中是一个比较特殊的类, 其有几个特性:

1. 对于类的所有构造参数允许外部访问
2. 创建case class的同时创建了伴生对象, 以及在伴生对象中定义了一个默认的`apply`方法, 创建实例的时候不需要new.

下面是一个case class的例子

```scala
class Nail(length: Int) // Regular class
val nail = new Nail(10) // Requires the `new` keyword
// println(nail.length) // Illegal! Class constructor parameters are not by default externally visible

class Screw(val threadSpace: Int) // By using the `val` keyword, threadSpace is now externally visible
val screw = new Screw(2)          // Requires the `new` keyword
println(screw.threadSpace)

case class Staple(isClosed: Boolean) // Case class constructor parameters are, by default, externally visible
val staple = Staple(false)           // No `new` keyword required
println(staple.isClosed)
```


Scala还支持**inline defining**, 这实际上创建了一个**anonymous class**匿名类:

```scala
val my_bundle = new Bundle {
	val a = Bool ()
	val b = UInt (32. W )
}
```

###  scala.collection

列举collection里面几个常用的类及其常用的methods

#### List

- 构造列表的两种基本方法: `list(x,y,z)`等价于`x::y::z::Nil`
- 方法`head`返回第一个元素, `tail`返回除了第一个元素外的其他元素, `isEmpty`判断是否为空
- 连接列表用`:::`, `++` 或者 `List.concat`
- 生成一个指定重复数量的元素列表用`List.fill[A](n: Int)(elem: => A): LazyList[A]`
- 通过给定函数来生成列表用`List.tabulate`
- 反转列表用`List.reverse`
- 取前若干个元素用`def take(n: Int): List[A]`

#### Map

这里列出一些常用的methods

- `Object Map`的*apply*: `val map=Map("a" -> 1)`

- `.get`


#### Seq

- `def foreach[U](f: ((K, V)) => U): Unit`

### *Option*, *Some* & *None*

`option`是Scala的一个泛型抽象类, 分别有两个子类`Some` 和 `None`, collection中类的方法的返回值中经常是`option`类型的

```scala
val map = Map("a" -> 1)
val a = map.get("a")
println(a)
val b = map.get("b")
println(b)

/*
Some(1)
None
*/
```

`Option`有方法`get`和`getOrElse`, 可用于决定取值的行为:

```scala
val some = Some(1)
val none = None
println(some.get)          // Returns 1
// println(none.get)       // Errors!
println(some.getOrElse(2)) // Returns 1
println(none.getOrElse(2)) // Returns 2
```

对于类的省缺参数, 一般用`None`. 在`match/case`中, 可以利用多态的性质进行匹配:

```scala
class DelayBy1(resetValue: Option[UInt] = None) extends Module {
  val io = IO(new Bundle {
    val in  = Input( UInt(16.W))
    val out = Output(UInt(16.W))
  })
  val reg = resetValue match {
    case Some(r) => RegInit(r)
    case None    => Reg(UInt())
  }
  reg := io.in
  io.out := reg
}

println(getVerilog(new DelayBy1))
println(getVerilog(new DelayBy1(Some(3.U))))
```

### Generic

Scala中, 我们可以在定义class和生成实例的时候指定泛型,

```scala
class Stack[A] {
  private var elements: List[A] = Nil
  def push(x: A) { elements = x :: elements }
  def peek: A = elements.head
  def pop(): A = {
    val currentTop = peek
    elements = elements.tail
    currentTop
  }
}

val stack = new Stack[Int]
stack.push(1)
stack.push(2)
println(stack.pop)  // prints 2
println(stack.pop)  // prints 1
```

除此以外, Scala关于generic还有一个非常有趣的概念: **Upper Type Bound** 和 **Lower Type Bound**, 实际上着两者规定了类之间的继承关系.

```scala
T <: A // T is the subclass of A
T >: B // B is the subclass of T
T >: B <: A // B is the subclass of T, T is the subclass of A
```

在generic class或generic的定义中, 我们可以用upper/lower type bound约束传入的类的类型:

```scala
abstract class Animal {
 def name: String
}

abstract class Pet extends Animal {}

class Cat extends Pet {
  override def name: String = "Cat"
}

class Dog extends Pet {
  override def name: String = "Dog"
}

class Lion extends Animal {
  override def name: String = "Lion"
}

class PetContainer[P <: Pet](p: P) {
  def pet: P = p
}

val dogContainer = new PetContainer[Dog](new Dog)
val catContainer = new PetContainer[Cat](new Cat)

// this would not compile
val lionContainer = new PetContainer[Lion](new Lion)
```

### About Debug

- 断言使用 `require(...)`

- `println(s"...$<...>")` 类似于Python, 是可执行的字符串.

  ```scala
  class ParameterizedScalaObject(param1: Int, param2: String) {
    println(s"I have parameters: param1 = $param1 and param2 = $param2")
  }
  val obj1 = new ParameterizedScalaObject(4,     "Hello")
  val obj2 = new ParameterizedScalaObject(4 + 2, "World")
  ```



## Chisel Primer

### Chisel Model

Chisel建立在Scala之上, 一个chisel model实际是一个继承一个被称为`Module`的`Scala object`, 其定义了`reset`, `clock`等硬件单元的基本接线. 需要时刻记住的是, Chisel毕竟是一门硬件描述语言, 我们需要时刻注意`Scala Object`与`Chisel Object`的区别与联系. 下面定义了一个简单的`Chisel Object`.

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

这个例子有几点值得注意:

- `Chisel Oject`描述的是一个一个硬件实体, 即使这个硬件存的值会变, 但是这个硬件本身不会变, 故需要用常量里面的`val`定义.

- `val io = IO(...)`. 输入和输出接口我们必须定义在一个特殊常量中, 且这个这个常量名必须是`io`, 并通过`Module`里面的单例对象`IO`的`apply`方法生成. 单例对象`IO`的`apply`方法声明见[此](https://www.chisel-lang.org/api/latest/chisel3/experimental/IO$.html).

- ```scala
  new Bundle {
      val in = Input(UInt(4.W))
      val out = Output(UInt(4.W))
  }
  ```

  利用**inline defining**创建了匿名类. `Input`和 `Output` 都是Module内部的Object, 可以理解为工厂模式.

  另外Chisel也有自己的Data Types. `UInt(4.W)`可以代表一个硬件, 而不一个是Scala的数字变量, 同时`4.W`也是一个Chisel定义的变量.

- ` io.out := io.in`. 注意我们是硬件电路意义上的连线不是赋值, 需要用`:=`而不是Scala的`=`(而且`val`也不支持赋值)

### Chisel Data Types , Conditionals & Operations

这部分毕竟简单, 只需要主要到Chisel和Scala的数据类型区别和联系就可以了(Chisel的基本数据类型会映射到硬件, 而Scala不会). 这部分直接看[Chisel CheatSheet](https://github.com/freechipsproject/chisel-cheatsheet/releases/latest/download/chisel_cheatsheet.pdf)即可. 这里提几个需要注意的点:

- `+` 不会考虑进位, 考虑进位需要用到 `+&`
- 判断是否相等是triple qeuals`===`
- Chisel常数定义类似于 `-3.S`, `1.U`等, `32.W`用于指定位宽

### Wire, Register & Memory

**wire**仅仅通过线连接, 可立刻更新.

```scala
// Allocate a as wire of type UInt ()
val x = Wire ( UInt ())
x := y // Connect wire y to wire x
```

**register**只有在时钟上升沿的时候才会更新, 常用的函数有:

```scala
RegInit(7.U(32.w)) // reg with initial value 7
RegNext(next_val) // update each clock, no init
RegEnable(next, enable) // update, with enable gate
```

### About Debug

- 编译时断言用Scala的`require(...)`, 模拟时的断言需要用Chisel的`assert(...)`

- 模拟时输出用Chisel的`printf(...)`,同时其也支持可执行字符串`printf(p"$<name>")`

#### Unit Test in Chisel

  常用单元测试导入的package和测试方法如下.

```scala
import chisel3.iotesters.{ChiselFlatSpec, Driver, PeekPokeTester}

Driver( () => new <ModuleName>){
    c => new PeekPokeTester(c) {
        poke(c.io.in, ...)
        expect(c.io.out, ...)
        step(<n_steps>)
    }
}
```

### Chisel Design Trick: Parameterization

Chisel利用参数化来生成硬件的tricks有:

- 利用Scala的`Option`作为缺省参数, 或选择性排除某些硬件设备
- 传入函数作为参数生成硬件设备

一个利用Scala的`Option`作为缺省参数的例子

```scala
class DelayBy1(resetValue: Option[UInt] = None) extends Module {
  val io = IO(new Bundle {
    val in  = Input( UInt(16.W))
    val out = Output(UInt(16.W))
  })
  val reg = resetValue match {
    case Some(r) => RegInit(r)
    case None    => Reg(UInt())
  }
  reg := io.in
  io.out := reg
}

println(getVerilog(new DelayBy1))
println(getVerilog(new DelayBy1(Some(3.U))))
```

有时，我们希望选择性地包括或排除IO. 有一些内部状态或许可以很好地进行调试，但是在最终的系统中我们希望可以隐藏起来. 这里举个参数化指定字段的例子, 还是使用`Option`, 可根据参数定义一个全加器或半加器.

```scala
class HalfFullAdder(val hasCarry: Boolean) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(1.W))
    val b = Input(UInt(1.W))
    val carryIn = if (hasCarry) Some(Input(UInt(1.W))) else None
    val s = Output(UInt(1.W))
    val carryOut = Output(UInt(1.W))
  })
  val sum = io.a +& io.b +& io.carryIn.getOrElse(0.U)
  io.s := sum(0)
  io.carryOut := sum(1)
}
```

当然上述例子, 也可以利用Chisel中<u>零宽度的wire恒为零</u>的特性来做:

```scala
class HalfFullAdder(val hasCarry: Boolean) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(1.W))
    val b = Input(UInt(1.W))
    val carryIn = Input(if (hasCarry) UInt(1.W) else UInt(0.W))
    val s = Output(UInt(1.W))
    val carryOut = Output(UInt(1.W))
  })
  val sum = io.a +& io.b +& io.carryIn
  io.s := sum(0)
  io.carryOut := sum(1)
}
println("Half Adder:")
println(getVerilog(new HalfFullAdder(false)))
println("\n\nFull Adder:")
println(getVerilog(new HalfFullAdder(true)))
```

下面是一个Mearly Machine的例子, 利用传入的状态转换函数来生成Mearly Machine.

```scala
// Mealy machine has
case class BinaryMealyParams(
  // number of states
  nStates: Int,
  // initial state
  s0: Int,
  // function describing state transition
  stateTransition: (Int, Boolean) => Int,
  // function describing output
  output: (Int, Boolean) => Int
) {
  require(nStates >= 0)
  require(s0 < nStates && s0 >= 0)
}

class BinaryMealy(val mp: BinaryMealyParams) extends Module {
  val io = IO(new Bundle {
    val in = Input(Bool())
    val out = Output(UInt())
  })

  val state = RegInit(UInt(), mp.s0.U)

  // output zero if no states
  io.out := 0.U
  for (i <- 0 until mp.nStates) {
    when (state === i.U) {
      when (io.in) {
        state  := mp.stateTransition(i, true).U
        io.out := mp.output(i, true).U
      }.otherwise {
        state  := mp.stateTransition(i, false).U
        io.out := mp.output(i, false).U
      }
    }
  }
}

// example from https://en.wikipedia.org/wiki/Mealy_machine
val nStates = 3
val s0 = 2
def stateTransition(state: Int, in: Boolean): Int = {
  if (in) {
    1
  } else {
    0
  }
}
def output(state: Int, in: Boolean): Int = {
  if (state == 2) {
    return 0
  }
  if ((state == 1 && !in) || (state == 0 && in)) {
    return 1
  } else {
    return 0
  }
}

val testParams = BinaryMealyParams(nStates, s0, stateTransition, output)

class BinaryMealyTester(c: BinaryMealy) extends PeekPokeTester(c) {
  poke(c.io.in, false)
  expect(c.io.out, 0)
  step(1)
  poke(c.io.in, false)
  expect(c.io.out, 0)
  step(1)
  poke(c.io.in, false)
  expect(c.io.out, 0)
  step(1)
  poke(c.io.in, true)
  expect(c.io.out, 1)
  step(1)
  poke(c.io.in, true)
  expect(c.io.out, 0)
  step(1)
  poke(c.io.in, false)
  expect(c.io.out, 1)
  step(1)
  poke(c.io.in, true)
  expect(c.io.out, 1)
  step(1)
  poke(c.io.in, false)
  expect(c.io.out, 1)
  step(1)
  poke(c.io.in, true)
  expect(c.io.out, 1)
}
val works = iotesters.Driver(() => new BinaryMealy(testParams)) { c => new BinaryMealyTester(c) }
assert(works) // Scala Code: if works == false, will throw an error
println("SUCCESS!!") // Scala Code: if we get here, our tests passed!
```

### Chisel Design Trick: Collection

我们可以利用scala.collection来管理一组硬件,  我们可使用`scala.collection.mutable.ArrayBuffer[A]`, 这个类可以动态增删元素, 比较方便.

下面是一个利用Scala的collection的一个FIR Filter generator.

```scala
class MyManyElementFir(consts: Seq[Int], bitWidth: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(bitWidth.W))
    val out = Output(UInt(bitWidth.W))
  })

  val regs = mutable.ArrayBuffer[UInt]()
  for(i <- 0 until consts.length) {
      if(i == 0) regs += io.in
      else       regs += RegNext(regs(i - 1), 0.U)
  }

  val muls = mutable.ArrayBuffer[UInt]()
  for(i <- 0 until consts.length) {
      muls += regs(i) * consts(i).U
  }

  val scan = mutable.ArrayBuffer[UInt]()
  for(i <- 0 until consts.length) {
      if(i == 0) scan += muls(i)
      else scan += muls(i) + scan(i - 1)
  }

  io.out := scan.last
}
```

有时候, 我们也需要Chisel提供的collection. 比如在Bundle里面需要提供Chisel instance的变量, 又比如我们访问collection的时候下标只能是一个Chisel的硬件(比如Register File). 实际上, Chisel也提供了collection 类型, 被称为`Vec`. Vec与Scala.collection具有相似的方法, 但是元素只能是Chisel instance.

这里列举一个Vec的常用用法, 实际上和`Reg(..)`和`RegInit(...)`类似, 详细可查文档.

- 初始化长度和初始值: `VecInit(...)`

- 创建但不初始化: `Vec(...)`

这里给了一个FIR Filter的例子, 这里的参数是通过IO传入而不是构造的时候指定的.

```scala
class MyManyDynamicElementVecFir(length: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
    val consts = Input(Vec(length, UInt(8.W)))
  })

  // Reference solution
  val regs = RegInit(VecInit(Seq.fill(length - 1)(0.U(8.W))))
  for(i <- 0 until length - 1) {
      if(i == 0) regs(i) := io.in
      else       regs(i) := regs(i - 1)
  }

  val muls = Wire(Vec(length, UInt(8.W)))
  for(i <- 0 until length) {
      if(i == 0) muls(i) := io.in * io.consts(i)
      else       muls(i) := regs(i - 1) * io.consts(i)
  }

  val scan = Wire(Vec(length, UInt(8.W)))
  for(i <- 0 until length) {
      if(i == 0) scan(i) := muls(i)
      else scan(i) := muls(i) + scan(i - 1)
  }

  io.out := scan(length - 1)
}
```

下面是另一个**Register File**的例子, 由于read的访问下标是传入的参数, 故需要使用Chisel的`Vec`

```scala
class RegisterFile(readPorts: Int) extends Module {
    require(readPorts >= 0)
    val io = IO(new Bundle {
        val wen   = Input(Bool())
        val waddr = Input(UInt(5.W))
        val wdata = Input(UInt(32.W))
        val raddr = Input(Vec(readPorts, UInt(5.W)))
        val rdata = Output(Vec(readPorts, UInt(32.W)))
    })

    // A Register of a vector of UInts
    val reg = RegInit(VecInit(Seq.fill(32)(0.U(32.W))))

    when(io.wen && io.waddr=/=0.U){
        reg(io.waddr) := io.wdata
    }

    for(i <- 0 until readPorts){
        io.rdata(i) := reg(io.raddr(i))
    }

}
```

### Chisel Standard Library

Chisel提供了若干个常用的标准库, flow control方面的有`Decoupled`, `FIFO`, `Arbiter`, `RRArbiter`等. 在[Chisel cheatsheet](https://github.com/freechipsproject/chisel-cheatsheet/releases/latest/download/chisel_cheatsheet.pdf)里面有简单的介绍.

#### Flow Control related

`Decoupled(...)` 提供了经典的**valid-ready**数据输出模型, 分别有`vaild`, `ready`和`bits`三个字段, 如果需要数据输入模型则使用`Flipped(Decoupled(...))`

`Queue` 实际是一个两端decoupled的synonymous FIFO.

一个Queue的例子是:

```scala
class MyQueue extends Module {
    // Example circuit using a Queue
    val io = IO(new Bundle {
        val in = Flipped(Decoupled(UInt(8.W)))
        val out = Decoupled(UInt(8.W))
    })
    val queue = Queue(io.in, 2)  // 2-element queue
    io.out <> queue
}
```

> **Bulk Connections**
>
> ​	`io.out <> queue` 实际上等价于
>
> ```scala
> io.out.valid := queue.valid
> io.out.bits := queue.bits
> queue.ready := io.out.ready
> ```

`Arbiter` 可以解决单生产者多消费者, 或者多生成者, 单消费者的问题(flip一下即可), 根据arbiter的规则还分为低下标优先`Arbiter`和轮询`RRArbiter`.

####  Function Blocks

提供了Bitwise Utilities, OneHot encoding utilities(在muxes中特别有用), 多选器Mux和计数器Count等.

### Chisel Design Trick: functional programming

#### Build-in Higher-Order Functions

函数式编程的第一个技巧是使用higher-order functions高阶函数, 如zip, `zipWithchIndex`, map, reduce(对于空list可能失败, 还能指定结合方向`reduceLeft`, `reduceRight`), flod(与reduce类似, 但是可用指定初值,对于空list不会失败), scan(生成每一步的结果,生成列表list)这些都是Scala.List中的函数, 当然我们也可用自己指定高阶函数.

关FIR Filter有一个一行解决的写法:

```scala
io.out := (taps zip io.consts).map { case (a, b) => a * b }.reduce(_ + _)
```

在这个例子中, 先讨论一下如何制定函数:

- 对于需要unpack的元组, 需要先使用`case` statement.
- 如果数组中每个元素只出现一次, 用下划线代替`_`
- 详细指定参数用匿名函数, 如上面的`_+_`可用`(a,b) => a+b`代替

这里用higher-order function的方法, 再次手写一个arbiter, 可用看到程序非常简洁:

```scala
class MyRoutingArbiter(numChannels: Int) extends Module {
  val io = IO(new Bundle {
    val in = Vec(numChannels, Flipped(Decoupled(UInt(8.W))))
    val out = Decoupled(UInt(8.W))
  } )
    val Valids = io.in.map(_.valid)
    io.out.valid := Valids.reduce(_ || _)
    val selected = PriorityMux(Valids.zipWithIndex.map{case (valid, index) => (valid,index.U)})
    io.out.bits := io.in(selected).bits
    io.in.foreach(x => x.ready := 0.B)
    io.in(selected).ready := io.in(selected).valid && io.out.ready
}
```

利用函数式编程, 在Chisel中一个神经元的写法可以很简单:

```scala
class Neuron(inputs: Int, act: FixedPoint => FixedPoint) extends Module {
  val io = IO(new Bundle {
    val in      = Input(Vec(inputs, FixedPoint(16.W, 8.BP)))
    val weights = Input(Vec(inputs, FixedPoint(16.W, 8.BP)))
    val out     = Output(FixedPoint(16.W, 8.BP))
  })
  io.out := act(io.in.zip(io.weights).map{case (x, y) => x * y}.reduce(_ +& _))
}
```

Object Oriented Programming

在Chisel中, 我们常用的基类有:

- `Module`: 每一个chisel module都继承自`Module`

- `Bundle`: 每一个chisel IO都继承自`Bundle`, 或某些特殊情况下继承自`Bundle`的超类`Record`

- `Data`: `UInt` 或 `Bundle` 都有`Data`作为超类.

#### Example: Aynonymous FIFO & Asynchronous FIFO

我们用面向对象的方法来实现一个asynchronous FIFO. 首先, 我们先看一下Chisel中的synonymous FIFO,即`Queue`, 源代码见[此](https://github.com/freechipsproject/chisel3/blob/master/src/main/scala/chisel3/util/Decoupled.scala)(不得不说, 感觉写得挺好的😀)

先是定义了两个decoupled的接口. 下面的`val enq = Flipped(EnqIO(gen))` 中需要Flipped的原因结合后面class `Queue`注释里面的exmaple比较容易理解.

```scala
/** Producer - drives (outputs) valid and bits, inputs ready.
  * @param gen The type of data to enqueue
  */
object EnqIO {
  def apply[T<:Data](gen: T): DecoupledIO[T] = Decoupled(gen)
}
/** Consumer - drives (outputs) ready, inputs valid and bits.
  * @param gen The type of data to dequeue
  */
object DeqIO {
  def apply[T<:Data](gen: T): DecoupledIO[T] = Flipped(Decoupled(gen))
}

/** An I/O Bundle for Queues
  * @param gen The type of data to queue
  * @param entries The max number of entries in the queue.
  */
class QueueIO[T <: Data](private val gen: T, val entries: Int) extends Bundle
{ // See github.com/freechipsproject/chisel3/issues/765 for why gen is a private val and proposed replacement APIs.

  /* These may look inverted, because the names (enq/deq) are from the perspective of the client,
   *  but internally, the queue implementation itself sits on the other side
   *  of the interface so uses the flipped instance.
   */
  /** I/O to enqueue data (client is producer, and Queue object is consumer), is [[Chisel.DecoupledIO]] flipped. */
  val enq = Flipped(EnqIO(gen))
  /** I/O to dequeue data (client is consumer and Queue object is producer), is [[Chisel.DecoupledIO]]*/
  val deq = Flipped(DeqIO(gen))
  /** The current amount of data in the queue */
  val count = Output(UInt(log2Ceil(entries + 1).W))
}
```

```scala
/** An I/O Bundle for Queues
  * @param gen The type of data to queue
  * @param entries The max number of entries in the queue.
  */
class QueueIO[T <: Data](private val gen: T, val entries: Int) extends Bundle
{ // See github.com/freechipsproject/chisel3/issues/765 for why gen is a private val and proposed replacement APIs.

  /* These may look inverted, because the names (enq/deq) are from the perspective of the client,
   *  but internally, the queue implementation itself sits on the other side
   *  of the interface so uses the flipped instance.
   */
  /** I/O to enqueue data (client is producer, and Queue object is consumer), is [[Chisel.DecoupledIO]] flipped. */
  val enq = Flipped(EnqIO(gen))
  /** I/O to dequeue data (client is consumer and Queue object is producer), is [[Chisel.DecoupledIO]]*/
  val deq = Flipped(DeqIO(gen))
  /** The current amount of data in the queue */
  val count = Output(UInt(log2Ceil(entries + 1).W))
}

/** A hardware module implementing a Queue
  * @param gen The type of data to queue
  * @param entries The max number of entries in the queue
  * @param pipe True if a single entry queue can run at full throughput (like a pipeline). The ''ready'' signals are
  * combinationally coupled.
  * @param flow True if the inputs can be consumed on the same cycle (the inputs "flow" through the queue immediately).
  * The ''valid'' signals are coupled.
  *
  * @example {% raw %}{{{
  * val q = Module(new Queue(UInt(), 16))
  * q.io.enq <> producer.io.out
  * consumer.io.in <> q.io.deq
  * }}}{% endraw %}
  */
@chiselName
class Queue[T <: Data](gen: T,
                       val entries: Int,
                       pipe: Boolean = false,
                       flow: Boolean = false)
                      (implicit compileOptions: chisel3.CompileOptions)
    extends Module() {
  require(entries > -1, "Queue must have non-negative number of entries")
  require(entries != 0, "Use companion object Queue.apply for zero entries")
  val genType = if (compileOptions.declaredTypeMustBeUnbound) {
    requireIsChiselType(gen)
    gen
  } else {
    if (DataMirror.internal.isSynthesizable(gen)) {
      chiselTypeOf(gen)
    } else {
      gen
    }
  }

  val io = IO(new QueueIO(genType, entries))

        // register
  private val ram = Mem(entries, genType)
  private val enq_ptr = Counter(entries)
  private val deq_ptr = Counter(entries)
  private val maybe_full = RegInit(false.B) // maybe_full的作用是在首位指针相同的时候判断是空还是满

	  // wire
  private val ptr_match = enq_ptr.value === deq_ptr.value
  private val empty = ptr_match && !maybe_full
  private val full = ptr_match && maybe_full
  private val do_enq = WireDefault(io.enq.fire())
  private val do_deq = WireDefault(io.deq.fire())

  when (do_enq) {
    ram(enq_ptr.value) := io.enq.bits
    enq_ptr.inc()
  }
  when (do_deq) {
    deq_ptr.inc()
  }
  when (do_enq =/= do_deq) {
    maybe_full := do_enq
  }

  io.deq.valid := !empty
  io.enq.ready := !full
  io.deq.bits := ram(deq_ptr.value)
	// 对于空的情况直接过去
  if (flow) {
    when (io.enq.valid) { io.deq.valid := true.B }
    when (empty) {
      io.deq.bits := io.enq.bits
      do_deq := false.B
      when (io.deq.ready) { do_enq := false.B }
    }
  }
	// 对于full的情况pipeline
  if (pipe) {
    when (io.deq.ready) { io.enq.ready := true.B }
  }
	// 计算队列内的元素个数, 当entries是2的方幂的时候, 组合逻辑可以简洁一点
  private val ptr_diff = enq_ptr.value - deq_ptr.value
  if (isPow2(entries)) {
    io.count := Mux(maybe_full && ptr_match, entries.U, 0.U) | ptr_diff
  } else {
    io.count := Mux(ptr_match,
                    Mux(maybe_full,
                      entries.asUInt, 0.U),
                    Mux(deq_ptr.value > enq_ptr.value,
                      entries.asUInt + ptr_diff, ptr_diff))
  }
}
```







###  Generate *Verilog* & *Firrtl*

- 分别用`getVerilog(<Module Instance>)`



## Reference

