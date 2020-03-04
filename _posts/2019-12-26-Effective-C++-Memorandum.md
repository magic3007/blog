---
layout: article
tag: [C++]
title: Effective C++ Memorandum
---

[TOC]

# Reference

[Effective C++: 55 Specific Ways to Improve Your Programs and Designs (3rd Edition)](https://www.amazon.com/Effective-Specific-Improve-Programs-Designs/dp/0321334876)

[Effective Modern C++: 42 Specific Ways to Improve Your Use of C++11 and C++14](https://www.amazon.com/Effective-Modern-Specific-Ways-Improve/dp/1491903996)



> 为了充分利用C++的特性, 如何看待今天的C++的编程范式?

今天C++以及是一个强大的多重范式编程语言(multiparadigm programming language), 可以同时支持过程形式(procedural), 面向对象形式(object-oriented), 函数形式(functional), 范式形式(generic), 元编程形式(meta-programming). C++可以视为一个语言联邦.

为了理解C++的各种语言范式, 我们可以从下面四个sub-language次语言的角度理解:

- C part of C++. 

- Object-Oriented C++. C++最初的名称即为C with Classes.

- Template C++. 这是C++的泛型编程(generic programming)的部分, 可以带来崭新的编程范式template-metaprogramming(TMP, 模板元编程).

- STL. STL是一个非常的特殊的template库

  

# 常量定义规范

第一个建议是尽可能用`const`, `enum`, `inline` 来 取代 `#define`, 这样的一个理由是可以变量或函数名可以存在于符号表, 方便调试.

> 辨析 `const char *` 和 `char * const`.
>
> - 前者是指针指向内容不可变, 后者是指针本身不可变
> - 头文件常量定义式通常结合两者 `const char * const = "Hello~"`

关于`enum`有一个概念是**enum hack**, 其理论基础是*一个枚举类型的数值可权充`int`被使用*, **enum hack**的行为与`#define`类似, 不会导致非必要的内存分配, 也不能取地址, 但是能更好约束区块作用域block scope.

最后对于形似函数的宏macros, 最好用inline和模板定义.

关于`const`的用法, 补充几点:
- 类似于`const char *` 和 `char * const`, STL的迭代器亦有类似使用方式, 如`const std::vector<int>::iterator`和`std::vector<int>::const_iterator`.

- 对于logical constness的成员函数在对象中尽量使const; 在const成员函数中为摆脱`bitwise constness`的死板限制可以使用`mutable`关键字.

最后关于对象初始化问题:
- 构造函数走好使用成员初始列(member initialization list)

- 注意构造顺序是在class中的声明顺序而不是成员初始列中顺序, 故成员初始列中的顺序最好与在class中的声明顺序相同

- 为解决跨编译单位的初始化次序, 推荐使用单体模式(Singleton).



# Constructors, Destructors & Assignment

## Disallow default copy & assign

注意到C++编译器可能暗自为class创建了默认构造函数, 复制构造函数, 赋值运算符和析构函数. 如果不想编译器默认构造复制函数和赋值运算符, 可以将相应的成员函数定义为private并不给予实现. 为此定义一个宏定义:

```c++
#define DISALLOW_COPY_AND_ASSIGN(className) \
  className(const className&) = delete; \
  className& operator=(const className&) = delete;
```

## Prevent exceptions from leaving destructors

防止异常逃离析构函数的理由在于, 析构函数的调用往往是隐式的并且有可能与其他对象的析构函数同时被调用析构函数, 如在离开作用域时该作用域内所有的临时对象都被调用析构函数,删除一个`std::vector`时每个元素都要调用析构函数.

一个好的写法是既要在析构函数中捕获异常, 也需要给用户提供一个捕获异常的机会.

```c++
class A{
    public:
    	void close(){
            b.close();
            closed = true;
        }
    	~A(){
            if(!closed){
                try{
                    b.close();
                }catch{
                    ...
                }
            }
        }
    private:
    	T b;
    	bool closed;
};
```

## Resrouce Management

常见的资源包括内存, 文件描述符, 锁等

对于内存资源, 我们可以用**智能指针管理**.

利用类的构造和析构函数, 我们也可以做一些资源管理. 比如对于static变量, 我们把资源的获取放在构造函数里面做, 这样只有第一次访问该变量的时候才会调用构造函数从而获取资源. 又例如对于文件描述符, 锁等在一个作用域内有效的, 我们可以利用把资源的释放分别放在一个类的析构函数内做, 当然这个函数可以是匿名函数. 

```c++
class DeferredAction {
 public:
  DISALLOW_COPY_AND_ASSIGN(DeferredAction)

  explicit DeferredAction(std::function<void(void)> f)
    : f_(std::move(f)) {}

  inline ~DeferredAction() {
    f_();
  }

 private:
  std::function<void(void)> f_;
};
```

如果某实例不一定在一个作用域内有效, 可以通过复制, 赋值等方式传递, 我们可以利用`std::shared_ptr`的`deleter`, 注意下面这个例子不需要析构函数.

```c++
class LockGuard {
 public:
   explicit LockGuard(Mutex *pm): mutex_ptr(pm, unlock) {
    	lock(mutex_ptr.get());
   }
 private:
    std::shared_ptr<Mutex> mutex_ptr;
};
```

当然为了让用户尽可能使用智能指针等模式, 可以采用**工厂模式**"先发制人", 返回一个智能指针, 甚至可以规定deleter.





