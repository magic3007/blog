#!https://zhuanlan.zhihu.com/p/579688074
# 讨论DREAMPlace中的静电场求解算法

> 由于知乎上部分markdown语法转化过程出现了错误，本文中的部分公式和引用可能无法正常显示，建议在[Magic Mai's Blog: 讨论DREAMPlace中的静电场求解算法](https://magic3007.github.io/blog/2022/11/02/Poission-in-DREAMPlace.html)上阅读。
## 问题描述

在[DREAMPlace (Yibo Lin, DAC'19)](https://dl.acm.org/doi/10.1145/3316781.3317803)这篇文章中，作者将器件近似类比为带电体，建立了静电场模型作为器件的密度模型，并使用了谱方法来求解静电场。当前，基于静电场系统的全局布局算法在学术界布局算法中获得了SOTA的性能和效率，其中关于如何高效计算静电场的电场强度和电势能在论文里面只是进行了简单的描述。最近组里来的本科实习生在研究这方面的算法，这里稍微写点东西讨论一下。

在DREAMPlace中，我们将整个版图$R$划分为$N \times M$个单元格，并根据类比为带电体的器件的位置分布得到版图上每个单元格上的电荷密度分布$\rho(x,y)$，其中$x=0, 1, \ldots, N-1, y = 0, 1, \ldots, M-1$。我们的目标是求解整个电场系统的电势能以及每个单元格X轴和Y轴方向上电场强度。

根据泊松方程，我们可以建立如下的方程组：


$$
\left\{\begin{array}{l}
\nabla \cdot \nabla \psi(x, y)=-\rho(x, y) \\
\hat{\mathbf{n}} \cdot \nabla \psi(x, y)=\mathbf{0},(x, y) \in \partial R \\
\iint_R \psi(x, y)=0 \\
\iint_R \rho(x, y)=0
\end{array}\right.
$$

其中第二行公式为Neumann边界条件[^1]，在这个边界条件下，静电场唯一性定理成立，并且我们求解的电势最多相差一个任意参数，而相差的参数无非就代表了电势零点的选取，其实还是对应于同一组电场。而这里第三行公式的作用其实就是为了选取电势零点。这里第四行关于电荷密度的公式有什么作用呢？我们可以先放下这个，后面再解释一下。

如何求解这个方程组呢？从简单情况出发，我们先从一维泊松方程开始讨论。


## Neumann边界条件的一维泊松方程

考虑如下Neumann边界条件的一维泊松方程：


$$
\left\{\begin{array}{l}
\psi''(x)=-\rho(x), & x \in (0,N) \\
\psi'(x) = 0, & x = 0 \text{ or } x = N\\
\int_0^N \psi(x)dx=0
\end{array}\right. \label{poisson_1d}
$$



傅立叶谱方法是求解此类偏微分方程的一种重要的数值方法，其主要包括傅立叶变换，余弦变换和正弦变换。一般来说，不同的变换方式适合不同的边界条件。这里的Neumann边界条件属于no-flux的边界条件，因此我们考虑使用离散余弦变换（DCT）来求解。

### 离散余弦变换DCT及其逆变换[^5]

假设我们有长度为$N$的序列$\{x(0), x(1), \ldots, x(N-1)\}$，我们对这个序列进行离散余弦变换和反变换。DCT形式有多种，其本质上都是对延拓为两倍长度的序列进行离散傅立叶变换（DFT），不同形式的差别其实是推导过程中延拓方式的差别。我们考虑使用下面的延拓方式：

记单位根为$W_{N}=e^{i\frac{2\pi}{N}}$. 我们构造新的序列$\widetilde{x}(n)$关于$x(n)$在$N-\frac{1}{2}$处对称表示, 即


$$
\widetilde{x}(n) = \left\{\begin{array}{l}
x(n), & \text{if}\  n = 0, 1, \ldots, N-1 \\
x(2N - n -1), & \text{if}\  n = N, N+1, \ldots, 2N-1
\end{array}\right.
$$


那么这个新构造的序列的DFT变换为：


$$
\begin{aligned}
\widetilde{X}(k) &=\sum_{n=0}^{2N-1} \widetilde{x}(n) W_{2 N}^{kn} \\
&=\sum_{n=0}^{N-1} \widetilde{x}(n) W_{2 N}^{kn}+\sum_{n=N}^{2N-1} \widetilde{x}(n) W_{2 N}^{kn} \\
&=\sum_{n=0}^{N-1} x(n) W_{2 N}^{kn}+\sum_{n=0}^{N-1} x(n) W_{2 N}^{k(2N-n-1)} \\
&=\sum_{n=0}^{N-1} x(n) [W_{2 N}^{nk} + W_{2 N}^{-(n+1)k}], \quad k=0,1, \ldots, 2 N-1
\end{aligned}
$$

两侧同乘$\frac{1}{2}W_{2N}^{k/2}$可得：

$$
\frac{1}{2}W_{2N}^{k/2}\widetilde{X}(k) = \frac{1}{2}\sum_{n=0}^{N-1} x(n) [W_{2 N}^{(n+\frac{1}{2})k} + W_{2 N}^{-(n+\frac{1}{2})k}]=\sum_{n=0}^{N-1} x(n)\cos(\frac{\pi}{N}(n+\frac{1}{2})k)
$$

我们令下述式子即为离散余弦变换(DCT)的形式，记为


$$
C(k)=\sum_{n=0}^{N-1} x(n)\cos(\frac{\pi}{N}(n+\frac{1}{2})k)=\frac{1}{2}W_{2N}^{k/2}\widetilde{X}(k) \label{dct}
$$


故$\widetilde{X}(k)=2W_{2N}^{-\frac{k}{2}}C(k)$。注意到，$\widetilde{x}(n)$是一个实数序列，由[^4]可得，**实数序列的离散傅里叶变换是埃尔米特序列；反之，一个埃尔米特序列的逆离散傅里叶变换是实序列**。因此$\widetilde{X}(k)$是一个埃尔米特序列，因此有


$$
\widetilde{X}(2N - k) = \widetilde{X}^*(k), \quad k = 1, \ldots, N
$$


另外由公式$\eqref{dct}$易得


$$
\widetilde{X}(N)=0
$$


同样，我们序列在$\widetilde{X}(k)$上进行离散傅立叶逆变换（IDFT）, 并将上述两个公式带入，得到离散余弦逆变换（IDCT)的形式：


$$
\begin{aligned}
\widetilde{x}(n) &=\frac{1}{2N}\sum_{k=0}^{2N-1} \widetilde{X}(k) W_{2 N}^{-kn} \\
&=\frac{1}{2N}[\widetilde{X}(0) + \sum_{k=1}^{N-1}(\widetilde{X}(k)W_{2 N}^{-kn} + \widetilde{X}^*(k)W_{2 N}^{-(2N-k)n})] \\
&=\frac{1}{2N}[\widetilde{X}(0) + \sum_{k=1}^{N-1}(\widetilde{X}(k)W_{2 N}^{-kn} + \widetilde{X}^*(k)W_{2 N}^{kn})] \\
&=\frac{1}{2N}(\widetilde{X}(0) + 2Re[\sum_{k=1}^{N-1}\widetilde{X}(k)W_{2N}^{-kn}]) \\
&=\frac{1}{2N}(2C(0) + 2\sum_{k=1}^{N-1}2C(k)Re[W_{2N}^{-k(n+\frac{1}{2})}]) \\
&=\frac{2}{N}\sum_{k=0}^{N-1}C'(k)\cos(\frac{\pi}{N}(n+\frac{1}{2})k) \\
\end{aligned}
$$

其中

$$
C'(k)=\left\{\begin{array}{l}
\frac{1}{2} C(k), & k = 0\\
C(k), & k = 1, 2, \ldots, N-1
\end{array}\right.
$$

即在$C(k)$的第零项要乘多一个系数$\frac{1}{2}$。因此我们可以得到如下的DCT和IDCT变换形式：
$$
\begin{align}
\text{DCT: } & C(k) = \sum_{n=0}^{N-1} x(n)\cos(\frac{\pi}{N}(n+\frac{1}{2})k)，& \quad k=0,1,\ldots,N-1 \\
\text{IDCT: } & x(n) = \frac{2}{N}(\frac{C(0)}{2}+\sum_{k=1}^{N-1}C(k)\cos(\frac{\pi}{N}(n+\frac{1}{2})k))，& \quad n=0,1,\ldots,N-1 \\
\end{align}
$$

因此上面的推导流程给了一个利用2N个点的DFT算法，在使用快速傅立叶变换算法的情况下，时间复杂度可以做到$O(N \log N)$。

### 快速余弦变换FDT及其逆变换

快速余弦变换由Narasimha与Peterson在1978年提出，此方法系借由巧妙的编排$y(n)$实现。假设：


$$
y(n) = x(2n), \text{ and } y(N-1-n) = x(2n+1), \quad n = 0,1, \ldots, \frac{N}{2}-1
$$


比如，当$N=7$的时候，重排后$\{y(n)\}=\{x(0), x(2), x(4), x(6), x(5), x(3), x(1)\}$。

对$x(n)$进行离散余弦变换并进行奇偶划分：


$$
\begin{aligned}
X(m) &= \sum_{n=0}^{N-1} x(n)\cos(\frac{\pi}{N}(n+\frac{1}{2})m) \\
&= \sum_{n=0}^{N/2-1}x(2n)cos(\frac{\pi}{N}(2n+\frac{1}{2})m) + \sum_{n=0}^{N/2-1}x(2n+1)cos(\frac{\pi}{N}(2n+\frac{3}{2})m) \\
&= \sum_{n=0}^{N/2-1}y(n)cos(\frac{\pi(4n+1)m}{2N})+\sum_{n=0}^{N/2-1}y(N-n-1)cos(\frac{\pi(4n+3)m}{2N}) \\
& = \sum_{n=0}^{N/2-1}y(n)cos(\frac{\pi(4n+1)m}{2N})+\sum_{n=\lceil N/2 \rceil}^{N-1}y(n)cos(\frac{\pi(4n+1)m}{2N}) \\
& = \sum_{n=0}^{N-1}y(n)cos(\frac{2\pi(n+\frac{1}{4})m}{N}), \quad m = 0, 1, \ldots, N-1
\end{aligned}
$$

因此$X(m)$是$y(n)$的某个scaled DFT的实部：


$$
\begin{align}
X(m) &= Re[H(m)] \\
H(m) &= W_{N}^{m/4}\sum_0^{N-1}y(n)W_N^{mn}=W_{N}^{m/4}Y(m)
\end{align}
$$

其中$Y(m)$是$y(n)$的DFT，同样由于$y(n)$是实数序列，故$Y(m)$是埃尔米特序列：


$$
Y(N-m) = Y^*(m)
$$

并进一步可以验证$H(m)$满足：$H(N-m)=iH^*(m)$。

因此欲求之$X(m)=Re[H(m)]$, $X(N-m)=Re[H(N-m)]=Im[H(m)], \quad m=0,1,\ldots, \frac{N}{2}$.

因此对$x(n)$做DCT，先进行重排，然后求某一个scaled DFT，最终scaled DFT的前半部分的实部就是$x(n)$的DCT的前部，后半部分的虚部就是$x(n)$的DCT的后部。在这里，我们只需要对N个点做FFT，计算量缩短了一半。

另外注意到我们的输入都是实数序列，因此可以使用实数傅立叶变换RFFT代替FFT（主要原理是实数序列进行离散傅立叶变换后是埃尔米特序列），计算量可以再进一步缩短一半，换算下来就是$N/2$的FFT的计算量。

对于逆离散余弦变换的快速算法，沿用FDT的符合，我们知道$X(m)=Re[H(m)]$, 不妨补充$H(m)$的虚部为$X_i(m)$, 即$X(m)+iX_i(m)=H(m)$. 这样就有$H(N-m)=iH^*(m)=X_i(m)+iX(m)$. 用$m$替换$N-m$则有$H(m)=X_i(N-m)+iX(N-m)$. 因此可得:


$$
X_i(m)=X(N-m)
$$

结合$H(m)=W_{N}^{m/4}Y(m)$可得：

$$
Y(m)=W_N^{-m/4}(X(m)+iX(N-m)), m=0,1,\ldots, N-1
$$

其中结合之前的推导过程我们知道$X(N)=0$. 从而我们可以通过IDFT得到$y(n)$： 

$$
y(n) = \frac{1}{N}\sum_{m=0}^{N-1}Y(m)W_N^{-nm}
$$


最后利用$y(n) = x(2n), y(N-1-n) = x(2n+1)$重排一下即可得到$x(n)$。因此我们只需要对N个点做IFFT，计算量缩短了一半。另外注意到$Y(m)$是埃尔米特序列，其IDFT为实数序列，我们可以直接调用逆实数傅立叶变换IRFFT即可，计算量等价于对N/2个点做IFFT。

### DCT求解PDE

接下来说明如何使用DCT对求解Neumann边界条件的泊松方程。我们对$\psi(x)$和$\rho(x)$做傅立叶级数展开。因为$\psi(x)$具有Neumann边界条件，因此其傅立叶级数展开只有$cos$项；另外因为由于$\psi''(x)=-\rho(x)$并且傅立叶级数展开具有唯一性，因此$\rho(x)$的傅立叶级数也只有$cos$项。因此傅立叶级数可以写成：


$$
\begin{align}
\rho(x) &= \frac{2}{N}\sum_{k=0}^{N-1}a'_k\cos(\frac{\pi}{N}kx) \label{fourier_rho}\\
\psi(x) &= \frac{2}{N}\sum_{k=0}^{N-1}b'_k\cos(\frac{\pi}{N}kx) \label{fourier_psi} \\
\end{align}
$$
将公式$\eqref{fourier_psi}$和$\eqref{fourier_rho}$代入泊松方程得到：


$$
（\frac{\pi k}{N})^2b'_k = a'_k,
$$


我们将$(0,N)$均分为$N$个单元格，并在各个单元格中心取值，即在点$x_n=\frac{1}{2}+n\ (n=0,1,\ldots,N-1)$处取值并对$p(x)$采样，记为$\rho_n=\rho(x_n)=\rho(n+\frac{1}{2})$. 因此有:

$$
\rho_n = \frac{2}{N}\sum_{k=0}^{N-1}a'_k\cos(\frac{\pi}{N}k(n+\frac{1}{2}))
$$

刚好是IDCT的形式。因此其对应的DCT形式为：




$$
a_k = \sum_{n=0}^{N-1}\rho_n\cos(\frac{\pi}{N}k(n+\frac{1}{2}))
$$


其中$a_k$和$a_k'$关系只有当$k=0$时候$a'_k$多乘了一个$1/2$。另外我们还关心$\psi(x)$的负梯度（在静电场里面就是电场强度），记$\xi_n = - \psi'(x_n) = - \psi'(n+\frac{1}{2})$. 我们有:


$$
\begin{align}
a_k &= \sum_{n=0}^{N-1}\rho_n\cos(\frac{\pi}{N}k(n+\frac{1}{2})) &= \{DCT(\rho_n)\}_k\\
\psi_n &= \frac{2}{N}\sum_{k=0}^{N-1}[\frac{a_k'}{(\frac{\pi k}{N})^2}]\cos(\frac{\pi}{N}k(n+\frac{1}{2})) &= \{IDCT(\frac{a_k'}{(\frac{\pi k}{N})^2})\}_n \\
\xi_n &= \frac{2}{N}\sum_{k=0}^{N-1}[\frac{a_k'}{(\frac{\pi k}{N})}]\sin(\frac{\pi}{N}k(n+\frac{1}{2})) &= \{IDXST(\frac{a_k'}{(\frac{\pi k}{N})})\}_n \\
\end{align}
$$



这里我们用了和DRAMPlace里面一样的一个新的运算IDXST表示一个序列和一个sin函数相同，原论文证明了该运算可以通过重排后用IDCT实现。再结束前，我们还有一些小问题没有解决，$\int_0^N \psi(x)dx=0$这个条件怎么解决呢？

通过傅立叶级数的计算方式我们知道$a_0'$其实被称为“直流分量”，是在积分区间上的均值，写成积分形式的时候其实$a_0'$和$\int_0^N \psi(x)dx$就差一个参数，而在我们的算法中$a_0'=\sum_{n=0}^{N-1}\rho_n$。当$N \rightarrow \infty$的时候，我们就可以认为$a_0'=0$，因此实际上计算的时候就是在中间把$a_0'$设为0。另外原论文在一维下的限制条件$\int_0^N \rho(x)dx=0$, 类似地这个值和$b_0'$相关，我们发现在我们的计算流程中如果$a_0'$设为0其实$b_0'$也设为了0而已。

不过，我们通过计算公式可以发现，不管是否修改$a'_0$和$b_0'$, 都对电场的计算没有影响，只是对电势能的计算有影响，我们只是在选取一个电势零点而已，这个和静电场唯一性定理是相符的。如果后续需要不需要用到电势能本身，这一步其实是可以去掉的。至此，我们解决了利用DCT求解一维Poisson方程的问题。



## 二维及以上的泊松方程

一维情况解决了，二维及以上的情况都能类似解决，比如二维情况下的解就是:


$$
\begin{aligned}
&a_{u, v}=\operatorname{DCT}\left(\operatorname{DCT}(\rho)^T\right)^T \\
&\psi_{x,y}=\operatorname{IDCT}\left(\operatorname{IDCT}\left(\left\{\frac{a_{u, v}}{w_u^2+w_v^2}\right\}\right)^T\right)^T \\
&\xi_{x,y}^X=\operatorname{IDXST}\left(\operatorname{IDCT}\left(\left\{\frac{a_{u, v} w_u}{w_u^2+w_v^2}\right\}\right)^T\right)^T \\
&\xi_{x,y}^Y=\operatorname{IDCT}\left(\operatorname{IDXST}\left(\left\{\frac{a_{u, v} w_v}{w_u^2+w_v^2}\right\}\right)^T\right)^T
\end{aligned}
$$

这里经历的过程无非就是重排->FFT->频域上的运算->IFFT->重拍。我们可以每个维度单独做FFT/IFFT，也可以两个维度一起重排，然后做FFT2/IFFT2。这两者时间复杂度一样，但是后者的参数可能更小一些。

对于三维甚至以上的情况也是一样的，读者可自行拓展。




[^1]: [四类边界条件下静电场唯一性定理的讨论](https://zhuanlan.zhihu.com/p/410633814)

[^2]: [详解离散余弦变换（DCT）](https://zhuanlan.zhihu.com/p/85299446)

[^3]: [离散余弦变换 - 维基百科](https://zh.wikipedia.org/wiki/%E7%A6%BB%E6%95%A3%E4%BD%99%E5%BC%A6%E5%8F%98%E6%8D%A2)

[^4]: [埃尔米特矩阵 - 维基百科](https://zh.wikipedia.org/wiki/%E5%9F%83%E5%B0%94%E7%B1%B3%E7%89%B9%E7%9F%A9%E9%98%B5) 

[^5]: [A Fast Cosine Transform in One and Two Dimensions](https://ieeexplore.ieee.org/document/1163351)
