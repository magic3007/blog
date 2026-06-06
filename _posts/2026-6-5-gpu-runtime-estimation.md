---
layout: article
title: 快速估计GPU程序的运行时间
date: 2026-1-7 12:00:00 +0800
categories:
  - LLM
tags:
  - lmsys
mathjax: true
mathjax_autoNumber: true
mermaid: true
chart: true
aside:
  toc: true
---
这篇文章主要讲了一下如何快速估计GPU程序的运行时间。GPU程序的运行时间主要分为两部分，一部分是CUDA Kernerl的运行时间，另外一部分是通信时间（通行渠道包括NVLink，RDMA等），我们来分别阐述。

# CUDA Kernel的运行时间估计

估算CUDA Kernel的运行时间的核心是Roofline 模型，每个 kernel 要么卡在算力上，要么卡在带宽上，取两者中较慢的。当然还有一些小型kernel实际上计算和访存量都特别小，这个时候其实是延迟占据主导。因此，总体来说，我们可以分为三种类型： 带宽瓶颈型，算力瓶颈型和延迟瓶颈型。

我们首先来看看经典的Roofline 模型。kernel的运行时间就是计算时间和数据传输时间的最大值。
$$
T_{kernel} = \max(T_{compute}, T_{memory}) = \max(FLOPs / peak\_FLOPS, Bytes / peak\_BW)
$$
思考这两者哪一个是取到最大值，我们其实关心的是算力和带宽的比值，也就是计算强度（Arithmetic Intensity，AI）：
$$
 AI = FLOPs / Bytes    (单位：FLOP/Byte)
$$
我们将计算强度和硬件的脊点 (Ridge Point)比较，如果小于脊点，那么就是带宽瓶颈类型，否则是算力瓶颈型。
$$
Ridge\ Point = peak\_FLOPS / peak\_BW
$$
一些常见硬件的脊点如下：

| GPU  | BF16峰值算力(TFLOPS) | HBM峰值带宽(TB/s) | 脊点(FLOP/Byte) |
| ---- | ---------------- | ------------- | ------------- |
| B200 | 2250             | 8             | 281           |
| H100 | 990              | 3.35          | 295           |
| H200 | 990              | 4.80          | 206           |
| H800 | 134              | 2             | 67            |
| A100 | 312              | 2.04          | 153           |
| A800 | 312              | 1.94          | 161           |


我们尤其关心算力瓶颈型的算子能不能打满带宽。在H100上，对于GEMM，假设M=N=K，N至少要xx才能打满带宽；对于Prefill Attention，假设d=（dpsk v4的数字），N至少要xx才能打满带宽。

从上表中我们可以看到B系列不愧是性能怪兽。我们可以得到下面的的表格，计算各类算子的计算强度公式：

| Kernel 类型                          | 计算量  | 访存量（BF16） | 计算强度 | 瓶颈  | 在H100上至少多大才能打满 算力/带宽 | 备注           |
| ---------------------------------- | ---- | --------- | ---- | --- | -------------------- | ------------ |
| GEMM (大矩阵)                         | 2N³  | 6N²       | N/3  | 算力  | 887                  | 假设M=N=K      |
| Prefill Attention (长序列 prefill)    | 4N²d | 8Nd       | N/2  | 算力  | 592                  | 计算强度和隐藏纬度d无关 |
| Decode Attention (decode, batch=1) | 2Nd  | 4Nd       | 0.5  | 带宽  | 无（始终带宽瓶颈）            | -            |
| Elementwise (add/relu/...)         | N    | 4N        | 0.25 | 带宽  | 无（始终带宽瓶颈）            | -            |
| Reduction (sum/max)                | N    | 2N        | 0.5  | 带宽  | 无（始终带宽瓶颈）            | -            |
| Softmax                            | 3N   | 6N        | 0.5  | 带宽  | 无（始终带宽瓶颈）            | -            |

我们可以看到，通常情况下，决定一个算子是计算瓶颈型还是访存瓶颈型，看的是当规模足够大的时候，计算强度和规模正相关还是无变化（不可能是负相关）。一般来说，element-wise 操作、reduction、decode 阶段的 attention是带宽瓶颈类型；GEMM、prefill 阶段的 attention、卷积是算力瓶颈型。

其中，“在H100上至少多大才能打满 算力/带宽”这一列的数值可以指导我们进行模型设计，比如在H100上，MoE的时候如果分到某个expert的token数量少于887就会无法打满带宽等。

在达到瓶颈的基础上，在我们可以通过下面的速查表来快速估计计算时间：

| Kernel 类型                          | 瓶颈  | 估算公式                     |
| ---------------------------------- | --- | ------------------------ |
| GEMM (大矩阵)                         | 算力  | 2MNK / peak_FLOPS        |
| Prefill Attention (长序列 prefill)    | 算力  | 4N²d / peak_FLOPS        |
| Decode Attention (decode, batch=1) | 带宽  | 2Nd × sizeof / peak_BW   |
| Elementwise (add/relu/...)         | 带宽  | N × sizeof × 2 / peak_BW |
| Reduction (sum/max)                | 带宽  | N × sizeof / peak_BW     |
| Softmax                            | 带宽  | 3N × sizeof / peak_BW    |

当然，还有一类算子，由于规模过小运行时间和**kernel launch**时间在同一个量级，**同步**操作过多，**原子操作**竞争等原因，是延迟计算型，算力和带宽都吃不满。这类算子的运行时间计算方式是：
```
T ≈ launch_overhead + serial_work
```
一般来说，可以按照kernel launch是4μs；一次`__syncthreads`和`atomicAdd`是16ns来估算。
H100的频率是1.83 GHz, 1 cycle大约是0.55 ns，我们这里分别给出片上操作延迟、片外操作和系统级延迟估计。

* 片上操作延迟

| 操作                           | 时钟周期   | 约 ns    | 备注                                     |
| ---------------------------- | ------ | ------- | -------------------------------------- |
| 寄存器读写                        | ~1     | ~0.5    | 最快，线程私有                                |
| `__shfl_sync` (warp shuffle) | ~1     | ~0.5    | warp 内直接交换寄存器，不走内存                     |
| Shared memory 读写             | 20-30  | ~11-16  | 无 bank conflict 时；每个 conflict +1 cycle |
| L1 cache hit                 | 20-30  | ~11-16  | 和 shared memory 共用硬件                   |
| `__syncthreads()`            | ~20-30 | ~11-16  | 轻量级，但阻塞整个 block                        |
| `__syncwarp()`               | ~几     | ~2-5    | 比 syncthreads 更轻                       |
| Shared memory atomicAdd      | ~30    | ~16     | 低竞争时；高竞争线性退化                           |
| DSMEM (跨 SM cluster)         | 33-213 | ~18-116 | Hopper 新特性                             |

* 片外操作延迟

| 操作                     | 时钟周期  | 约 ns  | 备注              |
| ---------------------- | ----- | ----- | --------------- |
| L2 cache hit (近分区)     | ~260  | ~140  | H100 L2 分两半，近的快 |
| L2 cache hit (远分区)     | ~740  | ~400  | 跨分区慢 3 倍        |
| Global memory (HBM3)   | ~500  | ~270  | L2 miss 后走 HBM  |
| Global atomicAdd (无竞争) | ~500+ | ~270+ | 走 L2 仲裁         |
| Global atomicAdd (高竞争) | 数千    | ~μs 级 | 大量线程抢同一地址时严重退化  |

* 系统级延迟

| 操作                       | 典型延迟    | 备注                     |
| ------------------------ | ------- | ---------------------- |
| Kernel launch (CPU 侧)    | 3-5 μs  | cudaLaunchKernel 本身的开销 |
| cudaMemcpyAsync (小数据)    | 5-10 μs | 启动 DMA 的固定开销           |
| cudaStreamSynchronize    | 3-10 μs | CPU 等 GPU 完成           |
| cudaDeviceSynchronize    | 5-15 μs | 等所有 stream             |
| NVLink 单次 P2P 访问         | ~1 μs   | 跨 GPU 单次 load/store    |
| PCIe D2H/H2D (小数据)       | 5-15 μs | 启动开销主导                 |
| `__threadfence_system()` | ~1-5 μs | 跨 GPU 内存序保证            |

我们可以看到甚至有的时候只要多几个Global atomicAdd运行时间kernel launch是差不多量级的了。根据这些数字，我们甚至能估算出，warp reduce sum如果用warp shuffle比shared memory其实快的数量级是100倍：
```
warp_reduce_sum 用 5 轮 __shfl_xor_sync:
  5 × 0.5 ns = 2.5 ns

如果改用 shared memory reduce:
  5 × (store 16ns + syncthreads 16ns + load 16ns) ≈ 240 ns

```
以此类推。


估算完后，我们还可以用NCU跑下面的命令来验证：

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained,  \
              dram__throughput.avg.pct_of_peak_sustained  \
    ./your_program
```

- 如果`sm__throughput` 接近 100% ，那么是算力瓶颈
- 如果`dram__throughput` 接近 100% → 那么是带宽瓶颈
- 如果都不高，那么是延迟瓶颈（launch overhead、同步、分支、occupancy 不足等）。

# 通行时间的估计

以MoE里面的weight_sync来说明如何进行通信时间的估计。

对于GB200 NVL72，通过NVL域内是第五代NVLink通行，每GPU的单向带宽是900GB/s；如果跨域就是基于InfiniBand的RDMA通行，每GPU的单向带宽只有25GB/s，后者是前者的36分之一。一般来说，大模型单层推理时间的量级是0.05到1ms级别。

|                 | GB200 NVL72 (NVLink 5)                 | 跨域 RDMA (NDR InfiniBand)              |
| --------------- | -------------------------------------- | --------------------------------------- |
| 每 GPU 双向带宽 | **1.8 TB/s** (18 条 NVLink × 100 GB/s) | ~50 GB/s (400Gbps NIC，通常多 GPU 共享) |
| 每 GPU 单向带宽 | **900 GB/s**                           | ~25 GB/s（且多 GPU 共享 NIC，实际更低） |
| 比值            | 基准                                   | NVLink 的 ~1/36 到 ~1/72                |

怎么估算通信时间？
直接用每张卡的通信量除以每张卡的带宽。

以 weight_sync 为例，假设一个 expert 的权重 W = 100MB (BF16)，master 有 3 个 replica (R=3)，那么weight_sync 时间 ≈ 100MB × 3 / 900 GB/s ≈ 0.33 ms，还是能接受的，但是跨NVL域就不能接受了。

如果需要更加精确的模型，我们还需要考虑：
1. 延迟项：`T = latency + data / bandwidth`。NVLink 延迟 ~1μs，RDMA ~5-10μs。数据量大时延迟项可忽略
2. 流水线效应：weight_sync 用 TMA 双缓冲，load 和 store 重叠。实际吞吐接近峰值带宽
3. 链路竞争：如果多个 master 同时向同一个 GPU 发数据，接收端带宽会成为瓶颈。NVSwitch 提供 full bisection bandwidth，但单个 GPU 的入向带宽仍有上限
4. 多 expert 累加：一张卡上可能有多个 master expert 需要同步，总通信量是所有 expert 的加和