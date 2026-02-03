---
layout: article
title: 深入理解 Megatron-LM 中的 Partial CUDA Graph：MoE 模型训练加速的关键技术
date: 2025-12-30 12:00:00 +0800
categories:
  - LLM
tags:
  - lmsys
mathjax: true
mermaid: true
aside:
  toc: true
---

# 为什么近期大模型训练需要用到CUDA Graph

CUDA Graph是NVIDIA推出的一项技术，它可以将一系列GPU操作，包括kernel launches，memory copies预先在capture阶段录制成一个图，然后在replay阶段一次性提交执行。CUDA Graph最大的好处是可以极大减少CPU开销，减少kernel launch latency，这在使用Grace CPU的B系列NV芯片上带来的好处尤其突出。同时GPU可以更好的优化执行顺序，减少运行队列中间的bubble。
> 针对B系列芯片CPU开销较大的问题，这里以MoE层中fc1的grouped GEMM操作为例进行说明。以下的实验均在B系列芯片上进行。在Transformer Engine中，默认的grouped GEMM其实是通过循环触发多个GEMM kernel，并分配到不同的CUDA stream上执行（如左图所示），本质上不是“真正”的grouped GEMM。而如果在B系列芯片上直接调用cutblass，只需启动一次kernel即可完成真正的grouped GEMM（如右图所示），这种方式能显著减少kernel launch次数，从而大幅降低overhead。同时可以看到，左图在GEMM计算结束后，队列中出现一段bubble才进入激活函数环节，也从侧面反映出CPU带来的额外延迟。值得一提的是，TE默认采用多stream实现grouped GEMM的做法在H系列芯片上通常更优，但迁移到B系列时就未必适合了。
> {% include img.html src="2025-12-30-training-cuda-graph.assets/1040025031s2kp8r11k0a6r18ok.webp" alt="1589443681801" %}

当然，使用 CUDA Graph 也引入了一些新的限制和约束：

1. 输入的 Tensor 形状必须保持一致 —— 每次调用时输入 shape 不能变化。
2. 禁止 CPU-GPU 同步操作 —— 在 graph 内部不能包含诸如 `.item()`、`.cpu()` 等涉及同步的指令。
3. 不能存在动态控制流 —— 执行路径必须是确定性的，不能依赖于运行时数据分支。
4. 内存地址需固定 —— 需采用静态分配的输入缓冲区。

其中，“内存地址需固定”这一点对MoE模型尤为突出。MoE 的核心机制在于 Token Dispatcher 根据路由器（router）的分配结果，动态决定每个 token 分派给哪个 expert。即每个 batch 的路由都可能不同，进而使 AlltoAll 通信所需的数据大小和内存布局发生变化。而 CUDA Graph 要求 capture 阶段所有操作涉及的 Tensor 地址均已确定，这与 MoE 这种动态分布形成冲突。

为了解决这一问题，工程实践中，我们采用“部分捕获”（Partial Capture）的方案：对于可以静态展现的部分我们用 CUDA Graph 捕获，其余涉及动态路径/动态资源分配的部分则按照传统方式执行。这样既能获得 graph 带来的 kernel launch/调度优化，也能兼顾 MoE 动态收发的灵活性。具体配置如下所示：
```yaml
cuda_graph_impl: transformer_engine
cuda_graph_scope:
  - attn            # 捕获 attention 层
  - moe_router      # 捕获 MoE 路由核心
  - moe_preprocess  # 捕获路由 token 的预处理环节
```
接下来将详细讲解实现原理与核心操作流程。
# TE Capture/Replay 工作流程

## 整体流程

Megatron-LM 支持两种 CUDA Graph 实现方式：其一为Megatron-LM自己的实现，通过设置 `cuda_graph_impl:local` 启用。这种方式由 Megatron 原生实现了 graph 的捕获及回放，在 dense 模型场景下，可以一次性录制并复现包含前向与反向传播（不含优化器操作）的完整迭代过程，相关核心代码位于 `megatron/core/transformer/cuda_graphs.py`，主要涉及 `CudaGraphManager` 与 `_CudaGraphRunner`。然而在 MoE 模型情形下，由于需要采用定长 padding，导致无论某个 expert 实际接收到的 token 数量多少，都必须填充到统一的最大容量，这不仅增加了不必要的计算消耗，还带来 AlltoAll 通信带宽的浪费，并且显存也需预留足够的最大 buffer。因此，目前该实现架构并不适合 MoE 场景下的 CUDA Graph 加速需求。

另一种对 CUDA Graph 的支持方式，是通过启用 Transformer Engine（TE）实现的，只需在配置中设置 `cuda_graph_impl:transformer_engine`。我们在partial cuda graph中使用这套方案。在该方案下，核心流程围绕 TE 的 CUDA Graph 捕获（Capture）与回放（Replay）机制展开。Megatron-LM 集成 TE CUDA Graph 的主要逻辑位于 `megatron/core/transformer/cuda_graphs.py` 的 `TECudaGraphHelper` 类内部，其核心方法是调用 TE 提供的 `make_graphed_callables()` API。从最简单的情况开始开始考虑，假设只有一个microbatch，那么整体的工作流程如下：

```python
# 1. 收集待捕获的层
flattened_callables = []
for layer in model.decoder.layers:
    if _layer_is_graphable(layer, config):
        flattened_callables.append(layer)

# 2. 构造示例输入
sample_args, kwargs = self._get_cuda_graph_input_data()
# 形如: ((hidden_states_layer0,), (hidden_states_layer1,), ...)

# 3. 捕获阶段
graphs = make_graphed_callables(
    tuple(flattened_callables),
    sample_args,
    kwargs,
    _order=order,  # 管道并行调度顺序
    _reuse_graph_input_output_buffers=True,  # TE 2.7+ 支持的 buffer 复用特性
)

# 4. 可cuda replay的 Graph分配给各层对象
for layer in layers:
    layer.cuda_graphs = [graphs[idx] for idx in ...]
```

**首先是第一阶段，收集待捕获的层**。由于我们是逐层进行捕获，因此首先会依据 config 配置筛选需要捕获 CUDA Graph 的层，依次存入 `flattened_callables`。注意这个时候虽然我们只捕获MoE layer的attn，moe_router以及moe_preprocess这三部分，但是这里我们还是将这个layer放进来，后面通过后面说的python的异常机制来实现部分捕获。

**接着是第二阶段，构造示例输入**。在这个阶段需为每一层准备一组示例输入 `sample_args` 与 `kwargs`，这个本质上是在调用layer的`get_layer_static_inputs`函数：
```python
args, kwargs = layer.get_layer_static_inputs(self.seq_length, self.micro_batch_size)
```
这套静态输入会作为forward graph的static input buffer，同时在它上面模拟forward和backward跑出来的结果会作为backward graph的static input buffer。涉及PP的调度与 buffer 管理会在后文详细介绍。

**第三阶段是捕获阶段**。在这个阶段我们调用了TE的`make_graphed_callables`这个API。在 `make_graphed_callables` 的内部，核心包括以下环节：

1. Warmup 阶段：多次执行 forward+backward，让类似`triton.compile`这样的操作完成lazy 初始化，确保后续捕获的稳定性。
2. Capture 阶段：依顺序对每个可调用对象依次执行 forward，再按逆序依次执行 backward。 捕获过程中将所有 CUDA kernel 操作录入 graph，同时固定输入和中间变量的 tensor 地址, 也就是说 forward 期间所有的中间 tensor buffer 其物理地址在 capture 当下即已确定。
```
# All captures here share a mempool. To avoid replays corrupting each other's memory,
# the safest approach is to capture all passes in the same order they'll run:
# fwd 1, fwd 2, ... fwd N, then bwd N, bwd N-1, ... bwd 1.
```
4. `make_graphed_callables` 最终返回一组包装好的 callable，每次直接调用这些 callable 的 forward/backward 方法，即等价于 replay 预先录制好的 graph。实际回放过程简化代码如下：
```python
class Graphed(torch.autograd.Function):
    """用于 graph 回放的自定义 Autograd Function。"""

    @staticmethod
    def forward(ctx, *inputs):
        # 将新输入的数据拷贝进预分配静态 tensor
        for i in range(len_user_args):
            if (
                isinstance(static_input_surface[i], torch.Tensor)
                and static_input_surface[i].data_ptr() != inputs[i].data_ptr()
            ):
                static_input_surface[i].copy_(inputs[i])

        # 回放 Graph 里的 forward kernel
        fwd_graph.replay()
        assert isinstance(static_outputs, tuple)
        return tuple(o.detach() for o in static_outputs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grads):
        # 回放 backward kernel
        assert len(grads) == len(static_grad_outputs)
        for g, grad in zip(static_grad_outputs, grads):
            if g is not None:
                # 若 grad 已处于目标 buffer, 则无需拷贝
                if g.data_ptr() != grad.data_ptr():
                    g.copy_(grad)
        bwd_graph.replay()

        # 对于不需要 grad 的输入，autograd 期望收到 None
        assert isinstance(static_grad_inputs, tuple)
        return (None,) + tuple(
            b.detach() if b is not None else b for b in static_grad_inputs
        )
```

**接着是最后阶段**，我们将可cuda replay的 Graph分配给各层对象作为一个成员变量，供后续调用。

## CUDA Graph在microbatch之间能否复用？
我们接下来思考另一个问题，现在有有多个micro batches，CUDA Graph在microbatch之间能否复用？不妨假设number of microbatches=4。实际上，这个和micro batch的forward和backward调用有关系。
* 第一种情况是，假设每个microbatch完成forward后立刻进行backward（也就是1F1B调度）。

```
Microbatch:  MB0    MB0    MB1    MB1    MB2    MB2    MB3    MB3
操作:        FWD    BWD    FWD    BWD    FWD    BWD    FWD    BWD
             └──┘   └──┘   └──┘   └──┘   └──┘   └──┘   └──┘   └──┘
              ↑      ↑      ↑      ↑      ↑      ↑      ↑      ↑
             立刻   立刻   立刻   立刻   立刻   立刻   立刻   立刻

order = [1, -1, 1, -1, 1, -1, 1, -1]
         ↑  ↑   ↑  ↑   ↑  ↑   ↑  ↑
       fwd bwd fwd bwd fwd bwd fwd bwd
       mb0 mb0 mb1 mb1 mb2 mb2 mb3 mb3

```

这种情况下显然CUDA Graph是可以复用的，同时input buffer也是可以复用的。那么这个时候构建的CUDA Graph的数量是2 × num_layers，其中2分别为是forward和backward。
* 另外一种情况是，所有microbatch都完成forward后，再依次进行backward。

```
Microbatch:  MB0    MB1    MB2    MB3    MB0    MB1    MB2    MB3
操作:        FWD    FWD    FWD    FWD    BWD    BWD    BWD    BWD
             ←── 所有 Forward ──→     ←── 所有 Backward ──→
             
order = [1, 1, 1, 1, -1, -1, -1, -1]
```

注意到中间变量的地址是存在CUDA Graph内部的，如果microbatch 1复用microbatch 0的CUDA Graph，那么在forward的时候会改掉microbatch 0的中间变量，这样在microbatch 0在backward的时候就没有中间变量就被破坏掉了。这个时候CUDA Graph是不可以被复用的，此时CUDA Graph的数量是2 × num_layers × num_microbatches。
很显然，第二个情况需要临时保存的activation数量太多了，正常Megatron-LM在不开流水线并行的情况下是不可能使用的。时间上我们可以从Megatron-LM的实际代码可以看到不开PP的时候就是第一种情况的1F1B调度：
```python
# megatron/core/pipeline_parallel/schedules.py: forward_backward_no_pipelining

with no_sync_func():  # 前 N-1 个 microbatch 不同步梯度
    for i in range(num_microbatches - 1):
        output_tensor = forward_step(...)   # Forward microbatch i
        if not forward_only:
            backward_step(...)               # Backward microbatch i（立刻！）

# 最后一个 microbatch 在 no_sync 外执行（触发梯度同步）
output_tensor = forward_step(...)           # Forward 最后一个
backward_step(...)                           # Backward 最后一个
```
## 流水线并行下的TE CUDA Graph解析

目前，Megatron-LM采用的流水线并行方案是1F1B interleaving，如下图所示：
![](https://developer-blogs.nvidia.com/wp-content/uploads/2021/03/interleaved_1F1B_schedule-1-1536x707.png)

为便于分析，我们先不考虑VPP，即采用上半图的标准的1F1B方案。这里number_of_microbatches为8，PP为4。结合前述分析，我们可以得出：microbatch 1 和 microbatch 4 之间的 CUDA Graph 不可以复用，但当 microbatch 1 的 backward 完成后，microbatch 5 在执行前向时，完全可以复用microbatch 1 对应的 CUDA Graph。在典型1F1B调度下，每个GPU上同时存在的activation峰值仅与PP相关，因此在理论上，最优情况下CUDA Graph数量下限为 2 × num_layers × PP。

然而，当前实现中CUDA Graph的实际数量不是随PP增长的，而是随num_of_microbatches线性增长。换言之，不同microbatch之间的CUDA Graph没有被复用。因此，目前系统中CUDA Graph的数量实际为 2 × num_layers × num_of_microbatches。

我们知道和num_of_microbatches成正比是一个不太好的事情，这意味和global batch size成正比，容易出现显存爆炸的问题。为了解决这个问题，我们首先需要明确两个概念：  
1. **CUDA Graph本身**：即对CUDA操作的录制，此部分显存消耗较小，每个microbatch都有一份独立的graph是没有问题的。
2. **中间变量显存（activation）**：在CUDA Graph捕获阶段产生。这部分显存可以在不同的CUDA Graph之间共享，经优化后实际占用量与PP成正比。这一优化空间及方法我们将在后文详细讨论。

我们回到考虑VPP的情况，假设num_of_microbathes=8, PP=4，VPP=2。我们首先学习Megatron-LM里面是如何表示1F1B interleaving的，可以跟着下面代码的思路：
```python
        # Get the PP and VPP scheduling order.
        from megatron.core.pipeline_parallel.schedules import (
            convert_schedule_table_to_order,
            get_pp_rank_microbatches,
            get_schedule_table,
        )

        _, _, num_warmup_microbatches, _ = get_pp_rank_microbatches(
            num_microbatches,
            self.num_model_chunks,
            self.config.microbatch_group_size_per_vp_stage,
            False,
        )
        schedule_table = get_schedule_table(
            num_microbatches,
            self.num_model_chunks,
            self.config.microbatch_group_size_per_vp_stage,
        )
        order = convert_schedule_table_to_order(
            num_warmup_microbatches, self.num_model_chunks, schedule_table
        )
```

首先，假设整个模型有32层（Layer 0-31），那么我们就将整个模型切分成PP × VPP个model chunk，每个PP rank就有VPP个model chunk：
```
模型: Layer 0-31 (32层)
PP = 4, VPP = 2 (num_model_chunks = 2)

PP Rank 0: [Layer 0-3]   ← Chunk 0
           [Layer 16-19] ← Chunk 1

PP Rank 1: [Layer 4-7]   ← Chunk 0
           [Layer 20-23] ← Chunk 1

PP Rank 2: [Layer 8-11]  ← Chunk 0
           [Layer 24-27] ← Chunk 1

PP Rank 3: [Layer 12-15] ← Chunk 0
           [Layer 28-31] ← Chunk 1
```
对于一个PP rank来说，其实它只能看到本地的VPP=2个model chunk，因此它其实依次这样的操作：选择本地的某一个model chunk进行fwd或者bwd。在Megatron-LM里面，我们用一个数字来表示这样的操作，"绝对值-1"表示本地的哪一个model chunk，正负号表示fwd还是bwd。
比如对于上面的PP Rank 0:
* 1表示[Layer 0-3] (chunk 0)做fwd；
* 2表示[Layer 16-19] (chunk 1) 做fwd;
* -1表示[Layer 0-3] (chunk 0) 做bwd；
* -2表示[Layer 16-19] (chunk 1) 做bwd;
每个PP rank都能拿到一个叫order的序列，只要编排好，每个PP rank按照本地编排的order依次执行，那么整体上就能形成一个完整的流水线：比如上图的Device 1的order就是
```python
order = [
    # Warmup (10 forwards)
    1, 1, 1, 1,      # chunk 0 fwd: mb 1,2,3,4
    2, 2, 2, 2,      # chunk 1 fwd: mb 1,2,3,4
    1, 1,            # chunk 0 fwd: mb 5,6
    
    # 1F1B steady state (交替 fwd/bwd)
    1, -1,           # F_mb7_c0, B_mb1_c1
    1, -1,           # F_mb8_c0, B_mb2_c1
    2, -2,           # F_mb5_c1, B_mb1_c0
    2, -2,           # F_mb6_c1, B_mb2_c0
    2, -1,           # F_mb7_c1, B_mb3_c1
    2, -1,           # F_mb8_c1, B_mb4_c1
    
    # Cooldown (全 backward)
    -2, -2,          # B_mb5,6_c0
    -1, -1,          # B_mb5,6_c1
    -2, -2,          # B_mb3,4_c0
    -1, -1,          # B_mb7,8_c1
    -2, -2,          # B_mb7,8_c0
]
```
其中我们可以看到分为三个阶段：warmup阶段（全是forward），1F1B交替阶段，以及cooldown阶段（全是backward）。那么对于每个PP rank怎么才能得到这个数组呢？我们跟着上面的代码来就行：
* 首先`get_pp_rank_microbatches` 告诉我们num_warmup_microbatches的计算公式就是

$$
(PP - rank - 1) * 2 + (VPP - 1) * \text{microbatch_group_size_per_vp_stage}
$$

这里microbatch_group_size_pervp_stage表示每个 virtual stage 连续执行的 microbatch 数量，默认值是PP。代入到上图里计算确实这样。
接着是`get_schedule_table`这个函数，这个函数是个生成 (microbatch_id, model_chunk_id) 的调度顺序，实际上是每个PP rank上fwd操作的顺序，这个数组每个PP rank看到的都是一样的，从上图中看到的确实是这样。同时这个顺序顺序也是每个PP rank上看到backward操作的顺序。
```python
def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    """
    生成 (microbatch_id, model_chunk_id) 的调度顺序
    """
    schedule_table = []
    
    # 按 group 遍历所有 microbatch
    for min_microbatch_id_in_group in range(0, num_microbatches, microbatch_group_size_per_vp_stage):
        
        # 对于每个 group：
        # 外层循环: model_chunk_id (0, 1, ..., VPP-1)
        # 内层循环: microbatch_id (group 内的连续 microbatch)
        
        schedule_table.extend([
            (microbatch_id, model_chunk_id)
            for model_chunk_id in range(num_model_chunks)      # 外层
            for microbatch_id in range(group_start, group_end)  # 内层
        ])
    
    return schedule_table
```
比如参数为num_microbatches=8, num_model_chunks=2, microbatch_group_size=4的时候，
```
# Group 0: microbatch 0-3
# Group 1: microbatch 4-7

# 遍历过程：
Group 0 (mb 0-3):
  chunk 0: mb 0, 1, 2, 3  →  (0,0), (1,0), (2,0), (3,0)
  chunk 1: mb 0, 1, 2, 3  →  (0,1), (1,1), (2,1), (3,1)

Group 1 (mb 4-7):
  chunk 0: mb 4, 5, 6, 7  →  (4,0), (5,0), (6,0), (7,0)
  chunk 1: mb 4, 5, 6, 7  →  (4,1), (5,1), (6,1), (7,1)

# 最终 schedule_table:
schedule_table = [
    (0,0), (1,0), (2,0), (3,0),  # chunk 0, mb 0-3
    (0,1), (1,1), (2,1), (3,1),  # chunk 1, mb 0-3
    (4,0), (5,0), (6,0), (7,0),  # chunk 0, mb 4-7
    (4,1), (5,1), (6,1), (7,1),  # chunk 1, mb 4-7
]
```

接着是`convert_schedule_table_to_order`这个函数，每个PP rank的num_warmup_microbatches是不一样的，都是都是全forward（warmup） + forward和backward交替 + 全backward三个阶段。只需要知道warmup有几个microbatches，后面的就能模拟出来了。
```python

# Step 1: schedule_table
schedule_table = [
    (0,0), (1,0), (2,0), (3,0),  # chunk 0, mb 0-3
    (0,1), (1,1), (2,1), (3,1),  # chunk 1, mb 0-3
    (4,0), (5,0), (6,0), (7,0),  # chunk 0, mb 4-7
    (4,1), (5,1), (6,1), (7,1),  # chunk 1, mb 4-7
]

# _, model_chunk_id_table = zip(*schedule_table)
model_chunk_id_table = [0,0,0,0, 1,1,1,1, 0,0,0,0, 1,1,1,1]

# Step 2: forward_order 和 backward_order
forward_order  = [1,1,1,1, 2,2,2,2, 1,1,1,1, 2,2,2,2]   # chunk_id + 1
backward_order = [-2,-2,-2,-2, -1,-1,-1,-1, -2,-2,-2,-2, -1,-1,-1,-1]  # chunk_id - 2

# Step 3: 生成 order
# Warmup: forward_order[:10]
order = [1,1,1,1, 2,2,2,2, 1,1]

# 1F1B: i = 10 to 15
# i=10: fwd[10]=1, bwd[0]=-2  → [1, -2]
# i=11: fwd[11]=1, bwd[1]=-2  → [1, -2]
# i=12: fwd[12]=2, bwd[2]=-2  → [2, -2]
# i=13: fwd[13]=2, bwd[3]=-2  → [2, -2]
# i=14: fwd[14]=2, bwd[4]=-1  → [2, -1]
# i=15: fwd[15]=2, bwd[5]=-1  → [2, -1]

# Cooldown: backward_order[-10:]
# = [-1,-1, -2,-2,-2,-2, -1,-1,-1,-1]

```

> 补充一句，VPP的本质是切更多的流水线阶段，只是将部分流水线阶段划分到同一个 GPU 上。如果我们的 GPU 数量足够，其实完全可以把这些被切分出来的子阶段分别放在不同的 GPU 上，只是受限于现实资源有限，只能将多个阶段合并放置于同一块 GPU罢了。因此从理论分析的角度，Bubble time fraction就是会下降至原先的 1/VPP，因为本质上流水线阶段就是变多了。

有了这个order之后，一个非常简单规则：当 order 中出现一个负数（backward），对应的 forward buffer 就可以被后续的 forward 复用了。这个规则将指导我们下面的显存复用，做到和PP成正比。

> 更加准确地说，同时因为进入稳定1F1B阶段正数和负数是交替的，是边使用边释放，因此更加准确来说，activation的峰值占用和(PP - rank - 1) × 2 + (VPP - 1) × PP + 1成正比。

## fwd graph的Static Input buffer的大小如何做到和PP成正比？

根据这个简单规则简单模拟即可。注意一个model chunk对应多个layer，我们首先获得每个layer的输入应该长什么样子（记为sample_keys), 然后模拟即可：
```python
# Forward: 尝试复用
if consumed_sample_queue.get(sample_keys, []):
    # 有可复用的 → 直接复用
    reuse_fwd_idx = consumed_sample_queue[sample_keys].pop(0)
    sample_args[per_callable_fwd_idx] = sample_args[reuse_fwd_idx]  # 复用！
else:
    # 没有可复用的 → 生成新的
    sample_args[per_callable_fwd_idx] = _get_layer_static_inputs(...)

# Backward: 释放 buffer
for sample_keys, fwd_idx in fwd_sample_queues[chunk][:num_layers]:
    consumed_sample_queue[sample_keys].append(fwd_idx)  # 标记可复用
```
这部分其实是TE的make_graphed_callables()的参数`_reuse_graph_input_output_buffers`对fwd graph的input buffer做的[事情](https://github.com/NVIDIA/TransformerEngine/blob/74faf7ec422229bcecf9e079d96b429da071e7b7/transformer_engine/pytorch/graph.py#L241-L292)。对于bwd graph的input buffer，这部分不需要像fwd那样保存中间结构，[直接检查key是否存在然后复用即可](https://github.com/NVIDIA/TransformerEngine/blame/74faf7ec422229bcecf9e079d96b429da071e7b7/transformer_engine/pytorch/graph.py#L615C18-L628)。这样就可以做到static Input buffer的大小和PP成正比。当然在Magatron-LM中准备sample args的时候，也要[参考TE这样做](https://github.com/NVIDIA/Megatron-LM/blob/708069774569517cc1802e6c339730a689179d4d/megatron/core/transformer/cuda_graphs.py#L1928-L2031)，不然在运行某个时候这部分大小会和num of microbatches成正比。

## 中间变量的显存占用如何做到和PP成正比？

Static Input buffer这部分我们可以自己通过模拟pipeline的调度显式控制，使得和PP而不是和num_of_microbatches成正比。
另外一个需要思考的如何让中间变量的显存占用做到和PP成正比。为了实现这个我们需要的使用到memory pool这个功能以及pytorch的`make_weak_ref`这个功能。
首先，我们需要让所有的graph都共享一个memory pool，
```python
# graph.py:359
mempool = graph_pool_handle() if pool is None else pool

# 所有 graph 捕获时都使用同一个 pool
with _graph_context_wrapper(fwd_graph, pool=mempool):  # forward graph
    outputs = func(*args, **kwargs)

with _graph_context_wrapper(bwd_graph, pool=mempool):  # backward graph
    ...
```
因为pytoch的API默认情况下，每个 CUDA Graph 有私有内存池，内存不能互相复用；所有 Graph 共享一个池才有复用的可能性。
```
┌─────────────────────────────────────────────────────────────┐
│  默认行为：每个 CUDA Graph 有私有内存池                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │ Graph 1 │  │ Graph 2 │  │ Graph 3 │  ...                │
│  │ Pool 1  │  │ Pool 2  │  │ Pool 3  │                     │
│  └─────────┘  └─────────┘  └─────────┘                     │
│  内存不能互相复用 → 总显存 = N * graph_size                 │
├─────────────────────────────────────────────────────────────┤
│  共享 Memory Pool：所有 Graph 共享一个池                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Shared Memory Pool                      │   │
│  │  ┌───────┐ ┌───────┐ ┌───────┐                      │   │
│  │  │Graph 1│ │Graph 2│ │Graph 3│  ...                 │   │
│  │  └───────┘ └───────┘ └───────┘                      │   │
│  │  内存可以复用 → 总显存 = max_concurrent * graph_size │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```
在复用了同一个memory pool之后，为了让中间变量释放，我们需要做的是及时触发python的gc自动删除不再需要的中间变量。在这里，我们需要用到Pytorch的`make_weak_ref`这个功能：
```python
class _WeakRefTensor:
    """只保存 data_ptr，不持有 tensor 引用"""
    def __init__(self, data_ptr, dtype, shape):
        self._data_ptr = data_ptr   # 只记录地址
        self.dtype = dtype
        self.shape = shape
        # 注意：没有保存对原 tensor 的引用！

def make_weak_ref(x):
    if isinstance(x, torch.Tensor):
        return _WeakRefTensor(x.data_ptr(), x.dtype, x.shape)
        # 原 tensor 的引用被丢弃 → PyTorch 认为该内存可以释放
```
其核心原理是：make_weak_ref 把"持有引用的 tensor 对象"替换成"只记录地址的整数"，原 tensor 失去所有引用后被 Python GC 回收，PyTorch 随之释放其 GPU 内存回 mempool。地址虽然还被记住，但内存已经可以被其他 graph 复用了。
```
原始状态：
┌─────────────────────────────────────────────┐
│  per_callable_static_outputs[idx]           │
│  = (tensor_A, tensor_B, tensor_C)           │
│        ↓         ↓         ↓                │
│   ┌────────┐ ┌────────┐ ┌────────┐          │
│   │ Memory │ │ Memory │ │ Memory │ (被占用) │
│   └────────┘ └────────┘ └────────┘          │
└─────────────────────────────────────────────┘

调用 make_weak_ref 后：
┌─────────────────────────────────────────────┐
│  per_callable_static_outputs[idx]           │
│  = (_WeakRefTensor, _WeakRefTensor, ...)    │
│        ↓ (只保存地址，不持有引用)            │
│   ┌────────┐ ┌────────┐ ┌────────┐          │
│   │ Memory │ │ Memory │ │ Memory │ (可复用) │
│   └────────┘ └────────┘ └────────┘          │
│   ↑ PyTorch 认为这些内存不再被使用          │
│   ↑ 可以被后续的 graph capture 复用         │
└─────────────────────────────────────────────┘
```
因此，当完成一个backward后，我们能将三部分tensor设置为week reference，从而触发python的自动gc：
```python
# graph.py:634-665
# Weak ref the static outputs and static grad inputs that are no longer needed
# in the following steps. These two type of tensors are both in cudagraph
# mempool, so we just deallocate them and let PyTorch's memory allocator
# reuse them elsewhere.
if _reuse_graph_input_output_buffers:
    # 1. Weak ref the static outputs of the forward pass of this backward. It's
    # no longer needed after the corresponding backward graph is built up.
    per_callable_static_outputs[per_callable_bwd_idx] = make_weak_ref(
        static_outputs
    )

    # 2. Weak ref the static grad inputs of the previous backward pass within the
    # same chunk.
    if previous_per_callable_bwd_idx is not None:
        idx = previous_per_callable_bwd_idx
        per_callable_static_grad_inputs[idx] = make_weak_ref(
            per_callable_static_grad_inputs[idx]
        )
    previous_per_callable_bwd_idx = per_callable_bwd_idx

    # 3. Weak ref the static grad inputs of the previous chunk's last backward
    # pass.
    # Note: After a chunk's backward pass, we assume Mcore will send the grad
    # input to another pipeline parallel rank and that the communication is
    # finished before the end of the next chunk's backward pass.
    if l_no == 0:
        if previous_chunk_last_callable_bwd_idx is not None:
            idx = previous_chunk_last_callable_bwd_idx
            per_callable_static_grad_inputs[idx] = make_weak_ref(
                per_callable_static_grad_inputs[idx]
            )
        previous_chunk_last_callable_bwd_idx = per_callable_bwd_idx
                if ceil(c_id) == c_id:
                    bwd_idx[m_chunk] += 1
```



# 基于 Python 异常机制实现 graph 边界的动态中断与恢复


## 显存占用与 CUDA Graph 数量分析

结合上文分析，下面以表格形式梳理不同配置下 CUDA Graph 的数量以及fwd graph static input buffer以及中间变量的显存大小。其中，CUDA Graph是 CUDA 操作的录制，录制部分占用的显存很小，在实现的时候每个 microbatch有独立的graph；但是fwd graph static input buffer以及中间变量的显存是需要根据PP实现microbatch之间共享的，使得和PP而不是num of microbatches成正比。

| 运行配置   | CUDA Graph 数量                     | fwd graph static input buffer显存大小 | 中间变量的显存大小           |
| ------ | --------------------------------- | --------------------------------- | ------------------- |
| 不启用 PP | 2 × num_layers                    | 与num_layers成正比                    | 与num_layers成正比      |
| 启用 PP  | 2 × num_layers × num_microbatches | 与num_layers × PP成正比               | 与num_layers × PP成正比 |

这里我们思考一个问题，相比于不使用CUDA Graph，使用partial CUDA Graph实际增加的显存占用在哪里？
通过上面的分析，我们知道相比于不使用CUDA Graph，Partial CUDA Graph 额外增加的显存主要来自：(1) PP 份 Static Input Buffers，(2) PP 份 MoE 中断点张量 (MoECudaGraphTensorStore)。

| 开销类型                 | 显存大小估算                                    | 是否可优化                 |
| -------------------- | ----------------------------------------- | --------------------- |
| CUDA Graph 对象        | < 100 MB                                  | 较小，可忽略                |
| Static Input Buffers | PP × layer × input_size<br>~128 MB - 1 GB | 通过 1F1B 复用已优化到 PP 份   |
| MoE 中断点张量            | PP × MoE_layers × store_size<br>~1-3 GB   | Partial Graph 特有，不可避免 |
| Mempool 碎片           | -                                         | -                     |

注意到很多地方都和num_layers相关，这是因为我们采用partial cuda graph逐layer进行capture，因此我们需要设置和num_layers成正比的同步点，这提示我们加入我们后面可以尽可能capture更完整的模型部分，我们额外需要的显存也会更小。

# 使用CUDA Graph时常见的易错点

1. CUDA Graph的本质是**捕捉某个CUDA stream上若干kernel launches和memory copy操作**，它无法捕捉到任何CPU上的操作或逻辑。

比如下述代码片段：
```python
    s += a.sum() # s和a为GPU上的tensor
    cnt += 1     # cnt是普通的Python计数器
```
在capture阶段，这两行代码都会被执行，但在replay阶段，`s += a.sum()`这样的GPU操作会被执行，而`cnt += 1`这类纯CPU的操作则不会再执行。换句话说，CUDA Graph的replay只会按图执行可捕获的GPU操作，至于代码表面上的其他Python逻辑（如计数器递增），在replay过程中是无效的。此外，诸如if分支等控制流结构，也只会按capture阶段实际走过的路径，把当时分支里的GPU操作捕捉进图，replay时不会根据新的条件判断重新分支。