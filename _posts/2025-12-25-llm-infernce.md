---
layout: article
title: "大模型推理关键技术备忘"
date: 2025-12-22 12:00:00 +0800
categories: [LLM]
tags: [lmsys]
mathjax: true
mermaid: true
aside:
  toc: true
---

# 简要介绍

[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 是一个仅约 1200 行代码的轻量级 vllm 实现，尽管体量精简，却高度还原了 vllm 的核心功能。vllm 本身集成了张量并行（tensor parallel）、前缀缓存（prefix cache）、Torch 编译（torch compile）、CUDA 图（cuda graph）、分页注意力（paged attention）等多项优化手段，底层 attention 引擎采用 Flash Attention。基准测试表明，nano-vllm 在推理性能上能够达到接近 vllm 的水平。

在深入学习 nano-vllm 项目后，对大模型推理中的关键技术与核心挑战有了更系统的理解。本文梳理并总结了相关的要点，作为后续查阅与参考的备忘。

# RoPE Embedding

关于RoPE的详细推导过程，推荐阅读苏剑林老师的[博客文章](https://kexue.fm/archives/8265)。
输入 hidden state 的维度为 $\text{seq\_len} \times \text{hidden\_dim}$。设第 $m$ 个位置的向量为 $\boldsymbol{q}$，我们可以将 $\boldsymbol{q}$ 的相邻两个元素视为一个复数的实部与虚部。这样，RoPE 其实是在对每对元素进行如下旋转操作：

$$
\boldsymbol{f}(\boldsymbol{q}, m)=\left(\begin{array}{cc}
\cos m \theta_0 & -\sin m \theta_0 \\
\sin m \theta_0 & \cos m \theta_0
\end{array}\right)\binom{q_0}{q_1}
$$

其中 $m$ 的取值范围为 $0$ 到 $\text{seq\_len}-1$，$\theta$ 的取值则从 $\theta_0$ 变化至 $\theta_{\text{hidden\_dim}/2-1}$。但在 Qwen 系列模型中，实际上是将向量的前半部分与后半部分分别作为实部和虚部，两两配对实现旋转操作：
```python
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

$\theta_i$ 的取值方式与经典的正弦位置编码方法保持一致，具体为：$\theta_i = 10000^{-2i/\text{hidden\_dim}}$。展开如下：

- $\theta_0 = 10000^{0/\text{hidden\_dim}} = 1$
- $\theta_1 = 10000^{-2/\text{hidden\_dim}}$
- $\theta_2 = 10000^{-4/\text{hidden\_dim}}$
- $\ldots$
- $\theta_{\text{hidden\_dim}/2-1} = 10000^{-(\text{hidden\_dim}-2)/\text{hidden\_dim}}$

可以看到，$\theta$ 的指数部分从 $0$ 开始，按照 $-2/\text{hidden\_dim}$ 递减，一直到 $-(\text{hidden\_dim}-2)/\text{hidden\_dim}$，即覆盖了从 $0$ 到 $-1$ 的区间。

在实际实现中，为了获得所有 $m\theta$ 的组合，可以借助 einsum 运算高效生成：

```python
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
t = torch.arange(max_position_embeddings, dtype=torch.float)
freqs = torch.einsum("i,j->ij", t, inv_freq)
cos = freqs.cos()
sin = freqs.sin()
cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
self.register_buffer("cos_sin_cache", cache, persistent=False)
```
这样即可通过广播实现不同位置与不同频率的高效旋转参数查找和计算。


# RMSNorm

最早，大家普遍采用的是 Batch Normalization。虽然起初normlization是为了让各层的输入分布“恒定不变”, 但是BatchNorm实际上训练时用小批量统计量、推理时用滑动平均，导致分布并不完全一致。BatchNorm 的真正价值是通过优化动力学的改善来加速训练。进入 NLP 时代后，相较于计算机视觉，自然语言场景下输入句子长度参差不齐，句末往往还存在大量 padding，这使得 BatchNorm 在 NLP 任务中表现不佳。

Layer Normalization应运而生。它在 hidden state 维度进行归一化，摆脱了对 batch size 的依赖。LN 通常包含两个步骤：一是中心化（re-centering），即减去均值；二是缩放（re-scaling），即除以标准差。

RMSNorm 则是在 LayerNorm 基础上的精简变体。研究表明，LN 中的中心化步骤（减去均值）对提升模型性能影响很小，主要起作用的是缩放操作——即对激活值幅度的控制。因此，RMSNorm 省略了中心化，仅保留缩放环节，用均方根（Root Mean Square）来标准化向量幅度。
```python
@torch.compile
def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)
    return x
```
在结构设计上，还有一种流行的“Pre-Norm”范式，即先进行残差加法再归一化（add→norm）。这种结构已证明能有效提升深层模型的训练稳定性,在qwen系列中常用。

```python
# 残差融合的 RMSNorm
# 输入：x, residual
# 输出: RMSNorm(x + residual), x + residual
# 返回的 residual 将作为下一层（如 FFN）输入，并再次参与残差连接。
@torch.compile
def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)
    return x, residual
```

# FlashAttention的API调用与Paged Attention

FlashAttention提供了高效的注意力机制API调用。针对推理场景下的prefill和decode，我们分别关注`flash_attn_varlen_func`和`flash_attn_with_kvcache`这两个API。这两个API虽然都能做attention，但是它们针对的是 LLM 推理过程中两种完全不同的计算模式。区分它们是为了极致的性能优化。
* `flash_attn_varlen_func`: 高并行度，计算密集, 适合prefill阶段
* `flash_attn_with_kvcache`: 访存密集，适合decode阶段

正常来说，输入的Q，K，V的tensor形状应该为`（batch_size, seq_len, num_heads, head_dim）` 。但是在NLP场景下，不同batch的实际sequence长度不尽相同，如果直接用四维数组保存容易产生太多padding造成冗余，flash attention的解决方式非常简单，直接用前缀和的思想即可（对应于底层就是grouped GEMM，解决同样的问题）：
```python
# 输入形状需要从（batch_size, seq_len, num_heads, head_dim）转换为（total_q, num_heads, head_dim），total_q是所有序列长度之和
# 输出的o是(total_q, num_heads, head_dim)
o = flash_attn_varlen_func(q, # (total_q, num_heads, head_dim)
k, # (total_k, num_kv_heads, head_dim)
v, # (total_v, num_kv_heads, head_dim)
max_seqlen_q=context.max_seqlen_q,  # max sequence length of query
cu_seqlens_q=context.cu_seqlens_q, # (batch_size + 1,) cumulative sequence lengths of query
max_seqlen_k=context.max_seqlen_k, # max sequence length of key and value
cu_seqlens_k=context.cu_seqlens_k, # (batch_size + 1,) cumulative sequence lengths of key and value
softmax_scale=self.scale,  # head_dim ** -0.5
causal=True, # whether to apply causal mask
)
```
当然，`flash_attn_varlen_func`这个API还支持paged attention的功能。当使用了block table这个参数后，k和v这两个参数就认为是KV cache, 形状是`(num_blocks, block_size, num_kv_heads, head_dim)`, 它们不是跟着 Batch Size 变的，而是一开始就申请好的固定大小的“内存池”。这个时候block table的形状是`(batch_size, max_num_blocks_per_seq)`。`max_num_blocks_per_seq`表示当前 Batch 中最长的那个序列所占用的 Block 数量。对于短序列，后面的空位会用 -1 进行 Padding。
```python
# 输出的o是(total_q, num_heads, head_dim)
o = flash_attn_varlen_func(q, # (total_q, num_heads, head_dim)
k=k_cache, # (num_blocks, block_size, num_kv_heads, head_dim)
v=v_cache, # (num_blocks, block_size, num_kv_heads, head_dim)
max_seqlen_q=context.max_seqlen_q,  # max sequence length of query
cu_seqlens_q=context.cu_seqlens_q, # (batch_size + 1,) cumulative sequence lengths of query
max_seqlen_k=context.max_seqlen_k, # max sequence length of key and value
cu_seqlens_k=context.cu_seqlens_k, # (batch_size + 1,) cumulative sequence lengths of key and value
softmax_scale=self.scale,  # head_dim ** -0.5
causal=True, # whether to apply causal mask
block_table=context.block_tables, # (batch_size, max_num_blocks_per_seq). max_num_blocks_per_seq表示当前 Batch 中最长的那个序列所占用的 Block 数量。对于短序列，后面的空位会用 -1 进行 Padding。
)
```
block table的计算逻辑如下：
假设 block_size = 16，正在计算第 0 个序列的第 20 个 Token (i=20)：
1.  逻辑位置：
	- logical_block_idx = 20 // 16 = 1 (属于该序列的第 2 个块)
	- offset = 20 % 16 = 4 (块内的第 5 个位置)
2. 查 Block Table:
	* 假设 `block_table[0]` 是 `[100, 55, 80, -1, ...]`
	* 查到 `block_table[0][1]` 的值是 55
3. 物理取值:
	* CUDA Kernel 会直接去读取`k_cache[55][4]`的数据

在decode阶段用的是`flash_attn_with_kvcache`，其输入输出的tensor形状如下：
```python
# o是(batch_size, 1, num_heads, head_dim)
o = flash_attn_with_kvcache(q.unsqueeze(1), # (batch_size, 1, num_heads, head_dim), 当前 Batch 中每个序列生成的最新 Token。注意第 1 维是 Sequence Length，Decode 阶段始终为 1。 
k_cache, # (num_blocks, block_size, num_kv_heads, head_dim)
v_cache, # (num_blocks, block_size, num_kv_heads, head_dim)
cache_seqlens=context.context_lens, # (batch_size + 1,) cumulative sequence lengths of key and value
block_table=context.block_tables, # (batch_size, max_num_blocks_per_seq). max_num_blocks_per_seq表示当前 Batch 中最长的那个序列所占用的 Block 数量。对于短序列，后面的空位会用 -1 进行 Padding。
softmax_scale=self.scale,  # head_dim ** -0.5
causal=True, # whether to apply causal mask
)
```

# KV Cache调度与Prefix Caching
从上面flash attention的参数传递可以看到，KV cache的总”内存池“是预先分配的，存在总大小限制的。当存在多个请求的时候，我们需要管理多个请求的kv cache，使得kv cache大小不超过总容量的限制，必要时刻可能还需要驱逐部分kv cache, 因此我们需要一个调度机制，即scheduler。
首先对于每个请求，我们用一个结构体`Sequence`来管理：
```python
class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        # 这个时候的block table是纯在CPU上的，在context上会组合batch内sequence的block table变成大小b（atch_size, max_num_blocks_per_seq). max_num_blocks_per_seq表示当前 Batch 中最长的那个序列所占用的 Block 数量。对于短序列，后面的空位会用 -1 进行 Padding。
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size
    
    @property
    def num_blocks(self):
	    # 当前这个序列需要多少个block来存储
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size
    
    def block(self, i):
        assert 0 <= i < self.num_blocks
        # Python 的 list 切片在右边界（甚至左边界）超出范围时，不会报错，而是自动截断到合法范围。
        return self.token_ids[i * self.block_size: (i+1) * self.block_size]
```

这里定义了paged attention的一个block size是256，说明每256个token的KV组成一个block。

在llm engine里面，每新来一个请求，我们先生成一个Sequence对象，然后放到调度器里面：
```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)
```
随后跑多轮迭代，每次迭代从调度器里面得到下一轮要跑的sequence batch，同时调度器会决定着一轮是跑prefill还是跑decode。
```python
def step(self):
    # 每个step，从scheduler里面得到下一轮迭代要跑的sequence, 可能是prefill, 也可能是decode
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    # postprocess主要是把token_ids添加到sequence中，并判断是否达到eos或max_tokens
    self.scheduler.postprocess(seqs, token_ids)
    # 返回所有已经完成的sequence的token_ids
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    # 返回这轮生成的token数量
    num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
    return outputs, num_tokens

def generate(self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams],
use_tqdm: bool = True) -> list[str]:
    ...
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)
    outputs = {}
    prefill_throughput = decode_throughput = 0.
    while not self.is_finished():
        t = perf_counter()
        output, num_tokens = self.step()
        ...
    return outputs
```

调度器scheduler的schedule这个函数选出一批 Sequence 来执行一次 forward。它优先服务 Prefill（新来的或被抢占的），如果没有 Prefill 任务，再服务 Decode。
第一步，scheduler尝试调度 Prefill 任务，这会关注`self.waiting`这个队列。如果成功调度了至少一个 Prefill 任务，直接返回 (seqs, True)。这意味着一旦有 Prefill 任务，这一轮就只跑 Prefill，不跑 Decode（因为两者计算特征不同，很少混着跑）。
```python
    def schedule(self) -> tuple[list[Sequence], bool]:
        # 这个函数的目标是选出一批 Sequence 来执行一次 forward。它优先服务 Prefill（新来的或被抢占的），如果没有 Prefill 任务，再服务 Decode（正在生成的）。
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        # 当前的这个batch中已经有的token数量
        num_batched_tokens = 0

        # waiting的意思是还没有进行过prefill或者被抢占的
        # self.waiting队列里面既有刚进来的全新请求，也有之前运行一半因为显存不够被抢占的
        while self.waiting and num_seqs < self.max_num_seqs:
            # 从waiting的队列中取出第一个sequence
            seq = self.waiting[0]
            # 如果当前的这个batch中已经有的token数量加上这个sequence的长度超过了max_num_batched_tokens，或者当前free_block_ids数量不足以分配给这个sequence，则break
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            # 给这个sequence分配更多的block用于用于存储KV
            # 注意如果之前这个sequence被抢占过，那么有可能部分前缀的KV没有被覆盖，可以重新复用, 这部分就不需要做prefill了
            self.block_manager.allocate(seq)
            # 对没有被缓存的token做prefill，说明支持增量prefill
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        # 如果成功调度了至少一个 Prefill 任务，直接返回 (seqs, True)。这意味着一旦有 Prefill 任务，这一轮就只跑 Prefill，不跑 Decode（因为两者计算特征不同，很少混着跑）。
        if scheduled_seqs:
            return scheduled_seqs, True
```

第二步会尝试调度 Decode 任务。这会关注`self.running`这个队列，这些是已经在生成 Token 的序列。这个时候主要是检查主要检查能不能再生成一个 Token. 因为生成一个新 Token 需要占用新的 KV Cache block或者占用当前 Block 的剩余空间。
```python
        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 主要检查能不能再生成一个 Token. 因为生成一个新 Token 需要占用新的 KV Cache block或者占用当前 Block 的剩余空间
            while not self.block_manager.can_append(seq):
                # “受害者选择”调度策略
                # 如果显存不足，尝试腾出空间
                if self.running:
                    # 如果队列里还有其他正在运行的序列，就执行 “牺牲他人” 策略。它会把队列尾部的序列弹出, 并对其执行挂起self.preempt(...)
                    self.preempt(self.running.pop())
                else:
                    # 如果队列空了（只剩当前这个 seq 了）但显存还是不够，那就说明连运行这一个序列的资源都不够。此时只能 “牺牲自己” 
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                # 添加一个token，如果当前最后一个block还有剩余空间，则直接添加，否则需要分配新的block
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

```
注意这里nano-vllm使用了“受害者选择”调度策略。如果发现如果发现显存不够让当前的 seq 生成下一个 Token，那么就把 running 队列末尾的序列（"受害者"）踢回 waiting 队列，释放它的显存，以此来满足当前 seq 的需求。
同时在self.preempt() 中释放资源，状态改回 WAITING，插队到 waiting 队列的最前面，保证下次优先恢复。

这里需要重点强调一下在block manager中的**prefix caching**的机制。这里的caching值得不是KV cache中的“cache”，它指的是**假如某个sequense的KV cache block被释放，当下次这个sequence的重新做prefill的时候，可能它的某个前缀在这段时间并没有被其他请求覆盖掉，那么我们就可以复用这段前缀的KV Cache**。
这个点的关键在于block manager的`_deallocate_block`函数中，虽然ref_count 降为 0，Block ID 会被放入 free_block_ids（空闲队列），但是`hash_to_block_id`里依然保留着记录（除非被别的请求覆盖了）；同时`ref_count`的机制也支持不同sequence之间相同prefix的复用。
```python
def _deallocate_block(self, block_id: int) -> Block:
    # 关键点：虽然ref_count 降为 0，Block ID 会被放入 free_block_ids（空闲队列），但是 hash_to_block_id里依然保留着记录（除非被别的请求覆盖了）。
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)

def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    # 在当前nano-vllm的实现中，如果被抢占了，那么num_cached_tokens会被设置为0
    seq.num_cached_tokens = 0
    seq.block_table.clear() 
```

因此，在`allocate(self, seq: Sequence)` 函数中，
1. 遍历 Sequence需要的每一个 Block。
2. 计算 Hash: 对于满的 Block（长度等于 block_size），基于内容 + 前缀 Hash 计算当前 Block 的 Hash。
3. 查缓存 (hash_to_block_id)。如果发现之前算过一模一样的数据，这个时候有两种情况。
	1. 那么如果block_id已经在used_block_ids中，则说明这个block被其他sequence正在使用，需要可以增加ref_count，实现不同sequence之间的复用；
	2. 如果block_id不在used_block_ids中，则说明这个block是空闲的，但是之前的记录没有被覆盖，只需要形式上重新分配一次即可，但是num_cached_tokens还是可以增加。

```python
def allocate(self, seq: Sequence):
    # Prefill 阶段分配
    # 当一个新的 Sequence（或者被抢占回来的）准备开始 Prefill 时调用。它的目标是为整个 Sequence 分配足够的 Block，并尽可能复用已有的 Block（Prefix Caching）
    assert not seq.block_table
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        # 如果当前的 Block 是填满的，就基于它的内容和前一个 Block 的哈希值计算新的哈希；否则（如果是未填满的尾部 Block），不计算哈希（设为 -1）, 也不做缓存
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        # 系统算出 Hash，去查表。
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        if cache_miss:
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # 如果发现发现 hash_to_block_id 里有记录(block_id!=-1), 并且刚好就是当前的token_ids，那么就可以直接复用这个block，就可以直接复用了
            seq.num_cached_tokens += self.block_size
            # 如果block_id已经在used_block_ids中，则说明这个block被其他sequence正在使用，需要增加ref_count
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            # 如果block_id不在used_block_ids中，则说明这个block是空闲的，但是之前的记录没有被覆盖，只需要形式上重新分配一次即可，但是num_cached_tokens还是可以增加
            else:
                block = self._allocate_block(block_id)
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```


# KV Cache物理存储与映射关系
实际上，上面讲到的block manger只是纯逻辑层的管理器，负责和schedule协调，其内部并没有实际申请大片的KV Cache内存。真实申请KV Cache的地方在model runner里面，它会根据我们参数指定最大内存占用量来申请对应数量的KV Cache内存池：
```python
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free # GPU 上当前已使用的总内存（包括所有进程和 PyTorch 的内存池）
        peak = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current = torch.cuda.memory_stats()['allocated_bytes.all.current'] # PyTorch 当前实际分配的内存（不包括内存池中的空闲块）
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
```

KV Cache 是一个统一的大张量，形状为：
```
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
 ↑      ↑           ↑           ↑
 K/V  每层独立   所有block   每block的token数
```
注意我们每一个layer都需要一组KV Cache。每个 Attention layer 获得的是这个大张量的一个 view（切片）：
```
k_cache = kv_cache[0, layer_id] → shape: [num_blocks, block_size, num_kv_heads, head_dim]
v_cache = kv_cache[1, layer_id] → shape: [num_blocks, block_size, num_kv_heads, head_dim]
```
KV Cache 统一大张量是所有请求共享的。**block manger实际上控制的是这个大张量的第三个维度**,另外注意在prefill阶段，KV Cache是在运行中存储的，在forward过程中，我们需要存储kv cache，这里通过一个自定义的triton kernel来实现。
```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
    
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N, )](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
    

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```
这里我们需要一个变量slot mapping来帮助我们写的triton kernel来完成存储。`slot_mapping`实际上就是把block table展开帮助我们写kernel：
```
slot_mapping[i] = block_table[block_idx] * block_size + offset_in_block
```
对应的代码如下：
```python
    def prepare_prefill(self, seqs: list[Sequence]):
		...
        for seq in seqs:
            seqlen = len(seq)
            ...
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
```
例如，如果 block_size=16，sequence 的 `block_table = [2, 5, 8]`：
```
token 0-15 → slot 32-47 (block 2)
token 16-31 → slot 80-95 (block 5)
token 32-47 → slot 128-143 (block 8)
```

# Prefill和Decode的输入输出准备

之前说过，schedule会决定当前step是做prefill还是做decode，并根据flash attention的对应API准备数据。如果做prefill，那么就要对当前没有cached的tokens计算KV Cache：
```python
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens # 只包含新 tokens 的数量（跳过已缓存的部分）
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        # 这个batch里面存在一个sequence有cache，就是cu_seqlens_k[-1] > cu_seqlens_q[-1]，需要使用prefix cache
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        # https://pytorch-tutorial.web.app/intermediate/pinmem_nonblock.html
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions
    
```

如果做decode，与 prefill 不同，我们每次只处理每个序列的一个新 token。
```python
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq)-1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True) 
        # (batch_size), 每个序列最新生成的 token
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True) 
        # (batch_size),每个序列最新token的位置索引（用于 RoPE）
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) 
        # (batch_size)，每个序列的完整长度（attention 需要关注的范围）
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) 
        # (batch_size)，新 token 的 KV cache 存储位置
        block_tables = self.prepare_block_tables(seqs) 
        # 因为 decode 阶段的每个新 token 的 Query 需要关注所有之前的 KV cache，
        # 必须通过 block_tables 告诉 attention kernel 
        # 如何在 paged memory 中找到这些缓存的 K、V。
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions
```

# 基于Gumbel Softmax的sampler
模型做完推理后，需要根据输出的logits生成下一个tokens：
```python
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill:bool):
        return self.model.compute_logits(self.model(input_ids, positions))
    
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids
```
虽然采样本身不可微，但可以通过 Gumbel-Softmax 的连续松弛实现可微, 同时数学上等价于从分类分布中采样:
```python
import torch
from torch import nn

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
    
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 温度缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # softmax归一化
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-Max采样
        # G_i ~ Exponential(1)  (指数分布)
        # 任何分类分布除以指数分布采样，等价于直接采样
        # sample = argmax_i (p_i / G_i) = argmax_i (log(p_i) - G_i)
        # 虽然采样本身不可微，但可以通过 Gumbel-Softmax 的连续松弛实现可微, 同时数学上等价于从分类分布中采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
```

# 进程间通信：共享内存和NCCL

# Tensor Parallel的实现

## VocabParallelEmbedding

## TP下的矩阵乘法

# Qwen3模型架构

# Model Warmup

# CUDA Graph的加速