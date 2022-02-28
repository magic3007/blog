最近被同学蛊惑了区块链, 于是看一些关于北京大学肖臻老师的[《区块链技术与应用》](https://www.bilibili.com/video/BV1Vt411X7JF)这门课. 肖老师讲得挺好, 这里记录一些BTC部分的笔记, 如果感觉笔记格式笔记乱的话可以看原版[notion](https://magic3007.notion.site/BlockChain-BTC-3d0a736d49b24357871f428224fb41ca)的笔记.
# BTC协议

- 挖矿实际在做的, 找到一个noise, 使得$Hash(block\ header, noise) \leq threshold$.

- 哈希指针 (Hash Pointer):除了指明存储位置，还对存储内容进行加密，保证内容没有被篡改.
    - 「例」区块链是使用哈希指针链接的. 对于单个节点而言, 我们可以只保留最近的若干个区块和tail哈希指针; 这样当我们问其他节点要前面的区块, 可以防止其他人篡改.

{% include img.html src="BlockChain%204d923/Untitled.png" alt="Untitled" %} 

- Merkle Tree: 和二叉树(binary tree)的区别是用哈希指针代替的普通指针.
- Merkle Proof(也称为proof of membership/inclusion): merkle tree的一个运用就是merkle proof. 比特币中的节点分为两类, 一列是全结点, 一个是轻结点. 全结点包括交易内容(block body)和哈希指针(block header); 一个是轻结点, 只保存merkle root, 比如比特币钱包.
    - 「例」对于轻结点, 利用merkle tree的性质可以以O(logn)的实际快速确认一笔交易是否已经写入区块链.
    - Proof of Non-membership: 如果对叶结点的排列顺序没有要求, 复杂度是O(n); 如果对叶结点排列顺序有要求(sorted merkle tree), 比如哈希值排序, 可以先找到查询交易的前后两个交易, 然后看这两个交易是否是相邻的即可.

{% include img.html src="BlockChain%204d923/Screen_Shot_2022-02-17_at_23.18.39.png" alt="Screen Shot 2022-02-17 at 23.18.39.png" %} 

「安全性分析」

- 「共识」最长合法链 ← 防止分叉攻击(forking attack)
    - 有可能会存在临时分支, 但是最早被拓展的才会被当作最长合法链
- 「共识」靠算力获得记账权 ← 防止女巫攻击(sybil attack, 防止如果按照账户投票的话, 某台计算机产生大量虚假账号从而控制整个区块链)
- 如何防止double spending? → 记录交易本身, 通过指向前面的交易来说明币的来源, 通过交易记录来计算每个账户的余额.
- 转账人如何说明这笔转账就是自己(即私钥)发出的? → 转账人用私钥对交易进行数字签名, 并公布公钥.
- 如何防止替身攻击, 比如A向B转账, 但是有个人直接用了用A的名义(A的公钥可能公布不及时), 但是用了自己的公钥和私钥进行数字签名制造虚假交易? → 每笔转账都要给出收款人的地址或者公钥, 这样新的交易指向前面的交易可以可以验证没有人在顶替A.
- 收款人为什么需要给地址(公钥的哈希)而不用公钥, 甚至可以每次转账都用一个新的地址? → 地址不会暴露公钥, 安全性更高, 而且反复用同一个地址会降低安全性和隐私.
- 如何避免临时分叉等导致的篡改风险?
    - 解决方法是多等几个区块才确定不可篡改, BTC是默认6个区块后.
    - 根据距离当前最新块的距离, 交易可以分为0-conformation, 1-conformation, 6-conformation. BTC是默认6-conformation后是不可篡改的.
- 「selfish mining」有没有可能一个矿工隐藏多个挖到的区块, 来制造最长链?
    - 这种情况是基本不可能出现的, 原因是1. 在诚实结点占多数的情况下, 成功概率很低 2. 就铸币奖励而言, 这样做是没有好处的.

# BTC实现

- UTXO(Unspent Transaction Output)
    - 记录每笔交易的收款者还没有花出去的BTC
    - 一笔交易可以有多个输出
    - 每一笔新的交易都需要说明支付者的BTC来源, 包括来源交易的哈希和是该交易的第几个输出

<aside>
🤔 基于通过遍历交易来获得余额的方式称为transaction-based ledger, 如BTC; 与之相对的是account-based ledger, 如ETH.

</aside>

- 交易费: 除此以外,每笔交易都需要支付一定的交易费给矿工
    - 总的来说, 矿工的收益来源于铸币收益和交易费

- 「Q」nonce域只有32位, 在挖矿难度需要不断提高的今天, 还有什么空间能进一步提高挖矿难度?

- 每个区块记录的交易中有一笔交易是转给矿工的铸币收入, 可以改这笔铸币交易的coinbase域, 进而可以修改merkle root hash.

「难度分析」

- 单个矿工挖矿是一个Bernoulli trial, 多个矿工挖矿是一个Bernoulli Process
- 当成功概率很低时, Bernoulli Process中整个系统当出块时间可以看作Poisson Process, 服从指数分布
- 指数分布具有“无记忆性”, 也就是说从任意中间时间开始后仍然是指数分布
- “无记忆性“对区块链非常重要, 因为其保证成功概率与算力成正比, 防止算力强的矿工有不成比例的优势.

[指数分布 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E6%8C%87%E6%95%B0%E5%88%86%E5%B8%83)

「数量分析」

- 21万个区块后每个区块的铸币收入减半, 初始时每个区块的铸币收益是50BTC.
- 每10min出一个新的区块, 每约四年每个区块的铸币收入减半.
- 总的比特币数量是有限的: $21万 \times 50 \times (1 + \frac{1}{2} + \frac{1}{4} + \cdots) = 2100万$.

<aside>
🤔 实际上, 稀缺的东西反而不适合作为货币. 一个好的货币应该具有通货膨胀的功能, 否则容易形成马太效益, 比如类似在现代中国购入的房产, 不利于鼓励生产力的发展. 更多相关的知识可以看《货币金融学》.

</aside>

# BTC网络

BTC网络的原则是easy, robust, but inefficient

- BTC网络用的是flooding的方式, 遇到新的消息就广播给其他邻居
- 这种方式是非常消耗带宽的, 因此初始BTC每个区块大小才1M
- 此外, 网络上的邻居并不对应地理位置上的邻居, 进一步造成带宽消耗的增加.

# 挖矿难度

- 区块中有一个域target, 表示区块哈希值二进制前多少位为零
- 每两个星期调整一次难度, 调整方式是$target'=target \times \frac{actual \ time}{expect\ time}$, 使得每个区块的出块时间大概在10min附近.
- 轻结点无法单独验证交易的合法性, 进而无法单独验证最长合法链, 但是可以验证最长链

「矿池」

- “矿主-矿工”是一种挖矿合作形式, 可以规定almost valid block来证明工作量, 每个矿工的收入用工作量划分.
- almost valid block指的是比当前区块链难度稍微简单一点的target
- 注意由于铸币交易的收款方指向了矿主, 因此矿工不可能暗地修改铸币收入方为自己.

# 比特币脚本

- 从概念上来说, 每笔交易的vin部分需要汇款者的数字签名和公钥, 以及BTC的来源; vout部分需要有收款人的地址或者公钥.
- 另外, 为了方便验证交易的合法性, BTC设计了一种比特币脚本来验证.
    - 比特币脚本是一种栈式语言, 中间任意一步出错那么验证就不会通过.
    - 每笔交易都有两个脚本, 分别是input script和output script.
- 需要注意的是, BTC是后一笔交易的input script拼接到前一笔的交易的output script后面执行的.
- 「推论」我们可以看到, 新的交易output script是在形成该笔交易的时候是不会被执行; 只有这笔新交易继续后结交易的时候才会被执行.
- 这里介绍三种比较经典的比特币脚本例子: P2PK, P2PKH, P2SH.
- 比特币脚本还有一个特殊的命令`RETURN`, 利用这个命令可以产生代币, 创造digit commitment等.

{% include img.html src="BlockChain%204d923/Untitled%201.png" alt="Untitled" %} 

## P2PK(Pay to Public Key)

- 这是最简单的一种交易模式, 我们可以看到, output script给出了前一个交易收款者的公钥, input script给出了后一个交易的支付者(也就是前一个交易的收款者)的签名, 这样可以防止有人冒名顶替前一个交易的收款者(冒名顶替者给不出前一个交易收款者的公钥对应的数字签名).
- 注意这里input script里面没有显式给出后一个交易的支付者(也就是前一个交易的收款者)的公钥, 但是结合利用input script里面的签名我们可以验证前一个交易的output script里面的公钥就是后一个交易的支付者的公钥.

{% include img.html src="BlockChain%204d923/Untitled%202.png" alt="Untitled" %} 

{% include img.html src="BlockChain%204d923/Untitled%203.png" alt="Untitled" %} 

## P2PKH(Pay to Public Key Hash)

- 所谓地址就是公钥的哈希, 这是用得最广泛的一种. 我们可以看到, output script给出前一个交易收款者的地址(公钥的哈希)以及哈希算法, inputscript给出了后一个交易支付者的签名和公钥.
- 就后一个交易的input script而言, P2PKH比P2PK多了一个后一个交易的支付者(也就是前一个交易的收款者)的公钥, 这是为了解决前一个交易用哈希算法隐藏了收款者公钥的问题.

{% include img.html src="BlockChain%204d923/Untitled%204.png" alt="Untitled" %} 

{% include img.html src="BlockChain%204d923/Untitled%205.png" alt="Untitled" %} 

## P2SH(Pay to Script Hash)

- P2SH分为两阶段验证. 第一次验证只会用序列化的redeem script(赎回脚本)的哈希值, 第二次继续在原来的栈的基础上验证赎回脚本. 只有两次验证都通过才算是通过验证.
- P2SH有很多用处, 比如用P2SH实现P2PK, 用P2SH实现P2PK实现P2PKH, 用P2SH实现多重签名.

「用P2SH实现P2PK」

- 我们可以看到, redeem script是由上一个交易的input script指定的, 新交易并不需要知道redeem script的具体内容, 只需要知道哈希值. 这样的一个好处是, 创建新交易的时候可以延期按照指定之前指定的脚本内容行动, 而不用知道脚本内容, 这样的利用常见如电商, 多重签名等.

{% include img.html src="BlockChain%204d923/Untitled%206.png" alt="Untitled" %} 

{% include img.html src="BlockChain%204d923/Untitled%207.png" alt="Untitled" %} 

{% include img.html src="BlockChain%204d923/Untitled%208.png" alt="Untitled" %} 

「用P2SH实现多重签名」

- 比如说一个公司内部有5个私钥, 规定只有给了其中3个的签名, 才允许交易.
- 在P2SH出现之前, 用的是`CHECKMULTISIG` 这个命令, 但是现在都是用P2SH实现.

{% include img.html src="BlockChain%204d923/Untitled%209.png" alt="Untitled" %} 

{% include img.html src="BlockChain%204d923/Untitled%2010.png" alt="Untitled" %} 

## Proof of Burn “死亡证明”

- 比特币脚本还有一个特殊的命令`RETURN`, 脚本遇到这个命令直接返回错误.
- 注意RETURN一般写在output script里面, 那么在创建这个交易的时候并不会执行, 但是可以保证这个交易后面不能再接交易了.
- 这个其实可以用来“销毁”一些比特币, 来获得一些功能, 就是所谓的“Proof of Burn”.
    - 虽然通过“交易费”这个手段, 我们可以实际上不销毁BTC.
- 利用这个“销毁”功能, 我们可以产生代币(Alternative Coin), 即通过证明你确实销毁了一些BTC, 从而获得一些另外的代币.
- 同时我们也可以利用这个功能创建digital commitment, 即往区块链里面写入永久保存的内容, 我们只需要在RETURN后写入需要永久保存的内容的哈希值即可.
    - 通过这个方式比利用coinbase的优势在与, coinbase只有记账结点才能用, 但是这个域所有人都能用.

# BTC分叉

- 一个关于分叉(fork)的taxonomy是fork分为state fork和protocol fork. state fork可能是本来临时性的多最长合法链的存在, 或者forking attack造成的; protocol fork是由于协议更新意见不一造成的.
- protocol fork进一步可以分成hard fork和soft fork. hard fork是发叉成了分别被两个社区认可的链; soft fork指的是在协议上”新认旧, 旧不认新“, 实际上出现也是临时性的.
- 在发生hard fork后, 原来分叉前的持币者, 会获得相应的分叉后的新币.
- 一些soft fork的例子如新增的P2SH脚本, 这是”新认旧, 旧不认新“的协议更新.

# 关于BTC匿名性的讨论

「一些可能破坏BTC匿名性的线索推断」

- 同一个交易的input可能同属于一个人
- 在多input, 多output的交易中, 其中一个output可能用于找零
- ...
- 只要与实体世界进行交互, 匿名性就很难获得保证.

「一些提高匿名性的手段」

分别从network layer和application layer角度出发.

- (network layer)可以使用TOR(洋葱路由), 防止IP跟踪
- (application layer): coin mixing服务(把不同人的币先混合在一起然后再分发)

<aside>
🤔 历史出现的一次BTC硬分叉是部分人想把原BTC的区块大小从1M改成4M, 这样修改了区块大小, 造成硬分叉. 新的用4M区块大小的链现在称为BCH(Bitcoin Cash), 原来仍然用1M区块大小的称为BTC.
以太坊也有类似的分叉, 即ETH(新链)和ETC(旧链).

</aside>

# 零知识证明

零知识证明的定义见wikipedia.

[零知识证明 - 维基百科，自由的百科全书](https://zh.wikipedia.org/zh/%E9%9B%B6%E7%9F%A5%E8%AF%86%E8%AF%81%E6%98%8E)

<aside>
🤔 基于公私钥的数字签名, 在部分人看来不算严谨的零知识证明.

</aside>

「同态隐藏」是零知识证明的数学证明.

1. (无冲突性) 若x≠y,则E(x)≠E(y) ⇒(推论) 若E(x)=E(y),则x=y.
2. (单向性) 给定E(x), 无法反推x
3. 给定E(x)和E(y), 容易计算关于x,y的加密函数值.
- 同态加法: E(x), E(y) ⇒ E(x+y)
- 同态乘法: E(x), E(y) ⇒ E(xy)
- 拓展到多项式.

「例」Alice如何向Bob证明知道一组数x和y使得x+y=7, 但是不让Bob知道x和y的具体数值.

一个简单的做法将是

1. Alice告诉Bob E(x)和E(y)
2. Bob根据E(x)和E(y)计算得到E(x+y)
3. Bob对比E(7)和E(x+y)是否相等.

在实际中, 还需要对x和y进行随机化出来, 防止Bob通过暴力枚举的方式获得x和y的具体数值.

「盲签方法」

- 一种中心化的记账方式, 但是又能防止央行知道每一笔交易.
- 具体是由用户来产生货币的“编号”, 但是不告诉央行.

{% include img.html src="BlockChain%204d923/Untitled%2011.png" alt="Untitled" %} 

「零币&零钞」

- 利用零知识证明的手段, 在协议层增加了匿名性的增强
- 零币基于基础币(如BTC)和零币之间的转换使用
- 零钞则不需要基础币.