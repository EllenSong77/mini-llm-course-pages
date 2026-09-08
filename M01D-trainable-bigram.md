# M1D：一段一段写出可训练的 Bigram

这一节直接跟着代码走。**每次只在 `m1/trainable_bigram.py` 末尾追加一段，运行整个文件，再看这一段的解释。** 第 1–11 步可以按顺序拼成完整脚本，没有需要猜的 TODO。用 CPU 即可。

我们继续使用“我爱猫、我爱狗、你爱猫”。这次要亲眼看到：模型给“猫”的概率，怎样从初始值变到接近 2/3。

先完成第 1–7 步，看到一次参数修改，再继续后面的完整训练。每段下面的“对照着看”是阅读提示，不用逐条交作业。

## 先看地图：整份脚本一共做什么

第 1–11 步的代码按顺序拼起来，就是一份完整的最小训练脚本。它总共做五件事：

1. **备料**（第 1–2 步）：三句话 → 词表和 ID → 12 对“当前 → 下一个”样本。
2. **建表**（第 3–4 步）：创建一张 7×7、先全零的分数表；输入“爱”，取出对应的一行。
3. **打分**（第 5–6 步）：这行分数 softmax 成概率；拿正确答案的概率算出 loss。
4. **学习**（第 7–9 步）：先亲眼看一次“算梯度 → 改表”，再把 12 条样本放在一起训练 1000 轮。
5. **验收**（第 10–11 步）：读出 P(猫|爱)，看它逼近 M1B 数出来的 2/3；训练前后各生成一批句子对照。

一句话：**备料 → 建表 → 打分 → 学习 → 验收**。以后 300M 模型的训练脚本再复杂，骨架仍然是这五件事。

## 第 1 步：先写语料和 token ID

写入：

```python
import torch                      # PyTorch 主库：张量 + 自动算梯度
import torch.nn as nn             # 模型组件库（第 3 步的 Embedding 在这里）
import torch.nn.functional as F   # 函数库（softmax、cross_entropy 在这里）

sentences = ["我爱猫", "我爱狗", "你爱猫"]      # 训练语料：还是那三句
BOS, EOS = "<BOS>", "<EOS>"                     # 句首牌 / 句尾牌
tokens = ["你", "我", "爱", "狗", "猫", BOS, EOS]  # 词表：顺序写死，数字好核对
stoi = {token: i for i, token in enumerate(tokens)}   # token → ID 对照表
vocab_size = len(tokens)          # 词表大小 = 7

print(stoi)                       # 看一眼每个 token 分到的编号
```

输出：

```text
{'你': 0, '我': 1, '爱': 2, '狗': 3, '猫': 4, '<BOS>': 5, '<EOS>': 6}
```

这一步和你之前做过的一样，只是固定了词表顺序，方便后面逐个核对数字。`stoi["爱"]` 得到 `2`，`tokens[4]` 得到“猫”。这两个编号接下来会不断出现。

`nn` 和 `F` 是 PyTorch 两组工具的简写。等用到某个函数时再解释它，这里不用先记。

## 第 2 步：把句子拆成输入和答案

接着追加：

```python
xs, ys = [], []                   # xs 收集“当前 token”，ys 收集“正确答案”
for text in sentences:
    sequence = [BOS] + list(text) + [EOS]             # 首尾加牌：我爱猫 → BOS 我 爱 猫 EOS
    for current, target in zip(sequence, sequence[1:]):  # 相邻配对：(BOS,我)(我,爱)…
        xs.append(stoi[current])   # 这道题：当前 token 的 ID
        ys.append(stoi[target])    # 这道题的答案：下一个 token 的 ID

x = torch.tensor(xs, dtype=torch.long)   # Python 列表 → PyTorch 整数数组
y = torch.tensor(ys, dtype=torch.long)

print("x:", x.tolist())           # 12 道题
print("y:", y.tolist())           # 对应的 12 个答案
for input_id, target_id in zip(xs, ys):
    print(tokens[input_id], "→", tokens[target_id])  # 人话版逐条打印
```

前两行输出：

```text
x: [5, 1, 2, 4, 5, 1, 2, 3, 5, 0, 2, 4]
y: [1, 2, 4, 6, 1, 2, 3, 6, 0, 2, 4, 6]
```

例如 `x[2]=2`、`y[2]=4`，对应“爱 → 猫”。`x` 和 `y` 的同一个位置是一道题和它的答案。

`torch.tensor` 把 Python 列表变成 PyTorch 数组；`dtype=torch.long` 表示里面存整数 ID。此时还没有模型，更没有开始训练。

**对照着看：** 12 条输出中，“爱 → 猫”出现两次，“爱 → 狗”出现一次。后面训练会反复用到这些样本。

## 第 3 步：创建模型实际要修改的那张表

追加：

```python
table = nn.Embedding(vocab_size, vocab_size)   # 一张 [7,7] 可训练查表（刚创建是随机数）
with torch.no_grad():             # 下面是手动改参数，不用记求梯度的草稿
    table.weight.zero_()          # 故意全部置零，方便核对数字（随机初始化也可以）

print("表的大小:", table.weight.shape)   # torch.Size([7, 7])
print(table.weight)               # 此刻应该看到 49 个 0
```

输出是一张 **7 行、7 列、全部为 0** 的表。

这张表怎么读？仍然使用第 1 步的编号：

- 第 2 行：当前 token 是“爱”时，对所有下一个 token 的打分。
- 第 4 列：候选答案“猫”。
- `table.weight[2, 4]`：当前是“爱”时，给“猫”的分数。

这里行列编号都从 0 开始。`nn.Embedding(7, 7)` 创建可训练的查表对象，真正存数字的地方叫 `table.weight`。第一个 7 决定有多少行，第二个 7 决定每行返回多少个数。本节这两个数都等于词表大小，因为我们直接给 7 个候选答案打分。

这里的 0 是**尚未变成概率的分数**，不是“出现 0 次”，也不是“概率为 0”。这种原始分数叫 logit；一组分数叫 logits。

我们故意全部置零，让下面的数字容易核对。`with torch.no_grad()` 表示这次手动初始化不用记录求梯度的过程；它本身不禁止修改参数。这种独立查表模型可以从全零开始学习，不能据此把未来整个 Transformer 都初始化为零。

## 第 4 步：输入“爱”，取出对应的一行

追加：

```python
one_x = torch.tensor([stoi["爱"]], dtype=torch.long)   # 一道题：输入“爱”（ID=2）

one_logits = table(one_x)         # 查表：按输入 ID 取出第 2 行 → 形状 [1, 7]
print("输入:", one_x)
print("模型给的分数:", one_logits)
print("分数的形状:", one_logits.shape)   # 1 条样本 × 7 个候选
```

你会看到：输入是 `[2]`，分数是 `[[0, 0, 0, 0, 0, 0, 0]]`，形状是 `[1, 7]`。

**`table(one_x)` 做的具体事情就是：按输入 ID 取行。** 输入 2 就取第 2 行。结果有一行，因为我们只输入了一条样本；这一行有七列，因为有七种候选答案。

这一步只需要输入。这道题的正确答案“猫”（ID=4）到第 6 步评分时才会出场——**预测时模型看不到答案**，这是训练的规矩：看到答案的预测没有意义。

这就是旧版代码中 `model(x)` 此刻承担的工作。先直接操作这张表，最后再说明如何包进 class。

## 第 5 步：把这一行分数转换成概率

追加：

```python
one_probs = F.softmax(one_logits, dim=-1)   # 这一行分数 → 一行加起来等于 1 的概率
for token, probability in zip(tokens, one_probs[0].tolist()):  # [0] 取出唯一那一条样本
    print(token, round(probability, 4))     # 每个候选分到多少概率
```

七个 token 的输出都是 `0.1429`，约等于 `1/7`。

`F.softmax` 把同一行中的分数转成概率。它的计算是“每个分数先取指数，再除以这些指数的总和”。这里七个分数都是 0，指数都是 1，所以各自分到 `1/7`。

`dim=-1` 指最后一个维度；在 `[1, 7]` 中就是这七列。意思是让**同一条样本的七个候选**一起比较。`one_probs[0]` 则取出唯一那条样本的概率行。

此时还没训练，“猫”和其他候选分到的概率完全相同。

## 第 6 步：拿出正确答案“猫”，计算 loss

追加：

```python
one_y = torch.tensor([stoi["猫"]], dtype=torch.long)   # 标准答案“猫”（ID=4）：评分才需要它

p_cat = one_probs[0, stoi["猫"]]            # 手动路线第 1 步：取出正确答案“猫”的概率
manual_loss = -torch.log(p_cat)              # 第 2 步：M1C 的公式 -log(真实答案概率)
one_loss = F.cross_entropy(one_logits, one_y)  # PyTorch 打包版：直接喂分数和答案 ID

print("给猫的概率:", p_cat.item())          # .item()：单元素张量 → 普通 Python 数字
print("按 M1C 算出的 loss:", manual_loss.item())
print("cross_entropy 算出的 loss:", one_loss.item())   # 两个数应该完全一样
```

两个 loss 都约为 `1.9459`。

现在把这三行对应起来：

1. `p_cat` 从七个概率里，选出正确答案“猫”的概率。
2. `-torch.log(p_cat)` 就是 M1C 的 `-log(真实答案概率)`。
3. `F.cross_entropy(one_logits, one_y)` 直接接收分数和答案 ID，用数值稳定的方式计算同一件事。

因此，**交给 `cross_entropy` 的是原始分数 `one_logits`**。上一节已经得到的 `one_probs` 是供我们观察和核对用的，不要把它当成 logits 再传进去。

`.item()` 只是把只有一个数的张量取成普通 Python 数字，便于打印。后面反向传播仍使用张量 `one_loss`。

## 第 7 步：让它针对“爱 → 猫”学习一次

先追加这一小段，运行并查看梯度：

```python
optimizer = torch.optim.SGD(table.parameters(), lr=1.0)  # 改表助手：管哪些参数、步子多大
optimizer.zero_grad()      # 先清空梯度（PyTorch 默认会把梯度累加）
one_loss.backward()        # 算梯度：问每个格子“加大一点，loss 会怎么变”

print("爱这一行的梯度:")
print(table.weight.grad[stoi["爱"]])       # .grad 里存着刚算出的梯度
print("更新前爱这一行的分数:")
print(table.weight[stoi["爱"]].detach())   # 表本身此刻应该还是全 0
```

梯度约为：

```text
[0.1429, 0.1429, 0.1429, 0.1429, -0.8571, 0.1429, 0.1429]
```

第 4 列对应“猫”，它是负数；其他列是正数。此时分数表仍然全零：**`backward()` 计算了梯度，还没有执行参数修改。**

接着追加：

```python
optimizer.step()           # 真正改表：新分数 = 旧分数 − lr × 梯度

print("更新后爱这一行的分数:")
print(table.weight[stoi["爱"]].detach())

with torch.no_grad():      # 下面只是看结果，不训练
    new_logits = table(one_x)                  # 再查一次“爱”这一行
    new_probs = F.softmax(new_logits, dim=-1)  # 再 softmax 成概率
    print("更新后给猫的概率:", new_probs[0, stoi["猫"]].item())
    print("更新后这道题的 loss:", F.cross_entropy(new_logits, one_y).item())
```

更新后的分数约为：

```text
[-0.1429, -0.1429, -0.1429, -0.1429, 0.8571, -0.1429, -0.1429]
```

“猫”的概率从 `0.1429` 增加到约 `0.3118`，loss 从 `1.9459` 降到约 `1.1654`。

为什么会这样？本节的 SGD 只做下面这个运算：

```text
新分数 = 旧分数 - 学习率 × 梯度
猫的分数 = 0 - 1.0 × (-0.8571) = 0.8571
狗的分数 = 0 - 1.0 × 0.1429 = -0.1429
```

`table.parameters()` 告诉优化器要修改哪些参数，这里就是那张分数表。`lr=1.0` 决定这次修改的步幅。`zero_grad()` 清掉此前留下的梯度，因为 PyTorch 默认把新算出的梯度累加起来。`.detach()` 在这里只用于打印一份不带梯度关系的视图。

**到这里，你已经亲眼看过一次训练：查表 → 用答案算 loss → 求梯度 → 修改表。** 因为这次只学了“爱 → 猫”，所以“狗”的分数下降了；下一步要把训练集里“爱 → 狗”的样本也一起考虑进去。

## 第 8 步：一次处理全部 12 条样本

单条样本演示结束。为了开始一次独立的正式实验，把表恢复为全零，并清掉演示梯度。追加：

```python
with torch.no_grad():
    table.weight.zero_()   # 单样本演示结束，表清零，重开正式实验
optimizer.zero_grad()      # 演示留下的梯度也清掉

logits = table(x)                   # 12 条样本一起查表 → [12, 7]
loss = F.cross_entropy(logits, y)   # 12 条各自的 loss 取平均

print("12 条样本的分数形状:", logits.shape)
print("全部样本的平均 loss:", loss.item())
print("下标 2 和 6 的样本查出的分数相同:",
      torch.equal(logits[2], logits[6]))   # 两条都是“爱”，查同一行，当然相同
```

输出形状是 `[12, 7]`，平均 loss 约 `1.9459`。这里没有创建 12 份模型：12 条样本都在查同一张 7×7 参数表。

`x[2]` 和 `x[6]` 都是“爱”，所以查出的分数相同；但 `y[2]` 是“猫”、`y[6]` 是“狗”。`cross_entropy` 会对每行选出该行正确答案的概率，再将 12 项损失取平均。

这就是为什么训练后“爱”这一行要同时给猫和狗分配概率。两次猫、一次狗，都通过各自的 loss 影响同一行参数。

## 第 9 步：重复训练 1000 次

追加：

```python
initial_loss = loss.item()                       # 训练前的平均 loss（约 1.95）
initial_weights = table.weight.detach().clone()  # 快照；clone 保证后续训练不会改到它

for step in range(1000):              # 重复训练 1000 轮
    logits = table(x)                 # ① 查表
    loss = F.cross_entropy(logits, y) # ② 打分

    optimizer.zero_grad()             # ③ 清掉上一轮梯度
    loss.backward()                   # ④ 算梯度
    optimizer.step()                  # ⑤ 改表

    if step in (0, 99, 499, 999):     # 抽查几个点，观察 loss 下降
        print("第", step + 1, "轮，更新前 loss:", loss.item())

with torch.no_grad():                 # 训练完，统一再测一次
    final_loss = F.cross_entropy(table(x), y).item()

print("训练前:", initial_loss)
print("训练后:", final_loss)          # 应降到约 0.32
```

循环里的六行正是前面已经运行过的动作。每轮重新查表、重新计算 loss，是因为参数已经被上一轮修改。这里每一轮都用全体 12 条样本。

`initial_weights` 保存一份训练前的数字，供第 11 步做生成对照；`.clone()` 确保以后改表时不会把快照一起改掉。

最终 loss 应降到约 `0.325`，接近 M1C 计数模型的 `0.318`。不用追求完全一致。即使充分训练，当前语料的最低平均 loss 也不是 0，因为“爱”后面同时有猫和狗，模型不可能对每条样本的不同答案都给概率 1。

## 第 10 步：读出训练后的“爱 → 猫/狗”概率

追加：

```python
with torch.no_grad():
    love_probs = F.softmax(table(one_x), dim=-1)[0]   # 训练后“爱”这一行的概率

p_cat = love_probs[stoi["猫"]].item()   # 应接近 2/3
p_dog = love_probs[stoi["狗"]].item()   # 应接近 1/3
print("P(猫 | 爱):", p_cat)
print("P(狗 | 爱):", p_dog)

assert final_loss < initial_loss        # 训练有效：loss 降了
assert abs(p_cat - 2 / 3) < 0.05        # 和 M1B 数出来的 2/3 对上了
assert abs(p_dog - 1 / 3) < 0.05        # 和 M1B 数出来的 1/3 对上了
```

你会得到大约 `0.664` 和 `0.330`。现在把它与第 5 步对照：初始时每个候选都是 `1/7`；1000 次更新后，概率已经接近语料中的 2:1 比例。

模型仍然只接收一个 token ID。以后“我爱”和“你爱”来到最后一个“爱”时，查到的仍然都是第 2 行。这解释了它为什么仍是 Bigram。

## 第 11 步：用学到的概率接龙

最后追加：

```python
@torch.no_grad()               # 整个函数只生成、不训练
def generate(score_table, seed=42, max_length=20, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    rng = torch.Generator().manual_seed(seed)   # 固定随机种子，结果可复现
    current_id = stoi[BOS]                      # 从“句子以什么开头”问起
    output_ids = []                             # 收集生成的 token

    for _ in range(max_length):                 # 最多接 max_length 个字
        current = torch.tensor([current_id], dtype=torch.long)
        scores = score_table(current)[0].clone()      # 查表取一行分数；clone 备份，不改到真表
        scores[stoi[BOS]] = -float("inf")       # 屏蔽 BOS：正文里不该出现句首牌
        probs = F.softmax(scores / temperature, dim=-1)   # 先除温度，再 softmax
        next_id = torch.multinomial(probs, 1, generator=rng).item()  # 按概率抽一个

        if next_id == stoi[EOS]:                # 抽到句尾牌：结束
            break
        output_ids.append(next_id)              # 否则记下这个字
        current_id = next_id                    # 以它为下一步输入，继续接龙

    return "".join(tokens[i] for i in output_ids)


before_table = nn.Embedding(vocab_size, vocab_size)   # 复刻一份“训练前的表”做对照
with torch.no_grad():
    before_table.weight.copy_(initial_weights)        # 把第 9 步存的快照装进去

for seed in range(10):                 # 同一组种子，训练前后各生成一遍
    print(seed, "训练前:", repr(generate(before_table, seed=seed)),
          "训练后:", repr(generate(table, seed=seed)))

assert len(generate(table, max_length=5)) <= 5    # 长度约束
```

按生成函数里的顺序看：

1. `current_id` 最初是 BOS 的 ID，也就是询问“句子以什么开头”。
2. `score_table(current)[0]` 查出这个 token 对应的一行分数。
3. 将 BOS 的候选分数设为负无穷，让它在生成时概率为 0，因为我们不把 BOS 当成正文输出。这里修改的是复制出的分数，参数表没有被修改。
4. `softmax` 得到概率，`multinomial` 按概率抽一个 ID。`temperature=1.0` 时就是直接使用分数。
5. 抽到 EOS 就结束；否则记下这个 ID，把它作为下一次输入，重复查表。

`@torch.no_grad()` 让整个生成函数不记录求梯度过程。`repr` 让空字符串显示为 `''`，便于看出模型是否一开始就抽到了 EOS。

循环最多记录 `max_length` 个正文 token。本节正文 token 都是单个汉字，所以可以用字符串长度核对；以后使用子词 tokenizer，要数 token ID 的个数，不能直接拿字符长度代替。

运行后，把最后的生成调用分别加上 `temperature=0.5` 和 `temperature=1.5`，各看一批。分数除以更小的正数，会扩大分数差，让高分候选更占优势；更大的温度则让概率更接近均匀。这是生成时的选择，不会重新训练模型。

## 回头看：旧版 class 与现在的代码有什么关系

这一段只读，不需要追加到脚本。旧版这样写：

```python
class TrainableBigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_ids):
        return self.table(token_ids)
```

`__init__` 把第 3 步的表放进一个模型对象；`super().__init__()` 初始化 PyTorch 的模块管理功能；`forward` 描述输入怎样算成输出，这里就一行查表。

调用 `model(x)` 时，PyTorch 会通过模块调用机制执行 `forward(x)`，最终执行 `self.table(x)`。**本节直接写的 `table(x)`，就是原来 `model(x)` 内部的核心计算。** class 是把已有计算组织起来，没多出另一套训练原理。

## 本节交什么

- 你的脚本，以及训练前后 loss、“猫/狗”概率和生成对照。
- 用自己的话解释：第 7 步里，为什么 `backward()` 之后表还没变，`step()` 之后才变？
- 解释第 8 步：相同输入“爱”遇到不同答案“猫/狗”，为什么必须共用同一行分数？这会限制模型记住多少上下文？

这次先按代码顺序看懂，每个函数的名字都可以查。能对应出“这一行读了谁、算了什么、改了谁”，就达到了本节要训练的能力。
