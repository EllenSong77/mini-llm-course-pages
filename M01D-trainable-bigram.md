# M1D · 这次不数答案，让模型自己学（人话版）

> 这是 [M01D-trainable-bigram.html](./M01D-trainable-bigram.html) 的手机阅读版，内容相同。
> 前置：M1B（计数 bigram）、M1C（loss / NLL）。完成后 M1 结束。

M1B 你亲手数出一张表；M1C 你给模型打的分算出了一个 loss。这一节把“数出来的表”换成“一开始随便填的表”，用 loss 当老师，一步一步把它教对。这个“教”的过程，就叫训练。

## 本节一句话

大模型里没有人在数“爱后面出现过几次猫”。它有的是一张**一开始随机填好、可以不断修改的表**，和一套自动改表的流程：

> 查表 → 算 loss → 算梯度 → 改表 → 再来一遍

本节用你已经熟悉的 Bigram 把这个流程完整走一遍。以后 300M 模型把表换成复杂的神经网络，但这套流程一步不多、一步不少。

## 1. 唯一的变化：表的来历

| | M1B 计数模型 | 本节可训练模型 |
|---|---|---|
| 表怎么来的 | 你一行行数出来的 | 一开始随机填，训练中慢慢改 |
| 表里的数字 | 真实出现次数：非负整数 | 模型自己打的分：可正可负，可大可小 |
| 怎么变成概率 | 次数 ÷ 行总和 | softmax（下面解释） |
| 完全一样的地方 | 都只看前 1 个 token，预测下一个 token | |

**logit 就是“还没变成概率的分数”。** 它可以是负数，一整行加起来也不必等于 1。很多个分数合起来叫 logits。

**softmax 是“次数 ÷ 行总和”的替身。** 次数都是正的，直接除就行；分数可能是负数，不能直接除。softmax 分两步：先用指数函数把每个分数变成正数（大的还是大，小的还是小，只是全变正了），再除以总和。你不需要手算它，记住“一行分数进去，一行加起来等于 1 的概率出来”就够了。

## 2. 训练一圈只有五步

拿一个训练样本“爱 → 猫”走一圈：

1. **查表**：把“爱”的 ID 交给模型，拿回一行 7 个分数。
2. **打分**：`F.cross_entropy(logits, 正确ID)` 做两件事：先把这行分数 softmax 成概率，再按 M1C 的算法算 `-log(真实答案的概率)`。它就是你手算过的那个 loss，被打包成了一个函数，数值上也更稳。
3. **算梯度**：`loss.backward()` 对表里每个数字问同一个问题：“把你稍微加大一点，loss 会变高还是变低？变多快？”答案——方向加幅度——就叫**梯度**。
4. **改表**：`optimizer.step()` 照着梯度，把每个数字往“loss 会降”的方向挪一小步。步子的大小叫 **lr**（learning rate，学习率），本节固定 1.0，照用即可；怎么选它是 M6 的事。
5. 用全部 12 个样本重复 1–4。重复很多圈，loss 越来越低。

> 每圈开头要 `optimizer.zero_grad()`：PyTorch 默认把新梯度**累加**在旧梯度上。不清零，这一圈的方向就和上一圈混在一起了。

> 只想看看 loss、不打算训练的时候（比如训练前后各测一次），用 `with torch.no_grad():` 把代码包起来，意思是“别准备改表要用的材料”。省内存，也算得快。训练循环里不需要它。

### 梯度不是答案

梯度不会告诉模型“把这个格子改成 2/3”。它只说方向和快慢。挪一小步、再看一次、再挪——绕，是这个方法的笨处，也是它能推广到几亿参数的原因。

## 3. 新面孔对照表

本节一口气出现不少 PyTorch 名字。它们不是新知识，大多是你已经写过的东西换了写法。对照着看，不用背：

| 你已经写过的（M1B / M1C） | 本节的写法 | 一句话 |
|---|---|---|
| list 套 list 的 counts 表 | `nn.Embedding(7, 7)` | 同一张 7×7 表，交给 PyTorch 管才能被训练 |
| 次数 ÷ 行总和 | `F.softmax(行, dim=-1)` | 一行分数 → 一行和为 1 的概率 |
| `-math.log(p)` | `F.cross_entropy(logits, y)` | softmax 加上 M1C 的 loss，打包成一个函数 |
| （没有对应，新能力） | `loss.backward()` | 算出每个数字该加还是减、动多少 |
| （没有对应，新能力） | `optimizer.step()` | 照梯度把数字挪一小步 |
| （没有对应） | `optimizer.zero_grad()` | 清掉上一轮梯度，防止累加 |
| `random.choices(候选, weights=概率)` | `torch.multinomial(probs, 1)` | 按概率抽 1 个，同一个意思 |
| `random.Random(seed)` | `torch.Generator().manual_seed(seed)` | 固定随机种子，同一个意思 |
| （没有对应） | `torch.no_grad()` | “只算不改”模式 |

## 4. 全节只需要盯住四个形状

**tensor（张量）就是 PyTorch 里的数组**：一维像 list，二维像嵌套 list。形状（shape）就是每一维的长度：`[12]` 是 12 个数排成一排，`[12, 7]` 是 12 行乘 7 列。

三句话的语料一共能切出 12 对“当前 → 下一个”，词表 7 个 token：

| 名字 | 形状 | 是什么 |
|---|---|---|
| `x` | `[12]` | 12 个当前 token 的 ID（M1B 数据集的 PyTorch 版） |
| `y` | `[12]` | 对应的 12 个正确答案 ID |
| 表（`model.table`） | `[7, 7]` | 那张可训练分数表，形状和 M1B 的 counts 一模一样 |
| `logits` | `[12, 7]` | 12 个样本各查一次表，各得一行 7 个分数 |

> `nn.Embedding(7, 7)` 在本节就是一张**能被训练的 [7, 7] 查表**：输入一个 ID，返回那一行。它的本职是“token → 向量”，那是 M5 的用法；本节借用它最简单的用法，直接当分数表。

## 5. 动手任务

新建 `m1/trainable_bigram.py`。脚手架如下：普通写法直接照用，三处 `TODO` 是你要补的（每一处都在上面的五步里讲过）。

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

sentences = ["我爱猫", "我爱狗", "你爱猫"]
BOS, EOS = "<BOS>", "<EOS>"

tokens = sorted(set("".join(sentences))) + [BOS, EOS]
stoi = {token: i for i, token in enumerate(tokens)}
itos = {i: token for token, i in stoi.items()}


def build_dataset(texts):
    xs, ys = [], []
    for text in texts:
        sequence = [BOS] + list(text) + [EOS]
        for current, target in zip(sequence, sequence[1:]):
            xs.append(stoi[current])
            ys.append(stoi[target])
    return torch.tensor(xs), torch.tensor(ys)


class TrainableBigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 每个 token 对所有 next token 的可训练分数
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_ids):
        # TODO：根据 token_ids 取出对应行并返回
        ...


torch.manual_seed(42)
x, y = build_dataset(sentences)
model = TrainableBigram(len(tokens))
untrained_model = copy.deepcopy(model)  # 留一份训练前快照做对照
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

with torch.no_grad():
    initial_loss = F.cross_entropy(model(x), y).item()

for step in range(1000):
    logits = model(x)
    loss = F.cross_entropy(logits, y)

    # TODO：清除上一次迭代留下的梯度
    # TODO：根据本次 loss 计算梯度
    # TODO：让优化器修改参数

    if step % 100 == 0:
        print(step, loss.item())

final_loss = F.cross_entropy(model(x), y).item()
print("initial loss:", initial_loss)
print("final loss:", final_loss)
```

### 检查模型学到了什么

```python
with torch.no_grad():
    love_logits = model(torch.tensor([stoi["爱"]]))
    love_probs = F.softmax(love_logits, dim=-1)[0]
    p_cat = love_probs[stoi["猫"]].item()
    p_dog = love_probs[stoi["狗"]].item()

print("P(猫 | 爱):", p_cat)
print("P(狗 | 爱):", p_dog)

assert len(x) == 12
assert model(x).shape == (12, len(tokens))
assert final_loss < initial_loss
assert abs(p_cat - 2 / 3) < 0.05
assert abs(p_dog - 1 / 3) < 0.05
```

最后两个概率不必精确等于 `2/3` 和 `1/3`：训练是逐步逼近，而且模型会给其他 token 留下一点点概率。

### 加入生成

流程和 M1B 的生成一模一样，只是概率的来历从“查计数表再除”变成“查分数表再 softmax”：

```python
def generate(model, max_length=20, seed=42, temperature=1.0):
    generator = torch.Generator().manual_seed(seed)
    current_id = stoi[BOS]
    output = []

    for _ in range(max_length):
        logits = model(torch.tensor([current_id]))[0]
        # TODO：temperature 应该怎样作用在 logits 上？
        probs = F.softmax(..., dim=-1)
        next_id = torch.multinomial(probs, 1, generator=generator).item()

        if next_id == stoi[EOS]:
            break
        output.append(itos[next_id])
        current_id = next_id

    return "".join(output)


for seed in range(10):
    print(seed, generate(model, seed=seed))
```

> **temperature（温度）是生成时的一个旋钮**：把分数先除以它，再做 softmax。小于 1（比如 0.5）会拉大分数差距，输出更保守；大于 1（比如 1.5）会把概率摊平，输出更放飞；1.0 就是原样。这里的 `max_length` 限的是最多生成多少个非 EOS token，验证 `len(generate(..., max_length=5)) <= 5` 即可。

### 对照实验

用 `untrained_model` 和训练后的 `model` 在相同 10 个 seed 下各生成一批：

```python
before = [generate(untrained_model, seed=i) for i in range(10)]
after = [generate(model, seed=i) for i in range(10)]

print("训练前:", before)
print("训练后:", after)
```

预期：训练前接近乱猜；训练后多数句子呈现“我/你 → 爱 → 猫/狗 → 结束”的局部规律。再把 temperature 换成 0.5 和 1.5 各试一遍，感受旋钮的作用。

> **训练前的输出可能看起来像出 bug 了。** 随机初始化的表约等于 7 选 1 乱猜，每个 token 都有约 1/7 的概率，所以“训练前”那批里可能直接吐出 `<BOS>` 字样、或迟迟不结束——这不是你写错了，是它还没学会。对比“训练后”，这种现象应该基本消失。

> 别对生成质量期待过高。它仍然只记得前一个 token。这个实验验证的是**训练机制有效**，不是让 Bigram 突然开窍。

## 6. 收官对照：梯度学到的，就是数出来的

跑完后看三个数字，它们串起了 M1B、M1C 和本节：

| 数字 | 实测值 | 说明 |
|---|---|---|
| 初始 loss | 约 1.95 | 恰好是“7 选 1 均匀瞎猜”的分数：-log(1/7) = log 7 ≈ 1.95。随机初始化 = 什么都不知道 |
| 最终 loss | 约 0.32 | M1C 你手算过：计数模型在训练集上的 loss 是 0.318。两个数字几乎重合 |
| P(猫 \| 爱) | 约 2/3 | 和 M1B 数出来的 2/3 对上了 |

可训练模型从头到尾没数过一次次数。它只是被 loss 推着一步步改表，最后停在了计数表的答案旁边——**同一个答案，两条路。**

> **这是 M1 的最后一块拼图，也是整门课最重要的一条结论：** loss 定了，模型学到的规律就是 loss 逼出来的。以后把这张 7×7 的表换成几亿参数的神经网络，这条结论一个字都不变。

## 7. 三个需要真正想明白的问题

**题 1** 计数模型的表和可训练模型的表形状相同，里面的数字为什么不是同一种东西？

**题 2** 用自己的话串起“查表 → cross_entropy → backward → optimizer.step”。每一步在解决什么？

**题 3** 训练 loss 明显下降后，为什么模型仍然无法根据“我爱”这个完整上下文做判断？

## 8. 提交与验收

1. `m1/trainable_bigram.py`。
2. 初始 loss、最终 loss，以及 `P(猫|爱)`、`P(狗|爱)`。
3. 训练前后各 10 条生成结果，并简单说出差异。
4. 三个问题的答案。

验收只看核心逻辑、输出是否合理，以及你能否解释关键步骤。忘记 API、格式问题或普通样板代码不会卡进度。

---

*M1D · 完成后 M1 结束，进入 M2 Tokenizer*
