# M4A：先查一小组特征，再算候选分数

前置：M3D。新建 `m4/embedding_and_linear.py`，按顺序追加代码。本节先使用固定小数字，随后接入自己的 tokenizer 数据；不需要提前学完整门线性代数。

上一节一次查表就得到 V 个分数。这一节改成两段：每个 token 先查出 C 个数，再把这 C 个数转换成 V 个候选分数。它们分别是 embedding 和线性层；先运行，再理解名字。

## 第 1 步：沿用 M3C 的输入

写入：

```python
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
tokens = ["你", "我", "爱", "狗", "猫", "<BOS>", "<EOS>", "<PAD>"]
V, C = len(tokens), 3
x = torch.tensor([[5, 1, 2, 4], [5, 1, 6, 7]], dtype=torch.long)
y = torch.tensor([[1, 2, 4, 6], [1, 6, 7, 7]], dtype=torch.long)
print("输入形状:", x.shape)
print("第 0 篇第 2 个输入:", tokens[x[0, 2].item()])
```

输出 `[2,4]` 和“爱”。`V=8` 表示词表里有八种 token；`C=3` 是我们指定“每个 token 用三个数来表示”。C 不是从句子长度算出来的，也不是三类答案。

本节 C 相当于以后模型中的隐藏宽度 `d_model`。现在取 3 只是为了看清运算，不是正式模型配置。

## 第 2 步：每个 token 查出三个数

追加：

```python
embedding = nn.Embedding(V, C)
with torch.no_grad():
    embedding.weight.copy_(torch.arange(V * C).reshape(V, C) / 10)

h = embedding(x)
print("embedding 表:", embedding.weight.detach())
print("输出形状:", h.shape)
print("爱对应的三个数:", h[0, 2].detach())
```

表有 `[8,3]` 个数，输出是 `[2,4,3]`，“爱”对应 `[0.6,0.7,0.8]`。

按坐标读：`h[0,2,:]` 是第 0 篇文档、第 2 个位置的全部三个特征。B 和 T 都没有变，每个输入 ID 被替换成一排 C 个数。

这些数字是我们为了教学手动填的，所以不能说 0.6 代表某种语义。真实训练会修改它们，但单个坐标通常也没有预先规定的“猫性”“语法”等人类含义。

M1D 的 `Embedding(V,V)` 直接返回候选分数；现在的 `Embedding(V,C)` 返回中间特征。**相同的查表工具，可以在模型中承担不同用途。**

## 第 3 步：先看一次乘法怎样得到“猫”的分数

追加：

```python
head = nn.Linear(C, V, bias=False)
with torch.no_grad():
    head.weight.copy_(torch.arange(V * C).reshape(V, C) / 10)

features = h[0, 2]
cat_weights = head.weight[4]
products = features * cat_weights
manual_cat_score = products.sum()
print("输入特征:", features.detach())
print("猫这一行权重:", cat_weights.detach())
print("逐项相乘:", products.detach())
print("猫的分数:", manual_cat_score.item())
```

对应的算式是：

```text
输入特征      [0.6, 0.7, 0.8]
猫的权重      [1.2, 1.3, 1.4]
逐项乘再相加   0.6×1.2 + 0.7×1.3 + 0.8×1.4 = 2.75
```

这叫点积。三个特征经过三次乘法和一次求和，得到一个候选的分数。

`head.weight` 形状是 `[V,C]`：每个候选有自己的一组 C 个权重。`nn.Linear(C,V,bias=False)` 让每组 C 个输入产生 V 个输出；`bias=False` 只表示暂时不额外加一个偏置数。

注意：embedding 和 head 的数值虽然都用 arange 填充，它们仍是两份不同参数，未共享存储。

## 第 4 步：矩阵乘法就是把刚才的计算做很多遍

追加：

```python
logits = head(h)
manual_logits = h @ head.weight.T
torch.testing.assert_close(logits, manual_logits)
torch.testing.assert_close(logits[0, 2, 4], manual_cat_score)
print("输出 logits:", logits.shape)
print("爱位置给猫的分数:", logits[0, 2, 4].item())
```

输出是 `[2,4,8]` 和 `2.75`。`head(h)` 对每条文档、每个位置，都执行第 3 步的计算，为八种候选各打一个分。

矩阵乘法的形状在这里是：

```text
h                     [2, 4, 3]
head.weight           [8, 3]
head.weight.T         [3, 8]
h @ head.weight.T     [2, 4, 8]
```

最后的 3 个特征分别与权重相乘并求和，得到 8 个输出；前面的 `[2,4]` 表示要对多少组特征重复同一种运算。

`.T` 将这张二维权重表的行列交换，让每一列对应一个候选的权重。对于以后多维张量，不要随意套 `.T`；先明确需要交换的两个轴。

## 第 5 步：transpose 和 reshape 为什么不能混用

追加：

```python
numbers = torch.arange(6).reshape(2, 3)
transposed = numbers.transpose(0, 1)
reshaped = numbers.reshape(3, 2)
print("原表:", numbers.tolist())
print("交换行列:", transposed.tolist())
print("重新分组:", reshaped.tolist())
assert transposed.shape == reshaped.shape
assert not torch.equal(transposed, reshaped)
```

输出：

```text
原表: [[0, 1, 2], [3, 4, 5]]
交换行列: [[0, 3], [1, 4], [2, 5]]
重新分组: [[0, 1], [2, 3], [4, 5]]
```

`transpose(0,1)` 把原来的列变成行，例如第一列 0、3 变成第一行。`reshape(3,2)` 在这个例子中按原来的数字顺序，改成每两个一组。两者 shape 都是 `[3,2]`，但数字对应关系不同。

以后拆多头时，“维度乘积没变”和“shape 对了”都不足以证明代码正确；要核对位置对应的数据有没有被混起来。

## 第 6 步：一行 loss 同时训练两张表

追加：

```python
optimizer = torch.optim.SGD(
    list(embedding.parameters()) + list(head.parameters()), lr=0.01
)
loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=7)
optimizer.zero_grad()
loss.backward()
print("embedding 梯度总量:", embedding.weight.grad.abs().sum().item())
print("head 梯度总量:", head.weight.grad.abs().sum().item())
assert embedding.weight.grad.abs().sum().item() > 0
assert head.weight.grad.abs().sum().item() > 0
optimizer.step()
```

两个梯度总量都大于零。loss 来自最终候选分数，但反向传播会沿计算过程同时影响 head 和 embedding。

此处一步更新只是验证“两个部分都参与训练”，不声称这份手动初始化已经能生成好文本。`0.01` 是演示步幅，也不是之后所有模型的默认学习率。

模型仍然逐位置独立计算：只要两个位置的 token ID 相同，它们就得到同样的特征和分数。**加入 embedding 与线性层，还没有让它读取其他位置。** 下一课 Attention 才开始让不同位置交换信息。

## 第 7 步：用你的真实 tokenizer 检查同一条路径

追加：

```python
payload = json.loads(Path("artifacts/m3/encoded.json").read_text(encoding="utf-8"))
real_V = payload["vocab_size"]
first = payload["splits"]["train"][0]["ids"]
real_x = torch.tensor([first], dtype=torch.long)
real_embedding = nn.Embedding(real_V, 4)
real_head = nn.Linear(4, real_V, bias=False)
real_h = real_embedding(real_x)
real_logits = real_head(real_h)
print("真实 IDs:", real_x.shape)
print("真实特征:", real_h.shape)
print("真实 logits:", real_logits.shape)
assert real_logits.shape == (1, len(first), real_V)
```

这里保留一篇完整序列，仅检查前向形状，不构造训练目标，因此没有在这段再次右移标签。

你会看到 `[1,T] → [1,T,4] → [1,T,真实词表大小]`。T 取决于你的 tokenizer；C=4 是我们为了小实验指定的。ID 数值、序列长度和特征宽度是不同概念。

## 本节只交这些

- 用三个数的乘法解释为什么“猫”的分数是 2.75。
- 展示 transpose 与 reshape 同形状、不同数值对应关系的结果。
- 展示真实 tokenizer 的三组 shape，以及两张表都有梯度的输出。
- 回答一句：这个模型现在会利用前面的其他 token 吗？从哪行代码看出来？

这批五节的学习结果是：你能顺着“原文 → ID → batch → 特征 → logits → loss”逐行找到对应数据。下一课 M4B 将从一个短句的一次加权求和讲单头 Attention。
