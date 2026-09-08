# M3D：模型怎样给整个 batch 评分

前置：M3C 的 `make_batch()` 和 M3B 的 `encoded.json`。新建 `m3/batch_bigram.py`，按顺序追加，运行 `python m3/batch_bigram.py`。

继续用 M1D 的查表模型，只改变输入的摆法：从 `[12]` 变成 `[B,T]`。模型输出会从 `[12,V]` 变成 `[B,T,V]`。字母依次代表文档条数、预测位置数、候选 token 数。

## 第 1 步：拿到训练和验证 batch

写入：

```python
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from build_batch import make_batch

payload = json.loads(Path("artifacts/m3/encoded.json").read_text(encoding="utf-8"))
V = payload["vocab_size"]
pad = payload["special_ids"]["pad"]
train_rows = payload["splits"]["train"]
val_rows = payload["splits"]["validation"]
x, y, valid = make_batch([r["ids"] for r in train_rows], pad)
val_x, val_y, val_valid = make_batch([r["ids"] for r in val_rows], pad)
print("训练 x/y:", x.shape, y.shape)
print("验证 x/y:", val_x.shape, val_y.shape)
```

训练第一维是 9，验证第一维是 3。两组的第二维可以不同，因为它们分别按各自最长文档补齐。

本课是教学 Bigram，参数量是 `V×V`。若你的 v0 词表特别大，这个表本身会占较多内存；先计算 `V*V*4` 字节的 FP32 参数大小。正式模型会在下一节把查特征和输出评分拆开，不会一直沿用这张平方大小的表。

## 第 2 步：每个输入位置都查出 V 个分数

追加：

```python
table = nn.Embedding(V, V)
with torch.no_grad():
    table.weight.zero_()
logits = table(x)
print("logits shape:", logits.shape)
print("一条文档所有位置:", logits[0].shape)
print("第 0 篇第 0 个位置:", logits[0, 0].shape)
assert logits.shape == (*x.shape, V)
```

如果 `x` 是 `[9,25]`，结果就是 `[9,25,V]`；25 只是举例，实际值取决于 tokenizer。

`logits[0,0,:]` 是一组候选分数；`logits[0,:,3]` 则是同一篇文档不同位置给“候选 ID 3”的分数。这两个切片含义完全不同，不能因为都是一排数字就混用。

对候选做 softmax 应沿最后一维。`dim=-1` 每次比较 V 个候选，共比较 B×T 组，输出仍为 `[B,T,V]`。

## 第 3 步：把每个位置当成一道分类题

追加：

```python
flat_logits = logits.reshape(-1, V)
flat_y = y.reshape(-1)
print("展平后:", flat_logits.shape, flat_y.shape)
print("第 0 篇第 2 个位置的答案:", y[0, 2].item(), flat_y[2].item())

def loss_stats(scores, targets, pad_id):
    total = F.cross_entropy(
        scores.reshape(-1, scores.shape[-1]), targets.reshape(-1),
        ignore_index=pad_id, reduction="sum",
    )
    count = (targets != pad_id).sum()
    if count.item() == 0:
        raise ValueError("这一批没有有效答案")
    return total, count

total, count = loss_stats(logits, y, pad)
loss = total / count
initial_loss = loss.item()
print("有效答案:", count.item(), "平均 loss:", initial_loss)
assert abs(initial_loss - math.log(V)) < 1e-5
```

`reshape(-1,V)` 保留每道题的 V 个分数，把前面 B×T 个位置排成一列题目；`-1` 让 PyTorch 自动计算题目数量。`y.reshape(-1)` 按相同顺序排列答案。

`ignore_index=pad` 只忽略**正确答案是 PAD 的位置**，不会从词表候选里删除 PAD 那一列。本例初始时每个候选分数都为零，所以真实答案概率是 `1/V`，平均 loss 为 `log(V)`。

`reduction="sum"` 先把每道有效题的损失加起来，再除以有效答案个数。没有把补齐位置计入分母。`cross_entropy` 接收原始 logits，不先做 softmax。

## 第 4 步：只训练训练集

追加：

```python
optimizer = torch.optim.SGD(table.parameters(), lr=1.0)
for step in range(200):
    scores = table(x)
    total, count = loss_stats(scores, y, pad)
    loss = total / count
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    train_sum, train_n = loss_stats(table(x), y, pad)
    val_sum, val_n = loss_stats(table(val_x), val_y, pad)
    train_mean = (train_sum / train_n).item()
    val_mean = (val_sum / val_n).item()
print("初始训练 loss:", initial_loss)
print("最终训练 loss:", train_mean)
print("验证 loss:", val_mean)
assert train_mean < initial_loss
```

循环仍是你已经会的训练六行。注意 `val_x/val_y` 没有出现在反向传播循环中。

训练 loss 应下降，但数值取决于你的 v0，不要求收敛到 M1D 的 0.318。这里真实文档更多、词表不同，不能比较成同一个考试成绩。也不把“验证 loss 必须高于训练 loss”写成普遍断言。

## 第 5 步：多补几个 PAD，成绩应该不变

追加：

```python
with torch.no_grad():
    more_x = F.pad(x, (0, 3), value=pad)
    more_y = F.pad(y, (0, 3), value=pad)
    extra_sum, extra_n = loss_stats(table(more_x), more_y, pad)
    extra_mean = extra_sum / extra_n
    torch.testing.assert_close(extra_mean, train_sum / train_n)
    assert extra_n.item() == train_n.item()
print("多补三个 PAD 后 loss:", extra_mean.item())
```

`(0,3)` 表示最后一维左边不补，右边补三格。相同文档只是矩形变宽了，真实题目没变，因此 loss 应在浮点容差内相同。这个实验可以抓住错误的分母或忘记忽略 PAD 的问题。

## 第 6 步：验证集分批计算时怎样合并成绩

追加：

```python
sum_of_losses, number_of_targets = 0.0, 0
with torch.no_grad():
    for group in (val_rows[:1], val_rows[1:]):
        group_x, group_y, _ = make_batch([r["ids"] for r in group], pad)
        group_sum, group_n = loss_stats(table(group_x), group_y, pad)
        sum_of_losses += group_sum.item()
        number_of_targets += group_n.item()
split_val_mean = sum_of_losses / number_of_targets
assert math.isclose(split_val_mean, val_mean, rel_tol=1e-5, abs_tol=1e-6)
print("分两批算验证 loss:", split_val_mean)
```

不是把两个 batch 的平均 loss 再简单平均。若一批有 10 道有效题，另一批有 100 道题，简单平均会让第一批每道题得到十倍权重。这里总损失相加、总题数相加，结果应与整批验证一致。

## 本节只交这些

- 训练/验证 `x/y/logits` 的 shape，初始与最终 loss。
- 两项检查通过：多补 PAD 不改成绩，验证拆批不改成绩。
- 用一个 `b,t` 坐标说明 logits 的一组分数如何与 y 的一个答案对应。

到这里你已经有小数据训练入口。M3 的规模化部分以后回访；下一课先把模型从直接查分数，改成“查一小组特征，再算分数”。
