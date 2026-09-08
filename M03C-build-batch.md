# M3C：把不同长度的文档摆成一个 batch

前置：M3B 的 `encoded.json`。新建 `m3/build_batch.py`，按顺序追加。先用熟悉的字符 ID 手动核对，再换成自己的 tokenizer 数据。

这一节只处理两个轴：`B` 是同时放进来的文档条数；`T` 是每行安排的预测位置数。`[2,4]` 就是两篇文档，每篇安排四个位置。

## 第 1 步：先摆两条长度不同的序列

写入：

```python
import json
from pathlib import Path
import torch

if __name__ == "__main__":
    toy_tokens = ["你", "我", "爱", "狗", "猫", "<BOS>", "<EOS>", "<PAD>"]
    toy_sequences = [[5, 1, 2, 4, 6], [5, 1, 6]]
    print(toy_sequences)
```

第一条是 `BOS 我 爱 猫 EOS`；第二条是 `BOS 我 EOS`。第二条故意短一点，方便看出补齐行为。这个小词表在 M1D 的七个 token 后加了 PAD，编号为 7。

`if __name__ == "__main__":` 表示这段演示只在直接运行本文件时执行。下一节会导入本文件的函数，不希望那时把演示又跑一遍。这里的缩进需要保留。

## 第 2 步：写一个补齐并错开答案的函数

在文件末尾追加，函数定义从最左边开始：

```python
def make_batch(sequences, pad_id):
    if not sequences or any(len(s) < 2 for s in sequences):
        raise ValueError("至少有一条序列，每条至少包含 BOS 和 EOS")
    max_length = max(len(s) for s in sequences)
    padded = [s + [pad_id] * (max_length - len(s)) for s in sequences]
    full = torch.tensor(padded, dtype=torch.long)
    x = full[:, :-1]
    y = full[:, 1:]
    valid = y != pad_id
    return x, y, valid


if __name__ == "__main__":
    x, y, valid = make_batch(toy_sequences, pad_id=7)
    print("x:", x.tolist())
    print("y:", y.tolist())
    print("valid:", valid.tolist())
    print("shape:", x.shape, y.shape)
```

输出：

```text
x: [[5, 1, 2, 4], [5, 1, 6, 7]]
y: [[1, 2, 4, 6], [1, 6, 7, 7]]
valid: [[True, True, True, True], [True, True, False, False]]
shape: torch.Size([2, 4]) torch.Size([2, 4])
```

按函数顺序看：先将两条完整序列补成长度 5；`full[:, :-1]` 取所有行、去掉最后一列作为输入；`full[:, 1:]` 取所有行、去掉第一列作为答案。完整序列长度 5，因此有 4 个预测位置。

冒号 `:` 表示这一维全部保留。`valid` 也是 `[2,4]`，每个位置说明**这道题是否应该算进 loss**。这是答案有效性标记，不是 Attention 的可见性规则。

## 第 3 步：把 ID 还原成一道道题

追加：

```python
if __name__ == "__main__":
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            current = toy_tokens[x[b, t].item()]
            target = toy_tokens[y[b, t].item()]
            print("文档", b, "位置", t, current, "→", target,
                  "计分" if valid[b, t] else "忽略")
    assert valid.sum().item() == 6
    assert y[0, 3].item() == 6 and valid[0, 3].item()
```

第 0 行有四道题：BOS→我、我→爱、爱→猫、猫→EOS。第 1 行只有两道真实题：BOS→我、我→EOS；余下两道答案是 PAD，不计分。

**EOS 要计分，PAD 不计分。** 结束是模型要学会的行为；补齐只是为了把数据摆成矩形。

第二条短文档的 `x` 里出现 EOS 和 PAD 是正常的，因为这些位置对应的答案都是 PAD，loss 不要求模型学习“结束后要预测占位符”。

## 第 4 步：接入 M3B 的真实编码

追加：

```python
if __name__ == "__main__":
    payload = json.loads(Path("artifacts/m3/encoded.json").read_text(encoding="utf-8"))
    real_pad = payload["special_ids"]["pad"]
    rows = payload["splits"]["train"][:2]
    real_x, real_y, real_valid = make_batch([r["ids"] for r in rows], real_pad)
    print("文档:", [r["doc_id"] for r in rows])
    print("实际 batch shape:", real_x.shape)
    print("有效答案数:", real_valid.sum().item())
    expected = sum(len(r["ids"]) - 1 for r in rows)
    assert real_valid.sum().item() == expected
    for b, row in enumerate(rows):
        length = len(row["ids"]) - 1
        assert real_x[b, :length].tolist() == row["ids"][:-1]
        assert real_y[b, :length].tolist() == row["ids"][1:]
```

运行 `python m3/build_batch.py`。实际 shape 的第一维仍为 2，第二维取决于这两篇文档的编码长度。两个断言逐个核对输入和答案是否对应原序列，能抓住“长度对了、顺序却错了”的问题。

小实验保留整篇短文档，不截断也不拼接文档。正式长文档会切窗口，届时不能把“达到窗口末尾”当成“文档真的结束”而伪造 EOS。

## 第 5 步：确认哪里已经做过了右移

你已经在函数里执行过：

```text
x = full[:, :-1]
y = full[:, 1:]
```

下一节模型看到 `x[b,t]` 后，应该预测 **同一坐标的 `y[b,t]`**。不要再写 `y[:,1:]`，否则会变成预测下下个 token。

## 本节只交这些

- `build_batch.py` 与两组演示输出。
- 指着短文档那一行说清：哪些位置参与 loss，为什么 EOS 要计分而 PAD 不计分。
- 能用 `x[0,2]` 和 `y[0,2]` 说出一组输入/答案即可，不要求背 slicing API。

下一节复用 `make_batch`，让 Bigram 一次给所有位置打分。
