# Generated from the lesson Markdown; learner files are separate.

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

table = nn.Embedding(V, V)
with torch.no_grad():
    table.weight.zero_()
logits = table(x)
print("logits shape:", logits.shape)
print("一条文档所有位置:", logits[0].shape)
print("第 0 篇第 0 个位置:", logits[0, 0].shape)
assert logits.shape == (*x.shape, V)

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

with torch.no_grad():
    more_x = F.pad(x, (0, 3), value=pad)
    more_y = F.pad(y, (0, 3), value=pad)
    extra_sum, extra_n = loss_stats(table(more_x), more_y, pad)
    extra_mean = extra_sum / extra_n
    torch.testing.assert_close(extra_mean, train_sum / train_n)
    assert extra_n.item() == train_n.item()
print("多补三个 PAD 后 loss:", extra_mean.item())

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
