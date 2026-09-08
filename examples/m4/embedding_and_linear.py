# Generated from the lesson Markdown; learner files are separate.

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

embedding = nn.Embedding(V, C)
with torch.no_grad():
    embedding.weight.copy_(torch.arange(V * C).reshape(V, C) / 10)

h = embedding(x)
print("embedding 表:", embedding.weight.detach())
print("输出形状:", h.shape)
print("爱对应的三个数:", h[0, 2].detach())

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

logits = head(h)
manual_logits = h @ head.weight.T
torch.testing.assert_close(logits, manual_logits)
torch.testing.assert_close(logits[0, 2, 4], manual_cat_score)
print("输出 logits:", logits.shape)
print("爱位置给猫的分数:", logits[0, 2, 4].item())

numbers = torch.arange(6).reshape(2, 3)
transposed = numbers.transpose(0, 1)
reshaped = numbers.reshape(3, 2)
print("原表:", numbers.tolist())
print("交换行列:", transposed.tolist())
print("重新分组:", reshaped.tolist())
assert transposed.shape == reshaped.shape
assert not torch.equal(transposed, reshaped)

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
