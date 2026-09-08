# M3B：一篇文档怎样变成带边界的 token 序列

前置：M3A 的 `artifacts/m3/prepared.json`，以及 M2C 选好的 tokenizer v0。新建 `m3/encode_documents.py`，按顺序追加代码。

本节把“文档”转换成“整数列表”，并记录这些整数属于哪个 tokenizer。具体 ID 和 token 数会随你的 tokenizer 不同而变化，不能照抄别人的数字。

## 第 1 步：加载你自己训练的 tokenizer

写入：

```python
import json
import sys
import hashlib
from pathlib import Path
import sentencepiece as spm

model_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "artifacts/tokenizers/bilingual_v0.model"
)
sp = spm.SentencePieceProcessor(model_file=str(model_file))
bos, eos, pad, unk = sp.bos_id(), sp.eos_id(), sp.pad_id(), sp.unk_id()
V = sp.get_piece_size()
assert len({bos, eos, pad, unk}) == 4
assert all(0 <= i < V for i in (bos, eos, pad, unk))
print("实际词表大小:", V)
print("BOS/EOS/PAD/UNK:", bos, eos, pad, unk)
```

从课程根目录运行：

```bash
python m3/encode_documents.py artifacts/tokenizers/bilingual_v0.model
```

如果 M2C 中使用了别的文件名，把命令最后的路径换成那个 `.model`。不要在找不到文件时临时换一个未知 tokenizer。

如果沿用 M2C 的特殊 ID 配置，会看到 `1 2 3 0`。这与 M1D 的手工词表不同，我们一直从 tokenizer 读取 ID。`get_piece_size()` 读取实际词表容量，未必等于训练时请求值。

## 第 2 步：先看一条熟悉的“我爱猫”

追加：

```python
text = "我爱猫"
body_ids = sp.encode(text, out_type=int)
sequence = [bos] + body_ids + [eos]

print("正文 pieces:", sp.encode(text, out_type=str))
print("正文 IDs:", body_ids)
print("完整序列:", sequence)
print("还原正文:", sp.decode(body_ids))
```

正文可能被切成单字，也可能有多字 token。我们使用 `encode` 的默认行为得到正文，然后**明确添加一次 BOS 和 EOS**，不再使用 `add_bos/add_eos`，避免重复。

如果正文是 `[a,b,c]`，完整序列就是 `[BOS,a,b,c,EOS]`。这里 a/b/c 只是示意，不是固定 ID。

三个特殊 token 的用途：BOS 让模型预测正文的开头；EOS 是模型应该学会预测的结束标记；PAD 是未来把短文档补齐时使用的占位符，目前还没加入。

## 第 3 步：编码全部文档，并打印还原结果

追加：

```python
prepared = json.loads(Path("artifacts/m3/prepared.json").read_text(encoding="utf-8"))
encoded = {}
for split, docs in prepared.items():
    rows = []
    for doc in docs:
        ids = sp.encode(doc["text"], out_type=int)
        assert not any(i in (bos, eos, pad) for i in ids)
        full_ids = [bos] + ids + [eos]
        assert all(0 <= i < V for i in full_ids)
        rows.append({"doc_id": doc["doc_id"], "ids": full_ids})
        print(split, doc["doc_id"], "正文 token 数:", len(ids),
              "UNK 数:", ids.count(unk))
        print("原文:", repr(doc["text"]))
        print("解码:", repr(sp.decode(ids)))
    encoded[split] = rows
```

两边分别编码，不合并训练和验证列表。这里没有再训练 tokenizer，所以也没有用验证集去修改它。

仔细看两种差异：

- 出现 UNK：词表无法完整表示某些字符，不同原字符可能丢成同一个 ID。
- 空格、缩进或全角字符变化：可能来自 tokenizer 的规范化规则；默认规范化不保证逐字节还原代码。

**不要把解码不同一律判成 encode 写错，也不要把代码缩进损失当成无关紧要。** 本节记录问题；正式 tokenizer 在真实训练划分上重新选择规范化、覆盖与 fallback 策略。

这个教学语料是在 M2C 后引入的；小规模 tokenizer 测试接近于外部数据检查。正式评测时还要确认 tokenizer 的训练语料没有包含评测文档。

## 第 4 步：把词表身份与数据一起保存

追加：

```python
payload = {
    "tokenizer_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
    "vocab_size": V,
    "special_ids": {"bos": bos, "eos": eos, "pad": pad, "unk": unk},
    "splits": encoded,
}
target = Path("artifacts/m3/encoded.json")
target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("已保存:", target)
print("tokenizer 指纹:", payload["tokenizer_sha256"][:12])
```

`tokenizer_sha256` 是 `.model` 文件内容的指纹。文件名可以改，但内容变了指纹就会改变。后续 token 数据和模型需要绑定这份词表；不能只看文件名都叫 `v0` 就认为兼容。

换 tokenizer 时，从这一节重新编码，重建后面的 batch，并重新训练关联模型；这份小实验文件可以重建，正式工程要保留版本目录。

## 第 5 步：确认 EOS 到底做了什么

不加代码，看当前保存的数据：每篇文档是一个独立整数列表，列表尾部是 EOS。

如果以后把 `[文档A, EOS, 文档B]` 拼成同一行，**EOS 本身不会阻止文档 B 的 Attention 读取文档 A**。EOS 是结束标记；哪些位置能读取哪些位置，由以后学习的 attention mask 决定。

下一课先保持一行一篇文档，便于看清数据，不在这里引入复杂拼接策略。

## 本节只交这些

- `encode_documents.py`，训练/验证两边的编码条数，tokenizer 指纹前 12 位。
- 一条中文、一条英文和一条代码的原文/pieces/IDs/解码对照；第 3 步已有 IDs 存盘，如需 pieces 可复用第 2 步。
- 记录实际遇到的 UNK 或规范化变化；没有遇到就说明“这些样本未观察到”，不强行制造错误。
- 回答一句：EOS 能否自动隔离跨文档 Attention？

下一节读取 `encoded.json`，把这些不同长度的列表组织成 `[B,T]` 的 batch。
