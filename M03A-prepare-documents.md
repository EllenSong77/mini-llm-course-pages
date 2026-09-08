# M3A：把文档变成干净的训练集和验证集

前置：完成 M2C。本节不需要 GPU。你将把 **16 条原始记录变成 12 篇不重复的文档，再划分为 9 篇训练、3 篇验证**。

新建 `m3/prepare_documents.py`，将下面的 Python 段落按顺序追加；每追加一段就从头运行。完整参考代码在页面顶部。清洗器只处理本节明确的几类问题，不是通用网页清洗器。

## 第 1 步：拿到这批课共用的语料

先在 `mini-llm-course/` 目录下载[教学语料](assets/m3_documents.jsonl)。已有这个文件就不必重新下载。

```bash
mkdir -p lessons/assets m3
curl -fL https://ellensong77.github.io/mini-llm-course-pages/assets/m3_documents.jsonl -o lessons/assets/m3_documents.jsonl
```

这 16 条是导师编写的中英双语教学样本，供本课程实验使用，不含私人材料，不代表真实语料分布。`jsonl` 表示每行各是一条 JSON 记录。写入脚本：

```python
import json
import hashlib
import random
from pathlib import Path

source = Path("lessons/assets/m3_documents.jsonl")
raw = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
print("原始条数:", len(raw))
print(raw[0])
print(raw[12])
```

输出中有：

```text
原始条数: 16
{'doc_id': 'd01', 'kind': 'text', 'text': '我爱猫'}
{'doc_id': 'd13', 'kind': 'text', 'text': '  我爱猫  '}
```

`doc_id` 标识原始记录，`kind` 区分普通文本和代码，`text` 才是要交给 tokenizer 的内容。两个不同 ID 可能装着同一份内容。

## 第 2 步：清洗之前，先看改动会碰到什么

追加：

```python
def clean_text(record):
    text = record["text"].replace("\r\n", "\n").replace("\r", "\n")
    if record["kind"] == "code":
        return text.strip("\n")
    return text.strip()

for record in (raw[12], raw[14], raw[15]):
    print(record["doc_id"], repr(record["text"]), "→", repr(clean_text(record)))
```

`d13` 两端空格消失；`d15` 的 Windows 换行 `\r\n` 变成 `\n`，但 `return` 前的四个空格保留；`d16` 变成空字符串。

`repr()` 把换行和空格显示得更清楚。普通文本去除两端空白，代码只去掉两端换行。**不能不分内容把所有空格压成一个空格**，否则代码缩进可能损坏。我们也没有改变英文大小写、小数点和中英文标点。

## 第 3 步：按内容去重，并保留来源

追加：

```python
unique = {}
for record in raw:
    text = clean_text(record)
    if not text.strip():
        continue
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if fingerprint not in unique:
        unique[fingerprint] = {
            "doc_id": record["doc_id"], "kind": record["kind"],
            "text": text, "fingerprint": fingerprint, "source_ids": [],
        }
    unique[fingerprint]["source_ids"].append(record["doc_id"])

documents = list(unique.values())
print("去空、去重后:", len(documents))
print("第一篇的来源:", documents[0]["source_ids"])
```

输出是 `12`，第一篇来源是 `['d01', 'd13']`。`d02/d14` 和 `d11/d15` 也分别合并了。

这里的 fingerprint 是由内容计算的固定指纹。同样的清洗后文本会得到相同指纹，所以字典只保留一篇。`source_ids` 留下原始记录的来源，方便查清一篇文档是怎么来的。

这只做**清洗后完全相同文本的精确去重**；“我喜欢猫”和“我爱猫”不会被判成重复。近似去重留到规模化数据阶段。

## 第 4 步：去重后再划分

追加：

```python
shuffled = sorted(documents, key=lambda d: d["doc_id"])
random.Random(42).shuffle(shuffled)
train_docs = shuffled[:9]
val_docs = shuffled[9:]

train_fingerprints = {d["fingerprint"] for d in train_docs}
val_fingerprints = {d["fingerprint"] for d in val_docs}
assert train_fingerprints.isdisjoint(val_fingerprints)

print("训练:", len(train_docs), "验证:", len(val_docs))
print("验证文档:", [d["doc_id"] for d in val_docs])
```

输出数量是 `9/3`。固定排序和随机种子，是为了每次对同一份输入得到同样划分；它不让三篇验证文档突然具备统计代表性。

为什么先去重？如果先按原始 ID 划分，`d01` 和 `d13` 可能一个在训练集，一个在验证集，模型就在考试时遇到了同样的内容。此处用内容指纹检查两边没有交集。

## 第 5 步：存下来，让下一节沿用

追加：

```python
out = Path("artifacts/m3")
out.mkdir(parents=True, exist_ok=True)
prepared = {"train": train_docs, "validation": val_docs}
(out / "prepared.json").write_text(
    json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("已保存:", out / "prepared.json")
```

从课程根目录运行：

```bash
python m3/prepare_documents.py
```

会生成 `artifacts/m3/prepared.json`。下一节从这里读数据，不重新随机划分。重复执行会重写本节生成的这个文件，不修改原始语料。

## 本节只交这些

- 脚本，以及 16 → 12 → 9/3 的运行结果。
- 用 `d01/d13` 解释为什么不同文档 ID 不足以保证训练和验证内容分开。
- 看一眼 `d11`：清洗后 `return` 的缩进是否仍在？

做完后进入 M3B，用 M2C 的 tokenizer v0 编码这同一批文档。完成这一课不代表已完成大规模数据清洗。
