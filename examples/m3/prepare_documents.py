# Generated from the lesson Markdown; learner files are separate.

import json
import hashlib
import random
from pathlib import Path

source = Path("lessons/assets/m3_documents.jsonl")
raw = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
print("原始条数:", len(raw))
print(raw[0])
print(raw[12])

def clean_text(record):
    text = record["text"].replace("\r\n", "\n").replace("\r", "\n")
    if record["kind"] == "code":
        return text.strip("\n")
    return text.strip()

for record in (raw[12], raw[14], raw[15]):
    print(record["doc_id"], repr(record["text"]), "→", repr(clean_text(record)))

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

shuffled = sorted(documents, key=lambda d: d["doc_id"])
random.Random(42).shuffle(shuffled)
train_docs = shuffled[:9]
val_docs = shuffled[9:]

train_fingerprints = {d["fingerprint"] for d in train_docs}
val_fingerprints = {d["fingerprint"] for d in val_docs}
assert train_fingerprints.isdisjoint(val_fingerprints)

print("训练:", len(train_docs), "验证:", len(val_docs))
print("验证文档:", [d["doc_id"] for d in val_docs])

out = Path("artifacts/m3")
out.mkdir(parents=True, exist_ok=True)
prepared = {"train": train_docs, "validation": val_docs}
(out / "prepared.json").write_text(
    json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("已保存:", out / "prepared.json")
