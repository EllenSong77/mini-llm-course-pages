# Generated from the lesson Markdown; learner files are separate.

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

text = "我爱猫"
body_ids = sp.encode(text, out_type=int)
sequence = [bos] + body_ids + [eos]

print("正文 pieces:", sp.encode(text, out_type=str))
print("正文 IDs:", body_ids)
print("完整序列:", sequence)
print("还原正文:", sp.decode(body_ids))

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
