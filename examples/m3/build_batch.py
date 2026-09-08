# Generated from the lesson Markdown; learner files are separate.

import json
from pathlib import Path
import torch

if __name__ == "__main__":
    toy_tokens = ["你", "我", "爱", "狗", "猫", "<BOS>", "<EOS>", "<PAD>"]
    toy_sequences = [[5, 1, 2, 4, 6], [5, 1, 6]]
    print(toy_sequences)

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

if __name__ == "__main__":
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            current = toy_tokens[x[b, t].item()]
            target = toy_tokens[y[b, t].item()]
            print("文档", b, "位置", t, current, "→", target,
                  "计分" if valid[b, t] else "忽略")
    assert valid.sum().item() == 6
    assert y[0, 3].item() == 6 and valid[0, 3].item()

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
