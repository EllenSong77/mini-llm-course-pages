# M3A–M3D / M4A 学习包

课程入口：https://ellensong77.github.io/mini-llm-course-pages/

先完成 M2C，再按 M3A、M3B、M3C、M3D、M4A 顺序学习。

包内：

- `lessons/assets/m3_documents.jsonl`：导师编写的 16 条教学数据，供课程实验使用。
- `lessons/*.md`：五节完整 Markdown 教学内容。
- `reference/m3/*.py`、`reference/m4/*.py`：与课件 Python 段落一致的完整参考代码。

推荐：在独立目录解压，然后把教学数据放入课程根目录的 `lessons/assets/`。按课件一步步写自己的 `m3/`、`m4/` 文件；需要对照时打开 reference。不要把参考代码覆盖到已经有作业的文件上。

所有命令都从 `mini-llm-course/` 运行。需要 Python、已有 PyTorch 和 M2C 的 SentencePiece；这些 CPU 小实验不要求升级服务器环境。

M3B 需要你自己的 `artifacts/tokenizers/bilingual_v0.model`。本包不提供替代你的 v0，也不含导师 QA tokenizer。没有完成 M2C 时可以阅读后续课程，但先不要执行 M3B 的模型加载。

实验会生成或重新生成 `artifacts/m3/prepared.json` 和 `encoded.json`，其余参考代码仅打印输出，不启动大规模训练。

导师验证环境：Python 3.13.1、PyTorch 2.10.0、SentencePiece 0.2.2、macOS CPU。服务器记录为 PyTorch 2.6.0+cu124，未在本次验证中连接服务器；实际执行如有兼容问题按安装版本检查。
