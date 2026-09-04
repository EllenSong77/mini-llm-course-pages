# M0 执行手册：服务器侦察与训练预算

## 本轮目标

只采集信息，不安装、升级、删除或修改服务器环境。

本轮结束时，我们应该能回答：

1. 服务器具有怎样的计算、显存、互联、主存和存储条件？
2. 哪个环节最可能成为训练瓶颈？
3. 教学模型和正式模型应采用什么初步规模？
4. 第一轮训练预算应该是多少？

## 步骤 1：采集服务器信息

在服务器终端执行：

```bash
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi

uname -a
python3 --version
python3 -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())" 2>/dev/null || true

free -h
df -h
lsblk
lscpu | head -n 30
```

注意：

- 可以隐去主机名、用户名、IP 地址和敏感挂载路径。
- 命令失败也是有效信息，请保留错误输出。
- 当前不要安装 PyTorch、CUDA、驱动或其他依赖。
- 当前不要下载训练数据。

## 步骤 2：回答环境问题

复制下面模板并填写：

```text
1. 目标语言：中文 / 英文 / 中英双语
2. 服务器能否访问互联网：
3. 可以连续使用服务器多久：
4. 可用于项目的数据盘空间：
```

## 步骤 3：提交本轮结果

提交内容包括：

- 所有命令的输出；
- 四项环境问题的答案；
- 任何你已经知道的使用限制，例如共享服务器、GPU 调度规则或网络限制。

## 下一轮导师工作

收到结果后，导师将：

- 逐项解释 `nvidia-smi` 和拓扑输出；
- 判断 GPU 是否具有 NVLink/NVSwitch；
- 解释 CPU、主存和磁盘对数据管线的影响；
- 带领完成训练显存的第一次手算；
- 制定教学模型、正式模型和 token budget；
- 把结论写入 `reports/M00-hardware-and-budget.md`；
- 更新 `PROGRESS.md` 中的任务状态。

## 验收清单

- [ ] GPU、拓扑和驱动信息已提交
- [ ] 操作系统、CPU、内存信息已提交
- [ ] Python、PyTorch、CUDA 信息已提交
- [ ] 磁盘与挂载信息已提交
- [ ] 四项环境问题已回答
- [ ] 未提前修改服务器环境

全部完成后，本模块进入“待验收”，但还不会立刻标记为“已完成”。硬件解读和预算报告通过后，M0 才正式完成。
