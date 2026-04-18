# Benchmarks

**简体中文** | [English](./benchmarks.en.md)

真实音频上的端到端耗时与资源占用。所有测量都是从外部 HTTP 客户端打到
`POST /api/transcribe` + 轮询 `/api/jobs/{id}` 的壁钟时间，配合容器和宿主机
侧的资源采样。

原始 5 秒一次的采样数据放在 [`benchmarks/`](./benchmarks/) 目录里的 CSV，
时间戳是"从提交到服务那一刻起"的相对秒数，列：
`t_offset_s, phase, cpu_pct, ram_gib, gpu_mem_mb, gpu_util_pct`。

---

## 1 小时中文会议（voscript 0.3.0，warm）

**输入**：57.6 分钟中文会议录音，mp3，40 kbps，mono，16 kHz，总 23.5 MB。

**环境**：
- 单张 24 GB 消费级 NVIDIA GPU
- voscript 0.3.0（WhisperX + faster-whisper large-v3 + pyannote 3.1 + WeSpeaker ResNet34 + sqlite-vec）
- 模型权重已在磁盘 warm-cache（**不**是冷启动数据）
- `WHISPER_MODEL=large-v3`，`DEVICE=cuda`

原始采样：[`benchmarks/1h-zh-meeting.csv`](./benchmarks/1h-zh-meeting.csv)（184 行）。

### 各阶段壁钟

| 阶段 | 相对时刻 | 耗时 |
| --- | --- | --- |
| 客户端 multipart 上传（23 MB） | `t = 0s → t = 1s` | **1 s** |
| `queued → transcribing`（容器调度） | `t = 1s → t = 22s` | 21 s |
| **Transcribe**（faster-whisper large-v3 + VAD） | `t = 22s → t = 14m 7s` | **14 m 45 s** |
| **Align + Embed + Identify**（wav2vec2 + WeSpeaker + 声纹识别） | `t = 14m 7s → t = 15m 42s` | **35 s** |
| **总壁钟** | `t = 0s → t = 15m 42s` | **15 m 42 s** |

**实时倍率 RTF ≈ 3.7**（57.6 min 音频 / 15.7 min 壁钟）。

> 冷启动（镜像刚拉完、alignment 模型未下载）首次会多出约 12–14 分钟下载
> 中文 wav2vec2 对齐模型（~1 GB）。第二次起缓存命中，和这里看到的 warm
> 数据基本相同。

### 产出

| 指标 | 值 |
| --- | --- |
| segments | **1 226** |
| 含词级时间戳（`words[]`）的 segments | **1 220 / 1 226**（99.5%） |
| 总词级时间戳条目 | **13 149** |
| 识别出的说话人 | **8**（外加 21 段 `UNKNOWN` 兜底，占比 1.7%） |
| 覆盖时长 | 11.4 s → 3 458.7 s（≈ 57.5 分钟，几乎满覆盖） |
| 主讲人段数占比 | 721 / 1 226 ≈ **59%**（一人主述，7 人穿插回应） |

### 资源占用（5 s 一采样，184 个样本）

| 阶段 | n | CPU 平均 | CPU 峰 | RAM | GPU 显存 | GPU util 平均 | GPU util 峰 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| idle（warm） | 10 | 72% | 287% | 1.92 GiB | 6.5 GiB | 19% | 98% |
| **transcribe** · 14 m 45 s | 130 | 121% | 586% | 2.08 GiB | 7.3 GiB | **21%** | 100% |
| **align + embed** · 35 s | 6 | 64% | 100% | 2.76 GiB | 8.1 GiB | **40%** | 63% |
| idle（完成后） | 38 | 0% | 0% | 2.77 GiB | 8.5 GiB | 0.1% | 1% |

### 可以看出什么

1. **瓶颈是 CPU + GIL，不是 GPU**。转录阶段 GPU 利用率平均只有 21%，峰值
   能打到 100% 但是很短暂。faster-whisper 的 VAD 预处理和 tokenizer 都在
   CPU 上，Python GIL 再加一层限制。想进一步压 RTF：
   - 调大 `batch_size`（目前 WhisperX 默认 16）
   - 给 faster-whisper 加多 CPU worker
   - 启用 flash attention
2. **大头在 transcribe，对齐 + 声纹几乎免费**。1 小时音频 whisper 跑 14 m 45 s，
   后面 wav2vec2 + WeSpeaker + 声纹 identify 合计只花 35 s。
3. **显存富余**。峰值 8.5 GiB / 24 GiB，同卡同时跑两个任务不会挤爆。
4. **完成后 GPU util 立即归零**，没有常驻推理或 memory leak。稳态 RAM
   2.77 GiB，不漂移。
5. **词级对齐成功率 99.5%**。剩下 0.5% 多半是纯数字或极短语气词，中文
   wav2vec2 对齐模型对这类 token 没可靠字符映射，失败时不 crash、直接跳过
   `words[]` 字段。

### 一个已知的阈值问题

这次音频里，有一位说话人在声纹库里已登记（1 个 sample）。引擎**正确**把
主讲人匹配到这条声纹（8 个候选里相似度最高）——但最高相似度是
**0.7472**，差 0.0028 过不了默认 0.75 阈值，被判成"未识别"而回退到
`SPEAKER_XX` 原始标签。

根因：
- 登记时样本数 = 1，averaged embedding 信息量不足
- 0.75 是 ECAPA-TDNN 时代沿用的阈值，WeSpeaker ResNet34 的 cosine 分布
  稍微收紧，对单样本的跨会议匹配偏严

缓解（0.3.1 起）：
- `VOICEPRINT_THRESHOLD` 作为环境变量可配（默认仍是 `0.75`）
- 建议：对同一人在新会议里多做几次 `update_speaker`，让 averaged
  embedding 收敛；两次以上采样通常能稳定跨 0.80
- 中期计划：**每说话人自适应阈值**（按该人已积累样本的 cosine 方差动态
  放松）
