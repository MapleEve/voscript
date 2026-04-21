# VoScript 🎙️

**简体中文** | [English](./README.en.md)

<p align="center">
  <img src="https://img.shields.io/github/actions/workflow/status/MapleEve/voscript/ci.yml?branch=main&style=for-the-badge" alt="CI">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Docker-ready-blue?style=for-the-badge&logo=docker" alt="Docker">
</p>

**自托管的 GPU 转录服务——带说话人记忆的会议逐字稿。**

```
音频  ──►  faster-whisper large-v3              （转录）
      ──►  pyannote 3.1                         （说话人分离）
      ──►  DeepFilterNet / noisereduce（可选降噪）
      ──►  WeSpeaker ResNet34                   （声纹提取）
      ──►  VoiceprintDB                         （与已注册声纹做余弦匹配）
      ──►  带时间戳和已识别说话人姓名的文本
```

[快速上手](#30-秒上手) · [API 参考](./doc/api.zh.md) · [安全策略](./doc/security.zh.md) · [Benchmarks](./doc/benchmarks.zh.md) · [更新日志](./doc/changelog.zh.md)

与"纯 whisper 包装"的区别：**持久化声纹库**。登记过一次，之后所有录音里这个
人都会被自动贴上真名，不需要每次人工贴标签。

> 用例参考：[OpenPlaud(Maple)](https://github.com/MapleEve/openplaud) 把本服务作为
> 会议录音的后端——把这个 repo 当作标准 HTTP 服务对接即可，不限特定客户端。

## 30 秒上手

> **安全警告**：未设置 `API_KEY` 时，服务接受**所有请求**（包括 DELETE 声纹库、上传文件触发 GPU 任务等高权限操作）。生产环境或公网暴露前**必须**设置 `API_KEY`，或将服务置于鉴权反代之后。本地开发可设置 `ALLOW_NO_AUTH=1` 明确确认无鉴权模式。

```bash
git clone https://github.com/MapleEve/voscript.git
cd voscript

cp .env.example .env
# 编辑 .env —— 至少要填 HF_TOKEN 和 API_KEY

docker compose up -d --build
curl -sf http://localhost:8780/healthz
```

完整步骤 + 排障清单看 [`doc/quickstart.zh.md`](./doc/quickstart.zh.md)。

## 功能亮点

- **异步任务流水线**：`queued → converting → denoising（可选）→ transcribing → identifying → completed`
- **中文 + 多语种转录**：WhisperX + faster-whisper large-v3，带词级时间戳的 forced alignment；`language` 省略时自动检测语言，普通话音频输出简体中文
- **说话人分离**：pyannote 3.1 + WeSpeaker ResNet34 声纹提取
- **自适应声纹阈值**：`VOICEPRINT_THRESHOLD`（默认 0.75）作为基准，实际阈值按每位说话人已注册向量的簇内标准差动态宽松：1 条样本固定宽松 −0.05，2 条及以上按 `min(3×std, 0.10)` 宽松，最低不低于 0.60。在 10 条真实录音上召回率从 50% 提升至 70%，零误识别
- **可选降噪 + SNR 门控**：`DENOISE_MODEL`（`none` | `deepfilternet` | `noisereduce`），`DENOISE_SNR_THRESHOLD`（默认 10.0 dB）——高于此 SNR 的录音视为干净音频自动跳过，防止 DeepFilterNet 劣化已清晰的录音
- **AS-norm 声纹评分**：启动时自动从已有转录的声纹 embedding 构建 impostor cohort，用自适应分数归一化（AS-norm）替代原始余弦，消除说话人依赖的基准偏差，相对 EER 降低 15–30%
- **持久化声纹**：一次登记，后续录音自动识别。底层 sqlite + sqlite-vec，top-k 近邻搜索 O(log N)，上千个声纹毫无压力
- **任务状态持久化**：重启后已完成任务仍可通过 `GET /api/transcriptions/{id}` 访问，进程内任务状态写入 `status.json`
- **结构化 API**：`app/` 按 `config / api/routers / services / pipeline` 分层，main.py 仅 ~160 行作为编排入口
- **路径遍历防护**：`_safe_tr_dir()` + 正则校验 job_id，杜绝目录穿越攻击
- **文件哈希去重**：相同文件重复提交时直接返回已有结果，不再重跑 Whisper GPU 推理
- **稳定的 HTTP 合同**：`/api/transcribe`、`/api/jobs/{id}`、`/api/voiceprints*` 等，任何 HTTP 客户端都能接入
- **容器以非 root 用户运行**；所有 `/api/*` 路由支持可选 Bearer / `X-API-Key` 鉴权（常量时间对比）；上传有 `MAX_UPLOAD_BYTES` 上限；声纹库并发安全、原子写入——完整硬化清单见 [`doc/security.zh.md`](./doc/security.zh.md)
- `/` 自带一个轻量 Web UI，方便单独测试

## 怎么接入

voscript 就是个普通的 HTTP 服务，没有特定客户端的强依赖。任何能发
`multipart/form-data` 的东西都能用（curl、axios、requests、网页上传框……）。

一个典型对接示例——OpenPlaud(Maple) 的"设置 → 转录"里配：

- **Private transcription base URL**：`http://<主机>:8780`
- **Private transcription API key**：跟 `.env` 里的 `API_KEY` 一致

之后它的 worker 会自动把每条录音都丢给这个服务。想自己写客户端的话，看
[`doc/api.zh.md`](./doc/api.zh.md) 的完整合同 + 错误码表。

## 文档

所有详细文档都在 [`doc/`](./doc/)，默认中文，每一份都有对应英文：

| 主题 | 中文 | English |
| --- | --- | --- |
| 快速安装 | [quickstart.zh.md](./doc/quickstart.zh.md) | [quickstart.en.md](./doc/quickstart.en.md) |
| API 参考 | [api.zh.md](./doc/api.zh.md) | [api.en.md](./doc/api.en.md) |
| **给 AI 的安装部署指南** | [ai-install.zh.md](./doc/ai-install.zh.md) | [ai-install.en.md](./doc/ai-install.en.md) |
| **给 AI 的接口使用指南** | [ai-usage.zh.md](./doc/ai-usage.zh.md) | [ai-usage.en.md](./doc/ai-usage.en.md) |
| 安全策略 | [security.zh.md](./doc/security.zh.md) | [security.en.md](./doc/security.en.md) |
| Benchmarks（真实音频耗时 + 资源占用） | [benchmarks.zh.md](./doc/benchmarks.zh.md) | [benchmarks.en.md](./doc/benchmarks.en.md) |
| 更新日志 | [changelog.zh.md](./doc/changelog.zh.md) | [changelog.en.md](./doc/changelog.en.md) |

人第一次部署 → [快速安装](./doc/quickstart.zh.md)；
AI agent 帮用户部署 → [给 AI 的安装部署指南](./doc/ai-install.zh.md)；
AI agent 调用接口 → [给 AI 的接口使用指南](./doc/ai-usage.zh.md)。

## 贡献

欢迎 PR！请先读 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MapleEve/voscript&type=date)](https://www.star-history.com/#MapleEve/voscript&type=date)

## License

MIT —— 看 [LICENSE](./LICENSE)。
