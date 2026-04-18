# voscript

**简体中文** | [English](./README.en.md)

自托管的 GPU 转录服务——**带说话人记忆的会议逐字稿**。一套简单的 HTTP API，
把音频转成带说话人名字的文字，同一个人再次出现会自动识别。

```
音频  ──►  faster-whisper large-v3  （转录）
      ──►  pyannote 3.1             （说话人分离）
      ──►  ECAPA-TDNN               （声纹提取）
      ──►  VoiceprintDB             （与已注册声纹做余弦匹配）
      ──►  带时间戳和已识别说话人姓名的文本
```

与"纯 whisper 包装"的区别：**持久化声纹库**。登记过一次，之后所有录音里这个
人都会被自动贴上真名，不需要每次人工贴标签。

> 用例参考：[OpenPlaud(Maple)](https://github.com/MapleEve/openplaud) 把本服务作为
> 会议录音的后端——把这个 repo 当作标准 HTTP 服务对接即可，不限特定客户端。

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

## 功能

- **异步任务流水线**：`queued → converting → transcribing → identifying → completed`
- **中文 + 多语种转录**（WhisperX + faster-whisper large-v3，**带词级时间戳**的 forced alignment）
- **说话人分离**（pyannote 3.1）+ **WeSpeaker ResNet34** 声纹提取
- **持久化声纹**：一次登记，后续录音自动识别（余弦相似度 ≥ 0.75 视为命中）。底层 sqlite + sqlite-vec，top-k 近邻搜索 O(log N)，上千个声纹毫无压力
- **稳定的 HTTP 合同**：`/api/transcribe`、`/api/jobs/{id}`、`/api/voiceprints*` 等，任何 HTTP 客户端都能接入
- **容器以非 root 用户运行**；所有 `/api/*` 路由支持可选 Bearer / `X-API-Key` 鉴权（常量时间对比）；上传有 `MAX_UPLOAD_BYTES` 上限；声纹库并发安全、原子写入——完整硬化清单见 [`doc/security.zh.md`](./doc/security.zh.md)
- `/` 自带一个轻量 Web UI，方便单独测试

## 30 秒上手

```bash
git clone https://github.com/MapleEve/voscript.git
cd voscript

cp .env.example .env
# 编辑 .env —— 至少要填 HF_TOKEN 和 API_KEY

docker compose up -d --build
curl -sf http://localhost:8780/healthz
```

完整步骤 + 排障清单看 [`doc/quickstart.zh.md`](./doc/quickstart.zh.md)。

## 怎么接入

voscript 就是个普通的 HTTP 服务，没有特定客户端的强依赖。任何能发
`multipart/form-data` 的东西都能用（curl、axios、requests、网页上传框……）。

一个典型对接示例——OpenPlaud(Maple) 的"设置 → 转录"里配：

- **Private transcription base URL**：`http://<主机>:8780`
- **Private transcription API key**：跟 `.env` 里的 `API_KEY` 一致

之后它的 worker 会自动把每条录音都丢给这个服务。想自己写客户端的话，看
[`doc/api.zh.md`](./doc/api.zh.md) 的完整合同 + 错误码表。

## License

MIT —— 看 [LICENSE](./LICENSE)。
