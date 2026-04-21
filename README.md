<div align="center">

# VoScript 🎙️

**简体中文** | [English](./README.en.md)

<a href="https://github.com/MapleEve/voscript/actions/workflows/ci.yml">
  <img src="https://img.shields.io/github/actions/workflow/status/MapleEve/voscript/ci.yml?branch=main&style=for-the-badge" alt="CI" />
</a>
<a href="https://github.com/MapleEve/voscript/releases">
  <img src="https://img.shields.io/github/v/release/MapleEve/voscript?style=for-the-badge" alt="Release" />
</a>
<a href="https://hub.docker.com/r/mapleeve/voscript">
  <img src="https://img.shields.io/badge/Docker-ready-blue?style=for-the-badge&logo=docker" alt="Docker" />
</a>
<a href="./LICENSE">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge" alt="License" />
</a>

**会议录音 → 逐字稿，带真名说话人标签。自托管，GPU 驱动，记得住每个人的声音。**

[快速上手](./doc/quickstart.zh.md) · [API 参考](./doc/api.zh.md) · [安全策略](./doc/security.zh.md) · [Benchmarks](./doc/benchmarks.zh.md) · [更新日志](./doc/changelog.zh.md)

</div>

---

开完会，录音里有六个人，你想知道谁说了什么。Whisper 只给你一段文字，pyannote 能告诉你"说话人A/说话人B"，但它不认识人——每次还是得手动贴名字。

VoScript 解决的就是这个：**登记一次声纹，之后所有录音里这个人都会被自动识别出来**。不是"说话人2"，是"Maple"。

```
音频  ──►  faster-whisper large-v3     转录 + 词级时间戳
      ──►  pyannote 3.1               说话人分离
      ──►  WeSpeaker ResNet34          声纹提取
      ──►  VoiceprintDB (AS-norm)      与已注册声纹匹配
      ──►  带时间戳 + 真名的逐字稿
```

## 30 秒上手

> **安全警告**：生产环境或公网暴露前**必须**在 `.env` 里设置 `API_KEY`，否则任何人都能删你的声纹库、触发 GPU 任务。

```bash
git clone https://github.com/MapleEve/voscript.git && cd voscript
cp .env.example .env        # 至少填 HF_TOKEN 和 API_KEY
docker compose up -d --build
curl -sf http://localhost:8780/healthz
```

完整步骤 + 排障清单 → [`doc/quickstart.zh.md`](./doc/quickstart.zh.md)

## 功能

- **持久化声纹库** — 登记一次，后续所有录音自动识别。底层 sqlite + sqlite-vec，top-k 近邻，上千声纹毫无压力
- **AS-norm 评分** — 启动时自动从历史转录构建 impostor cohort，消除说话人依赖的基准偏差，相对 EER 降低 15–30%
- **自适应阈值** — 每位说话人的实际识别阈值根据注册样本的方差动态宽松，10 条真实录音召回率从 50% → 70%，零误识别
- **说话人聚类合并** — 同一个人被分出多个聚类时自动合并为一个标签
- **词级时间戳** — WhisperX forced alignment，每个词都有精确时间
- **可选降噪 + SNR 门控** — DeepFilterNet / noisereduce，SNR 高于阈值的录音自动跳过（防止对干净音频劣化）
- **文件哈希去重** — 相同文件重复提交直接返回已有结果，不重跑 GPU
- **任务持久化** — 重启后已完成任务仍可访问
- **ngram 去重** — `no_repeat_ngram_size` 参数抑制转录中的口语重复（比如"就是就是就是"）
- **纯 HTTP 合同** — 任何能发 multipart/form-data 的客户端都能接入，不绑定特定框架

安全相关：路径遍历防护、非 root 容器、上传大小限制、常量时间鉴权、原子写入……完整清单 → [`doc/security.zh.md`](./doc/security.zh.md)

## 接入

就是个普通 HTTP 服务，没有特殊依赖。配两个值就行：

- **转录服务地址**：`http://<主机>:8780`
- **API Key**：`.env` 里设的那个 `API_KEY`

[BetterAINote](https://github.com/MapleEve/openplaud) 就是这样接的，其它客户端一样。完整接口合同 → [`doc/api.zh.md`](./doc/api.zh.md)

## 文档

| 主题 | 中文 | English |
| --- | --- | --- |
| 快速安装 | [quickstart.zh.md](./doc/quickstart.zh.md) | [quickstart.en.md](./doc/quickstart.en.md) |
| API 参考 | [api.zh.md](./doc/api.zh.md) | [api.en.md](./doc/api.en.md) |
| 给 AI 的安装指南 | [ai-install.zh.md](./doc/ai-install.zh.md) | [ai-install.en.md](./doc/ai-install.en.md) |
| 给 AI 的接口指南 | [ai-usage.zh.md](./doc/ai-usage.zh.md) | [ai-usage.en.md](./doc/ai-usage.en.md) |
| 安全策略 | [security.zh.md](./doc/security.zh.md) | [security.en.md](./doc/security.en.md) |
| Benchmarks | [benchmarks.zh.md](./doc/benchmarks.zh.md) | [benchmarks.en.md](./doc/benchmarks.en.md) |
| 更新日志 | [changelog.zh.md](./doc/changelog.zh.md) | [changelog.en.md](./doc/changelog.en.md) |

## 贡献

欢迎 PR，请先读 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MapleEve/voscript&type=date)](https://www.star-history.com/#MapleEve/voscript&type=date)

## License

Apache 2.0 — [LICENSE](./LICENSE)
