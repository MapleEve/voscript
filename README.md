<sub>🌐 <a href="README.en.md">English</a> · <b>中文</b></sub>

<div align="center">

# VoScript 🎙️

> *「录完音，你想知道谁说了什么——而不是"说话人 A"说了什么。」*

<a href="https://github.com/MapleEve/voscript/actions/workflows/ci.yml">
  <img src="https://img.shields.io/github/actions/workflow/status/MapleEve/voscript/ci.yml?branch=main&style=flat-square" alt="CI" />
</a>
<a href="https://github.com/MapleEve/voscript/releases">
  <img src="https://img.shields.io/github/v/release/MapleEve/voscript?style=flat-square" alt="Release" />
</a>
<a href="https://hub.docker.com/r/mapleeve/voscript">
  <img src="https://img.shields.io/badge/Docker-ready-blue?style=flat-square&logo=docker" alt="Docker" />
</a>
<a href="./LICENSE">
  <img src="https://img.shields.io/badge/License-个人免费%20·%20商业授权-orange?style=flat-square" alt="License" />
</a>

<br>

开完会，你想知道谁说了什么——不是手动回放对名字。<br>
声音登记一次，之后每个人都被自动认出来。数据留在自己的服务器，不上云，不按分钟收费。<br>
完整 HTTP 接口，可以接入任何工作流和 AI Agent。

<br>

[快速上手](./doc/quickstart.zh.md) · [API 参考](./doc/api.zh.md) · [Benchmarks](./doc/benchmarks.zh.md) · [更新日志](./doc/changelog.zh.md)

</div>

---

## 你是不是也遇到过这个

> 开完会，打开录音，一边播放一边手动加名字：「这段是 Maple，这段是 Tom……」90 分钟的会，整理要再花 45 分钟。

> 试过带说话人分离的方案，出来是 Speaker A、Speaker B——还是不知道谁是谁，该对应的还是得手动对。

VoScript 解决的就是这个。**把声音登记一次，之后所有录音自动打上真名**——不是「说话人 2」，是「Maple」。

---

## 开始用

```bash
git clone https://github.com/MapleEve/voscript.git && cd voscript
cp .env.example .env   # 至少填 HF_TOKEN 和 API_KEY
docker compose up -d --build
```

浏览器打开 `http://localhost:8780`，上传录音，等结果。

> 安全提醒：公网部署前务必在 `.env` 设置强 `API_KEY`，否则任何人都能操作你的声纹库。

完整安装步骤 + 排障 → [`doc/quickstart.zh.md`](./doc/quickstart.zh.md)

---

## 两种用法

### 直接用网页面板——打开浏览器就能干活

内置了一个轻量面板，不用写任何代码：

- **转录 tab**：上传音频文件，选参数，提交，等结果
- **声纹库 tab**：登记说话人（上传样本 → 命名 → 保存），删除，查看已有声纹

适合：偶尔用一次、临时整理录音、不想配置 API 的场景。

### 接入你的工具——全自动流水线

配置服务地址和 API Key，录音自动发来转录，结果直接落进你的工作流。[BetterAINote](https://github.com/MapleEve/BetterAINote) 就是这样接的，其它客户端同理。

适合：长期使用、团队共用、有现成录音工作流的场景。

---

## 你会得到什么

**转录结果**

- 带时间戳的逐字稿，每个词都精确对齐
- 真名说话人标签（没登记过的标 Unknown）
- 支持中文、英文等多语言混录

**声纹系统**

- 今天登记，三年后的录音还认识；数据库是普通文件，随时备份迁移
- 相同录音提交两次，第二次直接返回已有结果，不重跑 GPU
- 嘈杂录音自动降噪；干净录音自动跳过，不会越处理越差

**使用方式**

- 有网页面板，上传文件、查结果、管理声纹库，不用写代码
- 也支持 HTTP API 接入，任何能发请求的工具都行

---

## 核心流程

```
音频  ──►  faster-whisper large-v3     转录 + 词级时间戳
      ──►  pyannote 3.1               说话人分离
      ──►  WeSpeaker ResNet34          声纹提取
      ──►  VoiceprintDB (AS-norm)      与已注册声纹匹配
      ──►  带时间戳 + 真名的逐字稿
```

声纹匹配用 AS-norm 评分消除说话人依赖偏差，配合自适应阈值（每人根据登记样本方差动态调整）。实测 10 条真实录音：召回率 50% → 70%，零误识别。

技术细节 → [`doc/benchmarks.zh.md`](./doc/benchmarks.zh.md)

---

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

---

## 联系

微信公众号：**等枫再来**

有问题、有想法、想聊聊语音转录那些坑，欢迎来找我。

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MapleEve/voscript&type=date)](https://www.star-history.com/#MapleEve/voscript&type=date)

---

## 贡献 & License

欢迎 PR，请先读 [CONTRIBUTING.md](./CONTRIBUTING.md)。

个人随便用，企业要打招呼 — [LICENSE](./LICENSE)
