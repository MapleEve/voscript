<sub>🌐 <a href="README.en.md">English</a> · <b>中文</b></sub>

<div align="center">

# VoScript Skills 🧰

> *「你有 VoScript，也有 AI 代理——但它们还不互通。」*

<a href="https://www.npmjs.com/package/voscript-skills">
  <img src="https://img.shields.io/npm/v/voscript-skills?style=flat-square" alt="npm version" />
</a>
<a href="https://www.npmjs.com/package/voscript-skills">
  <img src="https://img.shields.io/npm/dm/voscript-skills?style=flat-square" alt="npm downloads" />
</a>
<a href="https://github.com/MapleEve/VoScript">
  <img src="https://img.shields.io/badge/VoScript-主项目-blue?style=flat-square" alt="主项目" />
</a>
<a href="./LICENSE">
  <img src="https://img.shields.io/badge/License-个人免费%20·%20商业授权-orange?style=flat-square" alt="License" />
</a>

<br>

一条命令装好，任意 AI 代理（Claude Code、Trae、Cursor……）立即能操作 VoScript。<br>
覆盖完整工作流：提交音频、轮询任务、导出字幕、管理声纹库。<br>
纯 Python + requests，零额外依赖。

<br>

[安装](#安装) · [快速开始](#快速开始) · [支持的代理](#支持的代理) · [主项目](https://github.com/MapleEve/VoScript)

</div>

---

## 你是不是遇到过这个

> 部署了 VoScript，也在用 AI 代理做任务——但要让代理自己提交录音、等结果、导出字幕，得先写一堆接口胶水代码。

> 或者——代理调了一下接口，发现不知道该用哪个 endpoint，不知道 job 什么时候完成，不知道声纹注册时 `speaker_label` 和显示名有什么区别。

这个技能包解决的就是这个。**加载一次，代理自动掌握完整 VoScript 工作流。**

---

## 安装

### 方式一：npx（推荐）

```bash
npx voscript-skills
```

默认安装到 `~/.claude/skills/voscript-api/`，Claude Code 自动识别。

**一键为所有已检测到的代理安装：**

```bash
npx voscript-skills --all
```

**指定代理或目录：**

```bash
npx voscript-skills --agent trae      # claude / trae / cursor / codex / gemini / hermes / openclaw / windsurf / cline / roo
npx voscript-skills --dir /custom/skills/path
```

**卸载：**

```bash
npx voscript-skills --uninstall
```

### 方式二：Git clone

```bash
git clone https://github.com/MapleEve/voscript-skills.git ~/.claude/skills/
```

克隆后 `~/.claude/skills/voscript-api/` 即为技能目录。

---

## 支持的代理

| 代理 | --agent | 技能目录 | 自动检测 |
| --- | --- | --- | --- |
| Claude Code | `claude` | `~/.claude/skills/` | ✅ |
| Trae | `trae` | `~/.trae/context/skills/` | ✅ |
| Cursor | `cursor` | `~/.cursor/rules/skills/` | ✅ |
| OpenAI Codex CLI | `codex` | `~/.codex/skills/` | ✅ |
| Gemini CLI | `gemini` | `~/.gemini/skills/` | ✅ |
| Hermes | `hermes` | `~/.hermes/skills/` | ✅ |
| OpenClaw | `openclaw` | `~/.openclaw/skills/` | ✅ |
| Windsurf | `windsurf` | `~/.codeium/windsurf/skills/` | ✅ |
| Cline | `cline` | `~/.cline/skills/` | ✅ |
| Roo Code | `roo` | `~/.roo/skills/` | ✅ |
| 其它 | — | 使用 `--dir` 指定 | — |

---

## 你会得到什么

**11 个即用脚本，覆盖完整工作流**

| 脚本 | 功能 |
| --- | --- |
| `submit_audio.py` | 提交音频文件，返回 job ID |
| `poll_job.py` | 带进度条轮询，自动等待完成 |
| `fetch_result.py` | 获取完整转写结果（含说话人） |
| `export_transcript.py` | 导出 SRT / TXT / JSON |
| `list_transcriptions.py` | 列出所有转写记录 |
| `enroll_voiceprint.py` | 注册声纹（含 speaker_label 避坑提示） |
| `list_voiceprints.py` | 列出已注册声纹 |
| `assign_speaker.py` | 手动分配片段说话人 |
| `manage_voiceprint.py` | 查看 / 重命名 / 删除声纹 |
| `rebuild_cohort.py` | 重建 AS-norm cohort |
| `common.py` | 基础客户端 + 诊断工具 |

每个脚本在失败时输出结构化诊断报告——明确告诉代理哪里出了问题、怎么修。

---

## 快速开始

### 前置条件

- Python 3.8+
- `pip install requests`
- 一个运行中的 [VoScript 服务](https://github.com/MapleEve/VoScript)

### 配置

```bash
export VOSCRIPT_URL=http://localhost:7880
export VOSCRIPT_API_KEY=your_api_key_here
```

或在每个脚本调用时传入 `--url` / `--api-key`。

### 示例

```bash
# 提交音频
python voscript-api/scripts/submit_audio.py --file meeting.mp3

# 轮询直到完成
python voscript-api/scripts/poll_job.py --job-id tr_xxxxxxxx

# 获取结果
python voscript-api/scripts/fetch_result.py --tr-id tr_xxxxxxxx

# 导出 SRT
python voscript-api/scripts/export_transcript.py --tr-id tr_xxxxxxxx --format srt

# 注册声纹（用 speaker_label，不是显示名！）
python voscript-api/scripts/enroll_voiceprint.py \
  --tr-id tr_xxxxxxxx \
  --speaker-label SPEAKER_00 \
  --speaker-name "张三"
```

---

## 在 AI 代理中加载

安装到代理的技能目录后，加载 `voscript-api/SKILL.md` 即可。代理会自动读取工作流定义和脚本用法，无需额外配置。

---

## 问题反馈 & 功能建议

本仓库禁用了 Issues 和 Discussions。请前往主项目提交：

→ **[github.com/MapleEve/VoScript/issues](https://github.com/MapleEve/VoScript/issues)**

---

## License

与 VoScript 主项目相同。详见 [LICENSE](LICENSE)。
