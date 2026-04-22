<sub>🌐 **中文** · [English](README.en.md)</sub>

# VoScript Skills

**VoScript 语音转写服务的 AI 代理技能包。**

本仓库为 [VoScript](https://github.com/MapleEve/VoScript) 提供可被任意 AI 代理加载的技能（Skill），涵盖音频提交、任务轮询、转写结果获取、字幕导出、声纹管理等完整工作流。

## 特性

- **代理无关（agent-agnostic）**：可在 Claude、Codex、Trae、Hermes、OpenClaw 等任意 AI 代理中使用，不依赖任何厂商专属特性。
- **自托管优先**：对接本地或私有部署的 VoScript 服务，无需云账号。
- **纯 Python + requests**：脚本只依赖标准库与 `requests`，无额外安装负担。
- **中英双语文档**：SKILL.md 及参考文档以中文编写，面向中文用户。

## 目录结构

```
voscript-api/
  SKILL.md                   # 主技能文档（所有工作流）
  .meta.json                 # 技能元数据
  references/
    configuration.md         # 服务地址与 API Key 配置说明
    job-lifecycle.md         # 任务状态机与生命周期文档
    voiceprint-guide.md      # 声纹注册与 AS-norm 评分指南
    export-formats.md        # SRT / TXT / JSON 格式说明
  scripts/
    common.py                # VoScriptClient 基础客户端
    submit_audio.py          # 提交音频转写
    poll_job.py              # 轮询任务状态
    fetch_result.py          # 获取完整转写结果
    export_transcript.py     # 导出字幕（SRT / TXT / JSON）
    list_transcriptions.py   # 列出所有转写记录
    enroll_voiceprint.py     # 注册声纹
    list_voiceprints.py      # 列出已注册声纹
    assign_speaker.py        # 手动分配片段说话人
    manage_voiceprint.py     # 查看 / 重命名 / 删除声纹
    rebuild_cohort.py        # 重建 AS-norm cohort
```

## 安装

### 方式一：npx 一键安装（推荐）

```bash
npx voscript-skills
```

默认安装到 `~/.claude/skills/voscript-api/`，Claude Code 会自动识别。

**一键为所有已检测到的代理安装：**

```bash
npx voscript-skills --all
```

**指定目标代理：**

```bash
npx voscript-skills --agent trae
npx voscript-skills --dir /custom/skills/path
```

**卸载：**

```bash
npx voscript-skills --uninstall
```

### 方式二：Git clone 手动安装

```bash
git clone https://github.com/MapleEve/voscript-skills.git ~/.claude/skills/
```

克隆后 `~/.claude/skills/voscript-api/` 即为技能目录。

### 已知代理目录

| 代理 | 技能目录 |
| --- | --- |
| Claude Code | `~/.claude/skills/` |
| Trae | `~/.trae/context/skills/` |
| 其它 | 使用 `--dir` 参数指定 |

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

或在每个脚本调用时传入 `--url` / `--api-key` 参数。

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

# 注册声纹（使用 speaker_label，而非显示名！）
python voscript-api/scripts/enroll_voiceprint.py \
  --tr-id tr_xxxxxxxx \
  --speaker-label SPEAKER_00 \
  --speaker-name "张三"
```

## 在 AI 代理中使用

将本仓库克隆到 AI 代理的技能目录，然后加载 `voscript-api/SKILL.md` 即可。

## 问题反馈 & 功能建议

本仓库已禁用 Issues 和 Discussions。请前往主项目提交：

→ **[github.com/MapleEve/VoScript/issues](https://github.com/MapleEve/VoScript/issues)**

## 许可证

与 VoScript 主项目相同。详见 [LICENSE](LICENSE)。
