---
name: voscript-api
description: |
  VoScript 自托管语音转写服务 API 技能包。提供音频上传、任务轮询、结果获取、
  字幕导出、声纹注册与管理、说话人分配、AS-norm cohort 重建等完整工作流。
  Triggers:
    - voscript
    - VoScript
    - voscript api
    - 提交转写
    - submit transcription
    - 上传音频
    - 查询任务
    - poll job
    - job status
    - 转写结果
    - get transcript
    - transcription result
    - 导出字幕
    - export transcript
    - export srt
    - 注册声纹
    - enroll voiceprint
    - speaker enrollment
    - 声纹列表
    - list speakers
    - voiceprint list
    - 分配说话人
    - assign speaker
    - 重建声纹库
    - rebuild cohort
---

# VoScript API 技能包

VoScript 是一个自托管的语音转写服务，支持多说话人分离、声纹识别、降噪、
多格式导出。本技能包封装了其 REST API 的全部主要工作流。

> 重要：本技能与代理无关（agent-agnostic），同等适用于 Claude、Codex、
> Trae、Hermes、OpenClaw 等任何 AI 代理，不依赖任何厂商专属特性。

## 1. 配置说明

VoScript 通过两个参数进行访问配置：

- `VOSCRIPT_URL`：服务地址，例如 `http://localhost:7880`
- `VOSCRIPT_API_KEY`：调用 API 所需的鉴权密钥

推荐通过环境变量设置，所有脚本也支持 `--url` / `--api-key` 命令行参数覆盖。

**当 `VOSCRIPT_URL` 或 `VOSCRIPT_API_KEY` 未配置时，代理必须：**

1. 先向用户索要服务地址与 API Key；
2. 告知用户配置方式：
   - 环境变量：`export VOSCRIPT_URL=...` / `export VOSCRIPT_API_KEY=...`
   - 或使用脚本的 `--url <URL>` / `--api-key <KEY>` 参数。

详见 `${SKILL_PATH}/references/configuration.md`。

## 2. 提交音频转写

上传音频文件并创建转写任务。

- 端点：`POST /api/transcribe`（`multipart/form-data`）
- 请求参数：
  - `file`（必填）：待转写音频文件
  - `language`（可选）：语言代码，例如 `zh` / `en`
  - `min_speakers`（默认 `1`）：最少说话人数
  - `max_speakers`（默认 `10`）：最多说话人数
  - `denoise_model`（默认 `none`）：可选 `none` / `deepfilternet` / `noisereduce`
  - `snr_threshold`（默认 `10.0`）：信噪比阈值
  - `no_repeat_ngram_size`（默认 `0`）：解码时抑制 n-gram 重复
- 响应：
  - 新任务：`{"id": "tr_xxx", "status": "queued"}`
  - 重复音频：`{"id": "tr_xxx", "status": "completed", "deduplicated": true}`
- SHA-256 去重：服务端会对音频内容计算 SHA-256，若曾提交过相同文件，
  将直接返回既有结果（`deduplicated: true`），无需再次处理。

执行脚本：

```bash
python ${SKILL_PATH}/scripts/submit_audio.py \
  --file <PATH> \
  [--language zh] \
  [--min-speakers 1] \
  [--max-speakers 10]
```

## 3. 轮询任务状态

- 端点：`GET /api/jobs/{job_id}`
- 状态机：`queued → converting → denoising → transcribing → identifying → completed | failed`
- 建议每 5 秒轮询一次，直到 `completed` 或 `failed`。
- 详细状态机与阶段耗时：`${SKILL_PATH}/references/job-lifecycle.md`

执行脚本：

```bash
python ${SKILL_PATH}/scripts/poll_job.py --job-id tr_xxx
```

## 4. 获取转写结果

- 端点：`GET /api/transcriptions/{tr_id}`
- 返回内容包括：`segments`、`speaker_map`、`params` 等完整结果。
- 每个 segment 字段：
  - `id`：片段序号
  - `start` / `end`：起止时间（秒）
  - `text`：转写文本
  - `speaker_label`：扬声器标签（如 `SPEAKER_00`）
  - `speaker_id`：已绑定的声纹 ID，可为空
  - `speaker_name`：说话人名字
  - `similarity`：AS-norm z-score（注意：不是 [0,1] 概率，取值无上界，可能大于 1.0）

**警告**：`similarity` 是 AS-norm 规范化后的 z-score，用于相对比较，
不能按概率解释，典型范围约 -1 到 2，典型匹配阈值 ~0.5。

执行脚本：

```bash
python ${SKILL_PATH}/scripts/fetch_result.py --tr-id tr_xxx
```

## 5. 导出转写

- 端点：`GET /api/export/{tr_id}?format=srt|txt|json`
- 支持格式：
  - `srt`：标准字幕文件
  - `txt`：带说话人前缀的纯文本
  - `json`：完整结构化数据
- 格式细节：`${SKILL_PATH}/references/export-formats.md`

执行脚本：

```bash
python ${SKILL_PATH}/scripts/export_transcript.py --tr-id tr_xxx --format srt
```

## 6. 转写列表

- 端点：`GET /api/transcriptions`
- 响应：`[{id, filename, created_at, segment_count, speaker_count}]`

执行脚本：

```bash
python ${SKILL_PATH}/scripts/list_transcriptions.py
```

## 7. 注册声纹

从已有转写中抽取某个 `speaker_label` 对应片段作为样本，注册或更新声纹。

- 端点：`POST /api/voiceprints/enroll`
- 请求参数：
  - `tr_id`（必填）：来源转写 ID
  - `speaker_label`（必填）：如 `SPEAKER_00`
  - `speaker_name`（必填）：说话人名字
  - `speaker_id`（可选）：传入已有声纹 ID 可更新该声纹
- 响应：`{"action": "created"|"updated", "speaker_id": "xxx"}`

执行脚本：

```bash
python ${SKILL_PATH}/scripts/enroll_voiceprint.py \
  --tr-id tr_xxx \
  --speaker-label SPEAKER_00 \
  --speaker-name "张三"
```

## 8. 声纹列表

- 端点：`GET /api/voiceprints`
- 响应：`[{id, name, sample_count, sample_spread, created_at, updated_at}]`
- `sample_spread`：样本间余弦相似度的标准差，单样本时为 `null`，数值越小表示
  样本一致性越高。

执行脚本：

```bash
python ${SKILL_PATH}/scripts/list_voiceprints.py
```

## 9. 分配说话人

手动为某个 segment 指定说话人（用于纠正分离错误或补齐未识别片段）。

- 端点：`PUT /api/transcriptions/{tr_id}/segments/{seg_id}/speaker`
- 请求参数：
  - `speaker_name`（必填）
  - `speaker_id`（可选）：若要绑定已注册声纹，传入声纹 ID

执行脚本：

```bash
python ${SKILL_PATH}/scripts/assign_speaker.py \
  --tr-id tr_xxx \
  --seg-id 5 \
  --speaker-name "李四"
```

## 10. 管理声纹

- `GET /api/voiceprints/{speaker_id}`：查看详情
- `PUT /api/voiceprints/{speaker_id}/name`（表单字段 `name`）：重命名
- `DELETE /api/voiceprints/{speaker_id}`：删除

执行脚本：

```bash
python ${SKILL_PATH}/scripts/manage_voiceprint.py \
  --action [get|rename|delete] \
  --speaker-id xxx \
  [--name "新名字"]
```

## 11. 重建声纹 cohort

AS-norm 评分依赖 cohort（对比样本集）。当累计注册达到 10+ 说话人，或
声纹库发生较大变动后，建议重建一次 cohort 以刷新评分基线。

- 端点：`POST /api/voiceprints/rebuild-cohort`
- 响应：`{"cohort_size": N, "skipped": M, "saved_to": "path"}`

执行脚本：

```bash
python ${SKILL_PATH}/scripts/rebuild_cohort.py
```

声纹完整工作流与阈值说明见 `${SKILL_PATH}/references/voiceprint-guide.md`。

## 错误响应规范

VoScript 返回标准 HTTP 状态码，代理在处理响应时应按下表做分支：

| 状态码 | 含义                   | 处理建议                                         |
| ------ | ---------------------- | ------------------------------------------------ |
| 200    | 成功                   | 正常解析响应                                     |
| 401    | API Key 无效           | 提示用户检查 `VOSCRIPT_API_KEY`                  |
| 404    | 资源不存在             | 核对 `tr_id` / `speaker_id` / `job_id`           |
| 422    | 请求参数校验失败       | 根据返回 `detail` 字段检查参数，常见于缺少 `file` |
| 500    | 服务端错误             | 收集 `error` 字段，必要时检查服务端日志          |

## 典型使用序列

1. 配置 `VOSCRIPT_URL` / `VOSCRIPT_API_KEY`。
2. `submit_audio.py` 上传音频，拿到 `tr_id`。
3. `poll_job.py` 轮询到 `completed`。
4. `fetch_result.py` 获取 segments，审阅 speaker 分离结果。
5. 对每个 `SPEAKER_xx` 调用 `enroll_voiceprint.py` 注册真实姓名。
6. 累计 10+ 声纹后运行 `rebuild_cohort.py` 刷新 AS-norm 基线。
7. 后续新音频转写会自动识别已注册说话人。
8. 需要字幕文件时使用 `export_transcript.py` 导出 SRT/TXT/JSON。

## 参考文档

- `${SKILL_PATH}/references/configuration.md` —— 配置与鉴权
- `${SKILL_PATH}/references/job-lifecycle.md` —— 任务状态机
- `${SKILL_PATH}/references/voiceprint-guide.md` —— 声纹与 AS-norm
- `${SKILL_PATH}/references/export-formats.md` —— 导出格式
