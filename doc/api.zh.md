# API 参考

**简体中文** | [English](./api.en.md)

所有接口都在 `http://<主机>:8780` 下。数据交换用 JSON，文件上传用
`multipart/form-data`。

## 鉴权

设了 `API_KEY` 之后，除了下面这几个路径，所有请求都必须带
`Authorization: Bearer <API_KEY>` **或** `X-API-Key: <API_KEY>`：

| 路径 | 无需鉴权 | 匹配方式 |
| --- | --- | --- |
| `GET /` | ✅ 返回内置 Web UI | 精确匹配 |
| `GET /healthz` | ✅ 健康检查 | 精确匹配 |
| `GET /docs` / `/redoc` / `/openapi.json` | ✅ FastAPI 自动文档 | 精确匹配 |
| `GET /static/*` | ✅ 静态资源 | `/static/` 前缀 |
| 其它 `/api/*` | ❌ 必须带 key | — |

没带、带错都会返回 `401 Unauthorized`。鉴权比较采用 `hmac.compare_digest`
常量时间，并且从 0.2.0 起 `/docs` 等路径是**精确匹配**——`/docsXYZ`
现在会 401，不再漏网。

## 任务生命周期

```
POST /api/transcribe
    ↓
queued → converting → denoising (if DENOISE_MODEL ≠ none) → transcribing → identifying → completed
                                                                                              ↘ failed
```

OpenPlaud(Maple) worker 每 5 秒轮询一次 `/api/jobs/{id}`，看到 `completed`
或 `failed` 就终止轮询。

## 接口清单

### `GET /healthz` — 健康检查

```bash
curl http://localhost:8780/healthz
# {"ok":true}
```

### `POST /api/transcribe` — 提交转录

表单字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `file` | file | 必填，音频文件（wav / mp3 / m4a / flac / ogg / webm） |
| `language` | string | 选填，ISO 639-1；留空则自动检测语言（对普通话音频会输出简体中文） |
| `min_speakers` | int | 选填，`0` 表示自动 |
| `max_speakers` | int | 选填，`0` 表示自动 |
| `denoise_model` | string | 选填。降噪后端：`none`（默认）、`deepfilternet`、`noisereduce`。仅对本次请求生效，覆盖容器环境变量 `DENOISE_MODEL`。 |
| `snr_threshold` | float | 选填。信噪比门限（dB），仅对本次请求生效。音频信噪比达到或超过此值时跳过降噪。覆盖 `DENOISE_SNR_THRESHOLD`。 |
| `no_repeat_ngram_size` | int | 选填，默认 `0`（不开启）。设置 ≥ 3 时抑制转录中的 n-gram 重复（如「比如比如」→「比如」）。值 < 3 等同于 `0`。非整数返回 422。 |
响应（200）：

```json
{ "id": "tr_20260418_080205_ea79b7", "status": "queued" }
```

如果上传的文件与已有的已完成任务完全一致（SHA256 相同），服务会直接返回那条任务的结果，
不再重跑 Whisper。此时响应包含额外字段 `deduplicated: true`：

```json
{ "id": "tr_existing_id", "status": "completed", "deduplicated": true }
```

客户端可以用这个 `id` 正常轮询 `/api/jobs/{id}` 或导出，无需任何特殊处理。

**上传大小**：服务端分块读取，累计超过 `MAX_UPLOAD_BYTES`（默认 2 GiB）
直接 `413`：

```json
{ "detail": "Upload exceeds MAX_UPLOAD_BYTES (2147483648 bytes)" }
```

同时把半截文件从 `data/uploads/` 删掉。用不了那么大空间的小机器可以在
`.env` 里把 `MAX_UPLOAD_BYTES` 调小（单位字节）。

**文件名**：multipart 里带的 `filename` 会先过
`PurePosixPath(filename).name`，只保留最末一段——客户端传
`filename=../../etc/passwd.wav` 也只会在磁盘上落为
`tr_<id>_passwd.wav`。

示例：

```bash
curl -X POST http://localhost:8780/api/transcribe \
     -H "Authorization: Bearer $API_KEY" \
     -F "file=@meeting.wav" \
     -F "language=zh" \
     -F "max_speakers=4" \
     -F "denoise_model=deepfilternet"
```

### `GET /api/jobs/{id}` — 查询任务

> **注意**：`/api/jobs/{id}` 优先读内存字典；内存中不存在时自动回落到 `data/transcriptions/<id>/status.json`。
> - 状态为 `completed` 时，连同 `result.json` 一起返回，行为等同 `GET /api/transcriptions/{id}`。
> - 状态为进行中（`converting / denoising / transcribing / identifying`）时，返回 `status=failed, error="Process restarted while job was in progress"`（启动时 `recover_orphan_jobs()` 已将孤儿任务标记为失败）。
> - `status.json` 不存在时才返回 404。
>
> 因此，**服务重启后旧 job 仍有确定终态**，前端不会再永久 pending。

```json
{
  "id": "tr_...",
  "status": "queued | converting | denoising | transcribing | identifying | completed | failed",
  "filename": "meeting.wav",

  "error": "...",     // 仅当 status = failed
  "result": {         // 仅当 status = completed
    "id": "tr_...",
    "language": "zh",
    "segments": [
      {
        "id": 0,
        "start": 0.0,
        "end": 4.32,
        "text": "一边是方便募取玩家自制的mix",
        "speaker_label": "SPEAKER_00",
        "speaker_id": "spk_...",
        "speaker_name": "张三",
        "similarity": 0.8421,
        "words": [
          { "word": "一边", "start": 0.02, "end": 0.35, "score": 0.93 },
          { "word": "是",   "start": 0.35, "end": 0.48, "score": 0.88 }
        ]
      }
    ],
    "params": {
      "language": "zh",  // 若提交时未指定语言，此处显示 "auto"
      "denoise_model": "none",
      "snr_threshold": 10.0,
      "voiceprint_threshold": 0.75,
      "min_speakers": 0,
      "max_speakers": 0
    }
  }
}
```

**`speaker_label` 是 pyannote 产出的原始标签**，不会因为匹配到已有声纹而变化。
这是做后续登记 / 重命名时必须用的 key。

`speaker_id` 和 `speaker_name`：匹配采用**自适应阈值**，不是固定 0.75。实际逻辑：

- 基础阈值为 `VOICEPRINT_THRESHOLD`（默认 0.75）。
- 每位说话人的有效阈值会根据已登记样本的余弦方差自动放松：单样本有效阈值约 0.70，
  spread 较大时进一步放宽（最多 0.10），**绝对下限 0.60**。
- AS-norm 模式激活（cohort ≥ 10）后改用归一化分数，操作点约 0.5。

只要通过了上述自适应阈值就匹配上已登记声纹；否则 `speaker_id = null`，
`speaker_name = speaker_label`（如 `SPEAKER_00`）。

`similarity`：说话人匹配相似度分数。
- **raw cosine 模式**（cohort < 10 或全新安装）：值域 [-1, 1]，通常为 [0, 1]，表示与已登记声纹均值的余弦相似度。
- **AS-norm 模式**（cohort ≥ 10）：归一化 z-score，**无界**（可大于 1.0 或为负数），代表相对于 impostor 分布的标准差倍数。
- 该值为 **说话人（speaker）级别聚合**，而非单段（segment）级别。
- `speaker_id` 非 null 表示通过了当前模式下的阈值。

**`words[]` 是 0.3.0 起新增的可选字段**（WhisperX forced alignment 输出）。
每个字/词有独立的 `start`/`end`/`score`。中文对齐模型有时会失败——失败时这
个字段缺失，不会阻塞任务完成。老客户端不认识这个字段时直接忽略即可。

**`params`** 记录本次任务实际采用的处理参数，包含所有请求级覆盖值，使每条结果
都可独立解读，无需再查原始请求。

### `GET /api/transcriptions` — 列出所有历史任务

```json
[
  { "id": "tr_...", "filename": "...", "created_at": "...",
    "segment_count": 42, "speaker_count": 3 }
]
```

### `GET /api/transcriptions/{tr_id}` — 单条任务详情

返回与 `GET /api/jobs/{id}` 里 `result` 字段相同的完整对象，另外包含两个方便 UI /
下游消费的聚合字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `speaker_map` | object | `speaker_label → {speaker_id, speaker_name}` 的映射，从 `segments` 聚合而来，便于前端一次性渲染人名下拉 / 统计 |
| `unique_speakers` | int | 去重后的说话人数（基于 `speaker_label`） |

与 `GET /api/jobs/{id}` 的 `result` 不同，本端点从磁盘读取持久化结果，**进程重启后
仍可访问**；`/api/jobs/{id}` 优先读内存，内存未命中时回落到磁盘（见上方注意事项）。

### `GET /api/export/{tr_id}` — 导出

query `format=srt | txt | json`。返回对应格式的下载响应。

### 声纹库

```
GET    /api/voiceprints
POST   /api/voiceprints/enroll
PUT    /api/voiceprints/{speaker_id}/name
DELETE /api/voiceprints/{speaker_id}
```

#### `GET /api/voiceprints`

```json
[
  { "id": "spk_61f24bd0", "name": "张三",
    "sample_count": 3,
    "created_at": "2026-04-18T08:06:41.951819",
    "updated_at": "2026-04-18T09:17:02.113207" }
]
```

#### `POST /api/voiceprints/enroll`

> **注意（enroll 幂等性）**：`add_speaker` 按 `name` 自动去重——同名的二次 enroll 会把新 embedding 合并到已有记录，**不会**再产生重复条目。
>
> 仍然建议已知 speaker 时传 `speaker_id` 走明确的更新路径（`update_speaker`），可以避免同名但不同人的歧义（例：两位都叫"张三"的发言人）。

表单字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `tr_id` | ✅ | 任务 id，对应 `result.id` |
| `speaker_label` | ✅ | **必须**是 `SPEAKER_XX` 这种原始标签，不是 `speaker_name` |
| `speaker_name` | ✅ | 展示用的人名，例如 "张三" |
| `speaker_id` | ❌ | 传了就是明确更新已有声纹；不传时按 `name` 自动去重（见上方注意） |

响应：

```json
{ "action": "created | updated", "speaker_id": "spk_..." }
```

示例：

```bash
curl -X POST http://localhost:8780/api/voiceprints/enroll \
     -H "Authorization: Bearer $API_KEY" \
     -F "tr_id=tr_20260418_080205_ea79b7" \
     -F "speaker_label=SPEAKER_00" \
     -F "speaker_name=张三"
```

#### `POST /api/voiceprints/rebuild-cohort`

从所有已处理的转录中重新构建 AS-norm 评分的上界矩阵（impostor cohort）。服务启动时会自动执行，若后续大量新增录音可手动触发。

响应：

```json
{ "cohort_size": 313, "skipped": 2, "saved_to": "/data/transcriptions/asnorm_cohort.npy" }
```

`skipped` — 无法加载 embedding 文件（`.npy` 损坏或缺失）的转录数量。

从 0.5.0 起，服务会在启动时尝试从已有转录中自动构建 AS-norm 评分矩阵。

**cohort 生命周期与行为**：

| cohort 规模 | identify 走的路径 | 有效阈值 |
| --- | --- | --- |
| 0（全新安装 / 无已有转录） | raw cosine | 基础 0.75 + 自适应放松，绝对下限 0.60 |
| 1–9（不足 10 条） | raw cosine（`score()` fallback） | 同上 |
| ≥ 10 | AS-norm 归一化分数 | 约 0.5（相对 impostor 分布，忽略 `VOICEPRINT_THRESHOLD`） |

**刷新时机**：cohort 仅在服务**启动时**构建一次；任务完成后**不会**自动把新
embedding 加入 cohort；必须显式调用 `POST /api/voiceprints/rebuild-cohort`
或重启服务才会更新。长期运行的服务在批量入库后应手动触发 rebuild。

#### `PUT /api/voiceprints/{id}/name`

表单 `name=新名字`，只改显示名，不动 embedding。

#### `DELETE /api/voiceprints/{id}`

从库里永久删除。被删掉的人后续录音就不会再被匹配上。

### `PUT /api/transcriptions/{tr_id}/segments/{seg_id}/speaker`

改某一条 segment 的说话人归属，用于手工纠正。

表单字段 `speaker_name`（必填）、`speaker_id`（选填）。

## 错误返回

| 状态码 | 原因 |
| --- | --- |
| 400 | 请求字段缺失或格式错误；job_id 格式非法（`^tr_[A-Za-z0-9_-]{1,64}$`）/ speaker_label 非法字符 / 路径穿越检测 |
| 401 | 缺 API key / key 不对 |
| 404 | tr_id / speaker_id / embedding 不存在 |
| 413 | 上传超过 `MAX_UPLOAD_BYTES`（默认 2 GiB），详见 `/api/transcribe` |
| 500 | 服务端异常（看 `docker logs voscript`） |
| 504 | ffmpeg 转码超时（超过 `FFMPEG_TIMEOUT_SEC`，默认 1800 秒） |

错误体结构：

```json
{ "detail": "..." }
```

## 与 OpenPlaud(Maple) 的对应关系

| OpenPlaud(Maple) 代码 | 调用的接口 |
| --- | --- |
| `submitVoiceTranscribeJob` | `POST /api/transcribe` |
| `pollVoiceTranscribeJob` | `GET /api/jobs/{id}` |
| `VoiceTranscribeClient.listVoiceprints` | `GET /api/voiceprints` |
| `VoiceTranscribeClient.enrollVoiceprint` | `POST /api/voiceprints/enroll` |
| `VoiceTranscribeClient.renameVoiceprint` | `PUT /api/voiceprints/{id}/name` |
| `VoiceTranscribeClient.deleteVoiceprint` | `DELETE /api/voiceprints/{id}` |

源码位置见 [OpenPlaud(Maple) 仓库](https://github.com/MapleEve/openplaud)下的
`src/lib/transcription/providers/voice-transcribe-provider.ts` 和
`src/lib/voice-transcribe/client.ts`。
