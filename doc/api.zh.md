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

`speaker_id` 和 `speaker_name`：如果 `similarity ≥ 0.75`，服务会自动匹配上已登记的声纹；
否则 `speaker_id = null`，`speaker_name = speaker_label`（如 `SPEAKER_00`）。

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

返回与 `GET /api/jobs/{id}` 里 `result` 字段相同的完整对象。

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

表单字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `tr_id` | ✅ | 任务 id，对应 `result.id` |
| `speaker_label` | ✅ | **必须**是 `SPEAKER_XX` 这种原始标签，不是 `speaker_name` |
| `speaker_name` | ✅ | 展示用的人名，例如 "张三" |
| `speaker_id` | ❌ | 传了就是更新已有声纹，不传就是新建 |

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
{ "cohort_size": 313, "saved_to": "/data/transcriptions/asnorm_cohort.npy" }
```

从 0.5.0 起，服务会在启动时从已有转录中自动构建 AS-norm 评分矩阵。启用后，声纹识别采用归一化分数（相对于 impostor 分布），有效阈值固定为 `0.5`，无视 `VOICEPRINT_THRESHOLD`。可通过 `/api/voiceprints/rebuild-cohort` 手动刷新。

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
| 400 | 请求字段缺失或格式错误 |
| 401 | 缺 API key / key 不对 |
| 404 | tr_id / speaker_id / embedding 不存在 |
| 413 | 上传超过 `MAX_UPLOAD_BYTES`（默认 2 GiB），详见 `/api/transcribe` |
| 500 | 服务端异常（看 `docker logs voscript`） |

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
