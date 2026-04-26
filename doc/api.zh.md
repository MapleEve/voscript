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
queued → converting → denoising (if effective denoise_model ≠ none) → transcribing → identifying → completed
                                                                                              ↘ failed
```

BetterAINote worker 每 5 秒轮询一次 `/api/jobs/{id}`，看到 `completed`
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
| `denoise_model` | string | 选填。降噪后端：`none`、`deepfilternet`、`noisereduce`。省略时使用服务端 `DENOISE_MODEL`（默认 `none`）；显式传 `none` 表示只对本次请求关闭降噪。 |
| `snr_threshold` | float | 选填。信噪比门限（dB），仅对本次请求生效。音频信噪比达到或超过此值时跳过降噪。覆盖 `DENOISE_SNR_THRESHOLD`（默认 `10.0`）。 |
| `no_repeat_ngram_size` | int | 选填，默认 `0`（不开启）。设置 ≥ 3 时抑制转录中的 n-gram 重复（如「比如比如」→「比如」）。值 < 3 等同于 `0`。非整数返回 422。 |
响应（200）：

```json
{ "id": "tr_example_id", "status": "queued" }
```

`POST /api/transcribe` 有两条去重路径，都是按上传文件的 SHA256 判断：

- **已完成结果去重**：如果完全相同的文件已经有完成态转录，接口会直接返回那条历史任务，
  不再重跑 Whisper：

```json
{ "id": "tr_existing_id", "status": "completed", "deduplicated": true }
```

- **并发 in-flight 去重**：如果完全相同的文件此刻已经被另一条在线请求处理，后到的请求
  不会再启动第二个 worker，而是直接复用第一条 job id，并返回当前排队态：

```json
{ "id": "tr_existing_inflight", "status": "queued", "deduplicated": true }
```

两种情况下，`deduplicated: true` 都表示**这次请求没有新建转录 worker**。客户端拿
到返回的 `id` 后，照常轮询 `/api/jobs/{id}` 或导出即可。

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

**503 场景**：`POST /api/transcribe` 也可能在真正开始处理前失败：

- `503 Failed to persist job state — disk error, retry later`
- `503 Failed to start background transcription — retry later`

示例：

```bash
curl -X POST http://localhost:8780/api/transcribe \
     -H "Authorization: Bearer $API_KEY" \
     -F "file=@meeting.wav" \
     -F "language=zh" \
     -F "max_speakers=4" \
     -F "denoise_model=deepfilternet"
```

降噪优先级是：API 显式字段优先，其次才是服务端 env。实际使用时，省略
`denoise_model` 表示继承 `DENOISE_MODEL`；传 `denoise_model=none` 表示本次请求关闭降噪；
只有当单个任务需要不同门限时才传 `snr_threshold`，它会覆盖
`DENOISE_SNR_THRESHOLD`。

### `GET /api/jobs/{id}` — 查询任务

> **注意**：`/api/jobs/{id}` 优先读内存字典；内存中不存在时自动回落到 `data/transcriptions/<id>/status.json`。
> - 如果完成态 job 还在内存里，`result` 直接从内存缓存返回。
> - 内存未命中时，完成态 job 才会从磁盘上的 `result.json` 读取。
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
    "speaker_map": {
      "SPEAKER_00": {
        "matched_id": "spk_...",
        "matched_name": "张三",
        "similarity": 0.8421,
        "embedding_key": "SPEAKER_00"
      }
    },
    "unique_speakers": ["张三"],
    "params": {
      "language": "zh",  // 若提交时未指定语言，此处显示 "auto"
      "denoise_model": "none",
      "snr_threshold": 10.0,
      "voiceprint_threshold": 0.75,
      "min_speakers": 0,
      "max_speakers": 0,
      "no_repeat_ngram_size": 0
    },
    "alignment": {
      "status": "succeeded",
      "language": "zh",
      "model": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
      "model_source": "whisperx_default",
      "cache_only": false
    }
  }
}
```

**`speaker_label` 是 pyannote 产出的原始标签**，不会因为匹配到已有声纹而变化。
这是做后续登记 / 重命名时必须用的 key。

**结果契约锚点**：完成态持久化转写对象会带 `status="completed"`。
`segments[].speaker_label` 永远是原始 diarization cluster 标签。
`segments[].words` 和顶层 `alignment` 都是可选元数据，客户端必须能接受字段缺失。

`speaker_id` 和 `speaker_name`：匹配采用**自适应阈值**，不是固定 0.75。实际逻辑：

- 基础阈值为 `VOICEPRINT_THRESHOLD`（默认 0.75）。
- 每位说话人的有效阈值会根据已登记样本的余弦方差自动放松：单样本有效阈值约 0.70，
  spread 较大时进一步放宽（最多 0.10），**绝对下限 0.60**。
- AS-norm 模式激活（cohort ≥ 10）后改用归一化分数，并围绕 0.5 操作点按样本数
  自适应：单样本更严格（默认至少 0.60），稳定多样本接近基准值；如果 top-1 与
  top-2 的 AS-norm 分数太接近，会保留为未命名供人工复核。

只要通过了上述自适应阈值就匹配上已登记声纹；否则 `speaker_id = null`，
`speaker_name = speaker_label`（如 `SPEAKER_00`）。

如果同一个结果里的多个 diarization 标签解析出相同展示名，服务会保留各自原始
`speaker_label`，并在 segment 输出里给展示名加序号区分，例如 `张三` 和
`张三 (2)`。声纹命名不会折叠 diarization cluster。

`similarity`：说话人匹配相似度分数。
- **raw cosine 模式**（cohort < 10 或全新安装）：值域 [-1, 1]，通常为 [0, 1]，表示与已登记声纹均值的余弦相似度。
- **AS-norm 模式**（cohort ≥ 10）：归一化 z-score，**无界**（可大于 1.0 或为负数），代表相对于 impostor 分布的标准差倍数。
- 该值为 **说话人（speaker）级别聚合**，而非单段（segment）级别。
- `speaker_id` 非 null 表示通过了当前模式下的阈值。

**`words[]` 是 0.3.0 起新增的可选字段**（WhisperX forced alignment 输出）。
每个字/词有独立的 `start`/`end`/`score`。如果某个语言的 alignment 模型不可用、
被配置禁用或加载失败，这个字段会缺失，不会阻塞任务完成。老客户端不认识这个
字段时直接忽略即可。

**`alignment`** 记录 forced alignment 状态（存在时）。常见值包括：
`status=succeeded`、`status=skipped` 且 `reason=language_disabled`、或
`status=failed` 且带脱敏后的 `error_type` 与 `actionable_hint`。默认中文
alignment 模型会记录为 `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`；
如果自定义旧运行时被 transformers 的 `torch.load` 安全检查拦截，`reason` 会是
`torch_version_blocked`，而不是 `not_found`。这里不会暴露 token、host 或本地路径。

**`params`** 记录本次任务实际采用的处理参数，包含所有请求级覆盖值，使每条结果
都可独立解读，无需再查原始请求。

`GET /api/jobs/{id}` 的完成态结果与 `GET /api/transcriptions/{id}` 使用同一份
持久化结果结构，因此完成态里同样会带上 `speaker_map` 和 `unique_speakers`：

- 如果你要拿**人工改过 segment 之后的最新持久化结果**，优先使用
  `GET /api/transcriptions/{id}`；`GET /api/jobs/{id}` 在缓存尚未淘汰前可能仍返回
  worker 结束时的内存副本。
- `speaker_map` 可能是空对象，例如整条音频里没有任何可用于登记的 speaker embedding
  （比如所有 diarization turn 都短于最小 embedding 时长）。
- `unique_speakers` 来自解析后的 `segments[].speaker_name`，所以匹配成功时会显示已登记
  人名，未匹配时则保留原始 diarization 标签。

### `GET /api/transcriptions` — 列出所有历史任务

```json
[
  { "id": "tr_...", "filename": "...", "created_at": "...",
    "segment_count": 42, "speaker_count": 3 }
]
```

### `GET /api/transcriptions/{tr_id}` — 单条任务详情

返回与 `GET /api/jobs/{id}` 完成态 `result` 字段同结构的完整对象，另外包含两个方便 UI /
下游消费的聚合字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `speaker_map` | object | `speaker_label → {matched_id, matched_name, similarity, embedding_key}` 的映射，反映 **diarization 模型的声纹匹配结果**，不随人工单段纠错变化；便于前端一次性渲染人名下拉 / 统计 |
| `unique_speakers` | array[string] | 去重后的说话人名列表，从持久化结果里的 `segments[].speaker_name` 重算，反映最新的人工纠错结果 |

与 `GET /api/jobs/{id}` 不同，本端点始终从磁盘读取持久化结果，**进程重启后仍可访问**，
也能反映最新的人工纠错；`/api/jobs/{id}` 优先读内存，内存未命中时才回落到磁盘（见上方注意事项）。

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
  { "id": "spk_example_id", "name": "张三",
    "sample_count": 3,
    "created_at": "2026-04-18T08:06:41.951819",
    "updated_at": "2026-04-18T09:17:02.113207" }
]
```

#### `POST /api/voiceprints/enroll`

> **注意（enroll 幂等性）**：`add_speaker` 按 `name` 自动去重——同名的二次 enroll 会把新 embedding 合并到已有记录，**不会**再产生重复条目。
>
> `speaker_id` 只在你明确要更新那条已有声纹时才传。如果传入的 `speaker_id` 格式合法
> 但库里不存在，接口**不会**返回 404，而是回落到创建 / 同名去重路径。

表单字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `tr_id` | ✅ | 任务 id，对应 `result.id` |
| `speaker_label` | ✅ | **必须**是 `SPEAKER_XX` 这种原始标签，不是 `speaker_name` |
| `speaker_name` | ✅ | 展示用的人名，例如 "张三" |
| `speaker_id` | ❌ | 显式更新目标。若该 id 存在，接口更新那条已有声纹并返回 `action: "updated"`；若省略，或该 id 格式合法但不存在，则走创建路径，而创建路径仍可能被 `add_speaker()` 的同名去重合并到已有记录。格式须符合 `^spk_[A-Za-z0-9_-]{1,64}$`（例如 `spk_example_id`）；不符合格式时返回 422。 |

响应：

```json
{ "action": "created | updated", "speaker_id": "spk_..." }
```

示例：

```bash
curl -X POST http://localhost:8780/api/voiceprints/enroll \
     -H "Authorization: Bearer $API_KEY" \
     -F "tr_id=tr_example_id" \
     -F "speaker_label=SPEAKER_00" \
     -F "speaker_name=张三"
```

#### `POST /api/voiceprints/rebuild-cohort`

从所有已处理的转录中重新构建 AS-norm 评分的上界矩阵（impostor cohort）。
0.7.1 仍支持手动触发，但也新增了自动加载与后台自动刷新。

响应：

```json
{ "cohort_size": 313, "skipped": 2, "saved_to": "/data/transcriptions/asnorm_cohort.npy" }
```

`skipped` — 无法加载 embedding 文件（`.npy` 损坏或缺失）的转录数量。

**cohort 生命周期与行为**：

| cohort 规模 | identify 走的路径 | 有效阈值 |
| --- | --- | --- |
| 0（全新安装 / 无已有转录） | raw cosine | 基础 0.75 + 自适应放松，绝对下限 0.60 |
| 1–9（不足 10 条） | raw cosine（`score()` fallback） | 同上 |
| ≥ 10 | AS-norm 归一化分数 | 约 0.5（相对 impostor 分布，忽略 `VOICEPRINT_THRESHOLD`） |

**启动行为**：

- 如果 `data/transcriptions/asnorm_cohort.npy` 已存在，服务启动时会直接加载该文件。
- 否则启动时会扫描持久化转录结果 / `emb_*.npy` 文件现场重建 cohort，并把结果保存回
  上述路径。

**刷新时机**：每次 enroll / update 都会增加 generation 计数。后台守护线程
`cohort-rebuild` 每 60 秒唤醒一次，在最近一次 enrollment 至少过去 30 秒后调用
`maybe_rebuild_cohort()`。重建过程有锁保护，因此后台线程与
`POST /api/voiceprints/rebuild-cohort` 不会并发执行同一次重建。**无需手动触发**，
新 embedding 通常会在 enrollment 后约 30-90 秒内进入 AS-norm 评分。
`POST /api/voiceprints/rebuild-cohort` 仍可用于立即强制重建。

#### `PUT /api/voiceprints/{id}/name`

表单 `name=新名字`，只改显示名，不动 embedding。

#### `DELETE /api/voiceprints/{id}`

从库里永久删除。被删掉的人后续录音就不会再被匹配上。

### `PUT /api/transcriptions/{tr_id}/segments/{seg_id}/speaker`

手工纠正单条 segment 的说话人归属。

表单字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `speaker_name` | ✅ | 新的说话人显示名 |
| `speaker_id` | ❌ | 已登记声纹的 ID（格式：`^spk_[A-Za-z0-9_-]{1,64}$`）；省略时清除该 segment 原有的 `speaker_id` |

行为说明：

- **只更新目标 segment**，其他 segment 不受影响。
- `speaker_map` **不会被修改**——它记录分离模型的声纹匹配结果，不随人工纠错变化。
- `unique_speakers` 在编辑后从全部 segment 重新计算，保持与当前内容一致。
- 省略 `speaker_id` 时，目标 segment 原有的 `speaker_id` 会被显式置为 `null`（防止过时声纹 ID 残留）。

错误：

- `422` — `speaker_id` 格式非法（不匹配 `^spk_[A-Za-z0-9_-]{1,64}$`）
- `404` — `speaker_id` 在声纹库中不存在
- `404` — `tr_id` 对应的转录不存在
- `404` — `seg_id` 在该转录中不存在

## 错误返回

| 状态码 | 原因 |
| --- | --- |
| 400 | 请求字段缺失或格式错误；job_id 格式非法（`^tr_[A-Za-z0-9_-]{1,64}$`）/ speaker_label 非法字符 / 路径穿越检测 |
| 422 | 字段值类型或取值校验失败；`speaker_id` 格式不符合 `^spk_[A-Za-z0-9_-]{1,64}$`；`no_repeat_ngram_size` 传入非整数 |
| 401 | 缺 API key / key 不对 |
| 404 | tr_id / speaker_id / embedding 不存在 |
| 413 | 上传超过 `MAX_UPLOAD_BYTES`（默认 2 GiB），详见 `/api/transcribe` |
| 503 | 初始 `queued` 状态落盘失败，或后台转录线程启动失败 |
| 500 | 服务端异常（看 `docker logs voscript`） |
| 504 | ffmpeg 转码超时（超过 `FFMPEG_TIMEOUT_SEC`，默认 1800 秒） |

错误体结构：

```json
{ "detail": "..." }
```

## 与 BetterAINote 的对应关系

| BetterAINote 代码 | 调用的接口 |
| --- | --- |
| `submitVoiceTranscribeJob` | `POST /api/transcribe` |
| `pollVoiceTranscribeJob` | `GET /api/jobs/{id}` |
| `VoiceTranscribeClient.listVoiceprints` | `GET /api/voiceprints` |
| `VoiceTranscribeClient.enrollVoiceprint` | `POST /api/voiceprints/enroll` |
| `VoiceTranscribeClient.renameVoiceprint` | `PUT /api/voiceprints/{id}/name` |
| `VoiceTranscribeClient.deleteVoiceprint` | `DELETE /api/voiceprints/{id}` |

源码位置见 [BetterAINote 仓库](https://github.com/MapleEve/BetterAINote)下的
`src/lib/transcription/providers/voice-transcribe-provider.ts` 和
`src/lib/voice-transcribe/client.ts`。
