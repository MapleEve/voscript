# openplaud-voice-transcribe

**简体中文** | [English](./README.en.md)

自托管的 GPU 转录服务，带持久化说话人声纹。为 [OpenPlaud](https://github.com/MapleEve/openplaud)
打造的私有后端，也可以独立作为一个 FastAPI 服务使用。

```
音频  ──►  faster-whisper large-v3  （转录）
      ──►  pyannote 3.1             （说话人分离）
      ──►  ECAPA-TDNN               （声纹提取）
      ──►  VoiceprintDB             （与已注册声纹做余弦匹配）
      ──►  带时间戳和已识别说话人姓名的文本
```

## 为什么单独放一个仓库

OpenPlaud 是一个单用户控制面板。把 whisper / pyannote 加载进显存、常驻 GPU、做说话人分离、
维护一套声纹库——这些重活都放在一个私有 HTTP API 后面，这样 OpenPlaud 本体就不用在浏览器
里跑 GPU 模型，也不用把原始声纹数据暴露出去。

这个仓库就是那个私有 API。OpenPlaud 上传音频、轮询任务、把转录结果存到本地数据库，
当用户做声纹登记时会调这里的 voiceprint 接口。

## 功能

- 异步任务流水线（`queued → converting → transcribing → identifying → completed`）
- 中文 + 多语种转录（faster-whisper large-v3）
- 说话人分离（pyannote 3.1）
- 持久化声纹：**一次登记，后续录音自动识别**（余弦相似度 ≥ 0.75 视为命中）
- 稳定 HTTP 合同，OpenPlaud 的
  [`voice-transcribe-provider.ts`](https://github.com/MapleEve/openplaud/blob/main/src/lib/transcription/providers/voice-transcribe-provider.ts)
  和 [`voice-transcribe/client.ts`](https://github.com/MapleEve/openplaud/blob/main/src/lib/voice-transcribe/client.ts)
  可以直接对接
- 所有 `/api/*` 路由支持可选 Bearer / `X-API-Key` 鉴权
- `/` 自带一个轻量 Web UI，方便单独测试

## 运行环境要求

- 带 NVIDIA GPU 的 Linux 主机（whisper large-v3 + pyannote + ECAPA-TDNN，大约吃 9 GB 显存。在 RTX 3090 上实测）
- Docker 24+ 且装了 NVIDIA Container Toolkit
- 一个 HuggingFace access token。需要先在 huggingface 接受两个 gated 模型的条款：
  [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)、
  [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)，
  然后在 <https://huggingface.co/settings/tokens> 生成 token

## 快速开始

```bash
git clone https://github.com/MapleEve/openplaud-voice-transcribe.git
cd openplaud-voice-transcribe

cp .env.example .env
# 编辑 .env —— 至少要填 HF_TOKEN 和 API_KEY

docker compose --env-file .env up -d --build
curl -sf http://localhost:8780/healthz
```

首次启动会下载约 5 GB 的模型权重到 `./models/`（或你在 `MODEL_CACHE_DIR` 指向的目录），
之后重启都走本地缓存。

## HTTP API

所有接口返回 JSON。设了 `API_KEY` 之后，每一个 `/api/*` 请求必须带其中一个 header：

```
Authorization: Bearer <API_KEY>
X-API-Key: <API_KEY>
```

`GET /healthz`、`GET /`、`/static/*` 永远开放。

### 提交转录任务

```
POST /api/transcribe
  multipart/form-data:
    file          音频文件（wav/mp3/m4a/...）
    language      "zh"（默认）、"en"、...
    min_speakers  整数，0 表示自动
    max_speakers  整数，0 表示自动
→ 200 { "id": "tr_YYYYMMDD_HHMMSS_XXXXXX", "status": "queued" }
```

### 轮询任务状态

```
GET /api/jobs/{id}
→ 200 {
    "id": "tr_...",
    "status": "queued" | "converting" | "transcribing" | "identifying"
            | "completed" | "failed",
    "filename": "...",
    "error": "...",                       // 仅在 status = failed 时返回
    "result": {                            // 仅在 status = completed 时返回
      "id": "tr_...",
      "language": "zh",
      "segments": [
        {
          "id": 0,
          "start": 0.0,          // 秒
          "end": 4.32,
          "text": "...",
          "speaker_label": "SPEAKER_00",   // pyannote 的原始标签（做声纹登记要用它）
          "speaker_id":    "spk_..." | null,
          "speaker_name":  "张三" | "SPEAKER_00",
          "similarity":    0.8421
        }
      ]
    }
  }
```

### 声纹管理

```
GET    /api/voiceprints
POST   /api/voiceprints/enroll       (tr_id, speaker_label, speaker_name, [speaker_id])
PUT    /api/voiceprints/{id}/name    (name)
DELETE /api/voiceprints/{id}
```

登记声纹时 `speaker_label` **必须**是任务结果里原始的 `SPEAKER_XX` 标签，不能是显示用的
`speaker_name`——服务端是按 diarization 产出的原始标签来存 embedding 的。

### 导出 / 片段管理

```
GET /api/transcriptions
GET /api/transcriptions/{tr_id}
GET /api/export/{tr_id}?format=srt|txt|json
PUT /api/transcriptions/{tr_id}/segments/{seg_id}/speaker  (speaker_name, [speaker_id])
```

## 和 OpenPlaud 对接

在 OpenPlaud 设置里配：

- **Private transcription base URL**：`http://<主机>:8780`
- **Private transcription API key**：跟 `.env` 里的 `API_KEY` 一致

OpenPlaud 会把每条录音都丢给这个服务，转录结果 + 说话人都落到它自己的本地数据库。
完整用户路径看 OpenPlaud 的 README。

## 数据在主机上的目录结构

```
data/
├── uploads/              # 原始上传，按 job id 分文件
├── transcriptions/
│   └── tr_.../
│       ├── result.json   # 完整转录结果 + 说话人映射
│       └── emb_SPEAKER_XX.npy   # 每个说话人的 embedding（做登记时要读）
└── voiceprints/
    ├── index.json
    ├── spk_xxx_avg.npy
    └── spk_xxx_samples.npy
```

**请务必备份 `data/voiceprints/`**。丢了就得从头重新登记所有人。其他目录只要原始音频
还在就都能重新生成。

## 安全

对外暴露之前请先读 [SECURITY.md](./SECURITY.md)。简要点：

1. **一定要设 `API_KEY`**，不要把 `:8780` 直接挂到公网
2. `.env` 已经 gitignore，`HF_TOKEN` 哪怕怀疑进过日志/镜像也要立刻换
3. 声纹是生物特征数据，`data/voiceprints/` 要按这个级别对待

## 给开发者看

- `app/main.py` 是 FastAPI 入口。鉴权中间件在所有路由之前跑；`/healthz`、`/`、`/static/*`、
  `/docs` 不需要鉴权。
- `app/pipeline.py` 包住 faster-whisper + pyannote + ECAPA-TDNN。HF 下载走
  `use_auth_token`（这是 pyannote 3.1.1 认的 kwarg 名）。
- `app/voiceprint_db.py` 是一个纯 numpy-on-disk 的声纹库，跑的是累加平均 embedding + 余弦相似度。
- `requirements.txt` 里的版本 pin 都是有意义的、不能随便放开：
  - `numpy<2` —— pyannote 3.1.1 用了 `np.NaN`，numpy 2.x 里被删掉了
  - `huggingface_hub<0.24` —— 保留 pyannote 3.1.1 还在调的 `use_auth_token` kwarg

## License

MIT —— 看 [LICENSE](./LICENSE)。
