# 给 AI 使用的接入指南

**简体中文** | [English](./ai-usage.en.md)

> 这篇文档是写给 **AI agent / LLM** 看的。给人看的文档请看
> [`quickstart.zh.md`](./quickstart.zh.md) 和 [`api.zh.md`](./api.zh.md)。
>
> 如果你是 AI，请把下面的内容整个读完再调接口。

## 你是谁，这个服务是什么

你是一个需要把音频转成带说话人名字的文字的 AI agent。这个服务
(`voscript`) 就是帮你做这件事的**有状态后端**，它：

1. 接收音频 → 跑 whisper 转录 + pyannote 说话人分离 + WeSpeaker ResNet34 声纹提取
2. 维护一个**持久化声纹库**，下次同一个人说话时自动把 `SPEAKER_XX` 标签
   升级成真实姓名
3. 让你在任务完成后**把 `SPEAKER_XX` 正式登记为某个人**，从此以后这个人
   的声音都会被自动识别

## 核心事实（最重要，务必记住）

1. **处理是异步的**。`POST /api/transcribe` 只返回一个 job id，你**必须**
   后续轮询 `/api/jobs/{id}` 直到 `status == "completed"` 或 `"failed"`。
   不要 sleep 等待，要 poll。
2. **短音频也可能要几十秒**。首次启动还要额外加上模型加载时间（可能 2+ 分钟）。
   不要因为一次 poll 没完成就判定失败。
3. **鉴权必须走 header，不是 query string**：
   ```
   Authorization: Bearer <API_KEY>
   或
   X-API-Key: <API_KEY>
   ```
4. **登记声纹时要用 `speaker_label`（原始 `SPEAKER_XX`），不是 `speaker_name`
   （显示名）**。这是最容易踩的坑：当服务自动匹配到已有声纹时，
   `speaker_name` 会变成"张三"，但 `speaker_label` 永远是 `SPEAKER_00`。
   拿 `speaker_name` 去 enroll 必然返回 404。
5. **声纹匹配阈值是自适应的**。基础阈值为 `VOICEPRINT_THRESHOLD`（默认 0.75），但每位
   说话人的实际阈值会根据已登记样本的余弦方差自动放松：单样本有效阈值约 0.70，
   样本方差大时进一步放宽，绝对下限 0.60。无论哪种模式，`speaker_id` 非 `null` 均
   表示通过了阈值，所以 `similarity` **不是**“>= 0.75 就一定匹配”的固定字段。

   AS-norm cohort 生命周期（重要）：
   - **全新安装（零转录）**：cohort size=0，`_asnorm=None`，`identify` 走 raw cosine +
     0.75 基础阈值 + 自适应放松，**不走 AS-norm**。
   - **cohort 规模 < 10**：`ASNormScorer.score()` 返回 raw cosine 而非真正的 AS-norm
     z-score（fallback 路径），阈值行为等同于 raw cosine 模式。
   - **cohort 规模 ≥ 10**：启用真正的 AS-norm，归一化分数阈值会围绕 0.5
     操作点按样本数自适应。单样本候选默认更严格（至少 0.60），稳定多样本候选接近
     基准值；如果 top-1 和 top-2 太接近，会保留为未命名供人工复核。
   - **启动路径**：如果 `data/transcriptions/asnorm_cohort.npy` 已存在，服务启动时会
     直接加载；否则启动时会扫描持久化转录结果 / `emb_*.npy` 文件，现场重建 cohort 并
     保存回该路径。
   - **刷新时机**：enroll / update 会推进 generation 计数。后台守护线程
     `cohort-rebuild` 每 60 秒唤醒一次，在最近一次 enrollment 至少过去 30 秒后自动
     触发重建。通常无需手动触发；新 embedding 一般会在 enrollment 后约 30-90 秒内
     进入匹配路径；只有 cohort 达到 10 条及以上时才进入完整 AS-norm 评分，否则仍是
     raw cosine fallback。`POST /api/voiceprints/rebuild-cohort` 仍可用于强制立即重建。
6. **省略 `language` 字段会触发自动检测**。Whisper 自行判断语言，服务同时注入
   `initial_prompt` 引导解码器输出简体中文（适用于普通话音频）。结果中
   `params.language` 会显示为 `"auto"`，而不是具体语言代码。显式传入 `language=zh`
   或 `language=en` 则按指定语言处理，行为与之前完全一致。
7. **重复提交相同文件时会命中去重**。服务对每份上传文件计算 SHA256；去重有两种结果：
   - 历史完成态命中 → `{"id": "...", "status": "completed", "deduplicated": true}`
   - 并发 in-flight 命中 → `{"id": "...", "status": "queued", "deduplicated": true}`
   两种情况都表示这次请求没有新启动 worker。拿到返回的 `id` 后正常轮询即可。
8. **`POST /api/voiceprints/enroll` 里的 `speaker_id` 是显式更新目标，不是严格的
   “传了就必须存在”的开关。** 如果该 `speaker_id` 存在，接口会更新那条已有声纹；
   如果省略，或该 id 格式合法但不存在，则会走创建路径，而创建路径仍可能因同名去重
   合并到已有记录。

## 推荐调用流程

```
[你有音频]
     │
     ▼
POST /api/transcribe           （拿到 job_id）
     │
     ▼
GET /api/jobs/{job_id}          （每 2~5 秒一次，直到 completed/failed）
     │
     ▼
解析 result.segments
     │
     ├── 用户已经在这条录音上给某个 SPEAKER_XX 贴了真名？
     │       └── POST /api/voiceprints/enroll   （登记 / 更新）
     │
     └── 需要导出给下游？
             └── GET /api/export/{tr_id}?format=srt|txt|json
```

## 伪代码模板

```python
import time, requests

BASE = "http://host:8780"
KEY = "<你的 API key>"
H = {"Authorization": f"Bearer {KEY}"}

# 1. 提交
with open("meeting.wav", "rb") as f:
    job = requests.post(
        f"{BASE}/api/transcribe",
        headers=H,
        files={"file": f},
        data={
            # "language": "zh",  # 可选；省略则自动检测（普通话音频输出简体中文）
            "max_speakers": "4",
            # optional: "denoise_model": "deepfilternet",
        },
    ).json()

job_id = job["id"]

# 2. 轮询
while True:
    r = requests.get(f"{BASE}/api/jobs/{job_id}", headers=H).json()
    if r["status"] == "completed":
        result = r["result"]
        break
    if r["status"] == "failed":
        raise RuntimeError(r.get("error", "unknown"))
    time.sleep(3)

# 3. 处理结果
for seg in result["segments"]:
    # 展示用 speaker_name，登记要用 speaker_label
    print(f"[{seg['start']:.1f}s] {seg['speaker_name']}: {seg['text']}")

# 4. 登记（假设用户告诉你 SPEAKER_00 是"张三"）
requests.post(
    f"{BASE}/api/voiceprints/enroll",
    headers=H,
    data={
        "tr_id": result["id"],
        "speaker_label": "SPEAKER_00",  # 原始标签！
        "speaker_name": "张三",
    },
).raise_for_status()
```

## 什么时候该登记声纹

**只在用户明确告诉你某个 `SPEAKER_XX` 是谁的时候**。不要自己猜。

推荐的触发时机：
- 用户说"SPEAKER_00 是张三"、"第一个说话的是老李"之类的
- 用户纠正了一段错误归属（比如说"这段其实是 Alice 说的"）
- 用户在 UI 里主动点了"登记"

**不要**主动发起登记，也不要把 `speaker_name` 当作用户确认的姓名去登记——
那可能只是上一次自动匹配的结果。

## 常见错误和怎么处理

| 现象 | 含义 | 怎么办 |
| --- | --- | --- |
| `401 Unauthorized` | key 没带或不对 | 检查 `Authorization` header |
| `404 Embedding not found for this speaker label` | enroll 时 `speaker_label` 用错了（传了显示名） | 改用原始 `SPEAKER_XX` |
| `deduplicated: true` 且 `status: "queued"` | 命中了并发中的重复提交，另一条请求已经拥有这条 job | 正常轮询返回的 id |
| 轮询一直 `transcribing` | 音频较长或首次加载模型 | 继续轮询，别超过 20 分钟 |
| `status = failed, error = "..."` | 容器内异常 | 直接把 `error` 字段报给用户，必要时看 `docker logs` |
| `503 Failed to persist job state...` / `503 Failed to start background transcription...` | 服务没能把 job 可靠地启动起来 | 稍后重试；这次请求没有启动 worker |
| `segments` 是空数组 | 音频静音 / 太短 / 采样率问题 | 告诉用户换一份音频，或确认文件没坏 |

## 不要做的事

- ❌ 不要把 `HF_TOKEN` 或 `API_KEY` 写进代码、日志、prompt 里
- ❌ 不要把 `:8780` 暴露给不信任的客户端
- ❌ 不要直接编辑 `data/voiceprints/voiceprints.db`，用 API 的 delete / rename
- ❌ 不要把一次音频重复提交很多次——每次都会跑一遍 whisper，浪费 GPU
- ❌ 不要把 `speaker_id` 和 `speaker_label` 搞混：
  - `speaker_label` = `SPEAKER_00`，录音内的本地标签
  - `speaker_id` = `spk_xxxx`，全局声纹库 id
- ❌ 不要重复提交同一份音频文件期望重新转录 — 服务端 SHA256 去重会直接返回已有完成结果，或把你挂到已经排队中的旧 job 上（`deduplicated: true`），都不会重跑 Whisper。若确实需要重新转录，先通过 `DELETE /api/transcriptions/{id}` 删除旧记录，再重新提交。

## 建议

- 如果你同时负责多条音频：先都 submit 完拿到所有 job_id，再并行 poll，
  不要一条完成再提交下一条。
- 如果你需要把转录结果注入后续 prompt，建议走
  `GET /api/export/{tr_id}?format=txt`，它已经把 segment 合并成按说话人分行
  的纯文本。
- 如果同一个说话人声纹已经登记过，新的一次录音里他依然会出现在 `speaker_label`
  为 `SPEAKER_XX` 下，但 `speaker_name` 会是已登记的名字。这不是 bug。
- 完成态结果里的 `speaker_map` 可能合法地是 `{}`，例如没有任何可持久化的 speaker
  embedding。`unique_speakers` 仍然会存在，并且来自 `segments[].speaker_name`。

## AI 代理技能包

如果你正在将 VoScript 集成到 AI 代理工作流（Claude、Codex、Trae、Hermes、OpenClaw 等），
可以直接使用官方技能包：

**[github.com/MapleEve/voscript-skills](https://github.com/MapleEve/voscript-skills)**

技能包包含：
- `SKILL.md`：全部 11 个工作流的完整文档（配置、提交、轮询、导出、声纹管理）
- `scripts/`：11 个可直接执行的 Python 辅助脚本（仅需 stdlib + `requests`）
- `references/`：任务状态机、声纹指南、AS-norm 评分说明、导出格式文档

## 相关文档

- 详细接口合同 → [`api.zh.md`](./api.zh.md)
- 部署和排障 → [`quickstart.zh.md`](./quickstart.zh.md)
- 安全注意事项 → [`security.zh.md`](./security.zh.md)
- AI 代理技能包 → [voscript-skills](https://github.com/MapleEve/voscript-skills)
