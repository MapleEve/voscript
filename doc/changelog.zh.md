# 更新日志

**简体中文** | [English](./changelog.en.md)

## 0.5.0 — AS-norm 声纹评分 + 片段级 sidetalk 分离 (2026-04-20)

### AS-norm 声纹评分

- 引入 `ASNormScorer`（`voiceprint_db.py`），对原始余弦分用 impostor cohort 做自适应评分归一化（AS-norm）
- AS-norm 消除说话人依赖的基准偏差，在同等精度下相对降低 EER 15–30%
- 服务启动时自动从已有转录的 embedding（`emb_*.npy`）构建 cohort，保存为 `data/transcriptions/asnorm_cohort.npy`；首次构建失败时静默降级为原始余弦
- AS-norm 启用后有效阈值固定为 `0.5`（经 cohort 归一化后的操作点）；未启用时仍走 0.4.0 的自适应余弦阈值
- 新增 `POST /api/voiceprints/rebuild-cohort` — 手动重建 impostor cohort

### `overlap_intervals` 持久化

- `POST /api/transcriptions/{tr_id}/analyze-overlap` 现在同时将 `overlap_intervals`（`[[start, end], ...]` 列表）写入 `result.json`
- 后续 `/separate-segments` 可直接读取缓存区间，无需重跑 OSD

### 片段级 MossFormer2 sidetalk 分离

- 新增 `POST /api/transcriptions/{tr_id}/separate-segments`
- 全文件分离（`/separate`）存在主导说话人坍塌问题：当 Maple 在整段录音中占主导时，MossFormer2 的第二轨退化为残留噪声
- 片段级方案：仅对 OSD 检测到的重叠窗口（双方说话人同时活跃）运行 MossFormer2，能量均衡 → 分离质量显著改善
- 57 条 PLAUD 录音实测：23/23 个重叠窗口成功返回双轨，多条检测到有意义的 sidetalk 内容

### 说话人分离引擎 Bug 修复

- **MossFormer2 输出路径修复**：正确路径为 `MossFormer2_SS_16K/{stem}_s{i}.wav`（原为 `{stem}_MossFormer2_SS_16K_spk{i}.wav`）
- **多 GPU 张量散落修复**：monkey-patch `SpeechModel.get_free_gpu` 使其始终返回配置设备索引，避免在第二个片段处发生 cuda:0 vs cuda:1 张量设备不一致错误
- **OSD 初始化修复**：在 `instantiate()` 之后调用 `initialize()`，修复 pyannote 3.1.1 的 `_binarize` 属性缺失错误

### 兼容性

- 所有已有接口行为不变
- `overlap_intervals`、`overlap_segments` 是 `result.json` 的新增顶层字段；老客户端忽略
- 未构建 cohort 时（零 transcription 环境），声纹识别自动回退到 0.4.0 的余弦逻辑

## 0.4.0 — 自适应声纹阈值 + 降噪 SNR 门限 + OSD (2026-04-19)

### 自适应声纹阈值

- `VOICEPRINT_THRESHOLD` 现在是可配环境变量（默认 0.75），作为基础阈值
- 每位说话人的实际阈值根据已登记样本的余弦方差自动放松：1 个样本固定 -0.05，2+ 个样本按 `min(3×std, 0.10)` 放松，绝对下限 0.60
- 10 条 PLAUD Pin 录音 A/B 测试：召回率从 50% 提升到 70%，零误识别
- 新增环境变量 `VOICEPRINT_THRESHOLD`（默认 `0.75`）

### 降噪处理 + SNR 门限

- 新增 `DENOISE_MODEL` 环境变量：`none`（默认）| `deepfilternet` | `noisereduce`
- 新增 `DENOISE_SNR_THRESHOLD` 环境变量（默认 `10.0` dB）：SNR 达到或超过此值时跳过降噪，避免对高质量录音做不必要处理
- 任务流水线新增 `denoising` 状态（converting 之后、transcribing 之前，仅在启用降噪时出现）
- `POST /api/transcribe` 新增 `denoise_model`（字符串）和 `snr_threshold`（浮点数）两个可选字段，支持单次请求级别覆盖
- DeepFilterNet 对高 SNR 录音（>10 dB）有害：段数增加 100-145%，代理 CER 劣化 20-91%。SNR 门限可自动保护干净音频
- CUDA OOM 修复：DeepFilterNet 处理长音频后（~15 GB PyTorch CUDA 保留），在调用 Whisper 前执行 `torch.cuda.empty_cache()` + `gc.collect()` 解决 ctranslate2 的 OOM 问题

### 重叠语音检测 OSD

- `POST /api/transcribe` 新增 `osd`（bool，默认 `false`）字段
- 启用时，每个 segment 包含 `has_overlap: bool` 字段，标记该片段中点是否存在多人同时说话
- 底层使用 `pyannote/segmentation-3.0`（与分离流水线共享，无需额外下载）
- 10 条真实会议录音平均 9.7% 片段重叠率，在正常对话范围内

### 结果结构变化

- 每个已完成任务的结果新增顶层 `params` 对象，记录本次转录实际使用的配置（语言、降噪模型、SNR 门限、声纹阈值、OSD 开关、说话人数约束）
- 移除 `/api/config` 全局接口——配置随任务结果一起返回，接口更独立

### 语言自动检测（不再默认 zh）

- `POST /api/transcribe` 的 `language` 字段默认值从 `"zh"` 改为空（自动检测）
- 省略 `language` 时，Whisper 自行判断语言，服务注入 `initial_prompt` 引导解码器输出简体中文，适用于普通话音频
- 显式传入 `language=zh` 或 `language=en` 行为不变
- 结果中 `params.language` 在自动检测时显示 `"auto"`，而非具体语言代码

### 文件哈希去重

- 每次上传文件时计算 SHA256；若相同文件已有完成的任务，`POST /api/transcribe` 直接返回该任务结果，不再重跑 Whisper
- 命中去重时响应包含 `deduplicated: true` 字段：`{ "id": "tr_existing_id", "status": "completed", "deduplicated": true }`
- 客户端可用返回的 `id` 正常轮询或导出，无需任何特殊处理

### 兼容性

- HTTP 合同完全兼容：`has_overlap` 和 `params` 都是新增字段，老客户端直接忽略
- `deduplicated` 是 `POST /api/transcribe` 的新增可选字段，老客户端忽略即可
- 除非启用降噪，否则对 PLAUD Pin 等高质量设备的录音处理方式不变
- 建议配置：`DENOISE_MODEL=none`（PLAUD Pin / 高质量麦克风）；`DENOISE_MODEL=deepfilternet` + `DENOISE_SNR_THRESHOLD=10.0`（噪声环境）

## 0.3.0 — WhisperX 强制对齐 + sqlite 声纹库 + WeSpeaker 升级 (2026-04-18)

三个独立的核心组件升级，一起发布：

### WhisperX 替换原来的自研对齐
- `app/pipeline.py` 的 `transcribe` + `align_segments` 改用 `whisperx` 的 forced alignment 管线
- 输出里每个 segment 多一个可选 `words: [{word, start, end, score}, ...]` 字段——**词级时间戳**
- 内部还是 CTranslate2 + faster-whisper，所以原来本地 `/models/faster-whisper-<size>` 缓存依然生效，冷启动不回退到 HF 拉模型
- 对齐模型（wav2vec2 一族）首次启动会下一个（缓存在 `/cache`）。对中文音频有时不可用——失败时自动降级回没有 `words[]` 的 segment 级结果，不整个失败
- 版本 pin：`whisperx==3.1.6`（3.1.x 是唯一和我们 `torch==2.4.1` + `pyannote==3.1.1` 兼容的 WhisperX 系列；这些 release 在 PyPI 是 yanked 状态，但 pip 在 `requirements.txt` 里显式写版本号时仍会装）

### 声纹库底层换成 sqlite + sqlite-vec
- `app/voiceprint_db.py` 不再用 `index.json + *.npy` 的文件格式
- 换成单个 `voiceprints.db`（sqlite 3）+ sqlite-vec 的 `vec0` 虚拟表做向量索引
- 查询复杂度从 O(N) 线性扫 `.npy` 降到 vec0 的 top-k 近邻检索——几百上千个声纹毫无压力
- 写入走 sqlite 原生事务，WAL 模式，不再需要我们自己手写 `os.replace` 原子替换
- 并发安全：线程级 `RLock` + sqlite WAL 组合
- **自动迁移**：第一次启动如果发现老的 `index.json` + `.npy` 文件，会一次性导入到 sqlite，然后把 `index.json` 改名成 `index.json.migrated.bak`。`.npy` 文件保留不动（可回滚）
- sqlite-vec 加载失败时自动降级到 Python 侧余弦全扫（仍然可用，只是慢）

### WeSpeaker ResNet34 替换 ECAPA-TDNN
- `pyannote/wespeaker-voxceleb-resnet34-LM` 替换 speechbrain 的 ECAPA
- 依赖简化：不再需要 `speechbrain` 包，WeSpeaker 的包装类本来就在 `pyannote.audio` 里
- embedding 维度从 192 变成 ~256
- **破坏性**：新的 embedding 空间和 ECAPA 的不兼容，**老版本的声纹需要重新登记**——同一个人用 0.2.x 的 ECAPA 生成的向量扔到 0.3.0 的 WeSpeaker 余弦对比里会拿到近乎随机的相似度
- **HF 门禁**：这个模型也是 gated 的，首次使用前要去 <https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM> 点"Agree and access repository"，和之前的 speaker-diarization-3.1 / segmentation-3.0 同一套流程
- 音频切分逻辑不变：每个说话人 ≥1s 的片段里挑最长的 10 段，做均值池化

### 升级的破坏性变化
- 容器镜像变大：WhisperX 额外带 wav2vec2 依赖 + 运行时下载的对齐模型（再加 ~1 GB 缓存）
- **老声纹必须全部重新登记**——升级后直接删 `data/voiceprints/voiceprints.db`（或者一个个 `DELETE /api/voiceprints/{spk_xxx}`）然后重新 enroll
- HF 需要多接受一个 gated 模型的条款（WeSpeaker）

### 保持兼容
- HTTP 合同完全不变（`segments[i].words` 是新增可选字段，老客户端忽略）
- 环境变量没新增
- 容器 `voscript` 名字、端口、数据目录结构、非 root 用户、API_KEY 鉴权全部不变

## 0.2.1 — 改名为 voscript (2026-04-18)

解耦和 OpenPlaud(Maple) 的绑定——服务本身可以独立使用，因此改名：

- **仓库**：`MapleEve/openplaud-voice-transcribe` → `MapleEve/voscript`
  （GitHub 老 URL 自动 301 重定向，老 clone 不会失效）
- **Docker 服务/容器名**：`voice-transcribe` → `voscript`
  （`docker logs voscript`、`docker exec voscript …`）
- **镜像名**：compose 自动产生的 `voscript-voscript:latest`
- **README/文档**：重写定位为独立转录服务，OpenPlaud(Maple) 改为
  "一个已知的接入方"而不是身份绑定
- **HTTP 合同、文件布局、环境变量、数据目录结构 全部不变**——现有调用方
  零修改

## 0.2.0 — 红队审计后的强化版 (2026-04-18)

在真实 Plaud 音频的端到端测试与独立渗透测试的基础上做了一轮全面硬化。

### 安全硬化
- **容器不再以 root 运行**：`app` 用户（默认 uid/gid 1000），通过 `APP_UID`/
  `APP_GID` 环境变量覆盖。任何容器内的代码执行漏洞都只能拿到 app 的权限，
  不能直接读取宿主机上其他 root 所有的文件。
- **上传大小限制**：`/api/transcribe` 现在做分块流式读取，累计超过
  `MAX_UPLOAD_BYTES`（默认 2 GiB，可配）直接返回 `HTTP 413` 并删掉半截
  文件，防止磁盘耗尽 DoS。
- **上传文件名清洗**：`PurePosixPath(filename).name` 把 `../../etc/passwd.wav`
  之类的路径片段剥掉，只保留最末一段再拼到上传目录。
- **ffmpeg argv 硬化**：在输入路径前插入 `--`，关掉选项解析，避免攻击者用
  `-y.mp4` 这类文件名注入 ffmpeg 标志。
- **鉴权采用常量时间比较**：`hmac.compare_digest` 代替 `!=`，消除理论上的
  时序侧信道。
- **`/docs`、`/redoc`、`/openapi.json` 改成精确匹配**：之前 `startswith("/docs")`
  会让 `/docsXYZ` 绕过鉴权；现在这些路径属于精确公开集合，只有 `/static/`
  仍保留前缀匹配。
- **`VoiceprintDB` 并发安全 + 原子化写入**：所有 mutation 在 `threading.Lock`
  里执行；`index.json` 和 `.npy` 通过 `tempfile + os.replace` 原子写入，防止
  崩溃写坏索引。
- **`np.load(..., allow_pickle=False)`**：默认关闭 numpy 反序列化 pickle 的
  路径，避免一条类似 `torch.load` 的 RCE 路径。

### 功能与配置
- 新增环境变量 `MAX_UPLOAD_BYTES`（默认 `2147483648`，即 2 GiB），可在
  `.env` / compose 中调整。
- 新增环境变量 `APP_UID` / `APP_GID`（默认 1000），给出宿主目录所有者不是
  1000 的场景留了口子。
- HF 模型缓存从容器里的 `/root/.cache/huggingface` 迁移到 `/cache`，
  `HF_HOME` / `HUGGINGFACE_HUB_CACHE` / `TORCH_HOME` / `XDG_CACHE_HOME`
  全部重指向，让非 root 用户也能写入。
- `docker-compose.yml` 同时挂载 `${MODEL_CACHE_DIR}` 到 `/cache`（读写、HF
  cache）和 `/models:ro`（只读，供 `pipeline.py` 走 local-first 解析），冷
  启动不再依赖 HuggingFace 网络。

### Bug 修复
- `_convert_to_wav` 改成直接调用 `ffmpeg` 子进程，替代 `pydub.AudioSegment`。
  修掉了 pydub 对新版 ffmpeg 在 Opus/部分容器场景下输出的 `codec_type`
  字段缺失的解析崩溃（[jiaaro/pydub#638](https://github.com/jiaaro/pydub/issues/638)）。
- `GET /` 现在对浏览器直接访问放行：配了 `API_KEY` 之前浏览器直接访问 `/`
  会 401，UI 不可用。鉴权保护实际落在 `/api/*`，由 UI 发起的 fetch 负责
  带 key。
- `VoiceTranscribeProvider` 在 OpenPlaud(Maple) 侧把 segment 的 `speaker`
  字段改回原始 `speaker_label`（`SPEAKER_XX`），修掉"自动匹配后就没法再
  登记"的断链。

### 破坏性变化
- 容器的 HF 缓存路径从 `/root/.cache/huggingface` 换到 `/cache`。如果你之前
  把宿主目录 mount 到 `/root/.cache/huggingface`，需要更新 compose（或直接
  用本仓库新版 compose，它自动兜底）。
- 上传超过 `MAX_UPLOAD_BYTES` 的请求现在会 413 而不是静默成功。默认 2 GiB
  对绝大多数音频足够。

## 0.1.0 — 首次公开发布

- 首次公开发布 [OpenPlaud(Maple)](https://github.com/MapleEve/openplaud) 的私有转录后端。
- 异步任务流水线：`queued → converting → transcribing → identifying → completed`。
- faster-whisper `large-v3` + pyannote `3.1` + ECAPA-TDNN 声纹提取。
- 持久化声纹库，基于余弦相似度自动匹配。
- 所有 `/api/*` 路由支持可选的 `API_KEY` Bearer 鉴权。
- 可移植的 `docker-compose.yml`（数据/模型路径都通过环境变量配置）。
- 必要的版本 pin，让 `pyannote.audio==3.1.1` 仍可用：
  - `numpy<2`（pyannote 3.1.1 用了 `np.NaN`，numpy 2.x 已移除）。
  - `huggingface_hub<0.24`（保留 pyannote 3.1.1 调用的 `use_auth_token` kwarg）。
