# 更新日志

**简体中文** | [English](./changelog.en.md)

## Unreleased

_暂无变更。_

## 0.7.5 — GPU 模型空闲卸载与 CI 质量门禁 (2026-04-29)

### Bug 修复

- 修复 faster-whisper CUDA 设备传参：内部仍可用 `cuda:0` / `cuda:1`
  表示 torch 设备，但加载 faster-whisper 时会转换为 `device="cuda"` 与
  对应 `device_index`，避免 `unsupported device cuda:0`。
- 修复 pyannote 本地缓存加载：当已有完整 Hugging Face snapshot 时，说话人分离
  会生成 runtime-localized config，把内嵌 segmentation / embedding 子模型也指向
  本地权重文件；缓存缺失仍回退到 Hub repo id，缺失本地工件会在加载前明确失败。

### 功能

- 新增可选 `MODEL_IDLE_TIMEOUT_SEC`。默认 `180` 秒（3 分钟）会在串行 GPU 运行时
  持续空闲达到该超时后卸载已加载 GPU 模型；显式设为 `0` 可关闭空闲卸载并保持模型常驻。
- 卸载后的下一次 lazy load 会在 CUDA 可用时选择当前可见设备里空闲显存最多的
  GPU；探测失败会安全回退到配置的 `DEVICE`。

### 可靠性

- 空闲卸载 daemon 与转写任务共用 GPU semaphore，并且在拿到 semaphore 后重新读取
  idle 时间戳，避免等待期间有新任务完成时仍按旧判断卸载模型。

### 文档

- 新增 [`configuration.zh.md`](./configuration.zh.md) /
  [`configuration.en.md`](./configuration.en.md)，补齐 v0.7.5 的完整配置与调参说明，
  覆盖服务 env、ASR 已支持与未暴露项、降噪覆盖语义、diarization / alignment、
  embedding、声纹 / AS-norm、结果契约和公开安全的 E2E 验证口径。
- 更新 README、quickstart、API、voiceprint tuning、`.env.example` 和 compose 链接，
  让用户能从公开文档入口找到完整配置参考。
- 收紧降噪和 AS-norm 验证文案：SNR gate 仅适用于 DeepFilterNet，
  `noisereduce` 选择后不按 SNR 跳过；cohort <10 时声纹匹配走 raw cosine fallback，
  完整 AS-norm 验证需要 cohort >=10。
- 在完整配置文档、quickstart、`.env.example` 和 compose 默认值中补充
  `MODEL_IDLE_TIMEOUT_SEC`。

### CI

- 在现有 CI test job 中接入 Codecov 覆盖率与测试结果上传，保持 required check
  名称不变。
- 新增非 required FOSSA 与 Claude Code review workflow；仓库 secret 未配置时会
  明确跳过并通过，不阻断 PR。
- 新增 `REVIEW.md`，让自动代码评审聚焦 VoScript 特有的可靠性、隐私、模型生命周期、
  API 和中英文文档同步风险。

## 0.7.4 — 环境默认值与契约准备 (2026-04-26)

### Bug 修复

- **说话人身份保留补齐 (#8)**：结果 artifact 会保留原始 diarization `speaker_label`，即使多个 cluster 匹配到同一个已登记声纹也不会折叠。展示名仍会自动加序号方便阅读，下游客户端可以继续把 `speaker_label` 当作稳定 cluster key。
- **AS-norm 按样本数校准**：AS-norm 激活后不再固定使用 `0.5` 阈值，而是使用 AS-norm
  专用的动态阈值，按候选声纹的登记样本数和 sample_spread 调整。单样本候选会比 `0.5`
  操作点更严格，因此 `0.5713` 这类弱分数不会再自动命名。
- **AS-norm 歧义保护**：自动命名前要求 top-1 与 top-2 保持最小 AS-norm margin。
  AS-norm 激活时会按 normalized score 重新排序候选，再基于 normalized top-1/top-2
  做阈值和 margin 判定；候选过近时保留为未命名，交给人工复核。
- **AS-norm cohort 保护**：启动时 direct-load 已持久化的 `asnorm_cohort.npy`
  后会把自动重建状态标记为 clean；后台自动重建不会用更少的转录 embedding 覆盖更大的
  已持久化 / 内存 cohort，避免清空转录结果后 AS-norm cohort 自动退化。
- **降噪 env/API 优先级修复**：`POST /api/transcribe` 省略 `denoise_model` 时现在使用服务端 `DENOISE_MODEL`；只有显式传 `denoise_model=none` 才会针对本次请求关闭降噪。显式 `snr_threshold` 仍优先覆盖 DeepFilterNet 的 `DENOISE_SNR_THRESHOLD` gate；`noisereduce` 不受该 gate 控制。

### 配置

- 在 `.env.example`、compose 和文档中统一公开 v0.7.4 默认值：`DENOISE_MODEL=none`、`DENOISE_SNR_THRESHOLD=10.0`、`VOICEPRINT_THRESHOLD=0.75`、`PYANNOTE_MIN_DURATION_OFF=0.5`、`MIN_EMBED_DURATION=1.5`、`MAX_EMBED_DURATION=10.0`。
- 将上述默认值收口到 `app/config.py`，pyannote off-turn 与 embedding 窗口调参不再由 provider 直接读取环境变量。

### 文档

- 新增 [`voiceprint-tuning.zh.md`](./voiceprint-tuning.zh.md) /
  [`voiceprint-tuning.en.md`](./voiceprint-tuning.en.md)，整理声纹相关环境变量、API
  参数、当前硬编码的 raw / AS-norm 阈值默认、cohort / top_n / margin 行为和调参建议。
- 更新 README、quickstart、API 参考和 voiceprint tuning 文档，补充 v0.7.4 公开 env 默认值、结果契约锚点和降噪覆盖语义。
- 修正 compose 中指向不存在 `docs/configuration.md` 的注释，改为指向已提交的声纹调参参考。

### 测试

- 新增本地契约测试覆盖 `status=completed`、原始 `segments[].speaker_label`、可选 `alignment` 元数据，以及降噪 env/API 优先级。
- 新增 AS-norm cohort 回归测试，覆盖 direct-load 后自动 tick 不重建、清空 / 小样本转录源不缩小既有 cohort，以及显式 rebuild 仍按手动操作执行。

## 0.7.3 — 运行时稳定性热修复 (2026-04-25)

### Bug 修复

- **Diarization 冷启动更稳**：pyannote 说话人分离和 WeSpeaker 声纹模型加载现在会先检查已有 Hugging Face snapshot 缓存，缓存完整时直接从本地加载，避免已部署环境重复联网下载。
- **默认绕开 Xet/CAS 下载链路**：运行时在导入 Hugging Face Hub 客户端前默认设置 `HF_HUB_DISABLE_XET=1`，除非运维显式覆盖。这样可避开部分远端环境中 hf-xet/CAS bridge 触发的 TLS EOF 失败。
- **更快回退到本地缓存**：Docker 和 compose 默认加入 `HF_HUB_ETAG_TIMEOUT=3`，网络慢或不稳定时 Hugging Face Hub 的元数据检查会更快回退到本地缓存。
- **ASR 重复幻觉过滤**：转写提示语污染导致的重复段（例如“请以简体中文输出”被反复识别为正文）会在进入 diarization、标点和 artifact 前被过滤。
- **中文 alignment 前置条件修复**：Docker base image 升级到 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`，满足 transformers 新安全检查对 WhisperX 默认中文 PyTorch `.bin` alignment 权重的加载要求。
- **中文 alignment 默认继续启用**：不再默认把 `zh` 放进 `WHISPERX_ALIGN_DISABLED_LANGUAGES`。该变量只作为明确的临时运营降级开关使用。
- **alignment 策略可配置**：新增 `WHISPERX_ALIGN_DISABLED_LANGUAGES`、`WHISPERX_ALIGN_MODEL_MAP`、`WHISPERX_ALIGN_MODEL_DIR`、`WHISPERX_ALIGN_CACHE_ONLY`。
- **pyannote checkpoint 安全加载**：针对 PyTorch 2.6 的 weights-only checkpoint 机制，pyannote diarization 模型加载只在 `from_pretrained` 调用期间临时信任必要的 checkpoint 元数据类型（`TorchVersion`、`Problem`、`Specifications`、`Resolution`），不使用全局 allowlist，也不关闭 weights-only 安全检查。
- **脱敏失败元数据**：完成结果可包含 `alignment` 对象，记录 `succeeded`、`skipped` 或 `failed`。torch 安全限制会分类为 `reason=torch_version_blocked`，日志不再输出可能包含本地路径或凭据的原始 alignment 异常文本。

### 部署

- 需要重建容器镜像以使用 torch 2.6 base image。已有模型缓存卷保持兼容。

### 兼容性

- 既有转写结果保持兼容。`alignment` 是新增字段，`words[]` 原本就是可选字段。

## 0.7.2 — 架构基础铺垫 + 稳定性加固 (2026-04-24)

### 架构

- **pipeline / provider / infra 分层**：原先平铺的 pipeline、任务运行时、音频处理和声纹数据库模块已拆到 `pipeline/`、`providers/`、`infra/`、`application/`、`voiceprints/` 边界下。对外 HTTP API 保持兼容，内部目录结构对齐后续 stage/provider 架构。
- **Canonical pipeline slot**：代码中已形成 normalize、enhance、ASR、diarization、speaker embedding、voiceprint matching、postprocess、artifacts 等稳定 stage/provider 边界。这些内部扩展点在 0.7.2 中不作为公开 API 合约。
- **发布卫生修复**：FastAPI metadata 现在返回 `0.7.2`；Docker healthcheck 不再依赖镜像里不存在的 `curl`，改用 Python 标准库访问 `/healthz`。

### 稳定性与验证

- **内部 live 验证**：`feat/v0.7.2` 候选已通过 live API 套件、overlap bench 和内部验证流程。
- **AS-norm 入库专项**：使用内部验证样本完成 enroll、cohort rebuild，并用另一段 probe 音频命中新入库 speaker。该验证只证明声纹 API、rebuild 入口和 raw cosine fallback 可用；没有可信的 >=10 cohort 证据时，不能声称 probe 已完整走 AS-norm 评分路径。
- **安全与失败路径加固**：新增测试覆盖损坏结果文件、partial upload 清理、导出名注入、状态写入失败、runner 失败路径和 in-flight dedup 清理。

### 已知取舍

- **GPU 串行范围**：v0.7.2 在当前架构下仍将完整转录 pipeline 放在既有 GPU 串行保护内。这个选择优先保证大重构后的稳定性，但会让部分 CPU/IO 工作也被串行化，吞吐可能下降。
- **任务重启语义**：进程重启时，queued / in-progress 任务仍会标记为 failed，而不是自动恢复。这是显式行为，不是静默恢复。

### 兼容性

- 现有 HTTP 接口和持久化的 `status.json` / `result.json` 结构保持兼容。
- 内部规划文档不属于公开发布文档。

## 0.7.1 — cohort 自动重建 + 线程安全修复 (2026-04-22)

### 新功能

- **AS-norm cohort 自动重建**：新增后台守护线程 `cohort-rebuild`，每 60 秒检查一次声纹库是否有新 enroll 未反映到 AS-norm cohort。若有脏数据且距最后一次 enroll 已超过 30 秒（防抖），自动触发重建。enroll 后新 embedding 最多约 90 秒内无需手动操作即可进入匹配路径；只有 cohort >=10 时才进入完整 AS-norm 评分，否则仍走 raw cosine fallback。
- **并发安全机制**：`_cohort_rebuild_lock`（非阻塞 acquire）防止守护线程与手动触发 `POST /api/voiceprints/rebuild-cohort` 并发执行两次重建。
- **ABA 防护**：`_cohort_generation` 版本计数替代 bool 脏标记，确保重建过程中发生的新 enroll 在下次 tick 仍会触发重建，不丢失 dirty。

### Bug 修复

- **守护线程优雅退出**：lifespan teardown 调用 `_stop_event.set()` 并 `join(timeout=5)` 等待线程退出，避免进程停止时残留后台线程。
- **原子写入异常清理**：`_atomic_write_json` 新增 `try/finally`，在 `json.dump`/`fsync` 抛异常时自动删除孤儿 `.tmp` 临时文件。
- **`speaker_id` 输入校验**：`POST /api/voiceprints/enroll` 的 `speaker_id` Form 字段新增格式校验（`^spk_[A-Za-z0-9_-]{1,64}$`），与路径参数端点保持一致；不合法格式返回 422。
- **pip-audit 硬门控**：CI 安全扫描移除 `|| echo`，漏洞检测失败时真正阻断构建。
- **CQ-C1 计数器移除**：移除 `job_service.py` 中每 10 次转录触发重建的旧机制，避免与守护线程并发重建。
- **并发去重保护**：新增 `_in_flight_hashes` 字典，防止内容完全相同的并发上传各自启动转录任务；第二个请求直接返回第一个任务的 ID，不重跑 GPU。
- **持久化窗口彻底关闭**：`register_in_flight` 现在在 `_write_status` 成功后才调用，保证任何通过并发去重路径拿到 job_id 的调用方都能在磁盘上找到对应的 `status.json`。若初始 `status.json` 写入失败，接口在注册任何 in-flight 记录前就以 `503` 终止请求。`_write_status` 返回值改为 `bool`，方便调用方检测写入失败。
- **Segment 说话人改写语义收紧**：`PUT /api/transcriptions/{tr_id}/segments/{seg_id}/speaker` 新增 `speaker_id` 格式校验（`^spk_[A-Za-z0-9_-]{1,64}$`，格式不合规返回 422）及数据库存在性校验（声纹不存在返回 404）。省略 `speaker_id` 时，旧值会被显式置为 `null`。该接口不修改 `speaker_map`（`speaker_map` 记录分离模型匹配结果，不受人工纠错影响）；每次编辑后 `unique_speakers` 从全部 segment 重新计算。
- **CI 单测门控**：CI 命令改为 `pytest tests/unit/ tests/test_security.py`，`test_job_service.py`、`test_main_lifespan.py`、`test_voiceprint_db.py` 现在都在 CI 门控范围内。

### 兼容性

- 所有已有 HTTP 接口行为不变。
- cohort 自动刷新为可观测的行为变化：长期运行的服务无需再手动 rebuild-cohort，但手动调用仍有效（立即重建，不等待防抖）。

## 0.7.0 — 说话人身份保留 + ngram 去重参数 (2026-04-21)

### Bug 修复

- **说话人身份保留**：说话人分离产生的多个聚类（如 `SPEAKER_00`、`SPEAKER_02`）即使匹配到同一个已登记声纹，每个原始 diarization 标签也会作为独立的输出说话人身份保留。显示名会做区分以便阅读（例如 `Maple`、`Maple (2)`）。

### 新功能

- **`no_repeat_ngram_size` 参数**：`POST /api/transcribe` 新增可选整数字段 `no_repeat_ngram_size`（默认 `0`，即不开启）。设置 ≥ 3 时传给 faster-whisper，抑制转录结果中的 n-gram 重复（如「比如比如比如」→「比如」）。设置 < 3 或省略时等同于不开启。
- **参数校验**：`no_repeat_ngram_size` 传入非整数值（如 `"banana"`）返回 HTTP 422。
- **params 记录**：已完成任务的 `params` 对象新增 `no_repeat_ngram_size` 键，值为实际使用的整数（未开启时为 `0`）。

### 测试

- 新增 43 个 E2E 测试，分布在 8 个测试类：`TestSpeakerIdentityPreservation`、`TestSecurity`、`TestSegmentReassignment`、`TestSpeakerManagement`、`TestExportFormats`、`TestOutputSchema`、`TestNoRepeatNgramSize`、`TestEdgeCases`、`TestLongChains`。
- 测试套件共 84 条（78 通过，6 预期跳过）。

### 部署

- `docker-compose.yml` 新增 `./app:/app` 卷挂载，本地代码变更通过 rsync 即可生效，无需重建镜像。

### 兼容性

- 所有已有 HTTP 接口行为不变。
- `no_repeat_ngram_size` 是新增可选字段，老客户端直接忽略。
- `params.no_repeat_ngram_size` 是新增字段；无该字段的历史转录视同 `0`。

## 0.6.0 — 安全硬化 + 架构重组 (2026-04-21)

### 安全

- **路径遍历防护**：`_safe_tr_dir()` 函数 + FastAPI `Path(pattern=r"^tr_[A-Za-z0-9_-]{1,64}$")` 参数校验，杜绝目录穿越攻击（SEC-C1）
- **Pickle RCE 修复**：`np.load(..., allow_pickle=False)` 防止恶意 `.npy` 文件执行任意代码（SEC-C2）
- **零向量防御**：`identify()` 对全零 embedding 提前返回，避免 AS-norm 分支语义错误
- **声纹去重**：`add_speaker()` 按名称去重，防止重复 enroll 污染声纹库（CQ-C3）
- **前端 CSP 收紧**：`Content-Security-Policy` meta 标签 + `sessionStorage` 替代 `localStorage` 存储 API Key（SEC-H2/H3）
- **安全响应头**：`X-Content-Type-Options`、`X-Frame-Options`、`Referrer-Policy`、`X-XSS-Protection` 中间件

### 架构重组

- **main.py 瘦身**：从 ~980 行拆分为 ~160 行编排入口；新增 `app/config.py`（所有环境变量集中管理）、`app/api/routers/`（transcriptions / voiceprints / health）、`app/api/deps.py`（FastAPI 依赖注入）、`app/services/audio_service.py`、`app/services/job_service.py`
- **任务状态持久化**：`_write_status()` 将 job 状态写入 `status.json`，`recover_orphan_jobs()` 在启动时修复孤儿任务；重启后已完成任务仍可通过 `GET /api/transcriptions/{id}` 访问（AR-C2）
- **LRU 任务缓存**：内存 job 字典改为 `_LRUJobsDict`（上限 200 条），防止长期运行内存泄漏

### 性能

- **异步上传**：`aiofiles` 流式写入 + 流式 SHA256，上传不再阻塞事件循环（PERF-C2）
- **分段音频加载**：`torchaudio.load(frame_offset, num_frames)` 按说话人分段读取，长音频显存占用下降 >1000×（PERF-H1）
- **NumPy BLAS 余弦扫描**：`_python_cosine_scan` 改用 `embs_normed @ q_normed`，批量余弦计算速度提升（PERF-H8）

### CI/CD

- **测试门控**：CI 移除 `|| true`，测试失败不允许合并（CD-H1）
- **pip-audit 安全扫描**：每次 CI 运行依赖漏洞扫描（CD-C2）
- **测试套件**：新增 `tests/test_security.py`、`tests/test_voiceprint_db.py`、`tests/test_job_service.py`（共 15 个测试）

### 兼容性

- 所有已有 HTTP 接口行为不变
- 升级无需数据迁移

## 0.5.0 — AS-norm 声纹评分 (2026-04-20)

### AS-norm 声纹评分

- 引入 `ASNormScorer`（`voiceprint_db.py`），对原始余弦分用 impostor cohort 做自适应评分归一化（AS-norm）
- AS-norm 消除说话人依赖的基准偏差，在同等精度下相对降低 EER 15–30%
- 服务启动时自动从已有转录的 embedding（`emb_*.npy`）构建 cohort，保存为 `data/transcriptions/asnorm_cohort.npy`；首次构建失败时静默降级为原始余弦
- AS-norm 启用后有效阈值固定为 `0.5`（经 cohort 归一化后的操作点）；未启用时仍走 0.4.0 的自适应余弦阈值
- 新增 `POST /api/voiceprints/rebuild-cohort` — 手动重建 impostor cohort

### 兼容性

- 所有已有接口行为不变
- 未构建 cohort 时（零 transcription 环境），声纹识别自动回退到 0.4.0 的余弦逻辑

### 升级迁移说明（从 0.4.x → 0.5.0）

- **0.4.x 历史转录不包含 `emb_*.npy` 时，AS-norm cohort 不会被自动激活。**
  启动日志会显示 `cohort_size=0` 或低于 10，`identify` 继续走 0.4.0 的 raw cosine +
  自适应阈值路径。
- 如果希望启用 AS-norm 归一化评分，0.5.0 升级后请：
  1. 确认 `data/transcriptions/` 下至少有 10 条历史转录包含 `emb_*.npy`；
  2. 调用 `POST /api/voiceprints/rebuild-cohort` 手动重建 cohort；
  3. 或让服务重新跑一批新转录再重启（启动时会重建 cohort）。
- 后续运行期新增的转录不会自动合入 cohort —— 需要再次触发 rebuild-cohort 或重启服务。

## 0.4.0 — 自适应声纹阈值 + 降噪 SNR 门限 + OSD (2026-04-19)

### 自适应声纹阈值

- `VOICEPRINT_THRESHOLD` 现在是可配环境变量（默认 0.75），作为基础阈值
- 每位说话人的实际阈值根据已登记样本的余弦方差自动放松：1 个样本固定 -0.05，2+ 个样本按 `min(3×std, 0.10)` 放松，绝对下限 0.60
- 10 条 PLAUD Pin 录音 A/B 测试：召回率从 50% 提升到 70%，零误识别
- 新增环境变量 `VOICEPRINT_THRESHOLD`（默认 `0.75`）

### 降噪处理 + SNR 门限

- 新增 `DENOISE_MODEL` 环境变量：`none`（默认）| `deepfilternet` | `noisereduce`
- 新增 `DENOISE_SNR_THRESHOLD` 环境变量（默认 `10.0` dB）：仅用于 DeepFilterNet，SNR 达到或超过此值时跳过 DeepFilterNet，避免对高质量录音做不必要处理；`noisereduce` 不受该 gate 控制
- 任务流水线新增 `denoising` 状态（converting 之后、transcribing 之前，仅在启用降噪时出现）
- `POST /api/transcribe` 新增 `denoise_model`（字符串）和 `snr_threshold`（浮点数）两个可选字段，支持单次请求级别覆盖
- DeepFilterNet 对高 SNR 录音（>10 dB）有害：段数增加 100-145%，代理 CER 劣化 20-91%。DeepFilterNet SNR 门限可自动保护干净音频
- CUDA OOM 修复：DeepFilterNet 处理长音频后（~15 GB PyTorch CUDA 保留），在调用 Whisper 前执行 `torch.cuda.empty_cache()` + `gc.collect()` 解决 ctranslate2 的 OOM 问题

### 重叠语音检测 OSD

> **注意**：OSD 功能已在 v0.5.x 中移除（参见上方 0.5.0 节与 git 历史中的 `Revert "feat: add overlapped speech detection ..."` 提交）。以下描述仅为历史记录，相关请求参数（`osd`）与响应字段（`has_overlap`）在当前版本中不再可用。

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
- 版本 pin：`whisperx==3.1.6`（匹配当前 `pyannote==3.1.1` pin，torch 由 Docker base image 提供；这些 release 在 PyPI 是 yanked 状态，但 pip 在 `requirements.txt` 里显式写版本号时仍会装）

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

解耦和 BetterAINote 的绑定——服务本身可以独立使用，因此改名：

- **仓库**：`MapleEve/openplaud-voice-transcribe` → `MapleEve/voscript`
  （GitHub 老 URL 自动 301 重定向，老 clone 不会失效）
- **Docker 服务/容器名**：`voice-transcribe` → `voscript`
  （`docker logs voscript`、`docker exec voscript …`）
- **镜像名**：compose 自动产生的 `voscript-voscript:latest`
- **README/文档**：重写定位为独立转录服务，BetterAINote 改为
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
- `VoiceTranscribeProvider` 在 BetterAINote 侧把 segment 的 `speaker`
  字段改回原始 `speaker_label`（`SPEAKER_XX`），修掉"自动匹配后就没法再
  登记"的断链。

### 破坏性变化
- 容器的 HF 缓存路径从 `/root/.cache/huggingface` 换到 `/cache`。如果你之前
  把宿主目录 mount 到 `/root/.cache/huggingface`，需要更新 compose（或直接
  用本仓库新版 compose，它自动兜底）。
- 上传超过 `MAX_UPLOAD_BYTES` 的请求现在会 413 而不是静默成功。默认 2 GiB
  对绝大多数音频足够。

## 0.1.0 — 首次公开发布

- 首次公开发布 [BetterAINote](https://github.com/MapleEve/BetterAINote) 的私有转录后端。
- 异步任务流水线：`queued → converting → transcribing → identifying → completed`。
- faster-whisper `large-v3` + pyannote `3.1` + ECAPA-TDNN 声纹提取。
- 持久化声纹库，基于余弦相似度自动匹配。
- 所有 `/api/*` 路由支持可选的 `API_KEY` Bearer 鉴权。
- 可移植的 `docker-compose.yml`（数据/模型路径都通过环境变量配置）。
- 必要的版本 pin，让 `pyannote.audio==3.1.1` 仍可用：
  - `numpy<2`（pyannote 3.1.1 用了 `np.NaN`，numpy 2.x 已移除）。
  - `huggingface_hub<0.24`（保留 pyannote 3.1.1 调用的 `use_auth_token` kwarg）。
