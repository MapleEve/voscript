# ADR-0012: 下一版本重构收敛职责边界和循环依赖

- 状态：已接受
- 日期：2026-06-11
- 范围：VoScript 下一版本架构重构、运行时检查、发布检查、文档/代码漂移检查

## 背景

本 ADR 使用几个会影响判断的架构词。Architecture ring（架构环）在本项目里指一组有共同职责和依赖方向的代码区域，例如 API/composition、application、pipeline、provider、domain 和 infra。Cycle（循环依赖）指两个或多个模块、层或职责互相依赖，导致修改一处时很难判断谁拥有规则、谁只能消费规则。Gate 指可重复执行的检查，不是人工印象。Boundary（边界）指两个 ring 或两个职责之间的可见接口。Owner（责任方）指某个规则或状态的唯一权威位置。Provider 指某个 pipeline step 的具体后端或模型实现。Lifecycle（生命周期）指 job、model、runtime resource 或 application startup/shutdown 的状态变化。先把这些词说清楚，是为了让后面的重构决策能落到 VoScript 的具体风险上：减少跨层互相调用、重复权威和发布前靠记忆查漏。

ADR-0001 已决定 Python 继续拥有 HTTP API、job lifecycle、pipeline runner、配置、模型 lifecycle 和 artifact/result contract，Rust 只作为 provider/kernel 内部实现。ADR-0003 已决定 provider capability 使用静态 metadata 表达。ADR-0010 和 ADR-0011 已分别固定 heavy CI gate 触发策略与 `RUST_KERNEL_MODE=off|required` 的显式启用/回滚语义。

下一版本重构要解决的问题不是单纯的 Python import cycle。在 VoScript 里，循环重要是因为它会让 API、application、pipeline、provider、infra 和 docs/release 之间的权威来回渗透；消除或收窄循环后，变更风险会从“牵一发影响整条链路”降低为“只影响明确 owner 的边界”。

cycle-analysis 重试结果确认：当前架构不能被描述为 cycle-free。已确认的 Python import SCC（strongly connected component，导入图里的强连通分量，表示这些模块在导入层面互相可达）是 `pipeline.contracts -> pipeline.contracts.context -> pipeline.contracts.requests -> pipeline.registry -> pipeline.contracts`。更宽的 layer-level SCC 是 `infra <-> pipeline <-> providers`。同时，本次分析没有发现 package-level cycle、API/application hard import cycle、uncontrolled restart/dedup infinite loop，也没有发现 silent Rust-to-Python fallback loop。这些否定项只能说明对应风险未被当前证据命中，不能抵消已确认的 import SCC 和 layer-level SCC。

当前代码中存在多个需要用架构 ring 和 gate 固化的证据：

- `app/pipeline/registry.py` 是真实 stage/provider registry，定义稳定 stage order 和 provider import path。
- `app/providers/capabilities.py` 已有静态 capability metadata，但覆盖面仍小于 registry stage/provider surface，且包含 `alignment` 这种不在 registry stage order 中的子能力。
- `app/pipeline/runner.py` 负责执行 stage order，并把 `request.provider_for(stage_name)` 记录到 `context.metadata["selected_providers"]`，但 runner 还没有在执行前强制调用 capability matching。
- `app/api/routers/transcriptions.py` 同时拥有 upload、job 查询、transcription 列表/读取、音频下载、speaker reassign 和 export 逻辑，路由文件已经成为多职责入口。
- `app/providers/normalize/default.py` 和 `app/infra/audio/paths.py` 在 API 层以下直接导入并抛出 FastAPI `HTTPException`，说明 HTTP error 类型已经泄漏到 provider/infra。
- `app/config.py` 仍允许 `API_KEY` 为空、`CORS_ALLOW_ORIGINS=*`、`MAX_UPLOAD_BYTES=2GB`、`RUST_KERNEL_MODE=off`；这些默认值可以用于本地/LAN 体验，但发布和公开部署必须通过文档、配置和 admission gate 明确边界。
- `app/providers/embedding/default.py`、`app/providers/enhance/default.py` 和 `app/providers/diarization/default.py` 仍在部分路径中加载完整音频或把整段 audio 交给下游库，说明 memory-sensitive provider 需要 explicit bounds，而不能只依赖上传大小。
- `app/voiceprints/db.py` 先由 Python repository 取出候选，再按 `RUST_KERNEL_MODE` 可选调用 Rust `voiceprint_score`；`app/voiceprints/repository.py` 仍拥有候选读取，`crates/voscript_core/src/voiceprint.rs` 只负责纯 scoring decision。
- `app/infra/job_persistence.py` 和 `app/pipeline/contracts/schema.py` 仍是 Python status/schema contract helper；`app/providers/kernel_bridge/runtime.py` 只是 Rust extension import/call 和 response validation bridge。
- `.github/workflows/ci.yml`、`.github/workflows/rust-foundation-heavy.yml` 和 `.github/workflows/release.yml` 把 public scan、lint/test/security、Rust wheel/Docker smoke、publish 分散在不同 workflow；发布 gate 需要同一个 exact ref 的自包含证据，而不是拼接过期或不同 ref 的绿灯。
- `docker-compose.yml`、`.env.example`、`README.md`、`README.en.md`、`doc/api.zh.md`、`doc/api.en.md`、`doc/configuration.zh.md` 和 `doc/configuration.en.md` 共同描述运行配置、API、鉴权、上传上限、Rust mode 和验证口径，必须被当作 public docs/code drift surface。

本 ADR 位于 ignored internal architecture docs。它只记录下一版本重构的长期约束和最终状态 contract。

## 术语约定

Architecture ring（架构环）不是目录装饰，而是判断“谁能依赖谁、谁拥有错误映射、谁拥有 runtime 决策”的依据；修复后风险会从跨层互相调用降低为单向、可审计的依赖。

Boundary（边界）指两个 ring 或两个职责之间的可见接口。在 VoScript 里，典型边界包括 route handler 到 usecase、pipeline 到 provider、Python runtime 到 Rust kernel。边界重要是因为越界会把 HTTP error、job lifecycle 或模型细节泄漏给不该拥有它们的代码；边界清楚后，测试和回滚可以按 owner 缩小范围。

Owner（责任方）指某个规则或状态的唯一权威位置，例如 `app/pipeline/registry.py` 拥有 stage/provider import registry，Python 拥有 job persistence 和 schema optionality。owner 不清会让同一规则在 router、runner、provider 和 docs 中重复出现；修复后可以通过一个 owner 变更和 gate 检查降低漂移风险。

Gate 指可重复执行的检查，不是人工印象。VoScript 的 architecture gate、release gate 和 docs/code drift gate 要把 import graph、forbidden dependency、exact-ref release evidence 和 public docs 同步变成可验证条件；这样可以把发布前风险从“靠记忆查漏”降到“证据缺失即失败”。

Provider 指某个 pipeline step 的具体后端或模型实现，例如 ASR、diarization、embedding、punc。provider 只应通过 stage contract 输入输出；如果 provider 反向拥有 job admission、HTTP error 或 thread/disk policy，风险会扩大到整个 runtime。

Repository 指领域或基础设施数据访问抽象，不等于 Git repository。在 VoScript 里，`app/voiceprints/repository.py` 这种 repository 应拥有候选读取接口，而不是让 Rust helper、router 或 provider 到处直接读取存储；修复后 candidate fetch 和 scoring decision 的责任边界更清楚。

Usecase 指 application 层的一条业务流程，例如 upload admission、job bootstrap、status recovery、export formatting。usecase 重要是因为它把多个底层能力编排成用户可见行为；把 usecase 从 router 中抽出后，HTTP 输入输出和业务流程可以分别测试。

Orchestration 指跨模块调度顺序和状态推进，例如 transcription job 如何经过 admission、dedup、pipeline stages、artifact write 和 status update。orchestration 如果散在 router、runner 和 provider 里，会让失败恢复和重试语义不稳定；集中到 application/pipeline owner 后，生命周期风险下降。

Adapter 指把外部系统或具体实现接入稳定边界的薄层，例如 filesystem、Rust extension bridge、CUDA/runtime helper。adapter 重要是因为它可以隔离副作用；如果 adapter 反向拥有业务决策，测试会被外部环境拖住，回滚范围也会变大。

Service 指对外提供能力的窄接口，不是把任意 helper 都命名成 service。VoScript 应避免恢复 `app/services/*` 这类扁平杂物层；窄 service 必须有明确 owner、输入输出和所在 ring，否则会制造新的结构债。

Lifecycle 指 job、model、runtime resource 或 application startup/shutdown 的状态变化。lifecycle 重要是因为 VoScript 同时有 job persistence、GPU/model load、idle unload 和 app lifespan；owner 混乱会造成重复启动、泄漏或错误恢复。

Import direction 和 dependency direction 分别指 Python `import` 图的方向和架构职责依赖方向。前者可以由静态脚本扫描，后者还要结合 ring owner 判断；两者都必须单向收敛，才能降低循环依赖和越界调用风险。

Facade 指对复杂子系统提供的简化入口。VoScript 只有在 facade 明确隐藏复杂性、且不吞掉 owner 和错误语义时才应使用；否则 facade 会变成新的大 router。

DTO 指跨边界传递的简单数据对象。它在 VoScript 中适合表达 API/application/pipeline 之间的稳定输入输出；如果 DTO 混入 persistence、FastAPI 类型或 provider 实现细节，会增加边界漂移风险。

Source-guard test 指用源码扫描守住架构规则的测试，例如禁止 API ring 之外导入 `fastapi.HTTPException`、禁止 provider 导入 router、禁止 docs 记录私有路径。它重要是因为这类违规不一定会被单元行为测试覆盖。

Structural debt / architecture debt 指当前还能运行、但职责和依赖已经让未来变更更危险的结构问题，例如大 router、多 owner、layer SCC 和 docs/code drift。偿还这类债务的收益不是立刻增加功能，而是降低后续修复、发布和回滚的不确定性。

## 决策

下一版本重构使用 architecture rings 作为默认边界模型，并把 ring/cycle gate 放入可重复验证链路。ring 是依赖方向、错误类型、配置权威和 runtime admission 的判定依据。

Architecture rings 定义如下：

- API/composition ring：`app/main.py`、`app/api/` 和 FastAPI wiring。只负责 request parsing、dependency wiring、auth、response shaping 和 HTTP error mapping。FastAPI `HTTPException` 只能在本 ring 或显式 composition boundary 中出现。
- Application ring：`app/application/`。负责 use-case orchestration、job lifecycle、status transition、dedup coordination 和 background execution admission。不得依赖 FastAPI 类型，不得直接拥有 provider/model implementation 细节。
- Pipeline ring：`app/pipeline/`。负责 stable stage order、`PipelineRequest`、`PipelineContext`、result/status/schema contract、stage dispatch 和 provider selection boundary。不得依赖 API/router，不得拥有 HTTP error mapping。
- Provider ring：`app/providers/`。负责每个 pipeline step 的具体 backend/model implementation。provider 必须通过 stage contract 输入输出，不能抛出 HTTP 类型，不能拥有 job/thread/disk admission policy。
- Domain ring：`app/voiceprints/` 等业务领域模块。负责 speaker enrollment、matching、cohort、scoring policy 和 repository abstraction。domain 可以调用明确的 kernel bridge，但不能把 Rust helper 描述成 domain 或 runtime owner。
- Infra ring：`app/infra/`。负责 filesystem、hash index、job persistence、runtime semaphore/cache、CUDA device selection、path safety 和 concrete adapters。infra 返回 domain/application 可映射的错误，不返回 HTTP error。

Ring gate 必须同时检查两类问题：

- import cycle gate：用可重复脚本扫描 `app/` Python import graph，禁止新增 SCC，并把已知 SCC 收敛为零。该 gate 的输出必须包含 module count、internal edge count、SCC 列表、layer edge 列表和 layer SCC 列表；不能依赖一次性人工统计，也不能在没有当前输出时声称具体计数。
- forbidden dependency gate：按 ring allowlist 检查禁止导入，例如 `fastapi` 不能出现在 API/composition ring 之外，provider/infra/domain 不能导入 router，provider 不能导入 application job orchestration，pipeline 不能导入 API。

Package-level graph、API/application hard import path、restart/dedup loop 和 Rust fallback loop 可以作为辅助检查，但不能替代 import cycle gate 和 forbidden dependency gate。没有命中这些辅助风险，只能说明当前证据未发现相应循环，不能证明整体架构 cycle-free。

Provider capability metadata 是 provider 可运行性的权威入口。`app/pipeline/registry.py` 仍是 stage/provider import registry，但任何 registry 中可被选择的 provider 都必须有 capability record 或显式 allowlisted exemption。runner 或 runner 前置 preflight 必须在 stage 执行前根据 `PipelineRequest.language`、stage criticality、provider name 和 Rust support 调用 capability matching；required mismatch fail closed，degradable/optional mismatch 只能通过明确 metadata 记录 skip reason。

Stage registry 与 capability metadata 必须互相校验。新增 stage、provider、language constraint、Rust-backed provider 或 alignment 子能力时，必须同时更新 registry、capability metadata、capability tests 和 docs/code drift gate。`alignment` 这类子能力可以保留为 stage 内 capability，但必须明确挂靠到拥有它的 registry stage，不能成为第二套隐式 stage order。

`PipelineContext` 仍可以作为 stage 间执行状态，但必须被 gate 和 tests 限制为稳定字段、稳定 metadata key 和单向 stage progression。新增 context 字段或 metadata key 需要说明 owner stage、读写 stage、是否进入 public result/status/artifact contract，以及是否允许下游覆盖。不得让任意 stage 通过自由写 `metadata` 反向改变 provider selection、job status、API response 或 artifact schema。

API/domain boundary cleanup 是下一版本重构的必做项。API ring 以下不得再抛 FastAPI `HTTPException`；provider、domain 和 infra 只能抛出 typed domain/application errors 或返回 typed result，API ring 统一映射为 HTTP status/detail。已有 `app/providers/normalize/default.py` 和 `app/infra/audio/paths.py` 的 HTTPException 泄漏必须被迁移到这个错误映射模型。

`app/api/routers/transcriptions.py` 必须瘦身。目标不是机械拆文件，而是让 route handler 只做 HTTP 输入输出和 dependency wiring。upload admission、job bootstrap、status read/recovery、transcription listing、audio lookup、speaker correction、export formatting 和 artifact read/write 应迁到 application/domain/infra 的窄接口；router 可以按 upload/jobs/transcriptions/export/speaker correction 拆分，但拆分后不能复制业务规则。

Runtime admission 和 memory bounds 必须成为显式 gate。当前 `MAX_UPLOAD_BYTES`、`UPLOAD_CHUNK`、`JOBS_MAX_CACHE`、serialized GPU semaphore、in-flight dedup 和 idle model unload 已经是基础约束，但下一版本不能继续用 per-request unbounded daemon thread 作为唯一 job admission。接受 job 前必须检查并记录 upload size、durable status write、in-flight/queued job bound、worker/thread bound、data disk pressure 和 configured memory-sensitive stage limits。超出 admission budget 必须在开始 GPU/model work 前返回可预期错误。

Memory-sensitive provider 必须有 size/duration/window policy。embedding、enhance、diarization/alignment 不能只依赖 2GB upload cap；对 full-audio load、resample、DeepFilterNet/noisereduce、WhisperX `load_audio`、speaker embedding chunking 等路径，必须定义可测试的 duration/sample/frame/memory guard 或 streaming/windowed strategy，并把默认值同步到 configuration docs。

Rust boundary wording 必须 truthful。Rust 只能被描述为 selected pure kernel/helper owner：voiceprint scoring decision、postprocess segment shaping、artifact/status helper contract 等。Python 仍拥有 candidate fetch、job persistence、schema optionality、pipeline runner、provider selection、artifact/result assembly 和 runtime mode。`RUST_KERNEL_MODE=off` 是默认业务路径；`required` 只表示被选中的 Rust-backed path 必须 import/call 成功并 fail closed，不表示 Rust 拥有整个 runtime。

Release gate 必须升级为 exact-ref self-contained release gate。发布镜像或 release artifact 前，必须对同一个 immutable ref/tag/SHA 取得以下证据：public release scan、lint/format、unit/security slice、Rust fmt/clippy/test、Rust wheel build、Docker image build with wheel、container Rust extension smoke、container `/healthz` smoke，以及要发布的 Docker tags/source ref。可以继续把 CI、heavy gate 和 publish 分在多个 workflow，但 release workflow 只能消费同一 exact ref 的不可变成功证据；不能用 stale PR 首轮结果、latest main 结果或手动输入未解析 SHA 的结果替代。

Docs/code drift gate 必须覆盖 public runtime surface。修改 `app/config.py`、`docker-compose.yml`、`.env.example`、API router public behavior、status/result/artifact contract、Rust mode、upload/job admission、voiceprint scoring semantics 或 release gate 时，必须校验 `README.md`/`README.en.md`、`doc/api.zh.md`/`doc/api.en.md`、`doc/configuration.zh.md`/`doc/configuration.en.md` 是否同步。public docs 只能记录 released behavior、配置和 API；internal validation wording 只能写行为类别，不能泄漏真实 job id、speaker id、样本、host、日志或路径。

## 被拒绝的方案

- 只用 import SCC 作为架构健康标准：它能发现一类循环导入，但发现不了 HTTPException 泄漏、provider capability 不权威、router 多职责、mutable context 反向改写、runtime admission 缺口或 release/docs drift。
- 忽略已确认 SCC、只记录未发现 package-level cycle：这会把证据口径从“当前有导入和层级循环需要治理”错误改写为“整体无循环”，导致 gate 目标失真。
- 保留大 router 并只在内部加 helper：helper 会降低局部函数长度，但不会把 upload/job/export/speaker correction 的权威移出 API ring，也不能防止业务规则继续在 route handler 中扩散。
- 让 provider 在运行时动态探测能力并自行决定是否运行：这会把模型加载、环境探测、副作用和 selection policy 混在一起，削弱 ADR-0003 的静态 metadata 决策。
- 用 best-effort Rust fallback 描述下一版本：这会违反 ADR-0011 的 explicit rollback 语义，也会掩盖 `RUST_KERNEL_MODE=off` 默认路径与 `required` hard-fail 路径的区别。
- 发布时只看 release workflow build/push 是否成功：这不能证明同一 exact ref 已通过 public scan、Python tests、Rust wheel/Docker smoke 和 runtime health smoke。
- 通过人工记忆同步 README、API docs、configuration docs 和 compose/env 示例：public docs/code drift surface 必须有 gate，不能靠维护者事后查漏。

## 后果

- 下一版本重构需要先落地可重复的 architecture gate，再用它约束后续代码迁移。任何 ring exception 都必须在 gate allowlist 中有 owner、原因和退出条件。
- 已确认的 `pipeline.contracts -> pipeline.contracts.context -> pipeline.contracts.requests -> pipeline.registry -> pipeline.contracts` 和 `infra <-> pipeline <-> providers` 必须被当作架构风险治理目标，而不是被 package-level 或 API/application 层面的未命中结果掩盖。
- Provider registry、capability metadata、runner preflight 和 capability tests 会成为同一组 contract；新增 provider 或 stage 时改一处不改另一处应当失败。
- API 层会变薄，application/domain/infra 的 typed error 和 use-case interface 会增加；这是为了让 HTTP 映射集中、测试更稳定，并避免 FastAPI 类型继续向内层扩散。
- Runtime admission 会把一部分失败提前到 job accepted 之前；这可能让某些原本排队很久才失败的请求更早返回 4xx/5xx，但能保护线程、磁盘、GPU 和内存预算。
- Rust 相关文档和 release notes 必须按 selected kernel/helper 精确描述，不能把 optional bridge 或 helper contract 夸大为 runtime rewrite。
- Release 成本会上升，因为发布必须证明 exact ref 自包含通过；换来的是 release artifact、Docker image、Rust wheel smoke 和 public docs scan 不再跨 ref 拼接。
- Public docs 和 internal ADR 的边界保持不变：ADR 可以记录架构决策和 gate contract，README/doc 只记录用户可用的 released behavior、配置和 API。
