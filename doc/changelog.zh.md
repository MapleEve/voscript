# 更新日志

**简体中文** | [English](./changelog.en.md)

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
