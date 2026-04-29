# 快速安装

**简体中文** | [English](./quickstart.en.md)

这篇面向第一次部署的人。走完大约需要 15~30 分钟，其中大部分时间在等模型下载。

## 0. 选你的部署路径

| 平台 | 路径 | 质量 | 备注 |
| --- | --- | --- | --- |
| Linux + NVIDIA GPU | docker-compose（下面主线） | 最好 | 推荐路径，本文档主流程 |
| Windows 11 + WSL2 + NVIDIA GPU | docker-compose（在 WSL2 里走 Linux 流程） | 最好 | 见 [0.2](#02-windows-11--wsl2) |
| macOS Apple Silicon（M1/M2/M3/M4） | **native venv，纯 CPU** | 可用但慢 | Docker Desktop 在 macOS **不能透传 GPU**，见 [0.3](#03-macos-apple-silicon) |
| macOS Intel | native venv，纯 CPU | 可用但非常慢 | 同 macOS Apple Silicon，见 [0.3](#03-macos-apple-silicon) |

CPU / 小显存场景**强烈建议**用 `WHISPER_MODEL=medium`（下面第 2 步会讲），速度差 3~4 倍而中文质量可接受。

### HuggingFace 准备（所有平台都要）

- 在 <https://huggingface.co/settings/tokens> 创建一个 **read** 权限的 token（以 `hf_` 开头）—— **什么时候建都行**，和下面的授权不冲突。
- 在 <https://huggingface.co/pyannote/speaker-diarization-3.1> 点 **Agree and access repository**。
- 在 <https://huggingface.co/pyannote/segmentation-3.0> 也点一下同意。
- 在 <https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM> 也点一下同意（**0.3.0 起新增**，声纹提取用）。

> token 和 gated 授权是**两件独立的事**，顺序无所谓，但**三个 gated 模型都要同意**+ token 都在位，才能拉到权重：只有 token 没接受条款会 403，接受了条款没 token 会 401。

## 0.1 Linux + NVIDIA GPU（主线路径）

- Docker 24+
- **NVIDIA Container Toolkit**（没装的话 compose 启动会报 `could not select device driver`）：
  ```bash
  # 以 Ubuntu 为例
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker

  # 验证
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

显存建议：
- ≥ 12 GB → 直接跑默认 `large-v3`
- 8~12 GB → 仍可跑 `large-v3`（实测约 9 GB），但和别的 GPU 任务抢显存要小心
- < 8 GB → `WHISPER_MODEL=medium`

然后跳到 [第 1 步](#1-克隆仓库)。

## 0.2 Windows 11 + WSL2

官方支持路径 = **WSL2 + NVIDIA Container Toolkit**。Docker Desktop 的 GPU passthrough 底层本身就是走 WSL2，所以等价。

前置：
- Windows 11 或 Windows 10 21H2+
- 装了 WSL2 Ubuntu（`wsl --install -d Ubuntu`）
- Windows 上装了 NVIDIA 驱动 ≥ 470
- WSL2 里装了 Docker（要么直接 WSL2 里装 docker，要么在 Windows 装 Docker Desktop 并开启 "Use WSL 2 based engine" + "Enable integration with my default WSL distro"）

之后**所有命令都在 WSL2 的 Ubuntu shell 里执行**，走 [0.1 Linux 的流程](#01-linux--nvidia-gpu主线路径)。验证：

```bash
# 在 WSL2 Ubuntu 里
nvidia-smi                        # 应该能看到你的 NVIDIA 卡
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

两个都过就跟 Linux 一样跑。

## 0.3 macOS（Apple Silicon / Intel）

**重要**：macOS 上的 Docker Desktop **不能透传 GPU**（无 CUDA、无 Metal），所以 docker-compose 路径在 macOS 上**只能跑 CPU**，跑 `large-v3` 基本不可用。
走 **native venv + CPU** 路径，并且把模型降到 `medium`。

前置：
- Python 3.11（推荐 `brew install python@3.11`）
- ffmpeg（`brew install ffmpeg`）
- libsndfile（`brew install libsndfile`）
- 至少 16 GB 内存（Apple Silicon 统一内存也算）

```bash
# 1. 克隆
git clone https://github.com/MapleEve/voscript.git
cd voscript

# 2. 建 venv（在仓库根目录）
python3.11 -m venv .venv
source .venv/bin/activate

# 3. 装依赖（macOS 上 torch 会自动装 CPU/MPS 版，不是 CUDA 版）
pip install --upgrade pip
pip install -r app/requirements.txt

# 4. 配置环境变量
export HF_TOKEN=hf_你的_token
export API_KEY=$(openssl rand -hex 32)
export DEVICE=cpu                 # macOS 必须 cpu（MPS 对 pyannote 支持不完整）
export WHISPER_MODEL=medium       # CPU 跑 large-v3 太慢
export DATA_DIR=$(pwd)/data
mkdir -p "$DATA_DIR"

# 记住这个 API_KEY，BetterAINote 要填一样的

# 5. 启动
cd app
uvicorn main:app --host 0.0.0.0 --port 8780
```

预期性能（参考值）：
- M2 Pro / M3 Pro + `medium` + 1 分钟音频 ≈ 30–60 秒
- M1 / Intel + `medium` + 1 分钟音频 ≈ 1.5–3 分钟
- 跑 `large-v3` CPU 版会慢 3~5 倍，**不建议**

已知限制：
- 不支持 docker-compose 路径
- 不支持 MPS 加速（pyannote 3.1 在 MPS 上有未实现算子，会报错或悄悄回落 CPU）
- 如果你有一台带 NVIDIA GPU 的 Linux / Windows 机器，强烈建议用那台跑服务，Mac 只当客户端

macOS 跑起来之后的流程（配置、对接 BetterAINote）和下面一致，但 **跳过** docker 相关步骤。

## 1. 克隆仓库

```bash
git clone https://github.com/MapleEve/voscript.git
cd voscript
```

## 2. 配置 .env

```bash
cp .env.example .env
```

编辑 `.env`，至少填这两项：

```env
HF_TOKEN=hf_你的_token
API_KEY=这里填一串长随机串_例如_openssl_rand_hex_32
```

如果你 GPU 显存不够（< 12 GB）或者你是 macOS / 纯 CPU 部署，把模型降一档：

```env
WHISPER_MODEL=medium
```

可选值：`tiny / base / small / medium / large-v3`。`medium` 在中文场景下质量损失小，速度大约快 3~4 倍。

如果你在中国大陆网络，建议同时加上：

```env
HF_ENDPOINT=https://hf-mirror.com
```

> 生成强随机 API key：`openssl rand -hex 32`

其他环境变量都有合理默认值，详见 [`.env.example`](../.env.example)。完整清单、
默认值、API 覆盖语义和未暴露调参边界见
[`configuration.zh.md`](./configuration.zh.md)。快速部署时值得留意的几个：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `MAX_UPLOAD_BYTES` | `2147483648`（2 GiB） | 单次上传最大字节数；超了直接 `HTTP 413` |
| `APP_UID` | `1000` | 容器以此 uid 运行，必须和宿主 `DATA_DIR` 的所有者一致 |
| `APP_GID` | `1000` | 同上，group id |
| `JOBS_MAX_CACHE` | `200` | 内存 job 字典 LRU 上限；超出后最旧的 job 从内存淘汰（磁盘 status.json 仍可查） |
| `FFMPEG_TIMEOUT_SEC` | `1800` | ffmpeg 转码超时秒数；超时返回 504，防止畸形音频卡死进程 |
| `MODEL_IDLE_TIMEOUT_SEC` | `0` | 可选 GPU 模型空闲卸载超时；0 表示关闭，正数表示空闲达到该秒数后卸载，并在下一次 lazy load 时重新选择最佳 CUDA 设备 |
| `ALLOW_NO_AUTH` | `0` | 设为 1 可在未配置 API_KEY 时抑制启动警告（明确确认无鉴权模式） |
| `DENOISE_MODEL` | `none` | 服务端默认降噪后端：`none`、`deepfilternet` 或 `noisereduce`；API 可按单次任务覆盖 |
| `DENOISE_SNR_THRESHOLD` | `10.0` | DeepFilterNet SNR 门限（dB）；选择 `deepfilternet` 时，音频信噪比达到或高于该值会跳过 DeepFilterNet；`noisereduce` 不受此 gate 控制 |
| `VOICEPRINT_THRESHOLD` | `0.75` | raw cosine 声纹匹配基础阈值，实际会按每位说话人自适应调整 |
| `PYANNOTE_MIN_DURATION_OFF` | `0.5` | pyannote 停顿合并参数，用于减少短暂停顿导致的过度切分 |
| `MIN_EMBED_DURATION` | `1.5` | 提取 speaker embedding 时接受的最短 diarization turn 时长 |
| `MAX_EMBED_DURATION` | `10.0` | 提取 speaker embedding 时单个 turn 使用的最长音频窗口 |
| `WHISPERX_ALIGN_DISABLED_LANGUAGES` | 空 | 逗号分隔的显式跳过 forced alignment 语言；只建议作为临时运营降级开关 |
| `WHISPERX_ALIGN_MODEL_MAP` | 空 | 逗号分隔的 `lang=model` 覆盖，例如 `zh=your-org/your-zh-align-model` |
| `WHISPERX_ALIGN_MODEL_DIR` | 空 | 可选 alignment 模型缓存目录；当前 WhisperX 支持时会透传 |
| `WHISPERX_ALIGN_CACHE_ONLY` | `0` | 设为 1 时，在当前 WhisperX 版本支持的情况下只从缓存加载 alignment 模型 |

对 `POST /api/transcribe` 来说，省略 `denoise_model` 表示使用服务端
`DENOISE_MODEL` 默认值；显式传 `denoise_model=none` 才表示本次请求关闭降噪。
显式传 `snr_threshold` 时，会只对本次请求覆盖 `DENOISE_SNR_THRESHOLD`。
该门限只影响 `deepfilternet`；`noisereduce` 一旦被选择就会运行。
所有可用配置项、哪些 Whisper / ASR 参数尚未暴露为 env，以及 AS-norm cohort
保护语义，见 [`configuration.zh.md`](./configuration.zh.md)。

中文词级 alignment 默认会尝试执行。Docker 镜像使用 PyTorch 2.6.0，可满足
transformers 新安全检查对默认中文 `.bin` alignment 权重的加载要求。如果你使用
自定义镜像且 torch 低于 2.6，请升级到 torch>=2.6，或改用提供 safetensors 的可信
替代 alignment 模型；只有确认要临时降级到段级时间戳时，才设置
`WHISPERX_ALIGN_DISABLED_LANGUAGES=zh`。

### 宿主目录所有者

容器默认用 uid 1000 跑，所以 `DATA_DIR`（默认 `./data`）和 `MODEL_CACHE_DIR`（默认 `./models`）**必须** 让 uid 1000 能写。大部分 Linux 发行版的首个普通用户刚好就是 1000，`docker compose up` 之前直接创建目录即可。如果你在别的 uid 下建的目录，两个办法二选一：

```bash
# A. 把宿主目录换成 1000:1000
sudo chown -R 1000:1000 ./data ./models

# B. 或者告诉容器跑成"你的" uid
echo "APP_UID=$(id -u)" >> .env
echo "APP_GID=$(id -g)" >> .env
```

## 3. 启动服务

```bash
docker compose up -d --build
```

第一次跑要下约 **5 GB** 的模型权重到 `./models/`，可以用下面这句跟进度：

```bash
docker logs -f voscript
```

看到 `Uvicorn running on http://0.0.0.0:8780` 就说明起来了。

也可以直接用仓库自带的脚本一把梭：

```bash
./scripts/deploy.sh
```

脚本会检查 `.env`、启容器、并等 `/healthz` 返回健康。

## 4. 验证部署

```bash
# 健康检查（永远无需鉴权）
curl -sf http://localhost:8780/healthz
# → {"ok":true}

# 需要 API_KEY 才能访问的端点
curl -sS http://localhost:8780/api/voiceprints \
    -H "Authorization: Bearer $API_KEY"
# → [] （首次一定是空数组）
```

浏览器打开 <http://localhost:8780/> 能看到一个简陋的 Web UI，可以直接上传音频测试。

## 5. 对接 BetterAINote

在 BetterAINote 的"设置 → 转录"里配：

- **Private transcription base URL**：`http://你部署的主机:8780`
- **Private transcription API key**：跟 `.env` 里 **完全一样** 的那串 `API_KEY`

配完后 BetterAINote 的 worker 会自动把每条录音提交到这个服务。
具体接口细节参考 [`api.zh.md`](./api.zh.md)。

## 升级

```bash
cd voscript
git pull
docker compose up -d --build
```

模型权重被缓存到 `./models/`，重建镜像不会重新下载。

### 从 0.2.x 升到 0.3.0

多做两件事：

1. **HF 多接受一个 gated 模型**：<https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM> 点
   "Agree and access repository"——WeSpeaker 替换了 ECAPA-TDNN 做声纹提取。没点
   同意会 403。
2. **老声纹需要全部重新登记**。0.3.0 的 embedding 空间和 0.2.x 的不兼容（WeSpeaker
   ≠ ECAPA，余弦对比结果毫无意义），所以：
   - 简单粗暴：`rm data/voiceprints/voiceprints.db` 然后让容器自己重建
   - 或者一个个：`curl -X DELETE -H "Authorization: Bearer $API_KEY" http://host:8780/api/voiceprints/<spk_id>`
   - 清完之后，把每个人的音频重新 transcribe + enroll 一次

老版本的 `index.json` + `.npy` 文件会被自动迁移成 sqlite DB 格式（不会丢数据，但
迁进来的 embedding 已经不能用，还是得重新登记）。

## 常见问题

### `nvidia-smi` 在容器里找不到
→ NVIDIA Container Toolkit 没装或者 Docker 没重启。回到第 0 步。

### 启动日志里看到 `403 Forbidden` 下载 pyannote 模型
→ 没点同意 gated 模型条款（回到 [第 0 步的 HuggingFace 准备](#huggingface-准备所有平台都要)）。

### 启动日志里看到 `401 Unauthorized` 下载 pyannote 模型
→ `HF_TOKEN` 没填、写错了、或者过期了，检查 `.env`。

### macOS 上跑起来了但巨慢
→ 先确认 `DEVICE=cpu` 且 `WHISPER_MODEL=medium`。大模型 CPU 跑就是这么慢，考虑换到一台带 NVIDIA 卡的机器。

### `np.NaN was removed` 崩溃
→ `requirements.txt` 被改坏了、numpy 被升到了 2.x。保持 `numpy<2.0` 的 pin 不要动。

### 服务起来了但 BetterAINote 调不通
→ 检查 `API_KEY` 两端是否一模一样（大小写、空格都不能差），以及 BetterAINote 主机能不能
访问到 `:8780`（防火墙、docker 网络）。

### 我要备份什么
→ 只用备份 `data/voiceprints/`。其他东西丢了都能从原始音频重建。

更多线上风险看 [`security.zh.md`](./security.zh.md)。
