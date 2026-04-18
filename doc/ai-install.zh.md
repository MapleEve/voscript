# 给 AI 用的安装部署指南

**简体中文** | [English](./ai-install.en.md)

> 这篇文档是写给 **AI agent / LLM** 看的，目标是"用户让你帮他部署这个服务"
> 这个场景。人工部署指引请看 [`quickstart.zh.md`](./quickstart.zh.md)。
>
> 配套阅读：部署完成后怎么用接口，看 [`ai-usage.zh.md`](./ai-usage.zh.md)。

## 你的工作边界

用户会让你在他的一台机器上部署 `voscript`。你能做的：
- 通过 shell 跑命令、读写文件
- 编辑 `.env`、`docker-compose.yml`
- 跑 `docker compose`

你**不能**擅自做的：
- 不能把 `HF_TOKEN` / `API_KEY` 写死进 commit、日志、聊天记录
- 不能跳过安全硬化（不设 `API_KEY` 就启动到公网可达的端口）
- 不能跑 `git reset --hard` / `docker system prune -a` 这类破坏性操作去"修问题"
- 不能自己瞎造 HF_TOKEN，必须让用户提供

## 决策树：先判断环境

```
检查 0：用户在什么系统？
    $ uname -s
    - Linux    → 继续走检查 1（GPU 分支）
    - Darwin   → macOS 分支（见下方「macOS 路径」），跳过 docker-gpu 检查
    - MINGW*/MSYS*/CYGWIN* → 用户可能在 Git Bash，真实部署目标多半是 WSL2。
      先问清楚：
        * "你希望部署到 WSL2 里吗？" → 让他进 WSL2，后续命令都在 WSL2 跑
        * "你就是想在原生 Windows 上跑？" → 不支持，推 WSL2
```

### Linux / WSL2 分支

```
检查 1：有没有 NVIDIA GPU？
    $ nvidia-smi
    - 能输出 → 继续
    - 找不到命令 / CUDA 不可用 → 告诉用户"这条路径需要 NVIDIA GPU + 驱动"，
      问他是否切到 macOS/CPU 分支，或修好驱动再来。

检查 2：显存够不够？
    $ nvidia-smi --query-gpu=memory.total --format=csv,noheader
    - ≥ 12 GB → 用默认 large-v3
    - 8–12 GB → 也能跑 large-v3（实测 ~9 GB），提醒用户别和其他大模型抢卡
    - < 8 GB → 在 .env 里设 WHISPER_MODEL=medium（速度 3~4x，质量差距可接受）

检查 3：Docker + NVIDIA Container Toolkit 可用吗？
    $ docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
    - 输出 GPU 信息 → OK
    - "could not select device driver ..." → 装 nvidia-container-toolkit（下面有脚本）

检查 3.5：用户的 uid/gid 是 1000 吗？
    $ id -u ; id -g
    - 都是 1000 → 继续，容器默认 APP_UID=1000 就够
    - 不是 1000 → 稍后在 .env 里追加 APP_UID=$(id -u) / APP_GID=$(id -g)，
      或者部署完之后 sudo chown -R 1000:1000 宿主数据目录

检查 4：用户的 HF_TOKEN 和 gated 授权都到位了吗？
    HF token 和 gated 授权是**两件独立的事**，顺序无所谓，但两件都要齐。
    - 用户有 token 并且已经接受 pyannote 两个仓库的条款 → 下一步
    - 只有其中一项 → 补另一项：
        * 缺 token：让用户去 https://huggingface.co/settings/tokens 建一个 read token
        * 缺授权：让用户分别去
            - https://huggingface.co/pyannote/speaker-diarization-3.1 → Agree
            - https://huggingface.co/pyannote/segmentation-3.0 → Agree
    - 都没有 → 两件一起补，顺序无所谓
    **不要**让用户把 token 贴到 git、commit message 或公开聊天，优先走
    私聊/终端粘贴。
```

### macOS 路径（Docker Desktop 不能透传 GPU）

Docker Desktop on macOS **无法把 GPU 透传进容器**（不管 CUDA 还是 Metal）。
所以 macOS 上**不要**走 docker-compose，直接走 native venv + CPU：

```
检查 A：Python 3.11 可用？
    $ python3.11 --version
    - 可用 → 继续
    - 不可用 → brew install python@3.11

检查 B：ffmpeg / libsndfile 有吗？
    $ which ffmpeg && which sndfile-info || brew install ffmpeg libsndfile

检查 C：磁盘和内存
    - 磁盘空余 ≥ 10 GB（模型权重 + 容器）
    - RAM ≥ 16 GB（Apple Silicon 统一内存也算）

检查 D：HF 授权（和 Linux 分支相同）
```

macOS 上**必须**设：
- `DEVICE=cpu`（MPS 对 pyannote 3.1 支持不完整，别用）
- `WHISPER_MODEL=medium`（CPU 跑 large-v3 太慢，不是"慢一点"而是"不可用"）

并且显式告诉用户：
> "这台 Mac 只能跑 CPU。1 分钟音频大约要 30 秒到 3 分钟不等，看 CPU 型号。
> 如果你有一台带 NVIDIA 卡的 Linux 或 Windows（WSL2）机器，把服务部署在那台
> 机器上会快 5~20 倍。"

## 安装 NVIDIA Container Toolkit（如果没装）

**Ubuntu / Debian**：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

其他发行版参考 [NVIDIA 官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

## 部署步骤（Linux / WSL2）

### 1. 选一个工作目录，克隆仓库

默认放在用户家目录下：

```bash
cd ~  # 或用户偏好的位置
git clone https://github.com/MapleEve/voscript.git
cd voscript
```

### 2. 生成并填 `.env`

**关键：API_KEY 必须是强随机串**。你应该主动给用户生成一个，不要让用户手写：

```bash
cp .env.example .env
API_KEY_VALUE=$(openssl rand -hex 32)
# 向用户确认是否用这个值，或让用户自带
```

然后用 Python 安全地改 `.env`——**不要**用 `sed + 字符串插值**，token 或 key 里
出现 `&`、`/`、换行之类的字符时 sed 会炸，而且一旦变量没设，还会静默把
`.env` 写成空值：

```bash
# 先 bail-out，保证两个变量都有值
: "${USER_SUPPLIED_HF_TOKEN:?need HF_TOKEN from user}"
: "${API_KEY_VALUE:?API_KEY_VALUE is empty}"

# 通过环境变量传值给 Python，Python 不做任何 shell 转义
HF_TOKEN="$USER_SUPPLIED_HF_TOKEN" API_KEY="$API_KEY_VALUE" python3 - <<'PY'
import os, pathlib
env = pathlib.Path(".env")
lines = []
for line in env.read_text().splitlines():
    if line.startswith("HF_TOKEN="):
        line = f"HF_TOKEN={os.environ['HF_TOKEN']}"
    elif line.startswith("API_KEY="):
        line = f"API_KEY={os.environ['API_KEY']}"
    lines.append(line)
env.write_text("\n".join(lines) + "\n")
PY
```

**在改完之后**立刻向用户展示 `.env` 里的 `API_KEY`（只在这一次露出明文），
让他把同一个 key 配到 OpenPlaud(Maple) 的"设置 → 转录"里。之后不要再把这个值
打印到日志/聊天。

**如果目标 GPU 显存 < 12 GB，或是 CPU / macOS 部署**，把模型降到 medium：

```bash
sed -i.bak "s|^WHISPER_MODEL=.*|WHISPER_MODEL=medium|" .env && rm .env.bak
```

**如果用户在中国大陆网络**，还要加一行镜像：

```bash
grep -q '^HF_ENDPOINT=' .env || echo 'HF_ENDPOINT=https://hf-mirror.com' >> .env
```

**如果用户的 uid 不是 1000**（见决策树检查 3.5），要么让容器跟用户 uid 一致：

```bash
UID_VAL=$(id -u); GID_VAL=$(id -g)
[ "$UID_VAL" = "1000" ] || {
    grep -q '^APP_UID=' .env && sed -i.bak "s|^APP_UID=.*|APP_UID=$UID_VAL|" .env || echo "APP_UID=$UID_VAL" >> .env
    grep -q '^APP_GID=' .env && sed -i.bak "s|^APP_GID=.*|APP_GID=$GID_VAL|" .env || echo "APP_GID=$GID_VAL" >> .env
    rm -f .env.bak
}
```

或者把数据目录所有者改成 1000：

```bash
sudo chown -R 1000:1000 ./data ./models 2>/dev/null || true
```

**磁盘小的机器**可以把上传上限调低（默认 2 GiB），比如把上限压到 500 MiB：

```bash
grep -q '^MAX_UPLOAD_BYTES=' .env \
  && sed -i.bak "s|^MAX_UPLOAD_BYTES=.*|MAX_UPLOAD_BYTES=524288000|" .env \
  || echo 'MAX_UPLOAD_BYTES=524288000' >> .env
rm -f .env.bak
```

### 3. 启动

```bash
docker compose up -d --build
```

### 4. 等模型下载完毕

首次启动会从 HuggingFace 下载约 5 GB 权重。你应该**周期性**（每 30 秒）检查日志：

```bash
docker logs --tail 20 voscript
```

关键信号：
- 看到 `Uvicorn running on http://0.0.0.0:8780` → 服务起来了
- 看到 `401 Client Error` 下载模型 → `HF_TOKEN` 错了
- 看到 `403 Forbidden` → 没接受 gated 模型条款，回到决策树的"检查 4"
- 看到 `np.NaN was removed` → 有人改了 `requirements.txt`，把 numpy 2.x 放进去了
- 超过 10 分钟还在下载 → 网络慢，建议加 `HF_ENDPOINT` 镜像

### 5. 健康检查

```bash
curl -sf http://localhost:8780/healthz
# 期望：{"ok":true}

# 鉴权有没有生效（不要用 `source .env`，值里若有 shell 元字符会被执行）
API_KEY=$(grep -E '^API_KEY=' .env | tail -n1 | cut -d= -f2-)
curl -sS http://localhost:8780/api/voiceprints -H "Authorization: Bearer $API_KEY"
# 期望：[]（首次一定是空）

# 确认没 key 会被拒
curl -sS -o /dev/null -w "%{http_code}\n" http://localhost:8780/api/voiceprints
# 期望：401
```

三个都符合预期 → 部署完成。

## 部署步骤（macOS，native venv）

不走 docker-compose，手动起一个 venv 服务。所有命令都在 **仓库根目录** 执行。

### 1. 克隆 + 装依赖

```bash
cd ~   # 或用户偏好位置
git clone https://github.com/MapleEve/voscript.git
cd voscript

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r app/requirements.txt
# 注意：torch==2.4.1 在 macOS 上解析到 CPU/MPS wheel，不是 CUDA。不用改 pin。
```

### 2. 准备运行环境

```bash
# 数据目录
export DATA_DIR="$(pwd)/data"
mkdir -p "$DATA_DIR"

# 密钥（向用户确认是否使用这个生成值）
export API_KEY_VALUE=$(openssl rand -hex 32)
export HF_TOKEN="${USER_SUPPLIED_HF_TOKEN:?need HF_TOKEN from user}"

# macOS 必须的固定项
export DEVICE=cpu
export WHISPER_MODEL=medium    # 不要偷懒用 large-v3
```

把这些写成一个 `run.sh` 放仓库根目录，让用户以后用同一个脚本启动：

```bash
cat > run.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export DATA_DIR="$(pwd)/data"
export DEVICE=cpu
export WHISPER_MODEL=medium
# 下面两行由部署时写入，不要提交到 git
export HF_TOKEN="__FILL_ME__"
export API_KEY="__FILL_ME__"
mkdir -p "$DATA_DIR"
cd app
exec uvicorn main:app --host 0.0.0.0 --port 8780
EOF
chmod +x run.sh

# 把 token / key 写进去（Python + 单次替换，避免 sed 转义问题）
: "${HF_TOKEN:?missing}"
: "${API_KEY_VALUE:?missing}"
HF_TOKEN="$HF_TOKEN" API_KEY="$API_KEY_VALUE" python3 - <<'PY'
import os, pathlib
p = pathlib.Path("run.sh")
text = p.read_text()
text = text.replace("__FILL_ME__", os.environ["HF_TOKEN"], 1)
text = text.replace("__FILL_ME__", os.environ["API_KEY"], 1)
p.write_text(text)
PY

# 加到 .gitignore，防止泄漏
grep -q '^run.sh$' .gitignore || echo 'run.sh' >> .gitignore
```

### 3. 启动 + 验证

```bash
./run.sh &    # 或者在 tmux / screen 里前台跑
sleep 30      # 等模型首次下载（5 GB，慢的话可能要更久）

# 健康检查
curl -sf http://localhost:8780/healthz
# → {"ok":true}

curl -sS http://localhost:8780/api/voiceprints -H "Authorization: Bearer $API_KEY_VALUE"
# → []
```

如果要做成**开机自启**，用户在 Apple Silicon Mac 上一般用 launchd。提供一份
`~/Library/LaunchAgents/com.openplaud.voscript.plist` 示例，但**不要**
替用户装上，让他自己决定。

> 提醒用户：Mac 合盖 / 睡眠时这个服务会停。持续跑的话需要"保持清醒"或改用
> 台式机。

## 验证容器不是 root 运行（0.2.0 新增）

```bash
docker exec voscript id
# 期望：uid=1000(app) gid=1000(app) groups=1000(app)
# 如果输出 uid=0(root) —— 说明镜像不是最新版，git pull + 重新 build。
```

## 验证 GPU 是真的用上了（Linux / WSL2 部署）

```bash
docker exec voscript python -c "import torch; print('cuda=', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# 期望：cuda= True NVIDIA ...
```

如果输出 `cuda= False`，检查 compose 里 GPU reservation、`nvidia-ctk runtime configure` 有没有执行。

## 向用户交付的清单

部署结束后，跟用户同步这些东西（一次性、清楚）：

1. ✅ **服务地址**：`http://<主机 IP 或域名>:8780`
2. ✅ **API_KEY**：把 `.env` 里生成的值完整告诉用户**一次**；提示 "后面请自己妥善保管"
3. ✅ 让用户在 OpenPlaud(Maple) "设置 → 转录" 里：
   - Private transcription base URL = 上面的服务地址
   - Private transcription API key = 同一个 API_KEY
4. ✅ **强提醒**：`:8780` **不要**直接暴露到公网，最好挂 VPN / 反代 + TLS /
   至少白名单。详见 [`security.zh.md`](./security.zh.md)。
5. ✅ 备份建议：提醒用户 `data/voiceprints/` 要定期备份

## 升级流程

当用户让你升级服务：

```bash
cd ~/voscript  # 或实际路径
git fetch origin
git diff --stat main origin/main   # 让用户看一下会变什么
git pull
docker compose up -d --build
docker logs --tail 40 voscript
curl -sf http://localhost:8780/healthz
```

如果 `git pull` 会覆盖用户本地未提交改动，**先停下来问用户**，不要 `git reset --hard`。

## 不要做的事

- ❌ 把 `HF_TOKEN`、`API_KEY` 回显到用户聊天记录之外的任何地方（日志、commit、PR）
- ❌ 把 `.env` `git add`（已经 gitignore，但别手动强加）
- ❌ 为了"启动成功"去掉 `requirements.txt` 里的版本 pin
- ❌ 为了节约磁盘删 `./models/` —— 那是模型权重缓存，删了下次要重下 5 GB
- ❌ 用 `docker rm -f voscript` 之后期待容器里手动装的包还在——记住
  `docker compose up --build` 之后会重建，一切以 `requirements.txt` 为准
- ❌ 不经用户同意就开一个 443 端口 / 反向代理 / 公网 DNS 记录

## 常见 followup

- **"OpenPlaud(Maple) 连不上这个服务"**
  → 检查：1) 两边 `API_KEY` 完全一致（无空格、无换行）；2) 主机防火墙放行了
  `8780`；3) OpenPlaud(Maple) 主机能不能 `curl` 通。
- **"能不能加个 HTTPS？"**
  → 推荐在前面挂一层 nginx/caddy/traefik 做 TLS 终止 + 证书自动续期。不要
  改 FastAPI 让它自己拿证书——维护成本高。
- **"GPU 0 被别的服务占了，能不能换 GPU 1？"**
  → 改 `.env` 里 `CUDA_VISIBLE_DEVICES=1`，然后重启 compose。

## 相关文档

- 给人看的安装文档 → [`quickstart.zh.md`](./quickstart.zh.md)
- 给 AI 看的接口使用 → [`ai-usage.zh.md`](./ai-usage.zh.md)
- 接口合同 → [`api.zh.md`](./api.zh.md)
- 安全策略 → [`security.zh.md`](./security.zh.md)
