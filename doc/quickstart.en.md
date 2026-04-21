# Quickstart

[简体中文](./quickstart.zh.md) | **English**

This guide is for first-time deployers. Expect 15–30 minutes, most of it
waiting for model weights to download.

## 0. Pick your deployment path

| Platform | Path | Quality | Notes |
| --- | --- | --- | --- |
| Linux + NVIDIA GPU | docker-compose (main flow below) | Best | Recommended, the main path of this doc |
| Windows 11 + WSL2 + NVIDIA GPU | docker-compose (Linux flow inside WSL2) | Best | See [0.2](#02-windows-11--wsl2) |
| macOS Apple Silicon (M1/M2/M3/M4) | **native venv, CPU-only** | Usable but slow | Docker Desktop on macOS **cannot pass through the GPU**; see [0.3](#03-macos-apple-silicon--intel) |
| macOS Intel | native venv, CPU-only | Usable but very slow | Same as Apple Silicon; see [0.3](#03-macos-apple-silicon--intel) |

CPU / low-VRAM deployments: **use `WHISPER_MODEL=medium`** (covered in step 2).
It's 3–4× faster than `large-v3` and quality stays acceptable, especially for
Chinese and English.

### HuggingFace prep (all platforms)

- Create a **read** token at <https://huggingface.co/settings/tokens> (starts
  with `hf_`). Tokens are account-level — **you can create it any time,
  order doesn't matter**.
- Click **Agree and access repository** at
  <https://huggingface.co/pyannote/speaker-diarization-3.1>.
- Do the same at <https://huggingface.co/pyannote/segmentation-3.0>.
- And at <https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM>
  (**added in 0.3.0** — used for speaker embeddings).

> Creating the token and accepting gated-model terms are **independent**.
> Order doesn't matter, but all three model agreements + a valid token
> must be in place to download weights: token without accepted terms →
> 403, accepted terms without token → 401.

## 0.1 Linux + NVIDIA GPU (main path)

- Docker 24+
- **NVIDIA Container Toolkit** (without it, compose fails with
  `could not select device driver`):
  ```bash
  # Ubuntu example
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker

  # verify
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

VRAM guidance:
- ≥ 12 GB → default `large-v3`
- 8–12 GB → `large-v3` still fits (~9 GB in practice), just don't share the
  GPU with another heavy job
- < 8 GB → set `WHISPER_MODEL=medium`

Then jump to [step 1](#1-clone-the-repo).

## 0.2 Windows 11 + WSL2

The officially supported path is **WSL2 + NVIDIA Container Toolkit**.
Docker Desktop's GPU passthrough is itself routed through WSL2 under the
hood, so the two are equivalent.

Prereqs:
- Windows 11, or Windows 10 21H2+
- WSL2 Ubuntu installed (`wsl --install -d Ubuntu`)
- NVIDIA driver ≥ 470 on Windows
- Docker available inside WSL2 (either install docker directly in WSL2, or
  install Docker Desktop on Windows and enable "Use WSL 2 based engine" +
  "Enable integration with my default WSL distro")

From then on, **every command runs inside the WSL2 Ubuntu shell**. Follow
[0.1 Linux](#01-linux--nvidia-gpu-main-path). Verify:

```bash
# inside WSL2 Ubuntu
nvidia-smi                        # should see your NVIDIA GPU
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If both work, the Linux flow applies verbatim.

## 0.3 macOS (Apple Silicon / Intel)

**Important**: Docker Desktop on macOS **cannot pass a GPU into containers**
(no CUDA, no Metal). The docker-compose path on macOS is CPU-only and
`large-v3` is effectively unusable there. Use **native venv + CPU** and
drop the model to `medium`.

Prereqs:
- Python 3.11 (recommend `brew install python@3.11`)
- ffmpeg (`brew install ffmpeg`)
- libsndfile (`brew install libsndfile`)
- 16 GB RAM or more (Apple Silicon unified memory counts)

```bash
# 1. clone
git clone https://github.com/MapleEve/voscript.git
cd voscript

# 2. create venv at the repo root
python3.11 -m venv .venv
source .venv/bin/activate

# 3. install deps (on macOS, torch==2.4.1 resolves to the CPU/MPS wheel, not CUDA)
pip install --upgrade pip
pip install -r app/requirements.txt

# 4. set env vars
export HF_TOKEN=hf_your_token
export API_KEY=$(openssl rand -hex 32)
export DEVICE=cpu                 # macOS must be cpu (pyannote's MPS support is incomplete)
export WHISPER_MODEL=medium       # large-v3 on CPU is too slow
export DATA_DIR=$(pwd)/data
mkdir -p "$DATA_DIR"

# Note this API_KEY — OpenPlaud(Maple) needs the exact same value

# 5. launch
cd app
uvicorn main:app --host 0.0.0.0 --port 8780
```

Expected performance (ballpark):
- M2 Pro / M3 Pro + `medium` + 1 minute of audio ≈ 30–60 s
- M1 / Intel + `medium` + 1 minute of audio ≈ 1.5–3 min
- `large-v3` on CPU is 3–5× slower. **Not recommended.**

Known limitations:
- Docker-compose path is not supported on macOS
- No MPS acceleration (pyannote 3.1 has unimplemented ops on MPS, either
  errors or silently falls back to CPU)
- If you have access to a Linux / Windows host with an NVIDIA GPU, run the
  service there and use Mac as the client

Everything after this point (config, wiring into OpenPlaud(Maple)) is the
same, just **skip every docker step**.

## 1. Clone the repo

```bash
git clone https://github.com/MapleEve/voscript.git
cd voscript
```

## 2. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env`. At minimum fill in:

```env
HF_TOKEN=hf_your_token
API_KEY=a_long_random_string_e.g._openssl_rand_hex_32
```

If you're short on VRAM (< 12 GB), or deploying on macOS / CPU-only, drop
the model one size:

```env
WHISPER_MODEL=medium
```

Choices: `tiny / base / small / medium / large-v3`. `medium` gives a
~3–4× speed-up with only a small quality drop, especially for Chinese and
English.

If you are on a China network, also add:

```env
HF_ENDPOINT=https://hf-mirror.com
```

> Generate a strong API key: `openssl rand -hex 32`

Every other env var has a sane default — see [`.env.example`](../.env.example)
for the full list. A few worth knowing about:

| Variable | Default | Effect |
| --- | --- | --- |
| `MAX_UPLOAD_BYTES` | `2147483648` (2 GiB) | Per-request upload cap; requests past this get `HTTP 413` |
| `APP_UID` | `1000` | uid the container runs as — must match the owner of `DATA_DIR` on the host |
| `APP_GID` | `1000` | same, gid |
| `JOBS_MAX_CACHE` | `200` | LRU cap for the in-memory job dictionary; evicted jobs remain queryable via disk status.json |
| `FFMPEG_TIMEOUT_SEC` | `1800` | Timeout in seconds for ffmpeg conversion; returns 504 on expiry |
| `ALLOW_NO_AUTH` | `0` | Set to 1 to suppress the startup warning when no API_KEY is configured (explicitly confirms unauthenticated mode) |

### Host directory ownership

The container runs as uid 1000 by default, so `DATA_DIR` (default
`./data`) and `MODEL_CACHE_DIR` (default `./models`) must be writable
by uid 1000 on the host. On most Linux distros the first regular user
is uid 1000, so plain `mkdir -p data models` before `docker compose
up` just works. If your user is a different uid, pick one:

```bash
# A. change the host dirs to 1000:1000
sudo chown -R 1000:1000 ./data ./models

# B. or tell the container to use YOUR uid
echo "APP_UID=$(id -u)" >> .env
echo "APP_GID=$(id -g)" >> .env
```

## 3. Start the service

```bash
docker compose up -d --build
```

The first boot downloads ~5 GB of model weights into `./models/`. Watch
progress with:

```bash
docker logs -f voscript
```

You are good when you see `Uvicorn running on http://0.0.0.0:8780`.

Or run the bundled helper:

```bash
./scripts/deploy.sh
```

It checks `.env`, starts the container, and waits for `/healthz`.

## 4. Verify the deployment

```bash
# Health check (always unauthenticated)
curl -sf http://localhost:8780/healthz
# → {"ok":true}

# Any /api/* call needs API_KEY
curl -sS http://localhost:8780/api/voiceprints \
    -H "Authorization: Bearer $API_KEY"
# → [] (empty on first boot)
```

Open <http://localhost:8780/> in a browser for a minimal web UI you can
upload audio to.

## 5. Wire it into OpenPlaud(Maple)

In OpenPlaud(Maple) → Settings → Transcription, set:

- **Private transcription base URL**: `http://<host>:8780`
- **Private transcription API key**: the **exact** `API_KEY` from `.env`

Once saved, the OpenPlaud(Maple) worker will route every recording through
this service. See [`api.en.md`](./api.en.md) for the full contract.

## Upgrades

```bash
cd voscript
git pull
docker compose up -d --build
```

Model weights in `./models/` are cached, rebuild won't redownload them.

### Upgrading from 0.2.x to 0.3.0

Two extra steps:

1. **Accept one more HuggingFace gated model** at
   <https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM> —
   WeSpeaker replaces ECAPA-TDNN for speaker embeddings. Without the
   click-through you'll get a 403 on first boot.
2. **Re-enroll every existing voiceprint.** 0.3.0 uses a new embedding
   space (WeSpeaker ≠ ECAPA, cosine distances between the two spaces
   are meaningless), so:
   - Quickest: `rm data/voiceprints/voiceprints.db`, let the container
     rebuild it empty, then re-enroll from fresh transcriptions.
   - Or per-speaker:
     `curl -X DELETE -H "Authorization: Bearer $API_KEY" http://host:8780/api/voiceprints/<spk_id>`
     for each enrolled id, then re-enroll.

Legacy `index.json` + `.npy` files from 0.2.x are auto-imported into the
new sqlite store on first boot — that doesn't lose data, but the
imported embeddings are still ECAPA-based and won't match any
WeSpeaker-generated queries. You still have to re-enroll.

## Troubleshooting

### `nvidia-smi` not found inside container
→ NVIDIA Container Toolkit missing or Docker wasn't restarted. Redo step 0.

### `403 Forbidden` downloading pyannote models
→ Gated-model terms not accepted. Revisit
[HuggingFace prep in step 0](#huggingface-prep-all-platforms).

### `401 Unauthorized` downloading pyannote models
→ `HF_TOKEN` missing, wrong, or expired. Check `.env`.

### macOS runs but is painfully slow
→ Confirm `DEVICE=cpu` and `WHISPER_MODEL=medium`. Large models on CPU
really are this slow — consider running the service on a Linux/Windows
host with an NVIDIA GPU instead.

### Crashes with `np.NaN was removed`
→ Your `requirements.txt` has been edited and numpy upgraded to 2.x. Keep
the `numpy<2.0` pin.

### Service is up but OpenPlaud(Maple) can't reach it
→ Check that `API_KEY` matches **exactly** on both sides (case/whitespace),
and that OpenPlaud(Maple)'s host can actually reach `:8780` (firewall,
docker networks).

### What do I back up?
→ Just `data/voiceprints/`. Everything else can be re-derived from the
original audio.

See [`security.en.md`](./security.en.md) for deployment-risk details.
