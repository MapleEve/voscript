FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Optional mirrors for users behind geographic restrictions.
#   docker build --build-arg APT_MIRROR=https://mirrors.aliyun.com \
#                --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple ./app
ARG APT_MIRROR=""
ARG PIP_INDEX_URL=""

# UID/GID the container runs as. Defaults to 1000 so bind mounts from a
# typical Linux host (where the first non-root user is uid 1000) work
# out of the box. Hosts with a different owner should pass a build arg
# matching `id -u` of the user who owns DATA_DIR.
ARG APP_UID=1000
ARG APP_GID=1000

RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://archive.ubuntu.com|${APT_MIRROR}|g; s|http://security.ubuntu.com|${APT_MIRROR}|g" /etc/apt/sources.list || true; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user. An RCE inside the container only owns the service's
# own uid — it cannot read root-owned files mounted elsewhere on the host.
RUN groupadd -g ${APP_GID} app \
    && useradd -u ${APP_UID} -g ${APP_GID} -m -s /sbin/nologin app \
    && mkdir -p /cache /data \
    && chown -R app:app /cache /data

WORKDIR /app

COPY --chown=app:app requirements.txt .
RUN if [ -n "$PIP_INDEX_URL" ]; then \
        pip install --no-cache-dir -i "$PIP_INDEX_URL" -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# clearvoice (MossFormer2): install --no-deps to avoid huggingface_hub version conflict
# with pyannote 3.1.1 (which requires huggingface_hub<0.24). clearvoice's runtime
# needs beyond torch/torchaudio/numpy/soundfile (already installed) are minimal;
# einops is the main extra required for MossFormer2 tensor ops.
RUN if [ -n "$PIP_INDEX_URL" ]; then \
        pip install --no-cache-dir -i "$PIP_INDEX_URL" --no-deps clearvoice && \
        pip install --no-cache-dir -i "$PIP_INDEX_URL" einops; \
    else \
        pip install --no-cache-dir --no-deps clearvoice && \
        pip install --no-cache-dir einops; \
    fi

COPY --chown=app:app . .
RUN chown -R app:app /app

# HuggingFace cache lives under the app user's reach, not /root.
ENV HF_HOME=/cache \
    HUGGINGFACE_HUB_CACHE=/cache \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache

USER app

EXPOSE 8780

VOLUME ["/data", "/cache"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8780"]
