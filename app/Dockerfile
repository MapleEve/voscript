# TODO: pin digest — 运行：
#   docker pull pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
#   docker inspect pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime --format='{{index .RepoDigests 0}}'
# 然后改为：FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime@sha256:<digest>
# 不 pin digest 的话 mutable tag 可能导致构建不可重现（对应评审 CD-H6 / BP-M6）。
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
    # TODO: 待确认 fake_git hack 删除后可移除此依赖（评审 BP-L5）。
    # DeepFilterNet 在初始化时可能探测 `git`，若已完全去除该 hack，可删除此行。
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

COPY --chown=app:app . .

# HuggingFace cache lives under the app user's reach, not /root.
ENV HF_HOME=/cache \
    HUGGINGFACE_HUB_CACHE=/cache \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache

USER app

EXPOSE 8780

VOLUME ["/data", "/cache"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8780"]
