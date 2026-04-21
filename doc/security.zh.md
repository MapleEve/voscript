# 安全策略

**简体中文** | [English](./security.en.md)

## 支持的版本

只支持 `main` 分支。请跑最新发布的镜像，或者从 `main` 重新构建。

## 威胁模型

这个服务会上传音频、运行说话人分离、存储说话人声纹。它的定位是**可信部署
环境**。默认情况下，任何能访问 `:8780` 的客户端在拿到 `API_KEY` 后可以：

- 上传音频到 `data/uploads/`，触发 GPU 推理
- 读取 `data/transcriptions/` 下所有转录结果（明文 + 时间戳）
- 操作所有已登记声纹（持久化的说话人 embedding，属于生物特征数据）

请把这个服务当成一个内部数据库对待。

## 内置的硬化（默认启用）

当前版本（0.6.0）默认开启以下保护：

1. **容器以非 root 用户运行**。Dockerfile 创建 `app` 用户（uid/gid 1000，
   可通过 `APP_UID`/`APP_GID` 覆盖），`USER app`。即使服务代码被 RCE，
   拿到的也只是这个低权限账号，读不到宿主机上 root 所有的敏感文件。
2. **上传体积上限 `MAX_UPLOAD_BYTES`**（默认 2 GiB）。分块流式读取，累计
   超限直接 `413` 并删除半截文件。磁盘耗尽型 DoS 被阻断。
3. **上传文件名清洗**。客户端提供的 `filename` 只留最末一段，
   `../../etc/passwd.wav` 之类的目录片段全部剥掉。
4. **ffmpeg argv 加 `--`**。关掉选项解析，阻断 `-Y.mp4` 这类文件名注入。
5. **鉴权常量时间比较**。`hmac.compare_digest`，消除时序侧信道。
6. **原子化、加锁的声纹库**。底层使用 SQLite WAL 模式，并发写入天然原子；
   进程级 `threading.RLock` 序列化多线程写操作，并发 enroll/delete 不丢数据。
7. **`np.load(..., allow_pickle=False)`**。默认关闭 numpy pickle 反序列化，
   堵掉类似 `torch.load` 的 RCE 路径。
8. **精确匹配 `/docs` / `/redoc` / `/openapi.json`**。前缀绕过
   （`/docsXYZ`）会被 401 拒绝。
9. **路径穿越防护**：`safe_tr_dir()` 对所有 `tr_id` 做正则 `^tr_[A-Za-z0-9_-]{1,64}$` + `resolve()` 前缀校验；`safe_speaker_label()` 同样限制允许字符集，防止 `../../etc/passwd` 类攻击
10. **日志注入防护**：`safe_log_filename()` 清除控制字符，防止恶意文件名污染日志
11. **路由参数强校验**：FastAPI `Path(pattern=...)` 直接拒绝格式非法的 id，路由函数体内无需二次检查
12. **ffmpeg 超时**：`FFMPEG_TIMEOUT_SEC`（默认 1800 s）防止畸形音频使 ffmpeg 卡死占用进程
13. **Pickle 防护**：`np.load(allow_pickle=False)` 防止恶意 `.npy` 执行任意代码（声纹向量加载）
14. **零向量防御**：声纹 `identify()` 对全零 embedding 提前返回，避免 AS-norm 分支产生错误匹配

## 部署侧必须做的硬化

代码里做不到的事情，部署方要补齐：

1. **设 `API_KEY`**。不设的话服务会无认证接收所有请求，并在启动时打 warning。
   任何不是纯内网可信网段的部署都 **必须** 把这个环境变量设为一串长随机字符串。
   客户端通过 `Authorization: Bearer <key>` 或 `X-API-Key: <key>` 带上它。
   - 生成：`openssl rand -hex 32`
   - 如果是可信内网环境确实不需要鉴权，设 `ALLOW_NO_AUTH=1` 可抑制启动 warning（该变量不提供任何鉴权，仅声明"我了解无鉴权的含义"）。
2. **`.env` 永远不要提交到 git**。仓库里只该有 `.env.example`。
3. **不要把 `:8780` 直接暴露到公网**。请挂在 VPN 后面，或加一层带 TLS 的
   反向代理，或者至少加 IP 白名单。`API_KEY` 单独用不能替代传输加密。
4. **HuggingFace token 不要进日志、不要打进镜像**。它只在运行时通过
   `HF_TOKEN` 被读一次，用来下载 pyannote 模型，除此之外无其他用途。
5. **请备份 `data/voiceprints/`**。声纹属于生物特征数据，丢了不光要重新
   登记，泄漏也更严重。
6. **宿主目录所有者要匹配 `APP_UID`/`APP_GID`**。容器默认以 uid 1000 运行，
   如果你的 `DATA_DIR` 是其他用户创建的，请 `chown -R 1000:1000`，或者
   在 `.env` 里把 `APP_UID`/`APP_GID` 改成实际所有者。

## 已知限制 / 暂不覆盖

- **没有内置的失败次数锁定 / 速率限制**。单租户 + 长随机 API_KEY 这个
  模型下可接受，key 一旦泄漏爆破拦不住。真要分担风险，在反向代理上
  加限速。
- **没有 TLS**。内网部署默认不加密，务必把这个端口只暴露在可信网段。
  上到公网请走 nginx/caddy/traefik 反代。
- **`server: uvicorn` 响应头**。小的指纹信息泄漏，无实际利用面，没做
  特殊处理。

## 漏洞上报

请在 GitHub 提交 private security advisory，或者发邮件给仓库维护者。
**不要** 在公开 issue 里提未修复的漏洞。
