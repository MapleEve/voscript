# 声纹调参参考

本文整理会影响说话人匹配的公开调参项，以及当前仍属于内部实现默认值、
尚未作为稳定 API 参数暴露的阈值。

如果你需要所有服务 env、Whisper/ASR、降噪、alignment、结果契约和
v0.7.4 验证口径的完整索引，请先看
[`configuration.zh.md`](./configuration.zh.md)。

## 环境变量

| 名称 | 默认值 | 作用范围 | 说明 |
| --- | ---: | --- | --- |
| `VOICEPRINT_THRESHOLD` | `0.75` | raw cosine 匹配 | raw cosine 模式的基础阈值。实际生效阈值会按已登记说话人的样本数和 sample_spread 动态调整。 |
| `DATA_DIR` | `/data` | 存储 | 转写结果、上传文件和 `voiceprints/` 的父目录。 |
| `EMBEDDING_DIM` | `256` | 声纹库 | 创建或加载向量索引时使用的 embedding 维度。不同维度的既有库不要混用。 |
| `DENOISE_MODEL` | `none` | 转写质量 | 会通过改变送入 diarization / embedding 的音频间接影响声纹 embedding。 |
| `DENOISE_SNR_THRESHOLD` | `10.0` | 转写质量 | 仅对 `deepfilternet` 生效，用于决定是否按 SNR 跳过 DeepFilterNet；`noisereduce` 不使用该 gate。 |
| `PYANNOTE_MIN_DURATION_OFF` | `0.5` | 说话人分离 | pyannote 停顿合并参数，用于减少短暂停顿附近的过度切分。 |
| `MIN_EMBED_DURATION` | `1.5` | 声纹 embedding | 短于该时长的 diarization turn 不参与 speaker embedding 提取。 |
| `MAX_EMBED_DURATION` | `10.0` | 声纹 embedding | 更长的 turn 会截断到该窗口后再提取 embedding。 |

## API 参数

| 端点 | 参数 | 默认值 | 说明 |
| --- | --- | ---: | --- |
| `POST /api/transcribe` | `language` | 自动检测 | 影响 ASR / alignment，不直接改变声纹阈值。 |
| `POST /api/transcribe` | `min_speakers`, `max_speakers` | `0` | 控制 diarization 说话人数范围；不合理的范围可能产生较差的说话人 embedding。 |
| `POST /api/transcribe` | `denoise_model`, `snr_threshold` | 服务默认值 | 省略 `denoise_model` 时使用 `DENOISE_MODEL`；显式 `denoise_model=none` 表示单次任务关闭降噪。显式 `snr_threshold` 只覆盖本次任务的 DeepFilterNet SNR gate。 |
| `POST /api/transcribe` | `no_repeat_ngram_size` | `0` | 仅用于 ASR 重复抑制，列在这里是为了完整覆盖转写调参。 |
| `POST /api/voiceprints/enroll` | `speaker_name`, `speaker_label`, 可选 `speaker_id` | 必填 / 可选 | 向声纹库添加样本。干净样本越多，校准越稳定。 |
| `POST /api/voiceprints/rebuild-cohort` | 无 | 不适用 | 从已持久化的转写 embedding 强制重建 AS-norm cohort。 |

## 当前内部默认值

这些值是当前代码默认值。它们面向运维和调参排查公开说明，但在明确暴露为参数前，
不应视为稳定公开 API。

| 调参项 | 默认值 | 影响 |
| --- | ---: | --- |
| raw 单样本放松 | `0.05` | raw cosine 单样本阈值为 `base - 0.05`，默认约 `0.70`。 |
| raw spread 放松 | `3.0 * sample_spread`，上限 `0.10` | raw cosine 多样本说话人的 sample_spread 越大，阈值越低。 |
| raw 绝对下限 | `0.60` | raw cosine 匹配永远不会接受低于该值的候选。 |
| AS-norm cohort 激活条件 | `10` 条 embedding | 低于该规模时，评分回退到 raw cosine 和 raw 动态阈值。 |
| AS-norm 操作阈值 | `0.5` | cohort 至少 `10` 条后使用的基础 z-score 阈值。 |
| AS-norm 单样本惩罚 | `+0.10` | AS-norm base 为 `0.5` 时，单样本说话人至少需要约 `0.60` 才会自动命名。 |
| AS-norm unknown-spread 惩罚 | `+0.05` | 没有 spread 元数据的历史多样本记录按保守方式处理。 |
| AS-norm 低样本惩罚 | 低于 `3` 个样本时，每少一个 `+0.025` | 两样本说话人需要比稳定说话人略强的证据。 |
| AS-norm spread 惩罚 | `0.50 * sample_spread`，上限 `0.10` | AS-norm 下更嘈杂的登记样本需要更强证据才会自动命名。 |
| AS-norm 稳定样本放松 | `-0.02` | 至少 `3` 个样本且 spread `<= 0.03` 的说话人可以略低于 base 命中。 |
| AS-norm top-1 / top-2 margin | `0.05` | normalized score 最高的候选必须与 normalized 第二名保持最小间隔，否则保持未命名。 |
| AS-norm cohort `top_n` | `200` | 用于 AS-norm 统计的最近 impostor 数量，上限为 cohort 实际大小。 |
| cohort 自动重建循环 | 每 `60s` 唤醒，`30s` 防抖 | 新登记样本通常会在约 `30-90s` 内进入匹配路径；cohort >=10 时才进入完整 AS-norm 评分，否则 raw cosine fallback。 |
| cohort 自动保护 | 保留更大 cohort | 后台自动重建不会用空转录源或更少的 embedding 覆盖已加载 / 已持久化的较大 cohort；手动 rebuild 仍按显式操作重建。 |

## AS-norm 调参建议

- raw cosine 阈值和 AS-norm 阈值要分开看。AS-norm score 是 z-score 类尺度，
  不要直接套用 raw cosine 的常量。
- AS-norm 激活时，候选会先计算 normalized score 并重新排序；阈值和 margin
  都基于 normalized top-1 / top-2，而不是 raw cosine top-1 / top-2。
- 为了保证生产精度，不建议在没有内部基准验证的情况下降低 AS-norm 单样本阈值。
  单样本候选的 `0.5713` 这类弱分数会被保守地留作未命名。
- 先补充干净样本，再考虑降低阈值。三到五条一致样本通常比激进调阈更可靠。
- 如果误识别集中发生在两个相似已登记说话人之间，应优先提高 AS-norm margin。
  只有在人工复核量过高且基准验证显示误识别仍可接受时，才考虑降低 margin。
- 只有 cohort 足够大且有代表性时才考虑提高 `top_n`。过小或偏置的 cohort
  应通过重建或扩充 cohort 解决，而不是单纯调大 `top_n`。
- `speaker_id = null` 表示需要人工复核，不应视作失败。服务会在样本稀疏或候选接近时保持保守。
