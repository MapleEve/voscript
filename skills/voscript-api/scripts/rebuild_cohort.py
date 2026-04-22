#!/usr/bin/env python3
"""重建声纹 cohort（自适应阈值背景集合）。

POST /api/voiceprints/rebuild-cohort
响应: {"cohort_size": N, "skipped": M, "saved_to": "path"}
"""

from __future__ import annotations

import argparse
import sys

from common import (
    VoScriptClient,
    VoScriptError,
    add_common_args,
    build_client_from_args,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="重建声纹 cohort（自适应阈值所用的背景嵌入集合）。"
    )
    add_common_args(parser)
    args = parser.parse_args()

    try:
        client: VoScriptClient = build_client_from_args(args)
        resp = client.post("/api/voiceprints/rebuild-cohort")
    except VoScriptError as exc:
        print(f"请求失败: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if not isinstance(resp, dict):
        print("服务器返回意外的数据格式：", file=sys.stderr)
        print(resp, file=sys.stderr)
        return 1

    cohort_size = resp.get("cohort_size", 0)
    skipped = resp.get("skipped", 0)
    saved_to = resp.get("saved_to", "")

    print("Cohort 重建完成")
    print(f"  cohort 大小 : {cohort_size}")
    print(f"  已跳过       : {skipped}")
    print(f"  保存路径     : {saved_to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
