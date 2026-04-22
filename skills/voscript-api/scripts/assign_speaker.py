#!/usr/bin/env python3
"""将某个片段分配给一位具名说话人。

PUT /api/transcriptions/{tr_id}/segments/{seg_id}/speaker
    form: speaker_name, speaker_id(optional)
响应: {"ok": true}
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

from common import (
    VoScriptClient,
    VoScriptError,
    add_common_args,
    build_client_from_args,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="将转写中的某个片段分配给指定说话人。")
    add_common_args(parser)
    parser.add_argument(
        "--tr-id",
        required=True,
        help="转写任务 ID。",
    )
    parser.add_argument(
        "--seg-id",
        required=True,
        type=int,
        help="片段 ID（整数）。",
    )
    parser.add_argument(
        "--speaker-name",
        required=True,
        help="要分配的说话人名字。",
    )
    parser.add_argument(
        "--speaker-id",
        default=None,
        help="可选：关联的声纹 ID。",
    )
    args = parser.parse_args()

    form: Dict[str, Any] = {"speaker_name": args.speaker_name}
    if args.speaker_id:
        form["speaker_id"] = args.speaker_id

    path = f"/api/transcriptions/{args.tr_id}/segments/{args.seg_id}/speaker"

    try:
        client: VoScriptClient = build_client_from_args(args)
        resp = client.put(path, data=form)
    except VoScriptError as exc:
        print(f"请求失败: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if not (isinstance(resp, dict) and resp.get("ok")):
        print("服务器未返回 ok=true：", file=sys.stderr)
        print(resp, file=sys.stderr)
        return 1

    print(f"片段 {args.seg_id} 已分配给 {args.speaker_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
