#!/usr/bin/env python3
"""根据某段转写中的说话人标签注册或更新声纹。

POST /api/voiceprints/enroll
    form: tr_id, speaker_label, speaker_name, speaker_id(optional)
响应: {"action": "created"|"updated", "speaker_id": "..."}
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
    parser = argparse.ArgumentParser(
        description="从转写结果注册/更新声纹（voiceprint enroll）。"
    )
    add_common_args(parser)
    parser.add_argument(
        "--tr-id",
        required=True,
        help="转写任务 ID。",
    )
    parser.add_argument(
        "--speaker-label",
        required=True,
        help="转写中原始的说话人标签，例如 SPEAKER_00。",
    )
    parser.add_argument(
        "--speaker-name",
        required=True,
        help="要给这位说话人的名字。",
    )
    parser.add_argument(
        "--speaker-id",
        default=None,
        help="已有声纹 ID（传入时会更新该声纹而非新建）。",
    )
    args = parser.parse_args()

    form: Dict[str, Any] = {
        "tr_id": args.tr_id,
        "speaker_label": args.speaker_label,
        "speaker_name": args.speaker_name,
    }
    if args.speaker_id:
        form["speaker_id"] = args.speaker_id

    try:
        client: VoScriptClient = build_client_from_args(args)
        resp = client.post("/api/voiceprints/enroll", data=form)
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

    action = resp.get("action", "unknown")
    speaker_id = resp.get("speaker_id", "")
    action_cn = {"created": "新建", "updated": "更新"}.get(action, action)
    print(f"声纹已{action_cn}: speaker_id = {speaker_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
