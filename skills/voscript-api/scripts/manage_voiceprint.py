#!/usr/bin/env python3
"""管理单个声纹：查询 / 改名 / 删除。

- get:    GET    /api/voiceprints/{speaker_id}
- rename: PUT    /api/voiceprints/{speaker_id}/name   form: name
- delete: DELETE /api/voiceprints/{speaker_id}
"""

from __future__ import annotations

import argparse
import sys

from common import (
    VoScriptClient,
    VoScriptError,
    add_common_args,
    build_client_from_args,
    print_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="管理单个声纹：查询 / 改名 / 删除。")
    add_common_args(parser)
    parser.add_argument(
        "--action",
        required=True,
        choices=["get", "rename", "delete"],
        help="要执行的操作：get / rename / delete。",
    )
    parser.add_argument(
        "--speaker-id",
        required=True,
        help="声纹 ID。",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="新名字（仅 action=rename 时必填）。",
    )
    args = parser.parse_args()

    if args.action == "rename" and not args.name:
        print("错误: action=rename 时必须提供 --name。", file=sys.stderr)
        return 1

    try:
        client: VoScriptClient = build_client_from_args(args)

        if args.action == "get":
            resp = client.get(f"/api/voiceprints/{args.speaker_id}")
            print_json(resp)
            return 0

        if args.action == "rename":
            resp = client.put(
                f"/api/voiceprints/{args.speaker_id}/name",
                data={"name": args.name},
            )
            print(f"声纹 {args.speaker_id} 已改名为: {args.name}")
            return 0

        if args.action == "delete":
            resp = client.delete(f"/api/voiceprints/{args.speaker_id}")
            print(f"声纹 {args.speaker_id} 已删除")
            return 0

    except VoScriptError as exc:
        print(f"请求失败: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
