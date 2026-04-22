#!/usr/bin/env python3
"""列出 VoScript 服务器上的所有声纹。

GET /api/voiceprints → [{id, name, sample_count, sample_spread, created_at, updated_at}]
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

from common import (
    VoScriptClient,
    VoScriptError,
    add_common_args,
    build_client_from_args,
    print_json,
)


COLUMNS = [
    ("id", "ID"),
    ("name", "名字"),
    ("sample_count", "样本数"),
    ("sample_spread", "样本离散度"),
    ("created_at", "创建时间"),
    ("updated_at", "更新时间"),
]


def _stringify(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _display_width(text: str) -> int:
    width = 0
    for ch in text:
        code = ord(ch)
        if (
            0x1100 <= code <= 0x115F
            or 0x2E80 <= code <= 0x303E
            or 0x3041 <= code <= 0x33FF
            or 0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xA000 <= code <= 0xA4CF
            or 0xAC00 <= code <= 0xD7A3
            or 0xF900 <= code <= 0xFAFF
            or 0xFE30 <= code <= 0xFE4F
            or 0xFF00 <= code <= 0xFF60
            or 0xFFE0 <= code <= 0xFFE6
        ):
            width += 2
        else:
            width += 1
    return width


def _pad(text: str, target_width: int) -> str:
    gap = target_width - _display_width(text)
    if gap <= 0:
        return text
    return text + " " * gap


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("（暂无声纹）")
        return

    widths = []
    for key, header in COLUMNS:
        max_value_width = max(
            (_display_width(_stringify(row.get(key))) for row in rows),
            default=0,
        )
        widths.append(max(_display_width(header), max_value_width))

    header_line = "  ".join(
        _pad(header, widths[i]) for i, (_, header) in enumerate(COLUMNS)
    )
    sep_line = "  ".join("-" * widths[i] for i in range(len(COLUMNS)))
    print(header_line)
    print(sep_line)

    for row in rows:
        cells = []
        for i, (key, _) in enumerate(COLUMNS):
            cells.append(_pad(_stringify(row.get(key)), widths[i]))
        print("  ".join(cells))


def main() -> int:
    parser = argparse.ArgumentParser(description="列出 VoScript 服务器上的所有声纹。")
    add_common_args(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 形式输出原始数据。",
    )
    args = parser.parse_args()

    try:
        client: VoScriptClient = build_client_from_args(args)
        data = client.get("/api/voiceprints")
    except VoScriptError as exc:
        print(f"请求失败: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print_json(data)
        return 0

    if not isinstance(data, list):
        print("服务器返回意外的数据格式：", file=sys.stderr)
        print_json(data)
        return 1

    _print_table(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
