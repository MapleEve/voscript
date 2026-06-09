"""Optional-first schema-version helpers for result/status artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

SCHEMA_VERSION_KEY = "schema_version"
OPTIONAL_FIRST_SCHEMA_POLICY = "optional_first"

_VERSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def read_optional_schema_version(payload: Mapping[str, Any] | None) -> str | None:
    """Read an optional schema_version without requiring legacy artifacts to set it."""

    if not isinstance(payload, Mapping):
        return None
    value = payload.get(SCHEMA_VERSION_KEY)
    if value is None:
        return None
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise ValueError("schema_version must be a short public-safe string")
    return value


def attach_optional_schema_version(
    payload: Mapping[str, Any],
    schema_version: str | None,
) -> dict[str, Any]:
    """Return a copy with schema_version only when a stable version is needed."""

    result = dict(payload)
    if schema_version is not None:
        if not _VERSION_RE.fullmatch(schema_version):
            raise ValueError("schema_version must be a short public-safe string")
        result[SCHEMA_VERSION_KEY] = schema_version
    return result


__all__ = [
    "OPTIONAL_FIRST_SCHEMA_POLICY",
    "SCHEMA_VERSION_KEY",
    "attach_optional_schema_version",
    "read_optional_schema_version",
]
