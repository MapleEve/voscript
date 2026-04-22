"""TEST-C1: _atomic_write_json orphan .tmp cleanup.

Verifies that the atomic write helper in job_service:
  - leaves no .tmp file behind on success
  - cleans up the orphan .tmp file when json.dump raises mid-write
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../app"))

import pytest
from unittest.mock import patch


def test_atomic_write_json_no_tmp_on_success(tmp_path):
    from services.job_service import _atomic_write_json

    target = tmp_path / "out.json"
    _atomic_write_json(target, {"key": "value"})
    assert target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_json_cleans_tmp_on_json_dump_failure(tmp_path):
    from services.job_service import _atomic_write_json

    target = tmp_path / "out.json"
    with patch("json.dump", side_effect=ValueError("mock fail")):
        with pytest.raises(ValueError):
            _atomic_write_json(target, {"key": "value"})
    assert list(tmp_path.glob("*.tmp")) == [], "Orphan .tmp file was not cleaned up"
    assert not target.exists(), "Target file should not exist after failed write"
