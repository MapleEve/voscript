"""E2E fixtures - 真实调用 localhost:8780，需要服务在线。"""

import pytest
import wave
import os
import tempfile
import time
import json
import urllib.request
import urllib.parse

BASE_URL = os.getenv("VOSCRIPT_URL", "http://localhost:8780")
API_KEY = os.getenv("VOSCRIPT_KEY") or os.getenv("VOSCRIPT_API_KEY") or ""
POLL_INTERVAL = 10  # 秒
POLL_TIMEOUT = 300  # 5 分钟


def _headers():
    if not API_KEY:
        raise RuntimeError("VOSCRIPT_KEY or VOSCRIPT_API_KEY is required")
    return {"X-API-Key": API_KEY}


def _get(path):
    req = urllib.request.Request(BASE_URL + path, headers=_headers())
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _upload(wav_path, extra_fields: dict):
    """multipart/form-data 上传"""
    boundary = "e2etestboundary"
    body = b""
    for k, v in extra_fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    with open(wav_path, "rb") as f:
        data = f.read()
    fn = os.path.basename(wav_path)
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fn}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode()
    body += data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE_URL + "/api/transcribe",
        data=body,
        headers={
            **_headers(),
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _poll(job_id, timeout=POLL_TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        data = _get(f"/api/jobs/{job_id}")
        if data["status"] == "completed":
            return data.get("result") or _get(f"/api/transcriptions/{job_id}")
        if data["status"] == "failed":
            raise RuntimeError(f"Job {job_id} failed: {data.get('error', '')}")
    raise TimeoutError(f"Job {job_id} timed out after {timeout}s")


@pytest.fixture(scope="session")
def server_url():
    """Verify server is accessible."""
    if not API_KEY:
        pytest.skip("VOSCRIPT_KEY or VOSCRIPT_API_KEY is required for live E2E")
    try:
        _get("/api/transcriptions")
        return BASE_URL
    except Exception as e:
        pytest.skip(f"voscript not accessible at {BASE_URL}: {e}")


@pytest.fixture(scope="session")
def test_wav(tmp_path_factory):
    """3 秒静音 WAV，用于验证 API schema（不用于内容质量测试）."""
    p = tmp_path_factory.mktemp("e2e") / "silence_3s.wav"
    with wave.open(str(p), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 3)
    return str(p)


@pytest.fixture(scope="session")
def method_a_result(server_url, test_wav):
    """Run Method A (OSD only) on test_wav, return full result."""
    resp = _upload(
        test_wav,
        {
            "language": "zh",
            "osd": "true",
            "osd_onset": "0.08",
            "separate_speech": "false",
        },
    )
    job_id = resp["id"]
    # If deduplicated, get existing result
    if resp.get("deduplicated"):
        return _get(f"/api/transcriptions/{job_id}")
    return _poll(job_id)


@pytest.fixture(scope="session")
def method_b_result(server_url, test_wav):
    """Run Method B (OSD + MossFormer2) on test_wav, return full result."""
    # Use different filename to avoid dedup with method_a
    import shutil

    tmp = tempfile.NamedTemporaryFile(suffix="_sep.wav", delete=False)
    shutil.copy(test_wav, tmp.name)
    try:
        resp = _upload(
            tmp.name,
            {
                "language": "zh",
                "osd": "true",
                "osd_onset": "0.08",
                "separate_speech": "true",
            },
        )
    finally:
        os.unlink(tmp.name)
    job_id = resp["id"]
    if resp.get("deduplicated"):
        return _get(f"/api/transcriptions/{job_id}")
    return _poll(job_id)
