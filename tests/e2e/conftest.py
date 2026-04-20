# tests/e2e/conftest.py
import os, time, requests, pytest

# Config from env vars
VOSCRIPT_URL = os.environ.get("VOSCRIPT_URL", "http://localhost:8780")
VOSCRIPT_API_KEY = os.environ.get("VOSCRIPT_API_KEY", "")
VOSCRIPT_TEST_AUDIO = os.environ.get("VOSCRIPT_TEST_AUDIO", "")  # path to real audio


@pytest.fixture(scope="session")
def server_url():
    return VOSCRIPT_URL


@pytest.fixture(scope="session")
def api_headers():
    if VOSCRIPT_API_KEY:
        return {"Authorization": f"Bearer {VOSCRIPT_API_KEY}"}
    return {}


@pytest.fixture(scope="session")
def base_transcription(server_url, api_headers):
    """Upload audio once, return (job_id, tr_id, result) - shared by both methods via deduplication."""
    audio_path = VOSCRIPT_TEST_AUDIO
    if not audio_path or not os.path.exists(audio_path):
        pytest.skip("VOSCRIPT_TEST_AUDIO not set or file not found")

    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{server_url}/api/transcribe",
            headers=api_headers,
            files={"file": f},
            data={"osd": "true", "max_speakers": "4"},
        )
    resp.raise_for_status()
    job_id = resp.json()["id"]

    # Poll until completed (max 20 min)
    for _ in range(240):
        r = requests.get(f"{server_url}/api/jobs/{job_id}", headers=api_headers)
        r.raise_for_status()
        data = r.json()
        if data["status"] == "completed":
            tr_id = data["result"]["id"]
            return job_id, tr_id, data["result"]
        if data["status"] == "failed":
            pytest.fail(f"Transcription failed: {data.get('error', 'unknown')}")
        time.sleep(5)

    pytest.fail("Transcription timed out after 20 minutes")


@pytest.fixture(scope="session")
def method_a_result(server_url, api_headers, base_transcription):
    """Method A: transcription + OSD analysis only."""
    job_id, tr_id, _ = base_transcription

    resp = requests.post(
        f"{server_url}/api/transcriptions/{tr_id}/analyze-overlap",
        headers=api_headers,
        data={"onset": "0.5"},
    )
    resp.raise_for_status()
    overlap_data = resp.json()

    # Fetch full updated result
    r = requests.get(f"{server_url}/api/jobs/{job_id}", headers=api_headers)
    result = r.json()["result"]
    result["_overlap_response"] = overlap_data
    return result


@pytest.fixture(scope="session")
def method_b_result(server_url, api_headers, base_transcription, method_a_result):
    """Method B: transcription + OSD analysis + MossFormer2 segment separation."""
    job_id, tr_id, _ = base_transcription

    resp = requests.post(
        f"{server_url}/api/transcriptions/{tr_id}/separate-segments",
        headers=api_headers,
        data={"onset": "0.08", "min_duration": "0.5"},
    )
    resp.raise_for_status()
    sep_data = resp.json()

    # Fetch full updated result
    r = requests.get(f"{server_url}/api/jobs/{job_id}", headers=api_headers)
    result = r.json()["result"]
    result["_separation_response"] = sep_data
    return result
