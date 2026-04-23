"""Upload persistence and content-addressed artifact indexing for audio."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path

from config import TRANSCRIPTIONS_DIR
from pipeline.contracts import (
    AudioArtifactIndex,
    SavedUploadArtifact,
    UploadPersistenceRequest,
)

_hash_index_thread_lock = threading.Lock()


def _atomic_write_json(path: Path, data: dict, **json_kwargs) -> None:
    """Write *data* to *path* via an atomic temp-file rename."""

    content = json.dumps(data, **json_kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        ) as file_obj:
            tmp_path = file_obj.name
            file_obj.write(content)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _with_file_lock(path: Path, func):
    """Execute *func* while holding an exclusive lock for *path*."""

    lock_path = str(path) + ".lock"
    with _hash_index_thread_lock:
        try:
            with open(lock_path, "w") as lock_file:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    return func()
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except (AttributeError, OSError):
            return func()


class JsonAudioArtifactIndex(AudioArtifactIndex):
    """Persist upload hashes in a JSON map guarded by a file lock."""

    def __init__(self, index_path: Path | None = None):
        self.index_path = index_path or (TRANSCRIPTIONS_DIR / "hash_index.json")

    def compute_file_hash(self, path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as file_obj:
            while chunk := file_obj.read(1 << 20):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def persist_upload(
        self, request: UploadPersistenceRequest
    ) -> SavedUploadArtifact:
        import aiofiles

        sha256 = hashlib.sha256()
        size = 0
        async with aiofiles.open(request.save_path, "wb") as file_obj:
            while True:
                chunk = await request.file.read(request.chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                if size > request.max_bytes:
                    raise ValueError(
                        f"Upload exceeds MAX_UPLOAD_BYTES ({request.max_bytes} bytes)"
                    )
                await file_obj.write(chunk)
                sha256.update(chunk)
        return SavedUploadArtifact(
            path=request.save_path,
            size_bytes=size,
            file_hash=sha256.hexdigest(),
        )

    def lookup(self, file_hash: str) -> str | None:
        def _do():
            if not self.index_path.exists():
                return None
            return json.loads(self.index_path.read_text()).get(file_hash)

        tr_id = _with_file_lock(self.index_path, _do)
        if tr_id and (TRANSCRIPTIONS_DIR / tr_id / "result.json").exists():
            return tr_id
        return None

    def register(self, file_hash: str, artifact_id: str) -> None:
        def _do():
            index = (
                json.loads(self.index_path.read_text())
                if self.index_path.exists()
                else {}
            )
            index[file_hash] = artifact_id
            _atomic_write_json(self.index_path, index, indent=2)

        _with_file_lock(self.index_path, _do)


default_audio_artifact_index = JsonAudioArtifactIndex()


def compute_file_hash(path: Path) -> str:
    return default_audio_artifact_index.compute_file_hash(path)


async def save_upload_and_hash(
    file, save_path: Path, max_bytes: int, chunk_size: int
) -> tuple[int, str]:
    request = UploadPersistenceRequest(
        file=file,
        save_path=save_path,
        max_bytes=max_bytes,
        chunk_size=chunk_size,
    )
    artifact = await default_audio_artifact_index.persist_upload(request)
    return artifact.size_bytes, artifact.file_hash


def lookup_hash(file_hash: str) -> str | None:
    return default_audio_artifact_index.lookup(file_hash)


def register_hash(file_hash: str, tr_id: str) -> None:
    default_audio_artifact_index.register(file_hash, tr_id)


__all__ = [
    "JsonAudioArtifactIndex",
    "compute_file_hash",
    "default_audio_artifact_index",
    "lookup_hash",
    "register_hash",
    "save_upload_and_hash",
]
