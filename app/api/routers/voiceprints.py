"""Voiceprint management endpoints.

All routes under /api/voiceprints/*.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi import Path as FPath

from api.deps import get_db
from config import TRANSCRIPTIONS_DIR
from services.audio_service import safe_speaker_label, safe_tr_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.post("/voiceprints/enroll")
async def enroll_speaker(
    request: Request,
    tr_id: str = Form(...),
    speaker_label: str = Form(...),
    speaker_name: str = Form(...),
    speaker_id: str = Form(None),
):
    """Enroll or update a voiceprint from a transcription's speaker embedding."""
    import numpy as np

    voiceprint_db = get_db(request)

    # SEC-C2: validate both tr_id and speaker_label before building any path.
    safe_label = safe_speaker_label(speaker_label)
    emb_path = safe_tr_dir(tr_id) / f"emb_{safe_label}.npy"
    if not emb_path.exists():
        raise HTTPException(404, "Embedding not found for this speaker label")
    # SEC-C1: allow_pickle=False prevents arbitrary code execution via
    # a crafted .npy file that embeds a pickle payload (CVSS 9.1).
    embedding = np.load(emb_path, allow_pickle=False)

    if speaker_id and voiceprint_db.get_speaker(speaker_id):
        voiceprint_db.update_speaker(speaker_id, embedding, name=speaker_name)
        return {"action": "updated", "speaker_id": speaker_id}
    else:
        new_id = voiceprint_db.add_speaker(speaker_name, embedding)
        return {"action": "created", "speaker_id": new_id}


@router.get("/voiceprints")
async def list_voiceprints(request: Request):
    return get_db(request).list_speakers()


@router.post("/voiceprints/rebuild-cohort")
async def rebuild_cohort(request: Request):
    """Rebuild the AS-norm cohort from all processed transcriptions."""
    voiceprint_db = get_db(request)
    cohort_path = TRANSCRIPTIONS_DIR / "asnorm_cohort.npy"
    n = voiceprint_db.build_cohort_from_transcriptions(
        str(TRANSCRIPTIONS_DIR), save_path=str(cohort_path)
    )
    # [CQ-M10] 报告跳过/损坏的文件数，让调用方看到 cohort 的实际覆盖情况
    skipped = getattr(voiceprint_db, "last_cohort_skipped", 0)
    return {
        "cohort_size": n,
        "skipped": skipped,
        "saved_to": str(cohort_path),
    }


@router.get("/voiceprints/{speaker_id}")
async def get_voiceprint(
    speaker_id: Annotated[str, FPath(pattern=r"^spk_[A-Za-z0-9_-]{1,64}$")],
    request: Request,
):
    speaker = get_db(request).get_speaker(speaker_id)
    if not speaker:
        raise HTTPException(404, "Speaker not found")
    return speaker


@router.delete("/voiceprints/{speaker_id}")
async def delete_voiceprint(
    speaker_id: Annotated[str, FPath(pattern=r"^spk_[A-Za-z0-9_-]{1,64}$")],
    request: Request,
):
    try:
        get_db(request).delete_speaker(speaker_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"ok": True}


@router.put("/voiceprints/{speaker_id}/name")
async def rename_voiceprint(
    speaker_id: Annotated[str, FPath(pattern=r"^spk_[A-Za-z0-9_-]{1,64}$")],
    request: Request,
    name: str = Form(...),
):
    try:
        get_db(request).rename_speaker(speaker_id, name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"ok": True}
