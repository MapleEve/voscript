"""Health-check endpoint."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pathlib import Path

router = APIRouter()


@router.get("/healthz")
async def healthz():
    return {"ok": True}


@router.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text(encoding="utf-8")
