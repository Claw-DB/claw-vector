from __future__ import annotations

import os

import blake3
from fastapi import Header, HTTPException


def _load_valid_keys() -> set[str]:
    raw = os.getenv("CLAW_VECTOR_API_KEYS") or os.getenv("CLAW_API_KEYS") or ""
    return {value.strip().lower() for value in raw.split(",") if value.strip()}


def hash_api_key(api_key: str) -> str:
    return blake3.blake3(api_key.encode("utf-8")).hexdigest()


def get_api_key(x_claw_api_key: str | None = Header(default=None)) -> str:
    valid_hashes = _load_valid_keys()
    if not x_claw_api_key:
        raise HTTPException(status_code=401, detail="unauthorized")

    if hash_api_key(x_claw_api_key).lower() not in valid_hashes:
        raise HTTPException(status_code=401, detail="unauthorized")

    return x_claw_api_key
