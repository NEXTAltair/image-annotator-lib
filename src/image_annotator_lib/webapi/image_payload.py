"""Shared image payload helpers for WebAPI requests."""

from __future__ import annotations

import base64


def build_base64_data_url(payload: bytes, mime_type: str) -> str:
    """Return an RFC 2397 data URL for an image payload."""

    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


__all__ = ["build_base64_data_url"]
