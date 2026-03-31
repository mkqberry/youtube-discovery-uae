from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


_SHORTS_RE = re.compile(r"#shorts|#short|\bshorts?\b", re.IGNORECASE)


def parse_iso8601_duration_to_seconds(duration: str) -> int:
    """
    Parse YouTube ISO8601 duration (e.g., PT1H2M3S) -> seconds.
    """
    if not duration:
        return 0
    if duration.isdigit():
        return int(duration)
    d = duration.strip()
    if not d.startswith("PT"):
        return 0
    d = d[2:]
    total = 0
    num = ""
    for ch in d:
        if ch.isdigit():
            num += ch
            continue
        if not num:
            continue
        v = int(num)
        num = ""
        if ch == "H":
            total += v * 3600
        elif ch == "M":
            total += v * 60
        elif ch == "S":
            total += v
    return total


def basic_video_filters(
    video_item: Dict[str, Any],
    *,
    min_duration_seconds: int,
    max_duration_seconds: int,
    drop_if_live: bool,
    drop_shorts: bool,
) -> Tuple[bool, List[str], int]:
    """
    Pure function: decide keep/drop based on basic metadata.
    Returns (keep, reason_codes, duration_seconds).
    """
    reasons: List[str] = []
    sn = video_item.get("snippet", {}) or {}
    cd = video_item.get("contentDetails", {}) or {}
    st = video_item.get("status", {}) or {}

    title = str(sn.get("title", "") or "")
    live = str(sn.get("liveBroadcastContent", "none") or "none")
    privacy = str(st.get("privacyStatus", "public") or "public")

    dur_s = parse_iso8601_duration_to_seconds(str(cd.get("duration", "") or ""))

    if privacy != "public":
        return False, ["drop:not_public"], dur_s

    if drop_if_live and live in ("live", "upcoming"):
        return False, ["drop:live"], dur_s

    if dur_s and dur_s < int(min_duration_seconds):
        return False, ["drop:too_short"], dur_s
    if dur_s and dur_s > int(max_duration_seconds):
        return False, ["drop:too_long"], dur_s

    if drop_shorts and (_SHORTS_RE.search(title) is not None or (dur_s and dur_s < 90)):
        return False, ["drop:shorts"], dur_s

    if cd.get("caption") == "false":
        reasons.append("hint:no_captions_flag")

    return True, reasons, dur_s

