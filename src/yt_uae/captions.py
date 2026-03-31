from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CaptionTrack:
    id: str
    language: str
    name: str
    track_kind: str  # "standard" (manual) vs "ASR" (auto), sometimes empty
    is_cc: bool
    last_updated: str


@dataclass(frozen=True)
class CaptionSummary:
    has_any: bool
    has_arabic: bool
    has_manual_arabic: bool
    languages: List[str]
    tracks: List[CaptionTrack]
    preferred_track: Optional[CaptionTrack]


def parse_captions_list_response(resp: Dict[str, Any]) -> CaptionSummary:
    items = resp.get("items", []) or []
    tracks: List[CaptionTrack] = []
    langs: List[str] = []
    for it in items:
        sn = it.get("snippet", {}) or {}
        lang = str(sn.get("language", "") or "")
        if lang:
            langs.append(lang)
        tracks.append(
            CaptionTrack(
                id=str(it.get("id", "")),
                language=lang,
                name=str(sn.get("name", "") or ""),
                track_kind=str(sn.get("trackKind", "") or ""),
                is_cc=bool(sn.get("isCC", False)),
                last_updated=str(sn.get("lastUpdated", "") or ""),
            )
        )

    langs_unique = sorted({l for l in langs if l})
    has_any = len(tracks) > 0
    has_ar = any(t.language.lower().startswith("ar") for t in tracks)
    has_manual_ar = any(t.language.lower().startswith("ar") and t.track_kind == "standard" for t in tracks)

    preferred = None
    if tracks:
        # Prefer manual Arabic, else any Arabic, else anything manual.
        manual_ar = [t for t in tracks if t.language.lower().startswith("ar") and t.track_kind == "standard"]
        if manual_ar:
            preferred = manual_ar[0]
        else:
            any_ar = [t for t in tracks if t.language.lower().startswith("ar")]
            if any_ar:
                preferred = any_ar[0]
            else:
                manual_any = [t for t in tracks if t.track_kind == "standard"]
                preferred = manual_any[0] if manual_any else tracks[0]

    return CaptionSummary(
        has_any=has_any,
        has_arabic=has_ar,
        has_manual_arabic=has_manual_ar,
        languages=langs_unique,
        tracks=tracks,
        preferred_track=preferred,
    )


def caption_quality_hint(summary: CaptionSummary, prefer_languages: List[str], prefer_manual: bool) -> float:
    """
    Pure function: map caption metadata to [0..1] quality hint.
    """
    if not summary.has_any:
        return 0.0

    score = 0.2
    if summary.has_arabic:
        score = max(score, 0.55)
    if summary.has_manual_arabic:
        score = max(score, 0.90 if prefer_manual else 0.75)

    # Language preference bump if preferred_track is among preferred languages.
    if summary.preferred_track and prefer_languages:
        lang = summary.preferred_track.language
        if lang in prefer_languages or lang.lower() in {l.lower() for l in prefer_languages}:
            score = min(1.0, score + 0.05)

    return float(min(1.0, max(0.0, score)))

