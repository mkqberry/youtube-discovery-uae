from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from yt_uae.text_ar import count_keyword_matches, normalize_text_for_matching, unique_keyword_matches


@dataclass(frozen=True)
class ScoreResult:
    single_speaker_score: float
    uae_relevance_score: float
    overall_score: float
    reason_codes: List[str]
    matched_uae_terms: List[str]
    matched_single_speaker_pos: List[str]
    matched_single_speaker_neg: List[str]
    matched_banned: List[str]


def score_video_metadata(
    *,
    title: str,
    description: str,
    channel_title: str,
    channel_description: str,
    cfg: "ScoringKeywords",
    caption_quality_hint: float = 0.0,
) -> ScoreResult:
    """
    Pure function: compute UAE relevance + single speaker likelihood from metadata.
    """
    texts = [title or "", description or "", channel_title or "", channel_description or ""]
    hay = " ".join(normalize_text_for_matching(t) for t in texts)

    # UAE relevance
    uae_hits = unique_keyword_matches(texts, cfg.uae_keywords_ar + cfg.uae_keywords_latn + cfg.uae_channel_hints)
    uae_score = min(1.0, len(uae_hits) / max(6, cfg.uae_norm_divisor))
    reasons: List[str] = []
    if uae_hits:
        reasons.append("uae:keyword_hit")

    # Single speaker likelihood (metadata-only, best-effort)
    pos_hits = unique_keyword_matches([title, description], cfg.single_speaker_positive)
    neg_hits = unique_keyword_matches([title, description], cfg.single_speaker_negative)
    banned_hits = unique_keyword_matches([title, description], cfg.banned_keywords)

    base = 0.50
    base += min(0.45, 0.08 * len(pos_hits))
    base -= min(0.60, 0.12 * len(neg_hits))
    base -= 0.50 if banned_hits else 0.0

    single_speaker = max(0.0, min(1.0, base))

    if pos_hits:
        reasons.append("speaker:pos")
    if neg_hits:
        reasons.append("speaker:neg")
    if banned_hits:
        reasons.append("content:banned")

    # Weighted overall
    overall = (
        cfg.weight_single_speaker * single_speaker
        + cfg.weight_uae_relevance * uae_score
        + cfg.weight_caption_quality * max(0.0, min(1.0, caption_quality_hint))
    )

    return ScoreResult(
        single_speaker_score=round(single_speaker, 4),
        uae_relevance_score=round(uae_score, 4),
        overall_score=round(overall, 4),
        reason_codes=reasons,
        matched_uae_terms=uae_hits[:20],
        matched_single_speaker_pos=pos_hits[:20],
        matched_single_speaker_neg=neg_hits[:20],
        matched_banned=banned_hits[:20],
    )


@dataclass(frozen=True)
class ScoringKeywords:
    uae_keywords_ar: List[str]
    uae_keywords_latn: List[str]
    uae_channel_hints: List[str]
    single_speaker_positive: List[str]
    single_speaker_negative: List[str]
    banned_keywords: List[str]
    weight_single_speaker: float
    weight_uae_relevance: float
    weight_caption_quality: float
    uae_norm_divisor: int = 6


def build_scoring_keywords(cfg: "yt_uae.config_loader.ScoringConfig") -> ScoringKeywords:
    return ScoringKeywords(
        uae_keywords_ar=cfg.uae_keywords_ar,
        uae_keywords_latn=cfg.uae_keywords_latn,
        uae_channel_hints=cfg.uae_channel_hints,
        single_speaker_positive=cfg.single_speaker_positive,
        single_speaker_negative=cfg.single_speaker_negative,
        banned_keywords=cfg.banned_keywords,
        weight_single_speaker=cfg.weight_single_speaker,
        weight_uae_relevance=cfg.weight_uae_relevance,
        weight_caption_quality=cfg.weight_caption_quality,
    )

