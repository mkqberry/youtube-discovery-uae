import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence


_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL_RE = re.compile(r"\u0640")
_AR_LETTERS_RE = re.compile(r"[\u0600-\u06FF]+")
_NON_WORD_RE = re.compile(r"[^\w\u0600-\u06FF]+", re.UNICODE)


@dataclass(frozen=True)
class ArabicNormalizationConfig:
    remove_diacritics: bool = True
    remove_tatweel: bool = True
    normalize_alef: bool = True
    normalize_yaa: bool = True
    normalize_taa_marbuta: bool = True
    normalize_hamza: bool = False


def normalize_arabic(text: str, cfg: ArabicNormalizationConfig | None = None) -> str:
    """
    Best-effort Arabic normalization for keyword matching and scoring.

    Notes:
    - Keep it conservative; do NOT try to "dialect-normalize" (unsafe).
    - Focus on orthographic variants (alef/yaa/taa marbuta/diacritics).
    """
    if not text:
        return ""
    cfg = cfg or ArabicNormalizationConfig()

    t = text
    if cfg.remove_diacritics:
        t = _ARABIC_DIACRITICS_RE.sub("", t)
    if cfg.remove_tatweel:
        t = _TATWEEL_RE.sub("", t)

    if cfg.normalize_alef:
        t = t.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    if cfg.normalize_yaa:
        t = t.replace("ى", "ي")
    if cfg.normalize_taa_marbuta:
        t = t.replace("ة", "ه")
    if cfg.normalize_hamza:
        t = t.replace("ؤ", "و").replace("ئ", "ي").replace("ء", "")

    return t


def normalize_text_for_matching(text: str, cfg: ArabicNormalizationConfig | None = None) -> str:
    if not text:
        return ""
    t = normalize_arabic(text, cfg=cfg)
    t = t.lower()
    t = _NON_WORD_RE.sub(" ", t)
    return " ".join(t.split())


def contains_arabic(text: str) -> bool:
    if not text:
        return False
    return _AR_LETTERS_RE.search(text) is not None


def tokenize(text: str, *, cfg: ArabicNormalizationConfig | None = None) -> List[str]:
    t = normalize_text_for_matching(text, cfg=cfg)
    if not t:
        return []
    return t.split()


def any_keyword_match(text: str, keywords: Sequence[str], *, cfg: ArabicNormalizationConfig | None = None) -> bool:
    """
    Substring-based matching on normalized text.
    Good for multiword phrases; fast and robust enough for heuristics.
    """
    if not text or not keywords:
        return False
    hay = normalize_text_for_matching(text, cfg=cfg)
    for kw in keywords:
        if not kw:
            continue
        if normalize_text_for_matching(kw, cfg=cfg) in hay:
            return True
    return False


def count_keyword_matches(text: str, keywords: Sequence[str], *, cfg: ArabicNormalizationConfig | None = None) -> int:
    if not text or not keywords:
        return 0
    hay = normalize_text_for_matching(text, cfg=cfg)
    c = 0
    for kw in keywords:
        if not kw:
            continue
        if normalize_text_for_matching(kw, cfg=cfg) in hay:
            c += 1
    return c


def unique_keyword_matches(texts: Iterable[str], keywords: Sequence[str], *, cfg: ArabicNormalizationConfig | None = None) -> List[str]:
    """
    Return unique keywords (original form) that matched at least once.
    """
    if not keywords:
        return []
    hay = " ".join(normalize_text_for_matching(t or "", cfg=cfg) for t in texts)
    matched: List[str] = []
    seen_norm: set[str] = set()
    for kw in keywords:
        nkw = normalize_text_for_matching(kw, cfg=cfg)
        if not nkw or nkw in seen_norm:
            continue
        if nkw in hay:
            matched.append(kw)
            seen_norm.add(nkw)
    return matched

