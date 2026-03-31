from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


try:
    import whisper  # type: ignore

    _WHISPER_AVAILABLE = True
except Exception:  # pragma: no cover
    whisper = None
    _WHISPER_AVAILABLE = False


# Audio purity scorer is optional (only needed if validation.enabled=true)
try:
    from audio_purity_scorer import analyze_audio

    _AUDIO_PURITY_AVAILABLE = True
except ImportError:
    _AUDIO_PURITY_AVAILABLE = False

    def analyze_audio(wav_path: str, *args, **kwargs):  # type: ignore
        """Fallback when audio_purity_scorer is not available."""
        return {"final_score": 0.0, "content_type": "unknown"}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    audio_purity_score: float
    content_type: str
    language: Optional[str]
    reasons: List[str]


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def get_best_audio_url(video_id: str) -> Optional[str]:
    """
    Fast: uses yt-dlp to resolve a direct audio URL without downloading the whole file.
    """
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--no-check-certificate",
        "-f",
        "ba",
        "-g",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    p = _run(cmd)
    if p.returncode != 0:
        return None
    url = (p.stdout or "").strip().splitlines()
    return url[0].strip() if url else None


def extract_audio_segment_to_wav(url: str, *, start_sec: int, duration_sec: int, out_wav_path: str) -> bool:
    """
    Uses ffmpeg to fetch only a short segment and transcode to 16k mono wav.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(int(start_sec)),
        "-t",
        str(int(duration_sec)),
        "-i",
        url,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        out_wav_path,
    ]
    p = _run(cmd)
    return p.returncode == 0 and os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 0


class WhisperLanguageDetector:
    def __init__(self, model_name: str = "tiny"):
        if not _WHISPER_AVAILABLE:  # pragma: no cover
            raise RuntimeError("whisper is not installed. Run: pip install openai-whisper")
        self._model = whisper.load_model(model_name)

    def detect(self, wav_path: str) -> Optional[str]:
        try:
            res = self._model.transcribe(wav_path, task="transcribe", language=None, verbose=False)
            return res.get("language")
        except Exception:
            return None


async def validate_video_audio(
    *,
    video_id: str,
    offsets_seconds: List[int],
    segment_seconds: int,
    min_audio_purity_score: float,
    whisper_detector: Optional[WhisperLanguageDetector] = None,
) -> ValidationResult:
    """
    Async wrapper: resolves URL + samples multiple segments.
    """
    if not _AUDIO_PURITY_AVAILABLE:
        return ValidationResult(
            ok=False,
            audio_purity_score=0.0,
            content_type="error",
            language=None,
            reasons=["validation:audio_purity_scorer_not_available"],
        )

    reasons: List[str] = []
    url = await asyncio.to_thread(get_best_audio_url, video_id)
    if not url:
        return ValidationResult(ok=False, audio_purity_score=0.0, content_type="error", language=None, reasons=["validation:no_audio_url"])

    best_score = 0.0
    best_type = "error"
    detected_lang = None

    for off in offsets_seconds:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            ok = await asyncio.to_thread(
                extract_audio_segment_to_wav,
                url,
                start_sec=int(off),
                duration_sec=int(segment_seconds),
                out_wav_path=tmp.name,
            )
            if not ok:
                reasons.append(f"validation:ffmpeg_failed@{off}")
                continue

            res = await asyncio.to_thread(analyze_audio, tmp.name, False, False)
            score = float(res.get("final_score", 0.0))
            ctype = str(res.get("content_type", "unknown"))
            if score > best_score:
                best_score = score
                best_type = ctype

            if whisper_detector and detected_lang is None:
                detected_lang = await asyncio.to_thread(whisper_detector.detect, tmp.name)

    if best_score < min_audio_purity_score:
        reasons.append("validation:low_audio_purity")
    if best_type in ("music", "mixed"):
        reasons.append("validation:music_or_mixed")

    if whisper_detector:
        if detected_lang is None:
            reasons.append("validation:lang_unknown")
        elif detected_lang != "ar":
            reasons.append(f"validation:lang_not_ar:{detected_lang}")

    ok_final = best_score >= min_audio_purity_score and best_type not in ("music", "mixed")
    if whisper_detector and detected_lang is not None:
        ok_final = ok_final and (detected_lang == "ar")

    return ValidationResult(
        ok=ok_final,
        audio_purity_score=round(best_score, 2),
        content_type=best_type,
        language=detected_lang,
        reasons=reasons,
    )

