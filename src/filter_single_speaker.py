#!/usr/bin/env python3
"""
Strict Single-Speaker Video Filtering Pipeline

This script implements a conservative, high-precision filtering approach
to identify videos that are clearly single-speaker.

Key principles:
- Conservative by design: reject if uncertain
- Prefer precision over recall
- Aggressively eliminate multi-speaker content
- Record detailed reasons for accept/reject decisions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Import speaker diarization module
try:
    from speaker_diarization import analyze_video_speakers, SpeakerDiarizationResult
    _SPEAKER_DIARIZATION_AVAILABLE = True
except ImportError:
    _SPEAKER_DIARIZATION_AVAILABLE = False
    analyze_video_speakers = None
    SpeakerDiarizationResult = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of single-speaker filtering for a video."""
    video_id: str
    is_single_speaker: bool
    confidence: str  # "high", "medium", "low"
    reasons: List[str] = field(default_factory=list)
    warning_flags: List[str] = field(default_factory=list)
    caption_analysis: Optional[Dict[str, Any]] = None
    metadata_analysis: Optional[Dict[str, Any]] = None
    audio_diarization: Optional[Dict[str, Any]] = None


# Multi-speaker indicators in metadata (titles/descriptions)
# NOTE: Common words like "مع" (with) and "بين" (between) are NOT included
# as they appear in normal sentences. Only clear multi-speaker patterns are included.
MULTI_SPEAKER_KEYWORDS = {
    # English - clear multi-speaker indicators
    "podcast", "interview", "dialogue", "conversation", "discussion", "debate",
    "panel", "talk show", "q&a", "qa", "question and answer", "roundtable",
    "featuring", "guest", "host", "co-host", "cohost", "vs", "versus",
    "duo", "trio", "group discussion", "multiple speakers", "speakers",
    "phone call", "phone conversation", "remote", "zoom", "meeting",
    "collaboration", "collab", "joint", "shared", "exchange",
    # "with" is too common - only use in specific patterns
    
    # Arabic - clear multi-speaker indicators only
    "بودكاست", "مقابلة", "نقاش", "مناقشة", "محادثة",
    "لقاء", "ضيف", "استضافة", "حوار مع", "لقاء مع",
    "ندوة", "جلسة", "برنامج", "مقابلة مع", "حوار مع",
    "سؤال وجواب", "أسئلة وأجوبة", "س و ج",
    "مكالمة", "اتصال", "هاتفي", "عن بعد", "زوم", "اجتماع",
    "تعاون", "مشترك", "تبادل", "حوار بين", "نقاش بين",
    # "مع" alone is too common - only "حوار مع", "لقاء مع" patterns
    # "بين" alone is too common - only "حوار بين", "نقاش بين" patterns
    # "حلقة" (episode) is NOT multi-speaker - removed
}

# Single-speaker positive indicators (weak signals, not guarantees)
SINGLE_SPEAKER_POSITIVE = {
    "lecture", "محاضرة", "شرح", "درس", "تفسير", "تعليق", "تعليق صوتي",
    "narration", "narration", "monologue", "speech", "خطاب", "كلمة",
    "tutorial", "tutorial", "lesson", "دروس", "تعليم",
}

# Patterns in captions that indicate multiple speakers
CAPTION_MULTI_SPEAKER_PATTERNS = [
    # Speaker labels
    r'^\s*(?:speaker|sp|s|speaker\s*\d+|sp\d+)[\s:]+',
    r'^\s*(?:المتحدث|المتكلم|المحاور|الضيف|المضيف)[\s:]+',
    r'^\s*(?:[A-Z][a-z]+\s*:|\w+\s*:)',  # Name: pattern
    r'^\s*[A-Z][A-Z\s]+:',  # ALL CAPS NAME:
    
    # Dialogue patterns
    r'\?.*\?',  # Multiple question marks (Q&A)
    r'^\s*[Qq]:\s*',  # Q: question
    r'^\s*[Aa]:\s*',  # A: answer
    r'^\s*س:\s*',  # س: (Arabic Q)
    r'^\s*ج:\s*',  # ج: (Arabic A)
    
    # Turn-taking indicators
    r'\[.*speaker.*\]',
    r'\(.*speaker.*\)',
    r'\[.*متحدث.*\]',
    r'\(.*متحدث.*\)',
    
    # Interview patterns
    r'interviewer|interviewee|host|guest',
    r'محاور|مضيف|ضيف',
]


async def download_captions_ytdlp(video_id: str, lang: str = "ar", semaphore: Optional[asyncio.Semaphore] = None) -> Optional[str]:
    """
    Download captions using yt-dlp (async version).
    Returns caption text or None if unavailable.
    """
    # Use semaphore to limit concurrent downloads
    if semaphore:
        async with semaphore:
            return await _download_captions_ytdlp_impl(video_id, lang)
    else:
        return await _download_captions_ytdlp_impl(video_id, lang)


async def _download_captions_ytdlp_impl(video_id: str, lang: str = "ar") -> Optional[str]:
    """Internal implementation of caption downloading."""
    try:
        # Check if yt-dlp is available
        check_result = await asyncio.to_thread(
            subprocess.run,
            ["yt-dlp", "--version"],
            capture_output=True,
            timeout=5,
        )
        if check_result.returncode != 0:
            logger.debug("yt-dlp not available, skipping caption download")
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("yt-dlp not found, skipping caption download")
        return None
    
    try:
        # Try different caption sources in order of preference
        attempts = [
            (["--write-sub", "--sub-lang", lang], "manual"),
            (["--write-sub", "--sub-lang", "ar"], "manual_ar"),
            (["--write-sub", "--sub-lang", "en"], "manual_en"),
            (["--write-auto-sub"], "auto"),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for sub_args, attempt_type in attempts:
                cmd = [
                    "yt-dlp",
                    "--skip-download",
                    "--quiet",
                    "--no-warnings",
                    *sub_args,
                    "--sub-format", "vtt",
                    "--convert-subs", "srt",
                    "--output", f"%(id)s.%(ext)s",
                    f"https://www.youtube.com/watch?v={video_id}",
                ]
                
                try:
                    result = await asyncio.to_thread(
                        subprocess.run,
                        cmd,
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    
                    if result.returncode == 0:
                        # Find the downloaded subtitle file
                        srt_files = list(Path(tmpdir).glob(f"{video_id}*.srt"))
                        if not srt_files:
                            vtt_files = list(Path(tmpdir).glob(f"{video_id}*.vtt"))
                            if vtt_files:
                                srt_files = vtt_files
                        
                        if srt_files:
                            caption_file = srt_files[0]
                            content = await asyncio.to_thread(
                                caption_file.read_text,
                                encoding="utf-8",
                                errors="ignore"
                            )
                            if content.strip():
                                logger.debug(f"Downloaded captions for {video_id} ({attempt_type})")
                                return content
                except subprocess.TimeoutExpired:
                    logger.debug(f"Timeout downloading captions for {video_id} ({attempt_type})")
                    continue
                except Exception as e:
                    logger.debug(f"Error downloading captions for {video_id} ({attempt_type}): {e}")
                    continue
        
        return None
    except Exception as e:
        logger.debug(f"Failed to download captions for {video_id}: {e}")
        return None


def parse_srt_captions(srt_text: str) -> List[Dict[str, Any]]:
    """
    Parse SRT or VTT subtitle format into structured data.
    Returns list of caption entries with text and timing.
    """
    entries = []
    
    # Handle VTT format (starts with WEBVTT header)
    if srt_text.strip().startswith("WEBVTT"):
        # VTT format: remove header and parse
        lines = srt_text.split('\n')
        # Skip WEBVTT header and any metadata
        content_start = 0
        for i, line in enumerate(lines):
            if '-->' in line:
                content_start = i
                break
        
        blocks = re.split(r'\n\s*\n', '\n'.join(lines[content_start:]))
    else:
        # SRT format
        blocks = re.split(r'\n\s*\n', srt_text.strip())
    
    for block in blocks:
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            continue
        
        # Find time line (contains -->)
        time_line = None
        time_line_idx = -1
        for i, line in enumerate(lines):
            if '-->' in line:
                time_line = line
                time_line_idx = i
                break
        
        if not time_line:
            continue
        
        # Text lines are after the time line
        text_lines = lines[time_line_idx + 1:]
        
        # Parse time: 00:00:00,000 --> 00:00:05,000 or 00:00:00.000 --> 00:00:05.000
        time_match = re.match(
            r'(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})',
            time_line
        )
        if time_match:
            start_sec = (
                int(time_match.group(1)) * 3600 +
                int(time_match.group(2)) * 60 +
                int(time_match.group(3)) +
                int(time_match.group(4)) / 1000.0
            )
            end_sec = (
                int(time_match.group(5)) * 3600 +
                int(time_match.group(6)) * 60 +
                int(time_match.group(7)) +
                int(time_match.group(8)) / 1000.0
            )
        else:
            # Try alternative format: 00:00:00 --> 00:00:05
            time_match = re.match(
                r'(\d{1,2}):(\d{2}):(\d{2})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})',
                time_line
            )
            if time_match:
                start_sec = (
                    int(time_match.group(1)) * 3600 +
                    int(time_match.group(2)) * 60 +
                    int(time_match.group(3))
                )
                end_sec = (
                    int(time_match.group(4)) * 3600 +
                    int(time_match.group(5)) * 60 +
                    int(time_match.group(6))
                )
            else:
                start_sec = 0.0
                end_sec = 0.0
        
        # Clean text (remove HTML tags, speaker labels, etc.)
        text = ' '.join(text_lines).strip()
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove common VTT styling
        text = re.sub(r'::[^:]+::', '', text)
        
        if text:
            entries.append({
                "start": start_sec,
                "end": end_sec,
                "text": text,
                "duration": end_sec - start_sec,
            })
    
    return entries


def analyze_captions_for_speakers(caption_text: str) -> Dict[str, Any]:
    """
    Analyze caption text for indicators of multiple speakers.
    Returns analysis results.
    """
    if not caption_text:
        return {
            "has_captions": False,
            "multi_speaker_indicators": [],
            "speaker_labels_found": False,
            "dialogue_patterns_found": False,
            "turn_taking_indicators": 0,
            "confidence": "unknown",
        }
    
    entries = parse_srt_captions(caption_text)
    if not entries:
        return {
            "has_captions": True,
            "caption_count": 0,
            "multi_speaker_indicators": [],
            "confidence": "low",
        }
    
    indicators = []
    speaker_labels = False
    dialogue_patterns = False
    turn_taking_count = 0
    
    # Check each caption entry
    for entry in entries:
        text = entry["text"].strip()
        
        # Check for speaker labels
        for pattern in CAPTION_MULTI_SPEAKER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                if "speaker" in pattern.lower() or "متحدث" in pattern or "محاور" in pattern:
                    speaker_labels = True
                    indicators.append(f"speaker_label_pattern: {pattern}")
                elif "q:" in pattern.lower() or "a:" in pattern.lower() or "س:" in pattern or "ج:" in pattern:
                    dialogue_patterns = True
                    indicators.append(f"dialogue_pattern: {pattern}")
                else:
                    turn_taking_count += 1
                    indicators.append(f"turn_taking_pattern: {pattern}")
    
    # Check for rapid turn-taking (many short captions in sequence)
    if len(entries) > 10:
        short_entries = [e for e in entries if e["duration"] < 2.0]
        if len(short_entries) > len(entries) * 0.3:  # More than 30% are very short
            indicators.append("high_frequency_short_captions")
            turn_taking_count += 1
    
    # Check for question-answer patterns
    question_count = sum(1 for e in entries if "?" in e["text"])
    if question_count > len(entries) * 0.2:  # More than 20% have questions
        indicators.append("high_question_frequency")
        dialogue_patterns = True
    
    confidence = "high"
    if speaker_labels or dialogue_patterns:
        confidence = "very_low"
    elif turn_taking_count > 3:
        confidence = "low"
    elif turn_taking_count > 0:
        confidence = "medium"
    
    return {
        "has_captions": True,
        "caption_count": len(entries),
        "multi_speaker_indicators": indicators,
        "speaker_labels_found": speaker_labels,
        "dialogue_patterns_found": dialogue_patterns,
        "turn_taking_indicators": turn_taking_count,
        "question_frequency": question_count / len(entries) if entries else 0.0,
        "confidence": confidence,
    }


def analyze_metadata_for_speakers(title: str, description: str) -> Dict[str, Any]:
    """
    Analyze title and description for multi-speaker indicators.
    Uses context-aware pattern matching to avoid false positives.
    """
    text = f"{title} {description}".lower()
    
    multi_speaker_matches = []
    single_speaker_matches = []
    
    # Check for multi-speaker keywords
    # Some keywords are more significant in title than in description
    title_lower = title.lower()
    desc_lower = description.lower()
    
    for keyword in MULTI_SPEAKER_KEYWORDS:
        keyword_lower = keyword.lower()
        # Check if keyword appears in title (more significant)
        if keyword_lower in title_lower:
            multi_speaker_matches.append(keyword)
        # For description, be more careful with common words
        elif keyword_lower in desc_lower:
            # "بودكاست" in description might just be mentioning podcasts, not being a podcast
            # Only count if it's in a clear context
            if keyword == "بودكاست":
                # Only reject if it's clearly about being a podcast
                if re.search(r'(?:بودكاست|podcast)\s+(?:مع|with|عن|about)', desc_lower):
                    multi_speaker_matches.append(keyword)
            else:
                multi_speaker_matches.append(keyword)
    
    # Check for single-speaker positive indicators (weak signal)
    for keyword in SINGLE_SPEAKER_POSITIVE:
        if keyword.lower() in text:
            single_speaker_matches.append(keyword)
    
    # Check for "with [Name]" patterns (often indicates guest/interview)
    # But NOT "with ourselves", "with yourself", etc.
    if re.search(r'\bwith\s+[A-Z][a-z]+\s+(?:and|&)', text, re.IGNORECASE):
        multi_speaker_matches.append("with_and_pattern")
    # "with [Name]" in title is more significant than in description
    if re.search(r'\bwith\s+[A-Z][a-z]+', title, re.IGNORECASE):
        # But exclude common phrases
        if not re.search(r'\bwith\s+(?:yourself|ourselves|themselves|himself|herself)\b', text, re.IGNORECASE):
            multi_speaker_matches.append("with_name_in_title")
    
    # Check for Arabic "حوار مع" or "لقاء مع" patterns (interview/conversation with)
    # But NOT just "مع" alone (too common)
    if re.search(r'(?:حوار|لقاء|مقابلة)\s+مع\s+[أ-ي]+', text):
        multi_speaker_matches.append("interview_with_pattern")
    
    # Check for "vs" or "versus"
    if re.search(r'\b(vs|versus)\b', text, re.IGNORECASE):
        multi_speaker_matches.append("vs_pattern")
    
    confidence = "high"
    if multi_speaker_matches:
        confidence = "very_low"
    elif len(single_speaker_matches) > 0 and len(multi_speaker_matches) == 0:
        confidence = "medium"  # Weak positive signal
    
    return {
        "multi_speaker_keywords": multi_speaker_matches,
        "single_speaker_keywords": single_speaker_matches,
        "confidence": confidence,
    }


async def filter_video_strict(
    video_record: Dict[str, Any],
    download_captions: bool = True,
    use_audio_diarization: bool = True,
    source_file: Optional[str] = None,  # Track which input file this came from
    caption_semaphore: Optional[asyncio.Semaphore] = None,
    audio_semaphore: Optional[asyncio.Semaphore] = None,
) -> FilterResult:
    """
    Apply strict single-speaker filtering to a video record.
    
    Conservative approach: reject if there's any uncertainty.
    However, for manual_arabic.jsonl (manually curated), be more lenient.
    """
    video_id = video_record.get("video_id", "")
    title = video_record.get("title", "")
    description = video_record.get("description", "")
    
    reasons = []
    warning_flags = []
    is_single_speaker = True
    confidence = "high"
    
    # 0. Check for Latin characters in title (English/mixed content - stricter rules)
    has_latin_chars = bool(re.search(r'[a-zA-Z]', title))
    
    # Special handling for manual_arabic.jsonl - more lenient for non-Latin titles
    is_manual_arabic_source = source_file and "manual_arabic" in str(source_file).lower()
    is_non_latin_title = not has_latin_chars
    
    if has_latin_chars:
        warning_flags.append("title_contains_latin_chars")
    
    # 1. Metadata analysis (FAST - do this first)
    metadata_analysis = analyze_metadata_for_speakers(title, description)
    
    # Special handling: manual_arabic.jsonl with non-Latin titles - only reject clear multi-speaker keywords
    if is_manual_arabic_source and is_non_latin_title:
        # Very lenient: only reject if clear multi-speaker keywords found
        if metadata_analysis["multi_speaker_keywords"]:
            is_single_speaker = False
            confidence = "very_low"
            reasons.append(f"metadata:multi_speaker_keywords:{','.join(metadata_analysis['multi_speaker_keywords'][:3])}")
        # Don't reject for "suspicious_patterns" - trust manual curation
    else:
        # For other sources, be strict
        if metadata_analysis["multi_speaker_keywords"]:
            is_single_speaker = False
            confidence = "very_low"
            reasons.append(f"metadata:multi_speaker_keywords:{','.join(metadata_analysis['multi_speaker_keywords'][:3])}")
        elif metadata_analysis["confidence"] == "very_low":
            is_single_speaker = False
            confidence = "very_low"
            reasons.append("metadata:suspicious_patterns")
    
    # EARLY EXIT: If metadata already rejects, skip expensive operations
    skip_expensive_ops = not is_single_speaker and confidence in ("very_low", "low")
    
    # 2. Caption analysis (if available) - SKIP if already rejected by metadata
    caption_analysis = None
    caption_text = None
    if download_captions and is_single_speaker and not skip_expensive_ops:  # Only if still passing
        caption_text = await download_captions_ytdlp(video_id, semaphore=caption_semaphore)
        caption_analysis = analyze_captions_for_speakers(caption_text if caption_text else "")
        
        # Special handling: manual_arabic with non-Latin titles - only reject clear indicators
        if is_manual_arabic_source and is_non_latin_title:
            # Only reject if very clear multi-speaker indicators
            if caption_analysis.get("speaker_labels_found"):
                is_single_speaker = False
                confidence = "very_low"
                reasons.append("captions:speaker_labels_detected")
            # Don't reject for dialogue patterns or turn-taking - might be false positives
            # Only reject if confidence is very_low (very clear multi-speaker)
            if caption_analysis.get("confidence") == "very_low":
                is_single_speaker = False
                confidence = "very_low"
                reasons.append("captions:very_low_confidence_multi_speaker")
        else:
            # For other sources, be strict
            if caption_analysis.get("speaker_labels_found"):
                is_single_speaker = False
                confidence = "very_low"
                reasons.append("captions:speaker_labels_detected")
            
            if caption_analysis.get("dialogue_patterns_found"):
                is_single_speaker = False
                confidence = "very_low"
                reasons.append("captions:dialogue_patterns_detected")
            
            if caption_analysis.get("turn_taking_indicators", 0) > 2:
                is_single_speaker = False
                confidence = "low"
                reasons.append(f"captions:excessive_turn_taking:{caption_analysis['turn_taking_indicators']}")
            
            if caption_analysis.get("confidence") in ("very_low", "low"):
                if caption_analysis.get("confidence") == "very_low":
                    is_single_speaker = False
                    confidence = "very_low"
                    reasons.append("captions:low_confidence_multi_speaker")
                else:
                    warning_flags.append("captions:medium_confidence")
    
    # 3. Additional heuristics - STRICTER RULES
    
    # Check if video has no captions at all (uncertainty)
    caption_avail = video_record.get("caption_availability", {})
    has_captions = caption_avail.get("has_any", False) or (caption_text is not None and caption_text.strip())
    
    if not has_captions:
        warning_flags.append("no_captions_available")
        
        # STRICT: If no captions, require strong single-speaker indicators
        single_speaker_keywords = metadata_analysis.get("single_speaker_keywords", [])
        
        # Special handling for manual_arabic.jsonl with non-Latin titles
        # These are manually curated, so trust them more
        if is_manual_arabic_source and is_non_latin_title:
            # Very lenient: if it's from manual_arabic and non-Latin title, accept unless clear multi-speaker indicators
            if metadata_analysis["multi_speaker_keywords"]:
                # Still reject if clear multi-speaker keywords found
                is_single_speaker = False
                confidence = "low"
                reasons.append("manual_arabic_but_multi_speaker_keywords")
            # Otherwise, accept (trust the manual curation - these are already filtered)
            # Don't reject just because captions are missing
        # If Latin title + no captions = reject (can't verify)
        elif has_latin_chars:
            is_single_speaker = False
            confidence = "low"
            reasons.append("latin_title_no_captions_cannot_verify")
        # If no captions and only weak positive signals = reject
        elif len(single_speaker_keywords) == 0:
            is_single_speaker = False
            confidence = "low"
            reasons.append("no_captions_no_strong_indicators")
        # If only 1 weak keyword and no captions = reject (too uncertain)
        elif len(single_speaker_keywords) == 1:
            is_single_speaker = False
            confidence = "medium"
            reasons.append("no_captions_only_weak_indicator")
    
    # Check duration - very short videos might be clips from longer multi-speaker content
    duration = video_record.get("duration_seconds", 0)
    if duration < 60:  # Less than 1 minute
        warning_flags.append("very_short_duration")
        # Very short + no captions = reject (unless manual_arabic with non-Latin title)
        if not has_captions:
            if not (is_manual_arabic_source and is_non_latin_title):
                is_single_speaker = False
                confidence = "low"
                reasons.append("very_short_no_captions")
    
    # 4. Audio-based speaker diarization (PROFESSIONAL VERIFICATION) - SKIP if already rejected
    audio_diarization_result = None
    if use_audio_diarization and _SPEAKER_DIARIZATION_AVAILABLE and is_single_speaker and not skip_expensive_ops:
        # Only run audio analysis if video still passes other checks
        try:
            duration = video_record.get("duration_seconds", 0)
            if duration > 60:  # Only for videos longer than 1 minute
                # Use semaphore to limit concurrent audio processing
                if audio_semaphore:
                    async with audio_semaphore:
                        logger.debug(f"  → Starting audio diarization for {video_id} (duration: {duration}s)...")
                        # Sample segments: beginning, middle, end
                        if duration > 600:  # > 10 minutes
                            segments = [(60, 15), (duration/2, 15), (duration-60, 15)]
                        elif duration > 180:  # > 3 minutes
                            segments = [(30, 15), (duration/2, 15), (duration-30, 15)]
                        else:  # 1-3 minutes
                            segments = [(10, 10), (duration/2, 10)]
                        
                        audio_diarization_result = await analyze_video_speakers(
                            video_id,
                            sample_segments=segments,
                            method="auto",
                        )
                else:
                    logger.debug(f"  → Starting audio diarization for {video_id} (duration: {duration}s)...")
                    # Sample segments: beginning, middle, end
                    if duration > 600:  # > 10 minutes
                        segments = [(60, 15), (duration/2, 15), (duration-60, 15)]
                    elif duration > 180:  # > 3 minutes
                        segments = [(30, 15), (duration/2, 15), (duration-30, 15)]
                    else:  # 1-3 minutes
                        segments = [(10, 10), (duration/2, 10)]
                    
                    audio_diarization_result = await analyze_video_speakers(
                        video_id,
                        sample_segments=segments,
                        method="auto",
                    )
                
                if audio_diarization_result:
                    logger.debug(f"  → Audio diarization complete: {audio_diarization_result.num_speakers} speaker(s), confidence: {audio_diarization_result.confidence:.2f}")
                
                if audio_diarization_result:
                    # Audio diarization is the final authority
                    if not audio_diarization_result.is_single_speaker:
                        is_single_speaker = False
                        confidence = "very_low"
                        reasons.append(f"audio:detected_{audio_diarization_result.num_speakers}_speakers")
                    
                    if audio_diarization_result.confidence < 0.7:
                        is_single_speaker = False
                        confidence = "low"
                        reasons.append(f"audio:low_confidence:{audio_diarization_result.confidence:.2f}")
                    
                    if audio_diarization_result.reasons:
                        reasons.extend([f"audio:{r}" for r in audio_diarization_result.reasons])
        except Exception as e:
            logger.warning(f"Audio diarization failed for {video_id}: {e}")
            # On error, be conservative - if we can't verify, reject
            if not has_captions:
                is_single_speaker = False
                confidence = "low"
                reasons.append("audio_diarization_failed_no_captions")
    
    # STRICT: Medium confidence = reject (not just 2+ warning flags)
    # BUT: More lenient for manual_arabic.jsonl with non-Latin titles
    if confidence == "medium":
        if is_manual_arabic_source and is_non_latin_title:
            # For manual_arabic with non-Latin titles, accept medium confidence
            # (trust the manual curation)
            pass
        else:
            is_single_speaker = False
            if "no_captions_only_weak_indicator" not in reasons:
                reasons.append("medium_confidence_rejected")
    
    # Final decision: if any red flags, reject
    if not is_single_speaker:
        confidence = "very_low" if confidence != "low" else "low"
    
    # If we have warnings but no clear rejection, still be conservative
    # BUT: More lenient for manual_arabic.jsonl with non-Latin titles
    if warning_flags and is_single_speaker:
        if is_manual_arabic_source and is_non_latin_title:
            # For manual_arabic with non-Latin titles, only reject if multiple serious warnings
            if len(warning_flags) >= 3:
                is_single_speaker = False
                confidence = "medium"
                reasons.append("too_many_warning_flags")
            # Otherwise, accept (trust manual curation)
        else:
            # For other sources, be strict
            # Any warning flag = downgrade confidence
            if confidence == "high":
                confidence = "medium"
                is_single_speaker = False
                reasons.append("warning_flags_present")
            elif len(warning_flags) >= 2:
                is_single_speaker = False
                confidence = "low"
                reasons.append("too_many_warning_flags")
    
    # Prepare audio diarization data for output
    audio_diarization_data = None
    if audio_diarization_result:
        audio_diarization_data = {
            "num_speakers": audio_diarization_result.num_speakers,
            "is_single_speaker": audio_diarization_result.is_single_speaker,
            "confidence": audio_diarization_result.confidence,
            "total_speech_duration": audio_diarization_result.total_speech_duration,
            "overlap_duration": audio_diarization_result.overlap_duration,
            "reasons": audio_diarization_result.reasons,
        }
    
    return FilterResult(
        video_id=video_id,
        is_single_speaker=is_single_speaker,
        confidence=confidence,
        reasons=reasons,
        warning_flags=warning_flags,
        caption_analysis=caption_analysis,
        metadata_analysis=metadata_analysis,
        audio_diarization=audio_diarization_data,
    )


async def process_single_video(
    video: Dict[str, Any],
    video_index: int,
    total_videos: int,
    input_file: Path,
    download_captions: bool,
    caption_semaphore: asyncio.Semaphore,
    audio_semaphore: asyncio.Semaphore,
    output_f,
    file_lock: asyncio.Lock,
    accepted_count: List[int],  # Use list for thread-safe counter
    rejected_count: List[int],
) -> None:
    """Process a single video and write result to output file."""
    video_id = video.get("video_id", "unknown")
    title = video.get("title", "")[:60]
    
    try:
        # Process video with all filters
        result = await filter_video_strict(
            video,
            download_captions=download_captions,
            use_audio_diarization=True,
            source_file=str(input_file),
            caption_semaphore=caption_semaphore,
            audio_semaphore=audio_semaphore,
        )
        
        # Add filtering results to video record
        video["single_speaker_filter"] = {
            "is_single_speaker": result.is_single_speaker,
            "confidence": result.confidence,
            "reasons": result.reasons,
            "warning_flags": result.warning_flags,
            "caption_analysis": result.caption_analysis,
            "metadata_analysis": result.metadata_analysis,
            "audio_diarization": result.audio_diarization,
        }
        
        # Async-safe writing
        async with file_lock:
            if result.is_single_speaker:
                accepted_count[0] += 1
                output_f.write(json.dumps(video, ensure_ascii=False) + "\n")
                output_f.flush()
                if video_index % 10 == 0 or accepted_count[0] % 10 == 0:
                    logger.info(f"✓ ACCEPTED [{accepted_count[0]}/{video_index}]: {video_id} - {title} (confidence: {result.confidence})")
            else:
                rejected_count[0] += 1
                if video_index % 10 == 0:
                    logger.info(f"✗ REJECTED [{rejected_count[0]}/{video_index}]: {video_id} - {title}")
    
    except Exception as e:
        logger.error(f"Error processing {video_id}: {e}", exc_info=True)
        video["single_speaker_filter"] = {
            "is_single_speaker": False,
            "confidence": "very_low",
            "reasons": [f"error:{str(e)[:50]}"],
            "warning_flags": [],
        }
        async with file_lock:
            rejected_count[0] += 1


async def process_file(
    input_file: Path,
    output_file: Path,
    source_name: str,
    download_captions: bool = True,
    max_videos: Optional[int] = None,
    max_concurrent: int = 5,  # Number of videos to process concurrently
) -> Dict[str, Any]:
    """
    Process a single input file and generate filtered output.
    Uses parallel processing for better performance.
    """
    logger.info(f"=" * 60)
    logger.info(f"Processing {source_name}: {input_file}")
    logger.info(f"=" * 60)
    
    videos = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                videos.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
                continue
    
    if max_videos:
        videos = videos[:max_videos]
    
    logger.info(f"Loaded {len(videos)} videos from {input_file}")
    logger.info(f"Processing with {max_concurrent} concurrent workers...")
    
    accepted_count = [0]  # Use list for thread-safe counter
    rejected_count = [0]
    
    # Open output file for real-time writing
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_f = open(output_file, "w", encoding="utf-8")
    
    # Semaphores to limit concurrent operations
    caption_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent caption downloads
    audio_semaphore = asyncio.Semaphore(2)   # Max 2 concurrent audio analyses (very expensive)
    file_lock = asyncio.Lock()  # Lock for file writing
    
    try:
        # Process videos in batches for better progress tracking
        batch_size = max_concurrent
        for batch_start in range(0, len(videos), batch_size):
            batch = videos[batch_start:batch_start + batch_size]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(videos) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} videos)...")
            
            # Process batch concurrently
            tasks = [
                process_single_video(
                    video=video,
                    video_index=batch_start + i + 1,
                    total_videos=len(videos),
                    input_file=input_file,
                    download_captions=download_captions,
                    caption_semaphore=caption_semaphore,
                    audio_semaphore=audio_semaphore,
                    output_f=output_f,
                    file_lock=file_lock,
                    accepted_count=accepted_count,
                    rejected_count=rejected_count,
                )
                for i, video in enumerate(batch)
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info(f"Batch {batch_num}/{total_batches} complete. Progress: {accepted_count[0]} accepted, {rejected_count[0]} rejected")
    
    finally:
        # Close output file
        output_f.close()
    
    stats = {
        "source": source_name,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "total_videos": len(videos),
        "accepted": accepted_count[0],
        "rejected": rejected_count[0],
        "acceptance_rate": accepted_count[0] / len(videos) if videos else 0.0,
    }
    
    logger.info(f"=" * 60)
    logger.info(f"Results for {source_name}:")
    logger.info(f"  Total: {stats['total_videos']}")
    logger.info(f"  Accepted: {stats['accepted']} ({stats['acceptance_rate']*100:.1f}%)")
    logger.info(f"  Rejected: {stats['rejected']}")
    logger.info(f"  Output: {output_file}")
    logger.info(f"=" * 60)
    
    return stats


async def main():
    parser = argparse.ArgumentParser(
        description="Strict single-speaker video filtering pipeline"
    )
    parser.add_argument(
        "--high-single-speaker",
        type=Path,
        default=Path("high_single_speaker.jsonl"),
        help="Input file: high_single_speaker.jsonl",
    )
    parser.add_argument(
        "--manual-arabic",
        type=Path,
        default=Path("manual_arabic.jsonl"),
        help="Input file: manual_arabic.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for filtered results",
    )
    parser.add_argument(
        "--no-captions",
        action="store_true",
        help="Skip caption downloading/analysis (metadata only)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Limit number of videos to process (for testing)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of videos to process concurrently (default: 5)",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    
    # Process high_single_speaker.jsonl
    if args.high_single_speaker.exists():
        stats1 = await process_file(
            input_file=args.high_single_speaker,
            output_file=output_dir / "high_single_speaker_filtered.jsonl",
            source_name="high_single_speaker",
            download_captions=not args.no_captions,
            max_videos=args.max_videos,
            max_concurrent=args.max_concurrent,
        )
        all_stats.append(stats1)
    else:
        logger.warning(f"File not found: {args.high_single_speaker}")
    
    # Process manual_arabic.jsonl
    if args.manual_arabic.exists():
        stats2 = await process_file(
            input_file=args.manual_arabic,
            output_file=output_dir / "manual_arabic_filtered.jsonl",
            source_name="manual_arabic",
            download_captions=not args.no_captions,
            max_videos=args.max_videos,
            max_concurrent=args.max_concurrent,
        )
        all_stats.append(stats2)
    else:
        logger.warning(f"File not found: {args.manual_arabic}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("FILTERING COMPLETE - SUMMARY")
    logger.info("=" * 60)
    for stats in all_stats:
        logger.info(
            f"{stats['source']}: {stats['accepted']}/{stats['total_videos']} "
            f"accepted ({stats['acceptance_rate']*100:.1f}%)"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
