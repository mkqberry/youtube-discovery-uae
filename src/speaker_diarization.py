"""
Audio-based Speaker Diarization Module

This module provides speaker diarization capabilities to detect
the number of speakers in audio segments.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import pyannote.audio (optional but recommended)
try:
    from pyannote.audio import Pipeline
    _PYANNOTE_AVAILABLE = True
except ImportError:
    _PYANNOTE_AVAILABLE = False
    Pipeline = None

# Try to import resemblyzer (lighter alternative)
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    import numpy as np
    _RESEMBLYZER_AVAILABLE = True
except ImportError:
    _RESEMBLYZER_AVAILABLE = False
    VoiceEncoder = None
    preprocess_wav = None
    np = None


@dataclass
class SpeakerDiarizationResult:
    """Result of speaker diarization analysis."""
    num_speakers: int
    is_single_speaker: bool
    confidence: float  # 0.0 to 1.0
    speaker_segments: List[Dict[str, Any]]  # List of {start, end, speaker_id}
    total_speech_duration: float
    overlap_duration: float
    reasons: List[str]


def get_audio_url(video_id: str) -> Optional[str]:
    """Get direct audio URL from YouTube video using yt-dlp."""
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--no-check-certificate",
        "-f", "ba",  # Best audio
        "-g",  # Get URL only
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            url = result.stdout.strip().splitlines()
            return url[0].strip() if url else None
    except Exception as e:
        logger.debug(f"Failed to get audio URL for {video_id}: {e}")
    return None


def download_audio_segment(
    audio_url: str,
    start_sec: float,
    duration_sec: float,
    output_path: str,
    sample_rate: int = 16000,
) -> bool:
    """Download a segment of audio and convert to WAV format."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", audio_url,
        "-vn",  # No video
        "-ac", "1",  # Mono
        "-ar", str(sample_rate),  # Sample rate
        "-f", "wav",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        logger.debug(f"FFmpeg failed: {e}")
        return False


def analyze_speakers_pyannote(audio_path: str) -> Optional[SpeakerDiarizationResult]:
    """
    Analyze speakers using pyannote.audio (most accurate).
    Requires HuggingFace token and model access.
    """
    if not _PYANNOTE_AVAILABLE:
        return None
    
    try:
        # Initialize pipeline (requires HuggingFace token)
        # For production, you'd load a pre-trained model
        # Note: Requires HuggingFace token in environment: export HUGGINGFACE_TOKEN=your_token
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            logger.warning("HUGGINGFACE_TOKEN not set, pyannote.audio may not work")
            return None
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        
        # Run diarization
        diarization = pipeline(audio_path)
        
        # Extract speaker segments
        speaker_segments = []
        speakers = set()
        total_duration = 0.0
        overlap_duration = 0.0
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            duration = turn.end - turn.start
            total_duration += duration
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": duration,
            })
        
        num_speakers = len(speakers)
        is_single = num_speakers == 1
        
        # Calculate confidence based on speaker distribution
        if num_speakers == 1:
            confidence = 1.0
        elif num_speakers == 2:
            # Check if one speaker dominates (>90%)
            speaker_durations = {}
            for seg in speaker_segments:
                sp = seg["speaker"]
                speaker_durations[sp] = speaker_durations.get(sp, 0) + seg["duration"]
            
            max_duration = max(speaker_durations.values())
            dominance = max_duration / total_duration if total_duration > 0 else 0
            confidence = 1.0 - dominance  # Lower confidence if not dominant
        else:
            confidence = 0.0
        
        reasons = []
        if num_speakers > 1:
            reasons.append(f"detected_{num_speakers}_speakers")
        if overlap_duration > total_duration * 0.1:  # More than 10% overlap
            reasons.append("significant_overlap_detected")
        
        return SpeakerDiarizationResult(
            num_speakers=num_speakers,
            is_single_speaker=is_single,
            confidence=confidence,
            speaker_segments=speaker_segments,
            total_speech_duration=total_duration,
            overlap_duration=overlap_duration,
            reasons=reasons,
        )
    except Exception as e:
        logger.warning(f"pyannote.audio analysis failed: {e}")
        return None


def analyze_speakers_resemblyzer(audio_path: str, num_segments: int = 5) -> Optional[SpeakerDiarizationResult]:
    """
    Analyze speakers using resemblyzer (lighter, no external API needed).
    Compares speaker embeddings from different segments.
    """
    if not _RESEMBLYZER_AVAILABLE:
        return None
    
    try:
        import soundfile as sf
        
        # Load audio
        wav, sr = sf.read(audio_path)
        if len(wav) == 0:
            return None
        
        # Preprocess
        wav = preprocess_wav(wav)
        
        # Initialize encoder
        encoder = VoiceEncoder()
        
        # Get duration
        duration = len(wav) / sr
        
        # Sample segments from different parts of audio
        segment_duration = min(3.0, duration / (num_segments + 1))  # 3 seconds or proportional
        embeddings = []
        segment_times = []
        
        for i in range(num_segments):
            start_idx = int((i + 1) * duration / (num_segments + 1) * sr)
            end_idx = int(start_idx + segment_duration * sr)
            
            if end_idx > len(wav):
                end_idx = len(wav)
            if start_idx >= end_idx:
                continue
            
            segment = wav[start_idx:end_idx]
            if len(segment) < sr * 0.5:  # At least 0.5 seconds
                continue
            
            # Get embedding
            embedding = encoder.embed_utterance(segment)
            embeddings.append(embedding)
            segment_times.append((start_idx / sr, end_idx / sr))
        
        if len(embeddings) < 2:
            return None
        
        # Compare embeddings (cosine similarity)
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        
        if not similarities:
            return None
        
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        
        # Threshold: if average similarity > 0.7, likely same speaker
        # If min similarity < 0.5, likely different speakers
        is_single = avg_similarity > 0.7 and min_similarity > 0.5
        
        # Estimate number of speakers
        if is_single:
            num_speakers = 1
            confidence = min(1.0, avg_similarity)
        else:
            # If similarity varies a lot, multiple speakers
            similarity_std = np.std(similarities)
            if similarity_std > 0.2 or min_similarity < 0.4:
                num_speakers = 2  # Conservative estimate
                confidence = 1.0 - avg_similarity
            else:
                num_speakers = 1  # Uncertain, but might be same speaker
                confidence = 0.5
        
        reasons = []
        if not is_single:
            reasons.append(f"embedding_similarity_low:{avg_similarity:.2f}")
        if num_speakers > 1:
            reasons.append(f"estimated_{num_speakers}_speakers")
        
        return SpeakerDiarizationResult(
            num_speakers=num_speakers,
            is_single_speaker=is_single,
            confidence=confidence,
            speaker_segments=[],  # Resemblyzer doesn't provide segments
            total_speech_duration=duration,
            overlap_duration=0.0,
            reasons=reasons,
        )
    except Exception as e:
        logger.warning(f"resemblyzer analysis failed: {e}")
        return None


async def analyze_video_speakers(
    video_id: str,
    sample_segments: List[Tuple[float, float]] = None,
    method: str = "auto",
) -> Optional[SpeakerDiarizationResult]:
    """
    Analyze speakers in a video by sampling audio segments.
    
    Args:
        video_id: YouTube video ID
        sample_segments: List of (start_sec, duration_sec) tuples. If None, samples automatically.
        method: "pyannote", "resemblyzer", or "auto" (tries both)
    
    Returns:
        SpeakerDiarizationResult or None if analysis fails
    """
    # Get audio URL
    audio_url = await asyncio.to_thread(get_audio_url, video_id)
    if not audio_url:
        logger.debug(f"No audio URL for {video_id}")
        return None
    
    # Default: sample 3 segments (beginning, middle, end)
    if sample_segments is None:
        # We'll need video duration, but for now use fixed segments
        sample_segments = [
            (30, 10),   # 10 seconds starting at 30s
            (120, 10),  # 10 seconds starting at 2min
            (300, 10),  # 10 seconds starting at 5min
        ]
    
    # Try each method
    results = []
    
    for start_sec, duration_sec in sample_segments:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Download segment
            success = await asyncio.to_thread(
                download_audio_segment,
                audio_url,
                start_sec,
                duration_sec,
                tmp_path,
            )
            
            if not success:
                continue
            
            # Analyze with available method
            result = None
            
            if method == "pyannote" or (method == "auto" and _PYANNOTE_AVAILABLE):
                result = await asyncio.to_thread(analyze_speakers_pyannote, tmp_path)
            
            if result is None and (method == "resemblyzer" or (method == "auto" and _RESEMBLYZER_AVAILABLE)):
                result = await asyncio.to_thread(analyze_speakers_resemblyzer, tmp_path)
            
            if result:
                results.append(result)
        
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    if not results:
        return None
    
    # Aggregate results
    # If any segment shows multiple speakers, reject
    max_speakers = max(r.num_speakers for r in results)
    any_multi = any(not r.is_single_speaker for r in results)
    avg_confidence = sum(r.confidence for r in results) / len(results)
    
    all_reasons = []
    for r in results:
        all_reasons.extend(r.reasons)
    
    return SpeakerDiarizationResult(
        num_speakers=max_speakers,
        is_single_speaker=not any_multi and max_speakers == 1,
        confidence=avg_confidence,
        speaker_segments=[],
        total_speech_duration=sum(r.total_speech_duration for r in results),
        overlap_duration=sum(r.overlap_duration for r in results),
        reasons=list(set(all_reasons)),
    )
