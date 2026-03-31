#!/usr/bin/env python3
"""
Split Multi-Speaker Audio into Single-Speaker Segments

This script processes ASR merged data and uses pyannote speaker diarization
to split multi-speaker audio files into single-speaker monologue segments.

Features:
    - Detailed logging (console + rotating log file)
    - Progress tracking: already-processed directories are skipped on re-run
    - --force flag to reprocess everything from scratch

Requirements:
    pip install pyannote.audio torch torchaudio
    pip install pyannote.audio[onnx]  # Optional, for faster inference

Note: You may need to accept pyannote.audio model licenses:
    - Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
    - Accept the user conditions
    - Get your HuggingFace token: https://huggingface.co/settings/tokens
    - Set environment variable: export HUGGINGFACE_TOKEN=your_token_here
"""

import json
import logging
import os
import subprocess
import sys
import time
import wave
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import required libraries
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logger (module-level; configured lazily via ``setup_logging``)
# ---------------------------------------------------------------------------
logger = logging.getLogger("diarization")

# Progress-tracking filename (stored inside output directory)
PROGRESS_FILENAME = "diarization_progress.json"


# =========================================================================
# Logging setup
# =========================================================================

def setup_logging(
    output_dir: Path,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Configure the module logger with console *and* rotating file handlers.

    Args:
        output_dir: Directory where the log file will be stored.
        console_level: Minimum level for console output.
        file_level: Minimum level for file output.
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of rotated backups to keep.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "diarization.log"

    # Prevent adding duplicate handlers on repeated calls
    if logger.handlers:
        return

    logger.setLevel(logging.DEBUG)  # capture everything; handlers decide

    # --- Formatters ---------------------------------------------------------
    console_fmt = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    file_fmt = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(funcName)s:%(lineno)d │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler ----------------------------------------------------
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    # --- Rotating file handler ----------------------------------------------
    fh = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    fh.setLevel(file_level)
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    logger.info("Logging initialised  →  console=%s  file=%s", console_level, log_path)


# =========================================================================
# Progress tracking
# =========================================================================

def _progress_path(output_dir: Path) -> Path:
    """Return the path to the progress JSON file."""
    return Path("logs") / PROGRESS_FILENAME


def load_progress(output_dir: Path) -> Dict[str, Any]:
    """
    Load the progress file from *output_dir*.

    Returns:
        Dict with at least ``{"completed": {<dir_name>: {...info...}, ...}}``.
    """
    pp = _progress_path(output_dir)
    if pp.exists():
        try:
            with open(pp, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug("Progress file loaded: %s (%d completed entries)", pp, len(data.get("completed", {})))
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupted progress file (%s), starting fresh: %s", pp, exc)
    return {"completed": {}}


def save_progress(output_dir: Path, progress: Dict[str, Any]) -> None:
    """Atomically persist the progress dict to disk."""
    pp = _progress_path(output_dir)
    tmp = pp.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        # Atomic rename (works on POSIX; on Windows it replaces if target exists in Python ≥ 3.3)
        tmp.replace(pp)
        logger.debug("Progress saved (%d completed)", len(progress.get("completed", {})))
    except OSError as exc:
        logger.error("Failed to save progress: %s", exc)


def mark_completed(
    output_dir: Path,
    progress: Dict[str, Any],
    dir_name: str,
    stats: Dict[str, Any],
) -> None:
    """
    Mark *dir_name* as completed in the progress dict and persist immediately.

    Args:
        output_dir: Base output directory (where progress.json lives).
        progress: The in-memory progress dict (mutated in-place).
        dir_name: Name of the directory that finished (e.g. a yt_id).
        stats: Arbitrary stats dict to store alongside the completion record.
    """
    progress["completed"][dir_name] = {
        "completed_at": datetime.now().isoformat(),
        **stats,
    }
    save_progress(output_dir, progress)
    logger.info("✓ Marked '%s' as completed in progress tracker", dir_name)


def is_completed(progress: Dict[str, Any], dir_name: str) -> bool:
    """Return *True* if *dir_name* has already been processed."""
    return dir_name in progress.get("completed", {})


# =========================================================================
# Dependency check
# =========================================================================

def check_dependencies() -> Tuple[bool, Optional[str]]:
    """Check if required dependencies are available."""
    if not PYANNOTE_AVAILABLE:
        return False, "pyannote.audio is not installed"
    if not TORCH_AVAILABLE:
        return False, "PyTorch is not installed"
    return True, None


# =========================================================================
# Pipeline initialisation
# =========================================================================

def initialize_diarization_pipeline(
    model_name: str = "pyannote/speaker-diarization-3.1",
    use_auth_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Optional[Pipeline]:
    """
    Initialize the speaker diarization pipeline.

    Args:
        model_name: HuggingFace model name for diarization.
        use_auth_token: HuggingFace token (or set HUGGINGFACE_TOKEN env var).
        cache_dir: Custom cache directory for models.

    Returns:
        Pipeline object or None if initialization fails.
    """
    if not PYANNOTE_AVAILABLE:
        logger.error("pyannote.audio is not available – cannot initialise pipeline")
        return None

    try:
        # Get token from environment if not provided
        if use_auth_token is None:
            use_auth_token = os.getenv("HUGGINGFACE_TOKEN")

        # Set cache directory if provided
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            logger.debug("Model cache directory set to: %s", cache_dir)

        # Try to load pipeline with token
        logger.info("Loading pipeline: %s", model_name)
        if use_auth_token:
            logger.info("Using HuggingFace token for authentication")
            pipeline = Pipeline.from_pretrained(model_name, token=use_auth_token)
        else:
            logger.info("Attempting to load without token …")
            try:
                pipeline = Pipeline.from_pretrained(model_name)
            except Exception as e:
                if "403" in str(e) or "gated" in str(e).lower() or "token" in str(e).lower():
                    logger.error(
                        "Authentication required for gated models. "
                        "Accept licences at https://huggingface.co/pyannote/speaker-diarization-3.1 "
                        "and set HUGGINGFACE_TOKEN."
                    )
                raise

        # Move to GPU if available
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            logger.info("✓ Using GPU for speaker diarization")
        else:
            logger.info("✓ Using CPU for speaker diarization")

        logger.info("✓ Pipeline initialised successfully")
        return pipeline

    except Exception as e:
        logger.exception("✗ Error initialising diarization pipeline: %s", e)
        logger.info(
            "Troubleshooting:\n"
            "  1. Accept ALL model licences:\n"
            "       https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "       https://huggingface.co/pyannote/segmentation-3.0\n"
            "       https://huggingface.co/pyannote/embedding\n"
            "  2. Set your HuggingFace token:\n"
            "       export HUGGINGFACE_TOKEN=your_token_here\n"
            "       Or use: hf auth login"
        )
        return None


# =========================================================================
# Audio helpers
# =========================================================================

def extract_audio_segment(
    input_audio: Path,
    start_sec: float,
    end_sec: float,
    output_audio: Path,
    sample_rate: int = 16000,
) -> bool:
    """
    Extract an audio segment using ffmpeg.

    Args:
        input_audio: Path to input audio file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        output_audio: Path to output audio file.
        sample_rate: Sample rate for output (default: 16000).

    Returns:
        True if successful, False otherwise.
    """
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",  # overwrite without asking
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", str(input_audio),
        "-vn",   # No video
        "-ac", "1",   # Mono
        "-ar", str(sample_rate),
        "-f", "wav",
        str(output_audio),
    ]
    logger.debug("ffmpeg cmd: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
        success = result.returncode == 0 and output_audio.exists() and output_audio.stat().st_size > 0
        if not success:
            stderr_text = result.stderr.decode(errors="replace").strip()
            logger.warning(
                "ffmpeg failed (rc=%d) for %s [%.2f–%.2f]: %s",
                result.returncode, input_audio.name, start_sec, end_sec, stderr_text,
            )
        return success
    except Exception as e:
        logger.error("Exception extracting segment from %s: %s", input_audio.name, e)
        return False


def get_wav_duration_seconds(wav_path: Path) -> Optional[float]:
    """
    Get WAV duration in seconds using the standard library (no extra deps).

    Returns:
        Duration in seconds, or None if it cannot be determined.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except Exception:
        return None


def build_entries_from_wavs(audio_root: Path) -> List[Dict]:
    """
    Build metadata-like entries from WAV files found under *audio_root*.

    This is used when ``metadata.jsonl`` is not present.
    """
    wav_files = sorted(
        [p for p in audio_root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"],
        key=lambda p: str(p).lower(),
    )
    logger.debug("Found %d WAV files under %s", len(wav_files), audio_root)
    entries: List[Dict] = []
    for wav_path in wav_files:
        rel = wav_path.relative_to(audio_root).as_posix()
        dur = get_wav_duration_seconds(wav_path)
        entry: Dict = {"file": rel, "text": ""}
        if dur is not None:
            entry["duration"] = dur
        entries.append(entry)
    return entries


# =========================================================================
# Speaker detection
# =========================================================================

def detect_speakers(
    audio_path: Path,
    pipeline: Pipeline,
    min_speakers: int = 1,
    max_speakers: int = None,
) -> Tuple[List[Dict], int]:
    """
    Detect speakers in audio file and return segments.

    Supports classic pyannote Annotation output as well as Transformers
    diarization (DiarizeOutput / chunks) output.

    Returns:
        Tuple of (segments, num_speakers).
        Each segment dict: ``{'start': float, 'end': float, 'speaker': str}``.
    """
    try:
        logger.debug("Running diarization on %s", audio_path.name)
        t0 = time.perf_counter()
        diarization = pipeline(
            str(audio_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        elapsed = time.perf_counter() - t0
        logger.debug("Diarization finished in %.2fs for %s", elapsed, audio_path.name)

        segments: List[Dict] = []
        speakers: set = set()

        # --- helpers to reduce repetition ------------------------------------
        def _collect_from_annotation(ann: Any) -> None:
            for turn, _, speaker in ann.itertracks(yield_label=True):
                segments.append({
                    "start": float(getattr(turn, "start", 0.0)),
                    "end": float(getattr(turn, "end", 0.0)),
                    "speaker": str(speaker),
                })
                speakers.add(str(speaker))

        def _collect_from_chunks(chunks: list) -> None:
            for ch in chunks:
                ts = ch.get("timestamp") or ch.get("timestamps")
                if not ts or len(ts) != 2:
                    continue
                start, end = ts
                speaker = ch.get("speaker") or ch.get("label") or ch.get("text") or "UNK"
                segments.append({
                    "start": float(start),
                    "end": float(end),
                    "speaker": str(speaker),
                })
                speakers.add(str(speaker))

        # 1) Classic pyannote Annotation
        if hasattr(diarization, "itertracks"):
            _collect_from_annotation(diarization)

        # 1.b) DiarizeOutput wrapping an Annotation (.annotation)
        elif hasattr(diarization, "annotation") and hasattr(
            getattr(diarization, "annotation"), "itertracks"
        ):
            _collect_from_annotation(diarization.annotation)

        # 1.c) speaker_diarization attr
        elif hasattr(diarization, "speaker_diarization") and hasattr(
            getattr(diarization, "speaker_diarization"), "itertracks"
        ):
            _collect_from_annotation(diarization.speaker_diarization)

        # 1.d) exclusive_speaker_diarization attr
        elif hasattr(diarization, "exclusive_speaker_diarization") and hasattr(
            getattr(diarization, "exclusive_speaker_diarization"), "itertracks"
        ):
            _collect_from_annotation(diarization.exclusive_speaker_diarization)

        # 2) Transformers chunks
        elif hasattr(diarization, "chunks"):
            _collect_from_chunks(getattr(diarization, "chunks", []))

        # 3) Plain list[dict]
        elif isinstance(diarization, list):
            _collect_from_chunks(diarization)

        else:
            logger.warning("Unsupported diarization output type: %s", type(diarization))
            return [], 0

        logger.debug(
            "%s → %d segments, %d speakers detected", audio_path.name, len(segments), len(speakers)
        )
        return segments, len(speakers)

    except Exception as e:
        logger.error("Error detecting speakers in %s: %s", audio_path.name, e, exc_info=True)
        return [], 0


# =========================================================================
# Segment grouping
# =========================================================================

def group_segments_by_speaker(
    segments: List[Dict],
    min_duration: float = 22.0,
    gap_tolerance: float = 0.5,
) -> List[Dict]:
    """
    Group consecutive segments from the same speaker.

    Args:
        segments: List of speaker segments.
        min_duration: Minimum duration for a valid monologue segment.
        gap_tolerance: Maximum gap between segments to merge (seconds).

    Returns:
        List of grouped segments with keys
        ``{'start', 'end', 'speaker', 'duration'}``.
    """
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda x: x["start"])

    grouped: List[Dict] = []
    current_group: Optional[Dict] = None

    for seg in sorted_segments:
        if current_group is None:
            current_group = {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "duration": seg["end"] - seg["start"],
            }
        elif current_group["speaker"] == seg["speaker"]:
            gap = seg["start"] - current_group["end"]
            if gap <= gap_tolerance:
                current_group["end"] = seg["end"]
                current_group["duration"] = current_group["end"] - current_group["start"]
            else:
                if current_group["duration"] >= min_duration:
                    grouped.append(current_group)
                current_group = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg["speaker"],
                    "duration": seg["end"] - seg["start"],
                }
        else:
            if current_group["duration"] >= min_duration:
                grouped.append(current_group)
            current_group = {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "duration": seg["end"] - seg["start"],
            }

    if current_group and current_group["duration"] >= min_duration:
        grouped.append(current_group)

    logger.debug(
        "Grouped %d raw segments → %d valid monologue groups (min_dur=%.1fs, gap=%.1fs)",
        len(segments), len(grouped), min_duration, gap_tolerance,
    )
    return grouped


# =========================================================================
# Per-directory processing
# =========================================================================

def process_video_directory(
    video_dir: Path,
    output_base_dir: Path,
    pipeline: Pipeline,
    min_duration: float = 22.0,
    gap_tolerance: float = 0.5,
) -> Tuple[int, int, int]:
    """
    Process a single video directory, splitting multi-speaker audio into
    monologue segments.

    Args:
        video_dir: Input video directory (yt_id folder with WAV files).
        output_base_dir: Base output directory.
        pipeline: Initialised diarization pipeline.
        min_duration: Minimum duration for output segments.
        gap_tolerance: Maximum gap to merge segments from same speaker.

    Returns:
        Tuple of (total_processed, single_speaker_kept, multi_speaker_split).
    """
    dir_name = video_dir.name
    logger.info("─── START processing directory: %s ───", dir_name)
    t_start = time.perf_counter()

    # Two supported input layouts:
    #   1) legacy: metadata.jsonl + wavs/*.wav
    #   2) new:    **/*.wav (no metadata.jsonl required)
    metadata_path = video_dir / "metadata.jsonl"
    legacy_audio_dir = video_dir / "wavs"

    has_legacy_layout = metadata_path.exists() and legacy_audio_dir.exists()
    if has_legacy_layout:
        audio_root = legacy_audio_dir
        input_entries: List[Dict] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    input_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        logger.info("[%s] Legacy layout detected (metadata.jsonl + wavs/)", dir_name)
    else:
        audio_root = video_dir
        input_entries = build_entries_from_wavs(audio_root)
        if not input_entries:
            logger.warning("[%s] No .wav files found, skipping", dir_name)
            return 0, 0, 0
        logger.info("[%s] New layout detected (%d WAV files)", dir_name, len(input_entries))

    # Create output directory structure
    output_video_dir = output_base_dir / dir_name
    output_audio_dir = output_video_dir / "wavs"
    output_metadata_path = output_video_dir / "metadata.jsonl"
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    single_speaker_kept = 0
    multi_speaker_split = 0

    logger.info("[%s] Processing %d audio files …", dir_name, len(input_entries))

    output_entries: List[Dict] = []
    segment_counter = 0

    for entry_idx, entry in enumerate(input_entries, 1):
        audio_filename = entry.get("file", "")
        if not audio_filename:
            continue

        audio_path = audio_root / audio_filename
        if not audio_path.exists():
            logger.warning("[%s] Audio file not found: %s", dir_name, audio_filename)
            continue

        total_processed += 1

        # Detect speakers
        segments, num_speakers = detect_speakers(audio_path, pipeline)

        if num_speakers == 0:
            logger.warning("[%s] Could not detect speakers in %s, skipping", dir_name, audio_filename)
            continue

        if num_speakers == 1:
            # Single speaker – keep as-is
            output_audio_path = output_audio_dir / audio_filename
            output_audio_path.parent.mkdir(parents=True, exist_ok=True)
            if not output_audio_path.exists():
                import shutil
                shutil.copy2(audio_path, output_audio_path)

            if "duration" not in entry:
                dur = get_wav_duration_seconds(audio_path)
                if dur is not None:
                    entry["duration"] = dur

            entry["diarization"] = {"num_speakers": 1, "original": True}
            entry["file"] = audio_filename
            output_entries.append(entry)
            single_speaker_kept += 1
            logger.debug(
                "[%s] %s → single speaker, kept as-is", dir_name, audio_filename
            )
        else:
            # Multiple speakers – split into monologue segments
            grouped_segments = group_segments_by_speaker(segments, min_duration, gap_tolerance)

            if not grouped_segments:
                logger.warning(
                    "[%s] No valid monologue segments in %s (all < %.1fs)",
                    dir_name, audio_filename, min_duration,
                )
                continue

            base_name = Path(audio_filename).stem
            for seg_idx, seg in enumerate(grouped_segments):
                segment_counter += 1
                output_filename = f"{base_name}_speaker_{seg['speaker']}_seg_{seg_idx:04d}.wav"
                output_audio_path = output_audio_dir / output_filename

                success = extract_audio_segment(
                    audio_path, seg["start"], seg["end"], output_audio_path
                )

                if success:
                    new_entry = {
                        "file": output_filename,
                        "text": entry.get("text", ""),
                        "duration": seg["duration"],
                        "diarization": {
                            "num_speakers": 1,
                            "speaker": seg["speaker"],
                            "original_start": seg["start"],
                            "original_end": seg["end"],
                            "original_file": audio_filename,
                            "split_from_multi_speaker": True,
                        },
                    }
                    output_entries.append(new_entry)
                    multi_speaker_split += 1
                    logger.debug(
                        "[%s] %s → segment %d (%.2f–%.2fs, speaker=%s) extracted",
                        dir_name, audio_filename, seg_idx, seg["start"], seg["end"], seg["speaker"],
                    )
                else:
                    logger.warning(
                        "[%s] Failed to extract segment %d from %s", dir_name, seg_idx, audio_filename
                    )

        # Progress heartbeat every 10 files
        if total_processed % 10 == 0:
            logger.info(
                "[%s] Progress: %d/%d files processed …", dir_name, total_processed, len(input_entries)
            )

    # Write output metadata
    with open(output_metadata_path, "w", encoding="utf-8") as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[%s] ✓ Done in %.1fs — %d files processed: %d single-speaker kept, %d multi-speaker segments created",
        dir_name, elapsed, total_processed, single_speaker_kept, multi_speaker_split,
    )

    return total_processed, single_speaker_kept, multi_speaker_split


# =========================================================================
# Main entry point
# =========================================================================

def main() -> None:
    """Main entry point with CLI argument parsing."""
    import argparse

    # Check dependencies early (before logging is set up)
    available, error = check_dependencies()
    if not available:
        print(f"Error: {error}")
        print("\nTo install dependencies:")
        print("  pip install pyannote.audio torch torchaudio")
        print("  pip install pyannote.audio[onnx]  # Optional, for faster inference")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Split multi-speaker audio into single-speaker monologue segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input directory containing yt_id subfolders.\n"
            "Each subfolder can be either:\n"
            "  - legacy layout: metadata.jsonl + wavs/*.wav\n"
            "  - new layout:    *.wav files directly under the folder\n"
            "Example: data/to_speaker_diarization"
        ),
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for monologue segments"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=22.0,
        help="Minimum duration for output segments in seconds (default: 22.0)",
    )
    parser.add_argument(
        "--gap-tolerance",
        type=float,
        default=0.5,
        help="Maximum gap between segments to merge (seconds, default: 0.5)",
    )
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="HuggingFace model name (default: pyannote/speaker-diarization-3.1)",
    )
    parser.add_argument("--cache-dir", help="Custom cache directory for models")
    parser.add_argument(
        "--video-dir",
        help="Process only a specific video directory (relative name inside input dir)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force reprocessing of ALL directories (ignore progress tracker)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console log level (default: INFO). File log is always DEBUG.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # --- Initialise logging -------------------------------------------------
    console_level = getattr(logging, args.log_level)
    setup_logging(output_dir, console_level=console_level)

    logger.info("=" * 70)
    logger.info("Speaker Diarization – run started at %s", datetime.now().isoformat())
    logger.info("=" * 70)

    # --- Load progress tracker ----------------------------------------------
    progress = load_progress(output_dir)
    already_done = len(progress.get("completed", {}))
    if already_done and not args.force:
        logger.info(
            "Progress tracker loaded: %d directories already completed (use --force to reprocess)",
            already_done,
        )
    elif args.force and already_done:
        logger.warning("--force flag set → ignoring %d previously completed directories", already_done)
        progress = {"completed": {}}
        save_progress(output_dir, progress)

    # --- HuggingFace token --------------------------------------------------
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning(
            "HUGGINGFACE_TOKEN not set. You may need to:\n"
            "  1. Accept the model licence: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  2. Get your token: https://huggingface.co/settings/tokens\n"
            "  3. Set: export HUGGINGFACE_TOKEN=your_token_here\n"
            "Trying without token (may fail) …"
        )

    # --- Initialise pipeline ------------------------------------------------
    logger.info("Initialising speaker diarization pipeline …")
    pipeline = initialize_diarization_pipeline(
        model_name=args.model, use_auth_token=hf_token, cache_dir=args.cache_dir
    )
    if pipeline is None:
        logger.critical("✗ Failed to initialise diarization pipeline. Exiting.")
        sys.exit(1)

    # --- Discover directories -----------------------------------------------
    if args.video_dir:
        video_dirs = [input_dir / args.video_dir]
    else:
        video_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())

    logger.info("Found %d video directories under %s", len(video_dirs), input_dir)
    logger.info("Output directory  : %s", output_dir)
    logger.info("Min duration      : %.1fs", args.min_duration)
    logger.info("Gap tolerance     : %.1fs", args.gap_tolerance)
    logger.info("Force reprocess   : %s", args.force)

    # --- Process each directory ---------------------------------------------
    total_processed = 0
    total_single_kept = 0
    total_multi_split = 0
    skipped_count = 0

    for idx, video_dir in enumerate(video_dirs, 1):
        dir_name = video_dir.name

        # Skip if already completed (and not forced)
        if is_completed(progress, dir_name) and not args.force:
            logger.info(
                "[%d/%d] SKIPPING '%s' (already completed on %s)",
                idx, len(video_dirs), dir_name,
                progress["completed"][dir_name].get("completed_at", "?"),
            )
            skipped_count += 1
            continue

        logger.info("[%d/%d] ▶ Processing: %s", idx, len(video_dirs), dir_name)

        try:
            processed, single_kept, multi_split = process_video_directory(
                video_dir,
                output_dir,
                pipeline,
                min_duration=args.min_duration,
                gap_tolerance=args.gap_tolerance,
            )

            total_processed += processed
            total_single_kept += single_kept
            total_multi_split += multi_split

            # Mark as completed in progress tracker
            mark_completed(output_dir, progress, dir_name, {
                "total_processed": processed,
                "single_speaker_kept": single_kept,
                "multi_speaker_split": multi_split,
            })

        except Exception as e:
            logger.error("✗ Error processing '%s': %s", dir_name, e, exc_info=True)
            continue

    # --- Summary ------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Total video directories found   : %d", len(video_dirs))
    logger.info("Skipped (already completed)      : %d", skipped_count)
    logger.info("Processed this run               : %d", len(video_dirs) - skipped_count)
    logger.info("Total audio files processed      : %d", total_processed)
    logger.info("Single-speaker files kept         : %d", total_single_kept)
    logger.info("Multi-speaker segments created    : %d", total_multi_split)
    logger.info("Output directory                  : %s", output_dir)
    logger.info("Progress file                     : %s", _progress_path(output_dir))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
