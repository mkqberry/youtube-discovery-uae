"""Merge audio segments by video ID in chronological order.

This script reads all WAV files from the ADI17_EXPORTED/audio directory,
groups them by video ID, sorts each group by start timestamp, and
concatenates them into a single WAV file per video ID.

File naming convention: {video_id}_{start_ms}-{end_ms}.wav

Usage:
    python merge_audio_by_id.py
    python merge_audio_by_id.py --input-dir data/downloaded/public/ADI17_EXPORTED/audio
    python merge_audio_by_id.py --output-dir data/downloaded/public/ADI17_EXPORTED/merged
    python merge_audio_by_id.py --gap-ms 100  # Add 100ms silence between segments
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Regex to parse: {video_id}_{start}-{end}.wav
# video_id may contain alphanumeric chars, hyphens, underscores
FILENAME_PATTERN = re.compile(r"^(.+)_(\d+)-(\d+)\.wav$")

DEFAULT_INPUT_DIR = Path(
    "data/downloaded/public/ADI17_EXPORTED/audio"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/downloaded/public/ADI17_EXPORTED/merged"
)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------
def parse_filename(filename: str) -> tuple[str, int, int] | None:
    """Parse a segment filename into (video_id, start_ms, end_ms).

    Args:
        filename: The WAV filename to parse.

    Returns:
        A tuple of (video_id, start_ms, end_ms) or None if parsing fails.

    Examples:
        >>> parse_filename("00FhWTim-h0_005040-006398.wav")
        ('00FhWTim-h0', 5040, 6398)
    """
    match = FILENAME_PATTERN.match(filename)
    if match is None:
        return None
    video_id = match.group(1)
    start_ms = int(match.group(2))
    end_ms = int(match.group(3))
    return video_id, start_ms, end_ms


def group_files_by_video_id(
    audio_dir: Path,
) -> dict[str, list[tuple[int, int, Path]]]:
    """Group audio segment files by their video ID.

    Args:
        audio_dir: Path to the directory containing WAV segment files.

    Returns:
        A dictionary mapping video_id -> sorted list of (start_ms, end_ms, filepath).
    """
    groups: dict[str, list[tuple[int, int, Path]]] = defaultdict(list)
    skipped = 0

    for wav_path in sorted(audio_dir.glob("*.wav")):
        parsed = parse_filename(wav_path.name)
        if parsed is None:
            skipped += 1
            logger.warning("Skipping unparseable filename: %s", wav_path.name)
            continue
        video_id, start_ms, end_ms = parsed
        groups[video_id].append((start_ms, end_ms, wav_path))

    # Sort each group by start timestamp
    for video_id in groups:
        groups[video_id].sort(key=lambda x: x[0])

    if skipped:
        logger.warning("Skipped %d unparseable files in total.", skipped)

    logger.info(
        "Found %d unique video IDs with %d total segments.",
        len(groups),
        sum(len(v) for v in groups.values()),
    )
    return dict(groups)


def merge_segments(
    segments: list[tuple[int, int, Path]],
    output_path: Path,
    gap_ms: int = 0,
    target_sr: int = 16_000,
) -> float:
    """Concatenate audio segments in order and write to a single WAV file.

    Args:
        segments: Sorted list of (start_ms, end_ms, filepath) tuples.
        output_path: Path where the merged WAV will be saved.
        gap_ms: Milliseconds of silence to insert between segments.
            Defaults to 0 (no gap).
        target_sr: Expected sample rate. Defaults to 16000.

    Returns:
        Total duration of the merged audio in seconds.

    Raises:
        ValueError: If a segment has an unexpected sample rate.
    """
    audio_chunks: list[np.ndarray] = []
    gap_samples = int(target_sr * gap_ms / 1000) if gap_ms > 0 else 0

    for idx, (start_ms, end_ms, filepath) in enumerate(segments):
        data, sr = sf.read(filepath, dtype="float32")
        if sr != target_sr:
            raise ValueError(
                f"Expected sample rate {target_sr}, got {sr} for {filepath}"
            )
        audio_chunks.append(data)

        # Insert silence gap between segments (not after the last one)
        if gap_samples > 0 and idx < len(segments) - 1:
            silence = np.zeros(gap_samples, dtype=np.float32)
            audio_chunks.append(silence)

    # Concatenate all chunks
    merged = np.concatenate(audio_chunks)
    sf.write(str(output_path), merged, target_sr)

    duration_s = len(merged) / target_sr
    return duration_s


def main() -> None:
    """Main entry point for merging audio segments by video ID."""
    parser = argparse.ArgumentParser(
        description="Merge audio segments by video ID in chronological order."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing WAV segment files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where merged WAV files will be saved.",
    )
    parser.add_argument(
        "--gap-ms",
        type=int,
        default=0,
        help="Milliseconds of silence to insert between segments (default: 0).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Expected sample rate of the audio files (default: 16000).",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    gap_ms: int = args.gap_ms
    target_sr: int = args.sample_rate

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Input:  %s", input_dir.resolve())
    logger.info("Output: %s", output_dir.resolve())
    if gap_ms > 0:
        logger.info("Gap between segments: %d ms", gap_ms)

    # Group and sort files
    groups = group_files_by_video_id(input_dir)

    # Merge each group
    total_duration = 0.0
    errors = 0

    for video_id in tqdm(sorted(groups.keys()), desc="Merging videos"):
        segments = groups[video_id]
        output_path = output_dir / f"{video_id}.wav"

        try:
            duration = merge_segments(
                segments=segments,
                output_path=output_path,
                gap_ms=gap_ms,
                target_sr=target_sr,
            )
            total_duration += duration
            logger.debug(
                "Merged %d segments -> %s (%.1fs)",
                len(segments),
                output_path.name,
                duration,
            )
        except Exception:
            errors += 1
            logger.exception("Failed to merge video ID '%s'", video_id)

    # Summary
    logger.info("=" * 60)
    logger.info("Merge complete!")
    logger.info("  Videos merged : %d / %d", len(groups) - errors, len(groups))
    logger.info("  Total duration: %.1f s (%.1f min)", total_duration, total_duration / 60)
    logger.info("  Output dir    : %s", output_dir.resolve())
    if errors:
        logger.warning("  Errors        : %d", errors)


if __name__ == "__main__":
    main()
