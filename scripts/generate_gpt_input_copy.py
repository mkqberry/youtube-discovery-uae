"""
GPT Input Generator Script

Generates gpt_input.jsonl from a data directory (e.g. data/asr/youtube) containing
metadata.jsonl files per segment. Supports full run or sampling N segments from N
different videos. Logs and sampled-segment tracking live under the script directory: <script_dir>/logs/.

Usage:
    # Full: all segments from all metadata.jsonl under data_dir
    python generate_gpt_input.py --data-dir data/asr/youtube

    # Sample: 100 segments from 100 different videos (1 per video)
    python generate_gpt_input.py --data-dir data/asr/youtube --sample 100

    # Exclude previously sampled segments when sampling again
    python generate_gpt_input.py --data-dir data/asr/youtube --sample 50 --exclude-sampled
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
METADATA_FILENAME = "metadata.jsonl"
LOG_DIRNAME = "logs"
LOG_FILENAME = "generate_gpt_input.log"
SAMPLED_SEGMENTS_FILENAME = "sampled_segments.txt"
DEFAULT_SYSTEM_INSTRUCTION = "gpt_evaluation_system_instruction.md"
DEFAULT_OUTPUT = "gpt_input.jsonl"


def _resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve path; if relative and base given, resolve against base."""
    p = Path(path)
    if not p.is_absolute() and base is not None:
        return (base / p).resolve()
    return p.resolve()


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to log_dir and console. Creates log_dir if needed."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILENAME

    logger = logging.getLogger("generate_gpt_input")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def discover_segments_by_video(data_dir: Path) -> dict[str, list[tuple[Path, list[dict[str, Any]]]]]:
    """
    Find all metadata.jsonl under data_dir and group by video ID (first parent dir).

    Returns:
        Mapping video_id -> [(metadata_path, [entry, ...]), ...]
    """
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        return {}

    video_to_segments: dict[str, list[tuple[Path, list[dict[str, Any]]]]] = {}
    for meta_path in data_dir.rglob(METADATA_FILENAME):
        try:
            rel = meta_path.relative_to(data_dir)
            # video_id is the first component (e.g. _4rs-znYoFM, uR-R438MEUE)
            parts = rel.parts
            if len(parts) < 2:
                continue
            video_id = parts[0]
            entries: list[dict[str, Any]] = []
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if not entries:
                continue
            key = (video_id, meta_path)
            if video_id not in video_to_segments:
                video_to_segments[video_id] = []
            video_to_segments[video_id].append((meta_path, entries))
        except (ValueError, OSError):
            continue
    return video_to_segments


def segment_identifier(data_dir: Path, video_id: str, meta_path: Path, wav_name: str) -> str:
    """Unique id for a single segment (one wav) for tracking sampled."""
    try:
        rel_meta = meta_path.parent.relative_to(data_dir)
        return f"{rel_meta.as_posix()}/{wav_name}"
    except ValueError:
        return f"{video_id}/{meta_path.parent.name}/{wav_name}"


def load_sampled_ids(sampled_file: Path) -> set[str]:
    """Load previously sampled segment IDs (one per line)."""
    if not sampled_file.exists():
        return set()
    ids: set[str] = set()
    with open(sampled_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def append_sampled_ids(sampled_file: Path, ids: list[str], logger: logging.Logger) -> None:
    """Append new sampled segment IDs to the state file."""
    sampled_file.parent.mkdir(parents=True, exist_ok=True)
    with open(sampled_file, "a", encoding="utf-8") as f:
        for sid in ids:
            f.write(sid + "\n")
    logger.info("Appended %d segment id(s) to %s", len(ids), sampled_file)


def collect_all_entries(
    data_dir: Path,
    video_to_segments: dict[str, list[tuple[Path, list[dict[str, Any]]]]],
    exclude_ids: set[str] | None,
) -> list[tuple[str, str]]:
    """
    Flatten to (user_prompt_json_str, segment_id) for every segment.
    user_prompt matches system instruction: {"audio_file": "...", "transcript": "..."}.
    If exclude_ids is set, skip segments whose id is in exclude_ids.
    """
    out: list[tuple[str, str]] = []
    for video_id, segments in video_to_segments.items():
        for meta_path, entries in segments:
            for entry in entries:
                wav = entry.get("file") or entry.get("audio_file")
                text = entry.get("text") or entry.get("transcript") or ""
                if not wav or text is None:
                    continue
                seg_id = segment_identifier(data_dir, video_id, meta_path, wav)
                if exclude_ids and seg_id in exclude_ids:
                    continue
                # audio_file: video_id/segment_folder/segment_00000.wav for traceability
                user_prompt = json.dumps({"audio_file": seg_id, "transcript": text}, ensure_ascii=False)
                out.append((user_prompt, seg_id))
    return out


def run_full(
    data_dir: Path,
    system_instruction_path: Path,
    output_path: Path,
    log_dir: Path,
    exclude_sampled: bool,
) -> int:
    """Process all segments under data_dir (optionally excluding already sampled)."""
    logger = setup_logging(log_dir)
    logger.info("Full run: data_dir=%s, output=%s", data_dir, output_path)

    video_to_segments = discover_segments_by_video(data_dir)
    if not video_to_segments:
        logger.warning("No metadata.jsonl found under %s", data_dir)
        return 0

    exclude_ids: set[str] | None = None
    if exclude_sampled:
        sampled_file = log_dir / SAMPLED_SEGMENTS_FILENAME
        exclude_ids = load_sampled_ids(sampled_file)
        logger.info("Excluding %d previously sampled segment(s)", len(exclude_ids))

    all_entries = collect_all_entries(data_dir, video_to_segments, exclude_ids)
    if not all_entries:
        logger.warning("No segments to write after exclusions")
        return 0

    with open(system_instruction_path, "r", encoding="utf-8") as f:
        system_instruction = f.read()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for user_prompt, _ in all_entries:
            obj = {"user_prompt": user_prompt, "system_instruction": system_instruction}
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Wrote %d record(s) to %s", count, output_path)
    return count


def run_sample(
    data_dir: Path,
    system_instruction_path: Path,
    output_path: Path,
    log_dir: Path,
    sample_size: int,
    exclude_sampled: bool,
    seed: int | None,
) -> int:
    """
    Sample `sample_size` segments from `sample_size` different videos (1 per video).
    Optionally exclude previously sampled; append new sampled IDs to log_dir/sampled_segments.txt.
    """
    logger = setup_logging(log_dir)
    logger.info("Sample run: data_dir=%s, sample=%d, output=%s", data_dir, sample_size, output_path)

    video_to_segments = discover_segments_by_video(data_dir)
    if not video_to_segments:
        logger.warning("No metadata.jsonl found under %s", data_dir)
        return 0

    sampled_file = log_dir / SAMPLED_SEGMENTS_FILENAME
    exclude_ids = load_sampled_ids(sampled_file) if exclude_sampled else set()
    if exclude_sampled and exclude_ids:
        logger.info("Excluding %d previously sampled segment(s)", len(exclude_ids))

    # Build per-video list of (user_prompt_str, segment_id) for segments not in exclude_ids
    video_candidates: dict[str, list[tuple[str, str]]] = {}
    for video_id, segments in video_to_segments.items():
        cands: list[tuple[str, str]] = []
        for meta_path, entries in segments:
            for entry in entries:
                wav = entry.get("file") or entry.get("audio_file")
                text = entry.get("text") or entry.get("transcript") or ""
                if not wav or text is None:
                    continue
                seg_id = segment_identifier(data_dir, video_id, meta_path, wav)
                if seg_id in exclude_ids:
                    continue
                # audio_file: video_id/segment_folder/segment_00000.wav for traceability
                user_prompt = json.dumps({"audio_file": seg_id, "transcript": text}, ensure_ascii=False)
                cands.append((user_prompt, seg_id))
        if cands:
            video_candidates[video_id] = cands

    available_videos = [v for v in video_candidates if video_candidates[v]]
    if seed is not None:
        random.seed(seed)
    random.shuffle(available_videos)

    # Take one random segment per video, up to sample_size
    chosen: list[tuple[str, str]] = []
    for video_id in available_videos[:sample_size]:
        cands = video_candidates[video_id]
        user_prompt, seg_id = random.choice(cands)
        chosen.append((user_prompt, seg_id))

    if not chosen:
        logger.warning("No segments to write (all excluded or no candidates)")
        return 0

    with open(system_instruction_path, "r", encoding="utf-8") as f:
        system_instruction = f.read()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for user_prompt, seg_id in chosen:
            obj = {"user_prompt": user_prompt, "system_instruction": system_instruction}
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    new_ids = [seg_id for _, seg_id in chosen]
    append_sampled_ids(sampled_file, new_ids, logger)
    logger.info("Wrote %d sampled record(s) to %s and updated %s", len(chosen), output_path, sampled_file)
    return len(chosen)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate gpt_input.jsonl from data dir (e.g. data/asr/youtube) with optional sampling.",
        epilog="Logs and sampled_segments.txt are written under the script directory: <script_dir>/logs/.",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        required=True,
        help="Root data directory (e.g. data/asr/youtube) containing video_id/.../metadata.jsonl",
    )
    parser.add_argument(
        "--system-instruction",
        "-s",
        default=DEFAULT_SYSTEM_INSTRUCTION,
        help=f"Path to system instruction markdown (default: {DEFAULT_SYSTEM_INSTRUCTION})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--sample",
        "-n",
        type=int,
        default=None,
        metavar="N",
        help="If set, sample N segments from N different videos (1 per video) instead of all",
    )
    parser.add_argument(
        "--exclude-sampled",
        action="store_true",
        help="Skip segments that were already sampled (read from data_dir/log/sampled_segments.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling (optional)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}")
        return 1

    log_dir = BASE_DIR / LOG_DIRNAME
    base = data_dir  # resolve paths relative to cwd for system_instruction and output
    system_instruction_path = _resolve_path(args.system_instruction, Path.cwd())
    output_path = _resolve_path(args.output, Path.cwd())

    if not system_instruction_path.exists():
        print(f"Error: system instruction file not found: {system_instruction_path}")
        return 1

    if args.sample is not None:
        if args.sample < 1:
            print("Error: --sample must be >= 1")
            return 1
        run_sample(
            data_dir=data_dir,
            system_instruction_path=system_instruction_path,
            output_path=output_path,
            log_dir=log_dir,
            sample_size=args.sample,
            exclude_sampled=args.exclude_sampled,
            seed=args.seed,
        )
    else:
        run_full(
            data_dir=data_dir,
            system_instruction_path=system_instruction_path,
            output_path=output_path,
            log_dir=log_dir,
            exclude_sampled=args.exclude_sampled,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
