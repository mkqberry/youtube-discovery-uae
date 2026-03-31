"""
GPT Input Generator Script

Generates gpt_input.jsonl from a data directory (e.g. data/asr/youtube) containing
metadata.jsonl files per segment folder.

This script processes entire videos (video-id folders). It keeps a state file under
<script_dir>/logs/ so that already processed video folders are skipped on subsequent runs.

Usage:
    # Full: process all not-yet-processed videos under data/asr/youtube
    python generate_gpt_input.py --asr-dir data/asr/youtube

    # By default, already processed videos are skipped. To process from all:
    python generate_gpt_input.py --asr-dir data/asr/youtube --include-all

Run examples (2-3 örnek):

    # Örnek 1: İlk kez çalıştır (skip state yok), çıktı gpt_input.jsonl; işlenen video-id'ler logs/processed_video_ids.txt'ye yazılır
    python generate_gpt_input.py --asr-dir data/asr/youtube -o gpt_input.jsonl

    # Örnek 2: İkinci çalıştırma – sadece daha önce işlenmeyen videoları işler
    python generate_gpt_input.py --asr-dir data/asr/youtube -o gpt_input_batch2.jsonl

    # Örnek 3: Video başına maksimum 100 segment yaz; özel system instruction ve çıktı
    python generate_gpt_input.py --asr-dir data/asr/youtube --max-segments-per-video 100 -s gpt_evaluation_system_instruction_v3.md -o gpt_input_full.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
METADATA_FILENAME = "metadata.jsonl"
LOG_DIRNAME = "logs"
LOG_FILENAME = "generate_gpt_input.log"
PROCESSED_VIDEOS_FILENAME = "processed_video_ids.txt"
DEFAULT_SYSTEM_INSTRUCTION = "gpt_evaluation_system_instruction_v3.md"
DEFAULT_OUTPUT = "gpt_input.jsonl"
DEFAULT_MAX_SEGMENTS_PER_VIDEO = 100


def _resolve_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
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


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping invalid lines."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def discover_metadata_by_video(root_dir: Path) -> Dict[str, List[Path]]:
    """Find all `metadata.jsonl` under `root_dir`, grouped by video-id folder.

    A video-id folder is assumed to be the first component under root_dir:
    `root_dir/<video_id>/**/metadata.jsonl`.

    Args:
        root_dir: Root directory to search.

    Returns:
        Mapping `video_id -> [metadata_path, ...]`.
    """
    root_dir = root_dir.resolve()
    if not root_dir.is_dir():
        return {}

    out: Dict[str, List[Path]] = {}
    for meta_path in root_dir.rglob(METADATA_FILENAME):
        try:
            rel = meta_path.relative_to(root_dir)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 2:
            continue
        video_id = parts[0]
        out.setdefault(video_id, []).append(meta_path)

    for video_id in out:
        out[video_id] = sorted(out[video_id])
    return out


def segment_identifier(asr_dir: Path, meta_path: Path, wav_name: str) -> str:
    """Build the segment index as a relative path under `asr_dir`.

    The user requested index format like:
    `-W2w0SKteuM/_ [-W2w0SKteuM]_speaker_SPEAKER_01_seg_0000/segment_00000.wav`.
    """
    try:
        rel_meta = meta_path.parent.relative_to(asr_dir)
        return f"{rel_meta.as_posix()}/{wav_name}"
    except ValueError:
        return f"{meta_path.parent.name}/{wav_name}"


def load_processed_video_ids(state_file: Path) -> Set[str]:
    """Load processed video IDs (one per line)."""
    if not state_file.exists():
        return set()
    out: Set[str] = set()
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.add(line)
    except OSError:
        return set()
    return out


def append_processed_video_id(state_file: Path, video_id: str, logger: logging.Logger) -> None:
    """Append a processed video ID to the state file (deduplicated)."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    existing = load_processed_video_ids(state_file)
    if video_id in existing:
        return
    try:
        with open(state_file, "a", encoding="utf-8") as f:
            f.write(video_id + "\n")
    except OSError as e:
        logger.warning("Failed to append processed video id to %s: %s", state_file, e)
        return
    logger.info("Marked video as processed: %s", video_id)


def run_full(
    asr_dir: Path,
    system_instruction_path: Path,
    output_path: Path,
    log_dir: Path,
    exclude_processed: bool,
    max_segments_per_video: int,
) -> int:
    """Process all videos under `asr_dir` into a GPT input JSONL file."""
    logger = setup_logging(log_dir)
    logger.info("Full run: asr_dir=%s, output=%s", asr_dir, output_path)

    meta_by_video = discover_metadata_by_video(asr_dir)
    if not meta_by_video:
        logger.warning("No metadata.jsonl found under %s", asr_dir)
        return 0

    state_file = log_dir / PROCESSED_VIDEOS_FILENAME
    processed_videos = load_processed_video_ids(state_file) if exclude_processed else set()
    if exclude_processed and processed_videos:
        logger.info("Excluding %d previously processed video(s)", len(processed_videos))

    with open(system_instruction_path, "r", encoding="utf-8") as f:
        system_instruction = f.read()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for video_id in sorted(meta_by_video.keys()):
            if exclude_processed and video_id in processed_videos:
                logger.debug("Skipping already processed video: %s", video_id)
                continue

            segment_count_for_video = 0
            wrote_any_for_video = False

            for meta_path in meta_by_video[video_id]:
                if segment_count_for_video >= max_segments_per_video:
                    break

                segment_folder_name = meta_path.parent.name
                for entry in iter_jsonl(meta_path):
                    if segment_count_for_video >= max_segments_per_video:
                        break

                    wav = entry.get("file") or entry.get("audio_file")
                    text = entry.get("text") or entry.get("transcript")
                    if not isinstance(wav, str) or not isinstance(text, str):
                        continue
                    if not wav:
                        continue

                    index = segment_identifier(asr_dir, meta_path, wav)
                    user_prompt = json.dumps(
                        {"index": index, "transcript": text},
                        ensure_ascii=False,
                    )
                    obj = {"user_prompt": user_prompt, "system_instruction": system_instruction}
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

                    wrote_any_for_video = True
                    written += 1
                    segment_count_for_video += 1

            if wrote_any_for_video:
                append_processed_video_id(state_file, video_id, logger)

    logger.info("Wrote %d record(s) to %s", written, output_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate gpt_input.jsonl from ASR dir (e.g. data/asr/youtube).",
        epilog="Logs and processed video-id state are written under the script directory: <script_dir>/logs/.",
    )
    parser.add_argument(
        "--asr-dir",
        required=True,
        help="Root ASR directory (e.g. data/asr/youtube) containing video_id/.../metadata.jsonl",
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
        "--max-segments-per-video",
        type=int,
        default=DEFAULT_MAX_SEGMENTS_PER_VIDEO,
        help=f"Maximum number of segments to include per video-id folder (default: {DEFAULT_MAX_SEGMENTS_PER_VIDEO})",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Process all videos (do not skip already processed). Default is to skip processed videos.",
    )
    args = parser.parse_args()

    asr_dir = Path(args.asr_dir).resolve()
    if not asr_dir.is_dir():
        print(f"Error: ASR directory not found: {asr_dir}")
        return 1

    if args.max_segments_per_video < 1:
        print("Error: --max-segments-per-video must be >= 1")
        return 1

    log_dir = BASE_DIR / LOG_DIRNAME
    system_instruction_path = _resolve_path(args.system_instruction, Path.cwd())
    output_path = _resolve_path(args.output, Path.cwd())

    if not system_instruction_path.exists():
        print(f"Error: system instruction file not found: {system_instruction_path}")
        return 1

    run_full(
        asr_dir=asr_dir,
        system_instruction_path=system_instruction_path,
        output_path=output_path,
        log_dir=log_dir,
        exclude_processed=not args.include_all,
        max_segments_per_video=args.max_segments_per_video,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
