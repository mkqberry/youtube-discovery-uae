"""
YouTube Video Metadata Fetcher
- Reads video URLs from a JSONL file
- Fetches metadata (duration, title, upload date, view count, etc.) using yt-dlp
- Saves results to a separate JSONL output file
- Supports resuming from where it left off
- Uses Chrome profile rotation for cookie management (same pattern as downloader)
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set

# ============== CONFIGURATION ==============
@dataclass
class Config:
    """Configuration for the metadata fetcher."""

    # Input / Output Paths
    INPUT_FILE: Path = Path("video_urls.jsonl")
    OUTPUT_FILE: Path = Path("video_metadata.jsonl")
    FAILED_FILE: Path = Path("logs/metadata_fetch_failed.jsonl")
    LOG_FILE: Path = Path("logs/metadata_fetch_log.txt")

    # Chrome Settings (same as downloader)
    CHROME_USER_DATA_DIR: str = "/root/.config/google-chrome"

    # yt-dlp timeout (seconds)
    YTDLP_TIMEOUT_SEC: int = 60

    # Retry settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5

    # Human-like delay between requests (seconds)
    MIN_DELAY: float = 10
    MAX_DELAY: float = 30

    # Cookie error → profile switch threshold
    MAX_COOKIE_ERRORS_BEFORE_SWITCH: int = 2

    # Batch size for periodic progress logs
    LOG_BATCH_SIZE: int = 50


config = Config()


# ============== LOGGING ==============
def setup_logging() -> logging.Logger:
    """Configure and return the logger instance.

    Returns:
        Configured logger with file and console handlers.
    """
    os.makedirs(config.LOG_FILE.parent, exist_ok=True)

    _logger = logging.getLogger("MetadataFetcher")
    _logger.setLevel(logging.DEBUG)

    # Detailed file handler
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)-8s - [%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Console handler (summary)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    _logger.addHandler(fh)
    _logger.addHandler(ch)
    return _logger


logger = setup_logging()


# ============== JSONL HELPER ==============
class JSONLHandler:
    """Thread-safe JSONL read/write operations."""

    def __init__(self) -> None:
        self.lock = Lock()

    def read_jsonl(self, filepath: Path) -> List[Dict]:
        """Read records from a JSONL file.

        Args:
            filepath: Path to the JSONL file.

        Returns:
            List of parsed JSON records.
        """
        if not filepath.exists():
            return []

        records: List[Dict] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSONL parse error: {line[:80]}... -> {e}")
        return records

    def append_jsonl(self, filepath: Path, data: Dict) -> None:
        """Append a single record to a JSONL file (thread-safe).

        Args:
            filepath: Path to the JSONL file.
            data: Dictionary to write as a JSON line.
        """
        with self.lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")


jsonl_handler = JSONLHandler()


# ============== CHROME PROFILE MANAGER ==============
class ProfileManager:
    """Discovers and rotates Chrome profiles for cookie-based auth."""

    def __init__(self, user_data_dir: str) -> None:
        self.base_dir = Path(user_data_dir)
        self.profiles = self._discover_profiles()
        self.current_index = 0
        self.cookie_error_count = 0
        self.current_profile = self.profiles[0] if self.profiles else "Default"
        logger.info(
            f"Chrome profiles discovered: {len(self.profiles)} -> {self.profiles}"
        )

    def _discover_profiles(self) -> List[str]:
        """Find Default and Profile X directories.

        Returns:
            Sorted list of profile directory names.
        """
        if not self.base_dir.exists():
            logger.warning(f"Chrome data dir not found: {self.base_dir}")
            return ["Default"]

        profiles: List[str] = []

        if (self.base_dir / "Default").exists():
            profiles.append("Default")

        for path in self.base_dir.glob("Profile *"):
            if path.is_dir():
                profiles.append(path.name)

        return sorted(profiles) if profiles else ["Default"]

    def get_current_profile(self) -> str:
        """Return the currently active profile name."""
        return self.current_profile

    def switch_to_next_profile(self) -> str:
        """Switch to the next profile (round-robin).

        Returns:
            New profile name.
        """
        old = self.current_profile
        self.current_index = (self.current_index + 1) % len(self.profiles)
        self.current_profile = self.profiles[self.current_index]
        self.cookie_error_count = 0
        logger.info(f"Profile switched: {old} -> {self.current_profile}")
        return self.current_profile

    def report_cookie_error(self) -> bool:
        """Increment cookie error counter; switch profile if threshold reached.

        Returns:
            True if a profile switch happened.
        """
        self.cookie_error_count += 1
        if self.cookie_error_count >= config.MAX_COOKIE_ERRORS_BEFORE_SWITCH:
            self.switch_to_next_profile()
            return True
        return False

    def reset_errors(self) -> None:
        """Reset cookie error counter (called on success)."""
        self.cookie_error_count = 0


# ============== ERROR CLASSIFICATION ==============
PERMANENT_ERRORS = [
    "video is private",
    "members-only",
    "video was removed",
    "channel has been terminated",
    "video not found",
    "not available in your region",
    "this video is not available",
    "video has been removed",
]

COOKIE_ERRORS = [
    "sign in",
    "cookie expired",
    "login required",
    "verify your age",
    "age-restricted",
    "403",
    "account issue",
    "cookie error",
    "confirm your age",
    "cookies from browser",
    "unable to extract",
    "cookies are no longer valid",
]


def classify_error(output: str) -> str:
    """Classify yt-dlp error output into PERMANENT, COOKIE, or RETRY.

    Args:
        output: Combined stdout+stderr from yt-dlp.

    Returns:
        Error category string.
    """
    lower = output.lower()
    for err in PERMANENT_ERRORS:
        if err in lower:
            return "PERMANENT"
    for err in COOKIE_ERRORS:
        if err in lower:
            return "COOKIE"
    return "RETRY"


# ============== VIDEO ID EXTRACTION ==============
def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats.

    Args:
        url: YouTube URL.

    Returns:
        11-character video ID or None.
    """
    patterns = [
        r"(?:v=|/v/|youtu\.be/|/embed/|/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# ============== METADATA FETCHER ==============
class MetadataFetcher:
    """Fetches video metadata using yt-dlp without downloading."""

    def __init__(self) -> None:
        self.profile_manager = ProfileManager(config.CHROME_USER_DATA_DIR)
        self.already_fetched: Set[str] = self._load_fetched_urls()

        logger.info(f"Already fetched: {len(self.already_fetched)} videos")

    def _load_fetched_urls(self) -> Set[str]:
        """Load URLs that have already been processed (success or permanent fail).

        Returns:
            Set of video URLs already in the output or permanent-fail file.
        """
        urls: Set[str] = set()

        # Successful fetches
        for record in jsonl_handler.read_jsonl(config.OUTPUT_FILE):
            url = record.get("video_url", "")
            if url:
                urls.add(url)

        # Permanent failures (no need to retry)
        for record in jsonl_handler.read_jsonl(config.FAILED_FILE):
            if record.get("error_type") == "PERMANENT":
                url = record.get("video_url", "")
                if url:
                    urls.add(url)

        return urls

    def _build_metadata_command(self, url: str, profile: str) -> List[str]:
        """Build the yt-dlp command to fetch metadata only.

        Args:
            url: YouTube video URL.
            profile: Chrome profile directory name.

        Returns:
            Command list for subprocess.
        """
        full_profile_path = os.path.join(config.CHROME_USER_DATA_DIR, profile)
        browser_arg = f"chrome:{full_profile_path}"

        return [
            "yt-dlp",
            "--no-check-certificate",
            "--dump-json",        # Output metadata as JSON
            "--no-download",      # Do NOT download the video
            "--socket-timeout", "30",
            "--cookies-from-browser", browser_arg,
            url,
        ]

    def _fetch_single(self, video_url: str, channel_url: str) -> Optional[Dict]:
        """Fetch metadata for a single video with retries and profile rotation.

        Args:
            video_url: YouTube video URL.
            channel_url: Channel URL (from input file).

        Returns:
            Metadata dict on success, None on permanent/exhausted failure.
        """
        video_id = extract_video_id(video_url) or "unknown"

        for attempt in range(1, config.MAX_RETRIES + 1):
            profile = self.profile_manager.get_current_profile()
            cmd = self._build_metadata_command(video_url, profile)

            logger.debug(
                f"[{video_id}] Attempt {attempt}/{config.MAX_RETRIES} "
                f"(profile: {profile})"
            )

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config.YTDLP_TIMEOUT_SEC,
                )

                # Success: parse JSON from stdout
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        raw = json.loads(result.stdout.strip())
                    except json.JSONDecodeError:
                        logger.warning(
                            f"[{video_id}] JSON parse failed on stdout"
                        )
                        continue

                    self.profile_manager.reset_errors()

                    # Build clean metadata record
                    metadata: Dict = {
                        "video_url": video_url,
                        "channel_url": channel_url,
                        "video_id": raw.get("id", video_id),
                        "title": raw.get("title"),
                        "duration": raw.get("duration"),          # seconds
                        "duration_string": raw.get("duration_string"),  # e.g. "12:34"
                        "upload_date": raw.get("upload_date"),    # YYYYMMDD
                        "view_count": raw.get("view_count"),
                        "like_count": raw.get("like_count"),
                        "comment_count": raw.get("comment_count"),
                        "channel": raw.get("channel"),
                        "channel_id": raw.get("channel_id"),
                        "description": raw.get("description"),
                        "categories": raw.get("categories"),
                        "tags": raw.get("tags"),
                        "language": raw.get("language"),
                        "thumbnail": raw.get("thumbnail"),
                        "availability": raw.get("availability"),
                        "age_limit": raw.get("age_limit"),
                        "live_status": raw.get("live_status"),
                        "fetched_at": datetime.now().isoformat(),
                    }
                    return metadata

                # Failure: classify error
                full_output = (result.stdout or "") + (result.stderr or "")
                error_type = classify_error(full_output)

                if error_type == "PERMANENT":
                    logger.warning(
                        f"[{video_id}] Permanent error – skipping"
                    )
                    self._log_failure(video_url, channel_url, video_id, full_output, "PERMANENT")
                    return None

                if error_type == "COOKIE":
                    switched = self.profile_manager.report_cookie_error()
                    label = "switched profile" if switched else "will retry"
                    logger.warning(
                        f"[{video_id}] Cookie error – {label}"
                    )
                    time.sleep(2)
                    continue

                # RETRY (transient error)
                logger.warning(
                    f"[{video_id}] Transient error on attempt {attempt}"
                )
                time.sleep(config.RETRY_DELAY * attempt)

            except subprocess.TimeoutExpired:
                logger.warning(
                    f"[{video_id}] Timeout on attempt {attempt}"
                )
                self.profile_manager.switch_to_next_profile()
                time.sleep(config.RETRY_DELAY)

            except Exception as exc:
                logger.exception(
                    f"[{video_id}] Unexpected error: {exc}"
                )
                self._log_failure(video_url, channel_url, video_id, str(exc), "EXCEPTION")
                return None

        # All retries exhausted
        logger.error(f"[{video_id}] All retries exhausted")
        self._log_failure(video_url, channel_url, video_id, "Max retries exceeded", "RETRY")
        return None

    def _log_failure(
        self,
        video_url: str,
        channel_url: str,
        video_id: str,
        error_msg: str,
        error_type: str,
    ) -> None:
        """Persist a failure record to the failed JSONL file.

        Args:
            video_url: YouTube video URL.
            channel_url: Channel URL.
            video_id: Extracted video ID.
            error_msg: Error description.
            error_type: PERMANENT | COOKIE | RETRY | EXCEPTION | TIMEOUT.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "video_url": video_url,
            "channel_url": channel_url,
            "video_id": video_id,
            "error": error_msg[:500],  # Truncate long error messages
            "error_type": error_type,
        }
        jsonl_handler.append_jsonl(config.FAILED_FILE, record)

    def run(self) -> None:
        """Main entry point: read input, fetch metadata, write output."""
        logger.info("=" * 60)
        logger.info("YOUTUBE METADATA FETCHER STARTING")
        logger.info("=" * 60)

        if not config.INPUT_FILE.exists():
            logger.error(f"Input file not found: {config.INPUT_FILE}")
            sys.exit(1)

        # Read input
        all_records = jsonl_handler.read_jsonl(config.INPUT_FILE)
        logger.info(f"Total records in input: {len(all_records)}")

        # Filter already-fetched
        queue: List[Dict] = []
        for rec in all_records:
            video_url = rec.get("video_url", "")
            if not video_url:
                continue
            if video_url in self.already_fetched:
                continue
            queue.append(rec)

        skipped = len(all_records) - len(queue)
        logger.info(f"Already fetched (skipping): {skipped}")
        logger.info(f"To fetch: {len(queue)}")
        logger.info("=" * 60)

        if not queue:
            logger.info("Nothing to fetch. Exiting.")
            return

        # Ensure output directories exist
        os.makedirs(config.OUTPUT_FILE.parent, exist_ok=True)
        os.makedirs(config.FAILED_FILE.parent, exist_ok=True)

        # Process
        success_count = 0
        fail_count = 0
        total = len(queue)

        for idx, rec in enumerate(queue, start=1):
            video_url = rec.get("video_url", "")
            channel_url = rec.get("channel_url", "")

            logger.info(f"[{idx}/{total}] Fetching: {video_url}")

            metadata = self._fetch_single(video_url, channel_url)

            if metadata is not None:
                jsonl_handler.append_jsonl(config.OUTPUT_FILE, metadata)
                success_count += 1
                duration = metadata.get("duration") or 0
                title = metadata.get("title", "N/A")
                logger.info(
                    f"[{idx}/{total}] OK – \"{title}\" "
                    f"({duration}s / {metadata.get('duration_string', 'N/A')})"
                )
            else:
                fail_count += 1
                logger.warning(f"[{idx}/{total}] FAILED – {video_url}")

            # Periodic progress report
            if idx % config.LOG_BATCH_SIZE == 0:
                logger.info(
                    f"\n--- Progress: {idx}/{total} "
                    f"(success={success_count}, fail={fail_count}) ---\n"
                )

            # Small human-like delay between requests
            if idx < total:
                delay = random.uniform(config.MIN_DELAY, config.MAX_DELAY)
                time.sleep(delay)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("METADATA FETCH COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total processed : {total}")
        logger.info(f"Successful      : {success_count}")
        logger.info(f"Failed          : {fail_count}")
        logger.info(f"Output file     : {config.OUTPUT_FILE}")
        logger.info(f"Failed log      : {config.FAILED_FILE}")
        logger.info("=" * 60)


if __name__ == "__main__":
    try:
        fetcher = MetadataFetcher()
        fetcher.run()
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
