#!/usr/bin/env python3
"""
Script to extract video IDs directly from Excel file with video links.
No YouTube API calls required - just extracts video IDs from URLs.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List
from urllib.parse import parse_qs, urlparse

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def extract_video_id_from_url(url: str) -> str | None:
    """Extract video ID from YouTube URL."""
    if not url or not isinstance(url, str):
        return None
    
    # Handle various YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Try parsing as URL
    try:
        parsed = urlparse(url)
        if parsed.hostname and ('youtube.com' in parsed.hostname or 'youtu.be' in parsed.hostname):
            if 'youtu.be' in parsed.hostname:
                return parsed.path.lstrip('/')
            query_params = parse_qs(parsed.query)
            if 'v' in query_params:
                return query_params['v'][0]
    except Exception:
        pass
    
    return None


def extract_video_ids_from_excel(excel_path: str) -> List[str]:
    """
    Read Excel file and extract video IDs from links.
    Returns list of video IDs (may contain duplicates).
    """
    logger.info(f"Reading Excel file: {excel_path}")
    df = pd.read_excel(excel_path)
    
    # Find the column with links (case-insensitive)
    link_column = None
    for col in df.columns:
        if 'link' in col.lower() or 'url' in col.lower() or 'iink' in col.lower():
            link_column = col
            break
    
    if link_column is None:
        raise ValueError(f"Could not find link column in Excel file. Columns: {df.columns.tolist()}")
    
    logger.info(f"Found link column: {link_column}")
    logger.info(f"Total rows: {len(df)}")
    
    # Extract video IDs
    video_ids: List[str] = []
    failed_rows = []
    
    for idx, link in enumerate(df[link_column], start=1):
        if pd.isna(link):
            logger.warning(f"Row {idx}: Empty link")
            failed_rows.append(idx)
            continue
            
        vid = extract_video_id_from_url(str(link))
        if vid:
            video_ids.append(vid)
        else:
            logger.warning(f"Row {idx}: Could not extract video ID from: {link}")
            failed_rows.append(idx)
    
    logger.info(f"Extracted {len(video_ids)} video IDs from {len(df)} rows")
    if failed_rows:
        logger.warning(f"Failed to extract video IDs from {len(failed_rows)} rows: {failed_rows}")
    
    if not video_ids:
        raise ValueError("No video IDs found in Excel file")
    
    return video_ids


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract video IDs directly from Excel file links (no YouTube API required)"
    )
    ap.add_argument(
        "--excel",
        required=True,
        help="Path to Excel file with video links (e.g., scripts/uae_channels.xlsx)",
    )
    ap.add_argument(
        "--output",
        default="outputs/uae_candidates.jsonl",
        help="Output JSONL file path (default: outputs/uae_candidates.jsonl)",
    )
    ap.add_argument(
        "--unique",
        action="store_true",
        help="Remove duplicate video IDs",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    args = ap.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    excel_path = Path(args.excel)
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        sys.exit(1)
    
    try:
        # Extract video IDs from Excel
        video_ids = extract_video_ids_from_excel(str(excel_path))
        
        # Remove duplicates if requested
        if args.unique:
            original_count = len(video_ids)
            video_ids = list(dict.fromkeys(video_ids))  # Preserves order
            logger.info(f"Removed {original_count - len(video_ids)} duplicate video IDs")
            logger.info(f"Unique video IDs: {len(video_ids)}")
        
        # Write to output file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Writing {len(video_ids)} video IDs to {output_path}")
        with output_path.open("w", encoding="utf-8") as f:
            for video_id in video_ids:
                record = {"video_id": video_id}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"Successfully wrote {len(video_ids)} video IDs to {output_path}")
        logger.info(f"Output file size: {output_path.stat().st_size} bytes")
        
        # Show sample
        sample_size = min(5, len(video_ids))
        logger.info(f"Sample video IDs: {video_ids[:sample_size]}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
