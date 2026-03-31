#!/usr/bin/env python3
"""
Merge ASR audio segments to ensure minimum duration of 22 seconds.
Preserves the same directory structure as the input dataset.

For RTL languages (e.g., Arabic), text is stored in logical order (the order
it's spoken), which matches the audio order. This script preserves that order
when merging segments to ensure text-audio alignment.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from pydub import AudioSegment
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_metadata(metadata_path: Path) -> List[Dict]:
    """Load metadata from JSONL file."""
    entries = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def merge_segments(
    input_dir: Path,
    output_dir: Path,
    min_duration: float = 22.0
) -> None:
    """
    Merge audio segments to ensure minimum duration.
    
    Args:
        input_dir: Input directory containing wavs/ and metadata.jsonl
        output_dir: Output directory (will be created with same structure)
        min_duration: Minimum duration in seconds for merged segments
    """
    input_wavs_dir = input_dir / "wavs"
    input_metadata = input_dir / "metadata.jsonl"
    
    output_wavs_dir = output_dir / "wavs"
    output_metadata = output_dir / "metadata.jsonl"
    
    # Create output directories
    output_wavs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    logger.info(f"Loading metadata from {input_metadata}")
    entries = load_metadata(input_metadata)
    logger.info(f"Loaded {len(entries)} entries")
    
    # Group segments to merge
    merged_groups = []
    current_group = []
    current_duration = 0.0
    
    for entry in entries:
        duration = entry.get('duration', 0.0)
        current_group.append(entry)
        current_duration += duration
        
        # If we've reached the minimum duration, finalize this group
        if current_duration >= min_duration:
            merged_groups.append(current_group)
            current_group = []
            current_duration = 0.0
    
    # Add remaining segments to the last group (if any)
    if current_group:
        # If the last group is too short, merge it with the previous group
        if merged_groups and current_duration < min_duration:
            merged_groups[-1].extend(current_group)
        else:
            merged_groups.append(current_group)
    
    logger.info(f"Created {len(merged_groups)} merged groups from {len(entries)} original segments")
    
    # Process each group
    merged_entries = []
    
    with open(output_metadata, 'w', encoding='utf-8') as f_out:
        for group_idx, group in enumerate(tqdm(merged_groups, desc="Merging segments")):
            # Load and concatenate audio segments in order
            # IMPORTANT: For RTL languages like Arabic, we preserve the logical order
            # (the order segments appear in the audio) to maintain text-audio alignment
            audio_segments = []
            texts = []
            total_duration = 0.0
            
            # Process entries in the exact order they appear in the group
            # This ensures text and audio are merged in the same sequence
            for entry in group:
                wav_path = input_wavs_dir / entry['file']
                
                if not wav_path.exists():
                    logger.warning(f"Audio file not found: {wav_path}, skipping...")
                    continue
                
                try:
                    audio = AudioSegment.from_file(str(wav_path))
                    # Append in order: first segment first, last segment last
                    audio_segments.append(audio)
                    # Append text in the same order as audio
                    texts.append(entry.get('text', ''))
                    total_duration += len(audio) / 1000.0  # Convert ms to seconds
                except Exception as e:
                    logger.error(f"Error loading {wav_path}: {e}, skipping...")
                    continue
            
            if not audio_segments:
                logger.warning(f"Group {group_idx} has no valid audio segments, skipping...")
                continue
            
            # Merge audio segments in order (first to last)
            merged_audio = sum(audio_segments)
            
            # Merge texts in the same order as audio
            # For Arabic/RTL: text is stored in logical order (spoken order),
            # which matches the audio order. Joining with space preserves this.
            merged_text = " ".join(texts).strip()
            
            # Verify alignment: ensure we have the same number of audio segments and texts
            if len(audio_segments) != len(texts):
                logger.warning(f"Group {group_idx}: Mismatch between audio segments ({len(audio_segments)}) and texts ({len(texts)})")
            
            # Generate output filename
            output_filename = f"merged_segment_{group_idx:05d}.wav"
            output_wav_path = output_wavs_dir / output_filename
            
            # Export merged audio
            merged_audio.export(str(output_wav_path), format="wav")
            
            # Create metadata entry
            merged_entry = {
                "file": output_filename,
                "text": merged_text,
                "duration": total_duration
            }
            
            # Write to metadata file
            f_out.write(json.dumps(merged_entry, ensure_ascii=False) + "\n")
            merged_entries.append(merged_entry)
            
            logger.debug(f"Group {group_idx}: {len(group)} segments -> {total_duration:.2f}s")
    
    logger.info(f"✓ Successfully merged {len(entries)} segments into {len(merged_entries)} merged segments")
    logger.info(f"✓ Output directory: {output_dir}")
    
    # Print statistics
    if merged_entries:
        durations = [e['duration'] for e in merged_entries]
        logger.info(f"  Min duration: {min(durations):.2f}s")
        logger.info(f"  Max duration: {max(durations):.2f}s")
        logger.info(f"  Avg duration: {sum(durations)/len(durations):.2f}s")
        logger.info(f"  Segments below {min_duration}s: {sum(1 for d in durations if d < min_duration)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge ASR audio segments to ensure minimum duration"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory containing wavs/ and metadata.jsonl'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory (will be created with same structure)'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=22.0,
        help='Minimum duration in seconds for merged segments (default: 22.0)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        exit(1)
    
    if not (input_dir / "metadata.jsonl").exists():
        logger.error(f"metadata.jsonl not found in {input_dir}")
        exit(1)
    
    if not (input_dir / "wavs").exists():
        logger.error(f"wavs directory not found in {input_dir}")
        exit(1)
    
    merge_segments(input_dir, output_dir, args.min_duration)
