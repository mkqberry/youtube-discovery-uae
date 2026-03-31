#!/usr/bin/env python3
"""
Calculate total audio duration for each folder in the merged dataset.
Reads metadata.jsonl files and sums up the duration values.
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.2f}s"
    else:
        return f"{secs:.2f}s"


def calculate_folder_duration(metadata_path: Path) -> Tuple[float, int]:
    """
    Calculate total duration and segment count from metadata.jsonl.
    
    Returns:
        (total_duration_seconds, segment_count)
    """
    if not metadata_path.exists():
        return 0.0, 0
    
    total_duration = 0.0
    segment_count = 0
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    duration = entry.get('duration', 0.0)
                    total_duration += duration
                    segment_count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"  Invalid JSON in {metadata_path}: {e}")
                    continue
    except Exception as e:
        logger.error(f"  Error reading {metadata_path}: {e}")
        return 0.0, 0
    
    return total_duration, segment_count


def calculate_all_durations(base_dir: Path, output_file: str = None) -> Dict[str, Dict]:
    """
    Calculate durations for all folders under base_dir.
    
    Returns:
        Dictionary mapping folder names to their duration info
    """
    results = {}
    total_all_duration = 0.0
    total_all_segments = 0
    
    if not base_dir.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        return results
    
    logger.info(f"Scanning directory: {base_dir}")
    logger.info("")
    
    # Get all subdirectories
    subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    if not subdirs:
        logger.warning("No subdirectories found!")
        return results
    
    logger.info(f"Found {len(subdirs)} subdirectories\n")
    
    # Process each subdirectory
    for subdir in subdirs:
        metadata_path = subdir / "metadata.jsonl"
        folder_name = subdir.name
        
        duration, segment_count = calculate_folder_duration(metadata_path)
        
        results[folder_name] = {
            'duration': duration,
            'segments': segment_count,
            'path': str(subdir)
        }
        
        total_all_duration += duration
        total_all_segments += segment_count
    
    results['_TOTAL_'] = {
        'duration': total_all_duration,
        'segments': total_all_segments,
        'path': str(base_dir)
    }
    
    return results


def print_results(results: Dict[str, Dict], output_file: str = None):
    """Print results in a formatted table."""
    lines = []
    
    # Header
    header = f"{'Folder Name':<60} {'Segments':<12} {'Duration':<20} {'Duration (s)':<15}"
    lines.append(header)
    lines.append("=" * 107)
    
    # Sort results (put _TOTAL_ at the end)
    sorted_items = sorted(
        [(k, v) for k, v in results.items() if k != '_TOTAL_'],
        key=lambda x: x[1]['duration'],
        reverse=True
    )
    
    # Add total at the end
    if '_TOTAL_' in results:
        sorted_items.append(('_TOTAL_', results['_TOTAL_']))
    
    # Print each folder
    for folder_name, info in sorted_items:
        duration = info['duration']
        segments = info['segments']
        duration_str = format_duration(duration)
        
        # Truncate long folder names
        display_name = folder_name if len(folder_name) <= 58 else folder_name[:55] + "..."
        
        if folder_name == '_TOTAL_':
            lines.append("=" * 107)
            display_name = "TOTAL"
        
        line = f"{display_name:<60} {segments:<12} {duration_str:<20} {duration:>14.2f}"
        lines.append(line)
    
    # Print to console
    for line in lines:
        print(line)
    
    # Write to file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                f.write('\n')
            logger.info(f"\nResults saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error writing to output file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate total audio duration for each folder in merged dataset"
    )
    parser.add_argument(
        '--merged-dir',
        type=str,
        default='Arabic_ASR_merged',
        help='Directory containing merged folders (default: Arabic_ASR_merged)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional output file to save results'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    base_dir = script_dir / args.merged_dir
    
    if not base_dir.exists():
        logger.error(f"Merged directory does not exist: {base_dir}")
        exit(1)
    
    # Calculate durations
    results = calculate_all_durations(base_dir, args.output)
    
    if not results or len(results) == 1:  # Only _TOTAL_ exists
        logger.warning("No valid folders found with metadata.jsonl files")
        exit(1)
    
    # Print results
    print_results(results, args.output)
    
    # Print summary
    if '_TOTAL_' in results:
        total_info = results['_TOTAL_']
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Total folders processed: {len(results) - 1}")
        logger.info(f"Total segments: {total_info['segments']}")
        logger.info(f"Total duration: {format_duration(total_info['duration'])} ({total_info['duration']:.2f} seconds)")
        logger.info("=" * 60)
