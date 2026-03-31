#!/usr/bin/env python3
"""
Sample monologue data uniformly from each dataset for evaluation.
Targets approximately 1 hour of total audio duration.
"""

import json
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
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
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {metadata_path}: {e}")
                    continue
    return entries


def calculate_dataset_stats(base_dir: Path) -> Dict[str, Dict]:
    """
    Calculate statistics for each dataset (subdirectory).
    
    Returns:
        Dictionary mapping dataset names to their stats
    """
    stats = {}
    
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        metadata_path = subdir / "metadata.jsonl"
        if not metadata_path.exists():
            logger.warning(f"Skipping {subdir.name}: metadata.jsonl not found")
            continue
        
        entries = load_metadata(metadata_path)
        if not entries:
            logger.warning(f"Skipping {subdir.name}: no entries found")
            continue
        
        total_duration = sum(entry.get('duration', 0.0) for entry in entries)
        
        stats[subdir.name] = {
            'path': subdir,
            'entries': entries,
            'total_duration': total_duration,
            'count': len(entries)
        }
    
    return stats


def sample_uniformly(
    entries: List[Dict],
    num_samples: int,
    seed: int = None
) -> List[Dict]:
    """
    Sample entries uniformly from the list.
    
    Args:
        entries: List of metadata entries
        num_samples: Number of samples to take
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled entries
    """
    if seed is not None:
        random.seed(seed)
    
    if num_samples >= len(entries):
        return entries.copy()
    
    # Use systematic sampling for better uniformity
    # This ensures samples are evenly distributed across the dataset
    step = len(entries) / num_samples
    indices = [int(i * step) for i in range(num_samples)]
    
    # Add some randomness to avoid always starting at the beginning
    offset = random.randint(0, max(1, int(step) - 1))
    indices = [(idx + offset) % len(entries) for idx in indices]
    
    sampled = [entries[i] for i in indices]
    
    return sampled


def sample_datasets(
    base_dir: Path,
    output_dir: Path,
    target_duration: float = 3600.0,  # 1 hour in seconds
    seed: int = None
) -> None:
    """
    Sample monologue data uniformly from each dataset.
    
    Args:
        base_dir: Base directory containing subdirectories with monologue data
        output_dir: Output directory for sampled data
        target_duration: Target total duration in seconds (default: 3600 = 1 hour)
        seed: Random seed for reproducibility
    """
    output_wavs_dir = output_dir / "wavs"
    output_metadata_path = output_dir / "metadata.jsonl"
    
    # Create output directories
    output_wavs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Scanning datasets in: {base_dir}")
    logger.info(f"Target total duration: {target_duration:.2f}s ({target_duration/3600:.2f} hours)")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    # Calculate statistics for all datasets
    stats = calculate_dataset_stats(base_dir)
    
    if not stats:
        logger.error("No valid datasets found!")
        return
    
    num_datasets = len(stats)
    logger.info(f"Found {num_datasets} datasets")
    
    # Calculate target duration per dataset (uniform distribution)
    target_per_dataset = target_duration / num_datasets
    logger.info(f"Target duration per dataset: {target_per_dataset:.2f}s ({target_per_dataset/60:.2f} minutes)")
    logger.info("")
    
    # Sample from each dataset
    all_sampled_entries = []
    dataset_summaries = []
    
    for dataset_name, dataset_stats in sorted(stats.items()):
        entries = dataset_stats['entries']
        total_dur = dataset_stats['total_duration']
        
        logger.info(f"Processing: {dataset_name}")
        logger.info(f"  Total entries: {len(entries)}")
        logger.info(f"  Total duration: {total_dur:.2f}s ({total_dur/60:.2f} minutes)")
        
        # Calculate how many samples to take
        # Estimate based on average duration
        if entries:
            avg_duration = total_dur / len(entries)
            num_samples = max(1, int(target_per_dataset / avg_duration))
            num_samples = min(num_samples, len(entries))  # Don't exceed available entries
        else:
            num_samples = 0
        
        logger.info(f"  Sampling {num_samples} entries (target: {target_per_dataset:.2f}s)")
        
        if num_samples == 0:
            logger.warning(f"  Skipping: no entries to sample")
            continue
        
        # Sample uniformly
        sampled = sample_uniformly(entries, num_samples, seed)
        
        # Copy audio files and collect metadata
        sampled_duration = 0.0
        copied_count = 0
        
        for entry in sampled:
            audio_filename = entry.get('file', '')
            if not audio_filename:
                continue
            
            source_audio_path = dataset_stats['path'] / "wavs" / audio_filename
            dest_audio_path = output_wavs_dir / audio_filename
            
            # Handle filename conflicts by prefixing with dataset name
            if dest_audio_path.exists():
                # File already exists from another dataset, rename it
                base_name = audio_filename.rsplit('.', 1)[0]
                ext = audio_filename.rsplit('.', 1)[1] if '.' in audio_filename else 'wav'
                new_filename = f"{dataset_name}_{base_name}.{ext}"
                dest_audio_path = output_wavs_dir / new_filename
                entry['file'] = new_filename
            
            if source_audio_path.exists():
                shutil.copy2(source_audio_path, dest_audio_path)
                sampled_duration += entry.get('duration', 0.0)
                copied_count += 1
                
                # Add dataset source info
                entry['source_dataset'] = dataset_name
                all_sampled_entries.append(entry)
            else:
                logger.warning(f"  Audio file not found: {source_audio_path}")
        
        logger.info(f"  Sampled {copied_count} entries, duration: {sampled_duration:.2f}s ({sampled_duration/60:.2f} minutes)")
        logger.info("")
        
        dataset_summaries.append({
            'dataset': dataset_name,
            'sampled': copied_count,
            'duration': sampled_duration
        })
    
    # Write output metadata
    logger.info("Writing output metadata...")
    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        for entry in all_sampled_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Calculate totals
    total_sampled_duration = sum(e.get('duration', 0.0) for e in all_sampled_entries)
    total_sampled_count = len(all_sampled_entries)
    
    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Sampling Summary")
    logger.info("=" * 70)
    logger.info(f"Total datasets processed: {num_datasets}")
    logger.info(f"Total samples selected: {total_sampled_count}")
    logger.info(f"Total sampled duration: {total_sampled_duration:.2f}s ({total_sampled_duration/3600:.2f} hours)")
    logger.info(f"Target duration: {target_duration:.2f}s ({target_duration/3600:.2f} hours)")
    logger.info(f"Difference: {abs(total_sampled_duration - target_duration):.2f}s")
    logger.info("")
    logger.info("Per-dataset breakdown:")
    for summary in dataset_summaries:
        logger.info(f"  {summary['dataset']:<50} {summary['sampled']:>4} samples, {summary['duration']:>8.2f}s")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Metadata file: {output_metadata_path}")
    logger.info(f"Audio directory: {output_wavs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample monologue data uniformly from each dataset for evaluation"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='Arabic_ASR_merged_monologue',
        help='Input directory containing monologue datasets (default: Arabic_ASR_merged_monologue)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for sampled data'
    )
    parser.add_argument(
        '--target-duration',
        type=float,
        default=3600.0,
        help='Target total duration in seconds (default: 3600 = 1 hour)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        exit(1)
    
    sample_datasets(input_dir, output_dir, args.target_duration, args.seed)
