#!/usr/bin/env python3
"""
Split Multi-Speaker Audio into Single-Speaker Segments

This script processes ASR merged data and uses pyannote speaker diarization
to split multi-speaker audio files into single-speaker monologue segments.

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
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import required libraries
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Install with:")
    print("  pip install pyannote.audio torch torchaudio")
    print("  pip install pyannote.audio[onnx]  # Optional, for faster inference")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch torchaudio")


def check_dependencies():
    """Check if required dependencies are available."""
    if not PYANNOTE_AVAILABLE:
        return False, "pyannote.audio is not installed"
    if not TORCH_AVAILABLE:
        return False, "PyTorch is not installed"
    return True, None


def initialize_diarization_pipeline(
    model_name: str = "pyannote/speaker-diarization-3.1",
    use_auth_token: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Optional[Pipeline]:
    """
    Initialize the speaker diarization pipeline.
    
    Args:
        model_name: HuggingFace model name for diarization
        use_auth_token: HuggingFace token (or set HUGGINGFACE_TOKEN env var)
        cache_dir: Custom cache directory for models
    
    Returns:
        Pipeline object or None if initialization fails
    """
    if not PYANNOTE_AVAILABLE:
        return None
    
    try:
        # Get token from environment if not provided
        if use_auth_token is None:
            use_auth_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Set cache directory if provided
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
        
        # Try to load pipeline with token
        print(f"Loading pipeline: {model_name}")
        if use_auth_token:
            print("Using HuggingFace token for authentication")
            pipeline = Pipeline.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
        else:
            # Try without token first
            print("Attempting to load without token...")
            try:
                pipeline = Pipeline.from_pretrained(model_name)
            except Exception as e:
                if "403" in str(e) or "gated" in str(e).lower() or "token" in str(e).lower():
                    print("\n⚠️  Authentication required for gated models.")
                    print("Please:")
                    print("1. Accept model licenses at:")
                    print("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
                    print("   - https://huggingface.co/pyannote/segmentation-3.0")
                    print("   - https://huggingface.co/pyannote/embedding")
                    print("2. Set your token: export HUGGINGFACE_TOKEN=your_token_here")
                    print("   Or use: hf auth login")
                raise
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            print("✓ Using GPU for speaker diarization")
        else:
            print("✓ Using CPU for speaker diarization")
        
        print("✓ Pipeline initialized successfully")
        return pipeline
        
    except Exception as e:
        print(f"\n✗ Error initializing diarization pipeline: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've accepted ALL model licenses:")
        print("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   - https://huggingface.co/pyannote/segmentation-3.0")
        print("   - https://huggingface.co/pyannote/embedding")
        print("2. Set your HuggingFace token   export HUGGINGFACE_TOKEN=your_token_here")
        print("   Or use: hf auth login")
        return None


def extract_audio_segment(
    input_audio: Path,
    start_sec: float,
    end_sec: float,
    output_audio: Path,
    sample_rate: int = 16000
) -> bool:
    """
    Extract an audio segment using ffmpeg.
    
    Args:
        input_audio: Path to input audio file
        start_sec: Start time in seconds
        end_sec: End time in seconds
        output_audio: Path to output audio file
        sample_rate: Sample rate for output (default: 16000)
    
    Returns:
        True if successful, False otherwise
    """
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", str(input_audio),
        "-vn",  # No video
        "-ac", "1",  # Mono
        "-ar", str(sample_rate),  # Sample rate
        "-f", "wav",
        str(output_audio),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
            check=False
        )
        return result.returncode == 0 and output_audio.exists() and output_audio.stat().st_size > 0
    except Exception as e:
        print(f"  Error extracting segment: {e}")
        return False


def detect_speakers(
    audio_path: Path,
    pipeline: Pipeline,
    min_speakers: int = 1,
    max_speakers: int = None
) -> Tuple[List[Dict], int]:
    """
    Detect speakers in audio file and return segments.
    
    Args:
        audio_path: Path to audio file
        pipeline: Initialized diarization pipeline
        min_speakers: Minimum number of speakers to detect
        max_speakers: Maximum number of speakers to detect (None = auto)
    
    Returns:
        Tuple of (segments: List[Dict], num_speakers: int)
        Each segment dict has: {'start': float, 'end': float, 'speaker': str}
    """
    try:
        # Run diarization
        diarization = pipeline(
            str(audio_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # Extract segments
        segments = []
        speakers = set()
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
            speakers.add(speaker)
        
        return segments, len(speakers)
        
    except Exception as e:
        print(f"  Error detecting speakers: {e}")
        return [], 0


def group_segments_by_speaker(
    segments: List[Dict],
    min_duration: float = 22.0,
    gap_tolerance: float = 0.5
) -> List[Dict]:
    """
    Group consecutive segments from the same speaker.
    
    Args:
        segments: List of speaker segments
        min_duration: Minimum duration for a valid monologue segment
        gap_tolerance: Maximum gap between segments to merge (seconds)
    
    Returns:
        List of grouped segments with {'start': float, 'end': float, 'speaker': str, 'duration': float}
    """
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x['start'])
    
    grouped = []
    current_group = None
    
    for seg in sorted_segments:
        if current_group is None:
            # Start new group
            current_group = {
                'start': seg['start'],
                'end': seg['end'],
                'speaker': seg['speaker'],
                'duration': seg['end'] - seg['start']
            }
        elif current_group['speaker'] == seg['speaker']:
            # Same speaker - check if we should merge
            gap = seg['start'] - current_group['end']
            if gap <= gap_tolerance:
                # Merge segments
                current_group['end'] = seg['end']
                current_group['duration'] = current_group['end'] - current_group['start']
            else:
                # Gap too large - save current group and start new one
                if current_group['duration'] >= min_duration:
                    grouped.append(current_group)
                current_group = {
                    'start': seg['start'],
                    'end': seg['end'],
                    'speaker': seg['speaker'],
                    'duration': seg['end'] - seg['start']
                }
        else:
            # Different speaker - save current group and start new one
            if current_group['duration'] >= min_duration:
                grouped.append(current_group)
            current_group = {
                'start': seg['start'],
                'end': seg['end'],
                'speaker': seg['speaker'],
                'duration': seg['end'] - seg['start']
            }
    
    # Don't forget the last group
    if current_group and current_group['duration'] >= min_duration:
        grouped.append(current_group)
    
    return grouped


def process_video_directory(
    video_dir: Path,
    output_base_dir: Path,
    pipeline: Pipeline,
    min_duration: float = 22.0,
    gap_tolerance: float = 0.5
) -> Tuple[int, int, int]:
    """
    Process a single video directory, splitting multi-speaker audio into monologues.
    
    Args:
        video_dir: Input video directory (contains metadata.jsonl and wavs/)
        output_base_dir: Base output directory
        pipeline: Initialized diarization pipeline
        min_duration: Minimum duration for output segments
        gap_tolerance: Maximum gap to merge segments from same speaker
    
    Returns:
        Tuple of (total_processed, single_speaker_kept, multi_speaker_split)
    """
    metadata_path = video_dir / "metadata.jsonl"
    audio_dir = video_dir / "wavs"
    
    if not metadata_path.exists():
        print(f"  Warning: metadata.jsonl not found in {video_dir.name}")
        return 0, 0, 0
    
    if not audio_dir.exists():
        print(f"  Warning: wavs directory not found in {video_dir.name}")
        return 0, 0, 0
    
    # Create output directory structure
    output_video_dir = output_base_dir / video_dir.name
    output_audio_dir = output_video_dir / "wavs"
    output_metadata_path = output_video_dir / "metadata.jsonl"
    
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    single_speaker_kept = 0
    multi_speaker_split = 0
    
    # Read metadata
    entries = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    print(f"  Processing {len(entries)} audio files...")
    
    output_entries = []
    segment_counter = 0
    
    for entry in entries:
        audio_filename = entry.get('file', '')
        if not audio_filename:
            continue
        
        audio_path = audio_dir / audio_filename
        if not audio_path.exists():
            print(f"    Warning: Audio file not found: {audio_filename}")
            continue
        
        total_processed += 1
        
        # Detect speakers
        segments, num_speakers = detect_speakers(audio_path, pipeline)
        
        if num_speakers == 0:
            print(f"    Warning: Could not detect speakers in {audio_filename}, skipping")
            continue
        
        if num_speakers == 1:
            # Single speaker - keep as is
            output_audio_path = output_audio_dir / audio_filename
            if not output_audio_path.exists():
                import shutil
                shutil.copy2(audio_path, output_audio_path)
            
            entry['diarization'] = {
                'num_speakers': 1,
                'original': True
            }
            output_entries.append(entry)
            single_speaker_kept += 1
            
        else:
            # Multiple speakers - split into monologue segments
            grouped_segments = group_segments_by_speaker(segments, min_duration, gap_tolerance)
            
            if not grouped_segments:
                print(f"    Warning: No valid monologue segments found in {audio_filename}")
                continue
            
            # Extract each monologue segment
            base_name = Path(audio_filename).stem
            for seg_idx, seg in enumerate(grouped_segments):
                segment_counter += 1
                output_filename = f"{base_name}_speaker_{seg['speaker']}_seg_{seg_idx:04d}.wav"
                output_audio_path = output_audio_dir / output_filename
                
                # Extract audio segment
                success = extract_audio_segment(
                    audio_path,
                    seg['start'],
                    seg['end'],
                    output_audio_path
                )
                
                if success:
                    # Create new metadata entry
                    new_entry = {
                        'file': output_filename,
                        'text': entry.get('text', ''),  # Keep original text (or could be empty)
                        'duration': seg['duration'],
                        'diarization': {
                            'num_speakers': 1,
                            'speaker': seg['speaker'],
                            'original_start': seg['start'],
                            'original_end': seg['end'],
                            'original_file': audio_filename,
                            'split_from_multi_speaker': True
                        }
                    }
                    output_entries.append(new_entry)
                    multi_speaker_split += 1
                else:
                    print(f"    Warning: Failed to extract segment {seg_idx} from {audio_filename}")
        
        if total_processed % 10 == 0:
            print(f"    Processed {total_processed}/{len(entries)} files...")
    
    # Write output metadata
    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"  ✓ Processed {total_processed} files:")
    print(f"    - Kept {single_speaker_kept} single-speaker files")
    print(f"    - Split {multi_speaker_split} segments from multi-speaker files")
    
    return total_processed, single_speaker_kept, multi_speaker_split


def main():
    """Main function."""
    import argparse
    
    # Check dependencies
    available, error = check_dependencies()
    if not available:
        print(f"Error: {error}")
        print("\nTo install dependencies:")
        print("  pip install pyannote.audio torch torchaudio")
        print("  pip install pyannote.audio[onnx]  # Optional, for faster inference")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Split multi-speaker audio into single-speaker monologue segments"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing video directories (e.g., data/asr_merged/manual_searched)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for monologue segments"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=22.0,
        help="Minimum duration for output segments in seconds (default: 22.0)"
    )
    parser.add_argument(
        "--gap-tolerance",
        type=float,
        default=0.5,
        help="Maximum gap between segments to merge (seconds, default: 0.5)"
    )
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="HuggingFace model name (default: pyannote/speaker-diarization-3.1)"
    )
    parser.add_argument(
        "--cache-dir",
        help="Custom cache directory for models"
    )
    parser.add_argument(
        "--video-dir",
        help="Process only a specific video directory (relative to input dir)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Get HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("Warning: HUGGINGFACE_TOKEN not set.")
        print("You may need to:")
        print("1. Accept the model license: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("2. Get your token: https://huggingface.co/settings/tokens")
        print("3. Set: export HUGGINGFACE_TOKEN=your_token_here")
        print("\nTrying without token (may fail)...")
    
    # Initialize pipeline
    print("\nInitializing speaker diarization pipeline...")
    pipeline = initialize_diarization_pipeline(
        model_name=args.model,
        use_auth_token=hf_token,
        cache_dir=args.cache_dir
    )
    
    if pipeline is None:
        print("\n✗ Failed to initialize diarization pipeline. Exiting.")
        sys.exit(1)
    
    print()
    
    # Find video directories
    if args.video_dir:
        video_dirs = [input_dir / args.video_dir]
    else:
        video_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(video_dirs)} video directories to process")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Min duration: {args.min_duration}s")
    print(f"Gap tolerance: {args.gap_tolerance}s")
    print()
    
    total_processed = 0
    total_single_kept = 0
    total_multi_split = 0
    
    for idx, video_dir in enumerate(video_dirs, 1):
        print(f"[{idx}/{len(video_dirs)}] Processing: {video_dir.name}")
        
        try:
            processed, single_kept, multi_split = process_video_directory(
                video_dir,
                output_dir,
                pipeline,
                min_duration=args.min_duration,
                gap_tolerance=args.gap_tolerance
            )
            
            total_processed += processed
            total_single_kept += single_kept
            total_multi_split += multi_split
            
        except Exception as e:
            print(f"  ✗ Error processing {video_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print()
    
    # Print summary
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total video directories processed: {len(video_dirs)}")
    print(f"Total audio files processed: {total_processed}")
    print(f"Single-speaker files kept: {total_single_kept}")
    print(f"Multi-speaker segments created: {total_multi_split}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

