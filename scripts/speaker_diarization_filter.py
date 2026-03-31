#!/usr/bin/env python3
"""
Speaker Diarization Filter for ASR Dataset

This script uses audio analysis to detect multiple speakers in audio files,
filtering out files with multiple speakers to keep only true monologue content.

Requirements:
    pip install pyannote.audio torch torchaudio
    pip install pyannote.audio[onnx]  # For ONNX runtime (optional, faster)

Note: You may need to accept pyannote.audio model licenses:
    - Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
    - Accept the user conditions
    - Get your HuggingFace token: https://huggingface.co/settings/tokens
    - Set environment variable: export HUGGINGFACE_TOKEN=your_token_here
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
        import os
        
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
        print("3. For fine-grained tokens, enable 'Access to public gated repositories'")
        print("   at: https://huggingface.co/settings/tokens")
        print("4. Or use a Classic token instead of fine-grained token")
        return None


def detect_multiple_speakers(
    audio_path: Path,
    pipeline: Pipeline,
    min_speakers: int = 1,
    max_speakers: int = None
) -> Tuple[bool, Dict]:
    """
    Detect if audio file has multiple speakers.
    
    Args:
        audio_path: Path to audio file
        pipeline: Initialized diarization pipeline
        min_speakers: Minimum number of speakers to detect
        max_speakers: Maximum number of speakers to detect (None = auto)
    
    Returns:
        Tuple of (has_multiple_speakers: bool, diarization_info: dict)
    """
    try:
        # Run diarization
        diarization = pipeline(
            str(audio_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # Extract speaker information
        speakers = set()
        segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        num_speakers = len(speakers)
        has_multiple = num_speakers > 1
        
        # Calculate statistics
        total_duration = sum(seg['end'] - seg['start'] for seg in segments)
        speaker_durations = {}
        for seg in segments:
            speaker = seg['speaker']
            duration = seg['end'] - seg['start']
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
        
        # Find dominant speaker (if multiple)
        dominant_speaker = None
        dominant_ratio = 0.0
        if speaker_durations:
            dominant_speaker = max(speaker_durations, key=speaker_durations.get)
            dominant_duration = speaker_durations[dominant_speaker]
            dominant_ratio = dominant_duration / total_duration if total_duration > 0 else 0.0
        
        info = {
            'num_speakers': num_speakers,
            'speakers': list(speakers),
            'segments': segments,
            'total_duration': total_duration,
            'speaker_durations': speaker_durations,
            'dominant_speaker': dominant_speaker,
            'dominant_ratio': dominant_ratio,
            'has_multiple_speakers': has_multiple
        }
        
        return has_multiple, info
        
    except Exception as e:
        print(f"Error processing {audio_path.name}: {e}")
        return False, {'error': str(e)}


def filter_with_diarization(
    source_dir: Path,
    output_dir: Path,
    pipeline: Pipeline,
    min_dominant_ratio: float = 0.85,
    min_duration: float = 22.0
):
    """
    Filter audio files using speaker diarization.
    
    Args:
        source_dir: Source directory with metadata.jsonl and audio/
        output_dir: Output directory for filtered files
        pipeline: Initialized diarization pipeline
        min_dominant_ratio: Minimum ratio for dominant speaker (0.85 = 85%)
        min_duration: Minimum audio duration in seconds
    """
    source_audio_dir = source_dir / "audio"
    source_metadata_path = source_dir / "metadata.jsonl"
    
    output_audio_dir = output_dir / "audio"
    output_metadata_path = output_dir / "metadata.jsonl"
    
    # Create output directories
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_metadata_path.exists():
        print(f"Error: Metadata file not found: {source_metadata_path}")
        return
    
    print(f"Filtering with speaker diarization")
    print(f"Reading from: {source_dir}")
    print(f"Writing to: {output_dir}")
    print(f"Minimum dominant speaker ratio: {min_dominant_ratio*100:.1f}%")
    print()
    
    total_count = 0
    kept_count = 0
    filtered_multiple_speakers = 0
    filtered_low_dominant = 0
    filtered_short_duration = 0
    skipped_count = 0
    
    with open(source_metadata_path, 'r', encoding='utf-8') as infile, \
         open(output_metadata_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                total_count += 1
                
                audio_filename = entry.get('audio_file', '')
                if not audio_filename:
                    skipped_count += 1
                    continue
                
                audio_path = source_audio_dir / audio_filename
                if not audio_path.exists():
                    skipped_count += 1
                    continue
                
                # Check duration first (quick check)
                duration_ms = entry.get('duration_ms')
                if duration_ms:
                    duration_sec = duration_ms / 1000.0
                    if duration_sec < min_duration:
                        filtered_short_duration += 1
                        continue
                
                # Run speaker diarization
                has_multiple, diarization_info = detect_multiple_speakers(
                    audio_path, pipeline
                )
                
                if 'error' in diarization_info:
                    print(f"  Warning: Error processing {audio_filename}, skipping")
                    skipped_count += 1
                    continue
                
                # Check if multiple speakers detected
                if has_multiple:
                    # Check if dominant speaker ratio is high enough
                    dominant_ratio = diarization_info.get('dominant_ratio', 0.0)
                    
                    if dominant_ratio < min_dominant_ratio:
                        filtered_low_dominant += 1
                        if (filtered_low_dominant + filtered_multiple_speakers) % 10 == 0:
                            print(f"  Filtered {filtered_multiple_speakers + filtered_low_dominant} files with multiple speakers...")
                        continue
                    else:
                        # Has multiple speakers but dominant speaker is clear
                        # Option: keep it but log the info
                        pass
                
                # Keep the file
                output_audio_path = output_audio_dir / audio_filename
                if not output_audio_path.exists():
                    import shutil
                    shutil.copy2(audio_path, output_audio_path)
                
                # Add diarization info to metadata
                entry['diarization'] = {
                    'num_speakers': diarization_info['num_speakers'],
                    'dominant_speaker': diarization_info['dominant_speaker'],
                    'dominant_ratio': round(diarization_info['dominant_ratio'], 3)
                }
                
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                kept_count += 1
                
                if has_multiple:
                    filtered_multiple_speakers += 1
                
                if kept_count % 10 == 0:
                    print(f"  Processed {total_count} files, kept {kept_count}...")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                skipped_count += 1
                continue
    
    # Print summary
    print()
    print("=" * 70)
    print("Speaker Diarization Filtering Summary:")
    print("=" * 70)
    print(f"Total entries processed: {total_count}")
    print(f"Entries kept: {kept_count}")
    print(f"Entries filtered:")
    print(f"  - Multiple speakers (low dominant ratio): {filtered_low_dominant}")
    print(f"  - Short duration (< {min_duration}s): {filtered_short_duration}")
    print(f"Entries skipped (errors): {skipped_count}")
    if total_count > 0:
        print(f"Keep percentage: {kept_count/total_count*100:.1f}%")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


def main():
    """Main function."""
    import os
    
    # Check dependencies
    available, error = check_dependencies()
    if not available:
        print(f"Error: {error}")
        print("\nTo install dependencies:")
        print("  pip install pyannote.audio torch torchaudio")
        print("  pip install pyannote.audio[onnx]  # Optional, for faster inference")
        sys.exit(1)
    
    # Define paths
    source_dir = Path("/bigstorage/workspace/z00836647/asr/mixat/monologue")
    output_dir = Path("/bigstorage/workspace/z00836647/asr/mixat/monologue_diarized")
    
    # Get HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("Warning: HUGGINGFACE_TOKEN not set.")
        print("You may need to:")
        print("1. Accept the model license: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("2. Get your token: https://huggingface.co/settings/tokens")
        print("3. Set: export HUGGINGFACE_TOKEN=your_token_here")
        print("\nTrying without token (may fail)...")
    
    # Check for local model cache
    local_model_dir = source_dir.parent / "models--pyannote--speaker-diarization-3.1"
    cache_dir = None
    if local_model_dir.exists():
        print(f"Found local model directory: {local_model_dir}")
        cache_dir = str(local_model_dir.parent)
        print(f"Using cache directory: {cache_dir}")
    
    # Initialize pipeline
    print("\nInitializing speaker diarization pipeline...")
    pipeline = initialize_diarization_pipeline(
        use_auth_token=hf_token,
        cache_dir=cache_dir
    )
    
    if pipeline is None:
        print("\n✗ Failed to initialize diarization pipeline. Exiting.")
        print("\nRun this to check token access:")
        print("  python check_token_access.py")
        sys.exit(1)
    
    print()
    
    # Filter with diarization
    filter_with_diarization(
        source_dir=source_dir,
        output_dir=output_dir,
        pipeline=pipeline,
        min_dominant_ratio=0.85,  # Keep if dominant speaker is >= 85%
        min_duration=22.0
    )


if __name__ == "__main__":
    main()

