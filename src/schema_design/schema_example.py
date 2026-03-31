#!/usr/bin/env python3
"""
Example script showing how to extract and populate metadata schema
from existing folder structure and metadata.jsonl files.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def detect_data_source(metadata_path: Path) -> str:
    """Detect the data source type from the path structure."""
    path_str = str(metadata_path)
    
    # Check for YouTube indicators
    if "youtube" in path_str.lower() or "/asr/youtube/" in path_str:
        return "youtube"
    
    # Check for HuggingFace indicators
    if "huggingface" in path_str.lower() or "hf_" in path_str.lower():
        return "huggingface"
    
    # Check for public repository indicators
    if "public_repo" in path_str.lower() or "repository" in path_str.lower():
        return "public_repo"
    
    # Check folder structure - YouTube typically has video IDs as folder names
    # Public repos might have dataset names or different structures
    parts = metadata_path.parts
    for part in parts:
        # YouTube video IDs are typically 11 characters
        if len(part) >= 10 and (part.startswith('_') or (part[0].isalnum() and len(part) == 11)):
            # Check if it looks like a YouTube ID
            if re.match(r'^[a-zA-Z0-9_-]{10,11}$', part):
                return "youtube"
    
    # Default to unknown, will be determined from metadata if available
    return "unknown"


def extract_video_id_from_path(path: Path) -> Optional[str]:
    """Extract YouTube video ID from folder path."""
    # Video ID is typically the parent folder name
    # Examples: _FlqmYFAabY, _4rs-znYoFM
    parts = path.parts
    for part in reversed(parts):
        # YouTube IDs are typically 11 characters, may start with underscore
        if len(part) >= 10 and (part.startswith('_') or part[0].isalnum()):
            # Remove brackets and other formatting
            clean_id = re.sub(r'[\[\]]', '', part)
            if len(clean_id) >= 10 and re.match(r'^[a-zA-Z0-9_-]{10,11}$', clean_id):
                return clean_id
    return None


def extract_repository_info_from_path(path: Path) -> Dict[str, Any]:
    """Extract repository information from folder path."""
    repo_info = {
        "repository_name": None,
        "dataset_name": None,
        "repository_url": None,
        "dataset_version": None,
        "original_dataset_id": None,
        "original_file_path": None
    }
    
    # Try to extract from path structure
    # Common patterns:
    # - huggingface/datasets/mozilla-foundation/common_voice/...
    # - public_repo/dataset_name/...
    parts = path.parts
    
    for i, part in enumerate(parts):
        if "huggingface" in part.lower() or "hf" in part.lower():
            # Try to extract repository name from next parts
            if i + 1 < len(parts):
                repo_info["repository_name"] = parts[i + 1]
            if i + 2 < len(parts):
                repo_info["dataset_name"] = parts[i + 2]
            repo_info["repository_url"] = f"https://huggingface.co/datasets/{repo_info['repository_name']}/{repo_info['dataset_name']}" if repo_info['repository_name'] and repo_info['dataset_name'] else None
            break
        elif "public_repo" in part.lower() or "repository" in part.lower():
            if i + 1 < len(parts):
                repo_info["dataset_name"] = parts[i + 1]
            break
    
    return repo_info


def extract_segment_info_from_path(path: Path) -> Dict[str, Any]:
    """Extract segment information from folder path."""
    folder_name = path.name
    
    # Pattern 1: "Episode_25_ORANGE-SAFFRON_CURD_ICE_CREAM_TART [_FlqmYFAabY]"
    # Pattern 2: "3 [_4rs-znYoFM]"
    # Pattern 3: "Vanlife [xSf6nYOKq0k]_speaker_SPEAKER_00_seg_0000"
    
    segment_info = {
        "segment_id": folder_name,
        "segment_name": folder_name,
        "speaker_id": None,
        "segment_number": None
    }
    
    # Extract speaker information
    speaker_match = re.search(r'speaker_(SPEAKER_\d+)', folder_name, re.IGNORECASE)
    if speaker_match:
        segment_info["speaker_id"] = speaker_match.group(1)
    
    # Extract segment number
    seg_match = re.search(r'seg_(\d+)', folder_name, re.IGNORECASE)
    if seg_match:
        segment_info["segment_number"] = int(seg_match.group(1))
    
    # Extract episode/title name (before brackets)
    name_match = re.match(r'^([^\[]+)', folder_name)
    if name_match:
        segment_info["segment_name"] = name_match.group(1).strip()
        segment_info["segment_id"] = name_match.group(1).strip()
    
    return segment_info


def parse_existing_metadata(metadata_path: Path) -> list:
    """Parse existing metadata.jsonl file."""
    entries = []
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return entries


def extract_video_title_from_segment(segment_info: Dict[str, Any]) -> str:
    """Extract video title from segment information."""
    segment_name = segment_info.get("segment_name", "")
    # Try to clean up the segment name to make it more readable
    # Replace underscores with spaces, remove brackets content
    title = segment_name.replace("_", " ").strip()
    # Remove video ID in brackets if present
    title = re.sub(r'\s*\[.*?\]\s*', '', title)
    
    # Try to format episode numbers nicely
    # "Episode 25 ORANGE-SAFFRON CURD ICE CREAM TART" -> "Episode 25: ORANGE-SAFFRON CURD ICE CREAM TART"
    title = re.sub(r'(Episode\s+\d+)\s+', r'\1: ', title, flags=re.IGNORECASE)
    
    return title if title else "Unknown Title"


def calculate_cumulative_times(entries: list, current_index: int) -> tuple:
    """Calculate cumulative start/end times for segments."""
    cumulative_start = 0.0
    for i in range(current_index):
        cumulative_start += entries[i].get("duration", 0.0)
    
    current_duration = entries[current_index].get("duration", 0.0)
    cumulative_end = cumulative_start + current_duration
    
    return cumulative_start, cumulative_end


def enhance_metadata_entry(
    old_entry: Dict[str, Any],
    metadata_path: Path,
    segment_index: int,
    total_segments: int,
    all_entries: list = None
) -> Dict[str, Any]:
    """Enhance existing metadata entry with new schema fields."""
    
    # Detect data source
    data_source = old_entry.get("data_source") or detect_data_source(metadata_path)
    
    # Extract information from path
    segment_folder = metadata_path.parent
    parent_folder = segment_folder.parent
    segment_info = extract_segment_info_from_path(segment_folder)
    
    # Calculate temporal information
    duration = old_entry.get("duration", 0.0)
    start_time = 0.0
    end_time = duration
    
    # Calculate original timestamps if we have all entries
    start_time_original = old_entry.get("start_time_original", 0.0)
    end_time_original = old_entry.get("end_time_original", duration)
    
    # Check for diarization info in old_entry (common in public repo data)
    diarization_info = old_entry.get("diarization", {})
    if diarization_info:
        start_time_original = diarization_info.get("original_start", start_time_original)
        end_time_original = diarization_info.get("original_end", end_time_original)
    
    # If not available, calculate cumulative times
    if start_time_original == 0.0 and end_time_original == duration and all_entries:
        start_time_original, end_time_original = calculate_cumulative_times(all_entries, segment_index)
    
    # Extract title/name based on data source
    if data_source == "youtube":
        video_id = extract_video_id_from_path(parent_folder)
        video_title = extract_video_title_from_segment(segment_info)
        segment_name = video_title
    else:
        video_id = None
        video_title = None
        segment_name = segment_info.get("segment_name", segment_info.get("segment_id", "Unknown"))
    
    # Determine speaker information
    speaker_id = segment_info.get("speaker_id") or diarization_info.get("speaker")
    if not speaker_id and diarization_info:
        # Try to extract from diarization
        num_speakers = diarization_info.get("num_speakers", 0)
        if num_speakers == 1:
            speaker_id = diarization_info.get("speaker")
    
    is_single_speaker = speaker_id is not None or (diarization_info.get("num_speakers", 0) == 1)
    speaker_count = diarization_info.get("num_speakers") or (1 if is_single_speaker else None)
    
    # Extract repository information if applicable
    repo_info = {}
    if data_source in ["huggingface", "public_repo"]:
        repo_info = extract_repository_info_from_path(metadata_path)
        # Also check if repository info is in old_entry
        repo_info["repository_name"] = old_entry.get("repository_name") or repo_info.get("repository_name")
        repo_info["dataset_name"] = old_entry.get("dataset_name") or repo_info.get("dataset_name")
        repo_info["repository_url"] = old_entry.get("repository_url") or repo_info.get("repository_url")
        repo_info["dataset_version"] = old_entry.get("dataset_version") or repo_info.get("dataset_version")
        repo_info["original_dataset_id"] = old_entry.get("original_dataset_id") or diarization_info.get("original_file")
        repo_info["original_file_path"] = old_entry.get("original_file_path")
    
    # Construct URLs based on data source
    video_url = None
    if data_source == "youtube" and video_id and video_id != "unknown":
        video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Build enhanced entry with all recommended fields
    enhanced = {
        # Audio file information
        "audio_file": old_entry.get("file", old_entry.get("audio_file", "")),
        "audio_path": f"wavs/{old_entry.get('file', old_entry.get('audio_file', ''))}",
        "audio_format": "wav",  # Default, can be detected from file
        "sample_rate": 16000,  # Default for ASR, can be detected
        "channels": 1,  # Default mono, can be detected
        "bit_depth": 16,  # Default bit depth for WAV files
        
        # Transcription
        "text": old_entry.get("text", ""),
        "language": old_entry.get("language", "ar"),  # Default for UAE dataset
        "language_code": old_entry.get("language_code", "ar-AE"),
        
        # Temporal information
        "duration": duration,
        "start_time": start_time,
        "end_time": end_time,
        "start_time_original": start_time_original,
        "end_time_original": end_time_original,
        
        # Data source information
        "data_source": data_source,
        
        # Segment information
        "segment_id": segment_info.get("segment_id", f"segment_{segment_index}"),
        "segment_index": segment_index,
        "segment_total": total_segments,
        "segment_name": segment_name,
        
        # Speaker information (if available)
        "speaker_id": speaker_id,
        "speaker_label": None,  # Not available from folder structure
        "is_single_speaker": is_single_speaker,
        "speaker_count": speaker_count,
        
        # Processing metadata
        "asr_model": old_entry.get("asr_model", "whisper"),  # Default based on pipeline
        "asr_model_version": old_entry.get("asr_model_version", "large-v3"),  # Default Whisper model version
        "asr_confidence": old_entry.get("asr_confidence", 0.95),  # Default if not present
        "processing_date": old_entry.get("processing_date", datetime.now().isoformat() + "Z"),
        "processing_pipeline": old_entry.get("processing_pipeline", "audio_dataset_generator.py"),
        "denoised": old_entry.get("denoised", True),  # Based on pipeline config
        "vad_checked": old_entry.get("vad_checked", True),  # Based on pipeline config
        "speech_ratio": old_entry.get("speech_ratio", 0.85),  # Default if not present
        
        # Quality metrics
        "quality_score": old_entry.get("quality_score", 0.92),  # Default if not present
        "rejected": old_entry.get("rejected", False),
        "rejection_reason": old_entry.get("rejection_reason"),  # null if not rejected
        
        # Additional metadata
        "metadata_version": "2.0",
        "schema_version": "1.0"
    }
    
    # Add YouTube-specific fields (only if data_source is youtube)
    if data_source == "youtube":
        enhanced["video_id"] = video_id or "unknown"
        enhanced["video_title"] = video_title
        enhanced["channel_id"] = old_entry.get("channel_id", "UC...")  # Placeholder if not available
        enhanced["channel_title"] = old_entry.get("channel_title", "Channel Name")  # Placeholder if not available
        enhanced["video_url"] = video_url
        enhanced["publish_date"] = old_entry.get("publish_date")  # None if not available
    else:
        # Set YouTube fields to null for non-YouTube sources
        enhanced["video_id"] = None
        enhanced["video_title"] = None
        enhanced["channel_id"] = None
        enhanced["channel_title"] = None
        enhanced["video_url"] = None
        enhanced["publish_date"] = None
    
    # Add repository-specific fields (only if data_source is huggingface or public_repo)
    if data_source in ["huggingface", "public_repo"]:
        enhanced["repository_name"] = repo_info.get("repository_name")
        enhanced["dataset_name"] = repo_info.get("dataset_name")
        enhanced["repository_url"] = repo_info.get("repository_url")
        enhanced["dataset_version"] = repo_info.get("dataset_version")
        enhanced["original_dataset_id"] = repo_info.get("original_dataset_id")
        enhanced["original_file_path"] = repo_info.get("original_file_path")
    else:
        # Set repository fields to null for YouTube sources
        enhanced["repository_name"] = None
        enhanced["dataset_name"] = None
        enhanced["repository_url"] = None
        enhanced["dataset_version"] = None
        enhanced["original_dataset_id"] = None
        enhanced["original_file_path"] = None
    
    # Add optional text_normalized if available
    if "text_normalized" in old_entry:
        enhanced["text_normalized"] = old_entry["text_normalized"]
    
    # Keep legacy 'file' field for backward compatibility
    if "file" not in enhanced and "audio_file" in enhanced:
        enhanced["file"] = enhanced["audio_file"]
    
    return enhanced


def migrate_metadata_file(metadata_path: Path, output_path: Optional[Path] = None):
    """Migrate a single metadata.jsonl file to new schema."""
    if output_path is None:
        output_path = metadata_path.parent / "metadata_v2.jsonl"
    
    # Parse existing entries
    old_entries = parse_existing_metadata(metadata_path)
    total_segments = len(old_entries)
    
    # Enhance each entry (pass all_entries for cumulative time calculation)
    enhanced_entries = []
    for idx, old_entry in enumerate(old_entries):
        enhanced = enhance_metadata_entry(
            old_entry, 
            metadata_path, 
            idx, 
            total_segments,
            all_entries=old_entries  # Pass all entries for time calculation
        )
        enhanced_entries.append(enhanced)
    
    # Write enhanced metadata
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in enhanced_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Migrated {total_segments} entries from {metadata_path} to {output_path}")
    return enhanced_entries


def example_usage():
    """Example of how to use the migration functions for both YouTube and public repository data."""
    
    # Example 1: YouTube data
    youtube_metadata_path = Path("/storage/workspace/m00836648/youtube_discovery_uae/data/asr/youtube/_FlqmYFAabY/Episode_25_ORANGE-SAFFRON_CURD_ICE_CREAM_TART [_FlqmYFAabY]/metadata.jsonl")
    youtube_output_path = Path("/storage/workspace/m00836648/youtube_discovery_uae/schema_design/example_formatted_metadata.jsonl")
    
    if youtube_metadata_path.exists():
        print("=" * 60)
        print("Example 1: YouTube Data Migration")
        print("=" * 60)
        enhanced = migrate_metadata_file(youtube_metadata_path, youtube_output_path)
        
        if enhanced:
            print("\nExample YouTube entry:")
            print(json.dumps(enhanced[0], ensure_ascii=False, indent=2))
    else:
        print(f"YouTube file not found: {youtube_metadata_path}")
    
    # Example 2: Public repository data (with diarization info)
    repo_metadata_path = Path("/storage/workspace/m00836648/youtube_discovery_uae/dataset_generator_input/3FNwJeiMRpE/metadata.jsonl")
    
    if repo_metadata_path.exists():
        print("\n" + "=" * 60)
        print("Example 2: Public Repository Data Migration")
        print("=" * 60)
        
        # Parse and enhance repository data
        old_entries = parse_existing_metadata(repo_metadata_path)
        total_segments = len(old_entries)
        
        enhanced_entries = []
        for idx, old_entry in enumerate(old_entries):
            # Mark as public_repo source
            old_entry["data_source"] = "public_repo"
            enhanced = enhance_metadata_entry(
                old_entry,
                repo_metadata_path,
                idx,
                total_segments,
                all_entries=old_entries
            )
            enhanced_entries.append(enhanced)
        
        if enhanced_entries:
            print(f"\nMigrated {total_segments} entries from public repository")
            print("\nExample public repository entry:")
            print(json.dumps(enhanced_entries[0], ensure_ascii=False, indent=2))
    else:
        print(f"\nPublic repository file not found: {repo_metadata_path}")
        print("\nExample of what a public repository entry would look like:")
        example_entry = {
            "audio_file": "segment_00000.wav",
            "audio_path": "wavs/segment_00000.wav",
            "audio_format": "wav",
            "sample_rate": 16000,
            "channels": 1,
            "bit_depth": 16,
            "text": "transcribed text",
            "language": "ar",
            "language_code": "ar-AE",
            "duration": 30.0,
            "start_time": 0.0,
            "end_time": 30.0,
            "start_time_original": 225.95,
            "end_time_original": 255.96,
            "data_source": "huggingface",
            "segment_id": "segment_0",
            "segment_index": 0,
            "segment_total": 1,
            "segment_name": "Unknown",
            "speaker_id": "SPEAKER_00",
            "speaker_label": None,
            "is_single_speaker": True,
            "speaker_count": 1,
            "asr_model": "whisper",
            "asr_model_version": "large-v3",
            "asr_confidence": 0.95,
            "processing_date": datetime.now().isoformat() + "Z",
            "processing_pipeline": "audio_dataset_generator.py",
            "denoised": True,
            "vad_checked": True,
            "speech_ratio": 0.85,
            "quality_score": 0.92,
            "rejected": False,
            "rejection_reason": None,
            "metadata_version": "2.0",
            "schema_version": "1.0",
            "video_id": None,
            "video_title": None,
            "channel_id": None,
            "channel_title": None,
            "video_url": None,
            "publish_date": None,
            "repository_name": "mozilla-foundation/common_voice",
            "dataset_name": "common_voice_13_0",
            "repository_url": "https://huggingface.co/datasets/mozilla-foundation/common_voice",
            "dataset_version": "13.0",
            "original_dataset_id": "original_file.wav",
            "original_file_path": None
        }
        print(json.dumps(example_entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    example_usage()
