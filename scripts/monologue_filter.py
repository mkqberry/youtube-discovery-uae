#!/usr/bin/env python3
"""
Filter extracted ASR data to keep only monologue content (single-speaker narratives).

This script reads the filtered data and filters out dialogue content,
keeping only monologue/reading style entries, then exports to the output directory.
"""

import json
import re
import shutil
import argparse
from pathlib import Path


def detect_dialogue_patterns(transcript):
    """
    Detect dialogue patterns in transcript.
    Returns True if dialogue patterns are found, False if monologue.
    """
    text = str(transcript).strip()
    
    if not text:
        return True  # Empty transcript treated as dialogue (to filter out)
    
    # Check for very short transcripts first (likely dialogue responses)
    words = text.split()
    if len(words) < 10:  # Very short transcript likely dialogue
        return True
    
    # Dialogue indicators - more specific patterns
    dialogue_patterns = [
        # Multiple consecutive question marks (not single rhetorical questions)
        r'[؟]\s*[؟]',  # Consecutive question marks
        r'\?.*[؟].*[؟]',  # Multiple question marks in sequence
        
        # Turn-taking indicators (multiple short responses)
        r'^[إأا]\s+[إأا]\s+[إأا]',  # Multiple "yes" responses in a row
        r'لا\s+لا\s+لا',  # Multiple "no" responses in a row
        r'صحيح\s+صحيح',  # Multiple confirmations
        
        # Very short standalone responses (typical of dialogue)
        r'^[إأا]\.?\s*$',  # Just "yes"
        r'^لا\.?\s*$',  # Just "no"
        r'^صحيح\.?\s*$',  # Just "correct"
        r'^[إأا]ه\.?\s*$',  # Just "yes" (variant)
        r'^بكمل\.?\s*$',  # Just "continue"
        
        # Conversational turn-taking patterns
        r'[إأا]\s*بس\s*[إأا]\s*بس',  # Yes but yes but (turn-taking)
        r'لا\s*[إأا]\s*لا',  # No yes no (back-and-forth)
        
        # Multiple very short sentences in sequence (typical of dialogue)
        r'\.\s+[إأا]\.\s+[إأا]\.',  # Period yes period yes
        r'\.\s+لا\.\s+لا\.',  # Period no period no
    ]
    
    # Check for dialogue patterns
    for pattern in dialogue_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Check for multiple very short sentences (likely dialogue responses)
    sentences = re.split(r'[\.\?؟]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    short_sentences = [s for s in sentences if len(s.strip()) < 15 and len(s.strip()) > 0]
    
    # If more than 3 very short sentences, likely dialogue
    if len(short_sentences) > 3:
        return True
    
    # If most sentences are very short, likely dialogue
    if len(sentences) > 0 and len(short_sentences) / len(sentences) > 0.5 and len(short_sentences) > 2:
        return True
    
    # Note: We're NOT filtering based on:
    # - Single question marks (rhetorical questions in monologues)
    # - Brackets [ ] (code-switching markers in monologues)
    # - Single instances of "لا" or "إأا" (can appear in monologues)
    
    return False


def is_monologue(transcript):
    """
    Determine if transcript is a monologue.
    Returns True if monologue, False if dialogue.
    """
    return not detect_dialogue_patterns(transcript)


def filter_monologues(
    filtered_dir: Path,
    monologue_dir: Path
):
    """
    Filter audio files to keep only monologue content.
    
    Args:
        filtered_dir: Path to the filtered data directory (input)
        monologue_dir: Path to the monologue output directory
    """
    # Define paths
    filtered_audio_dir = filtered_dir / "wavs"
    filtered_metadata_path = filtered_dir / "metadata.jsonl"
    
    monologue_audio_dir = monologue_dir / "wavs"
    monologue_metadata_path = monologue_dir / "metadata.jsonl"
    
    # Create output directories
    monologue_audio_dir.mkdir(parents=True, exist_ok=True)
    monologue_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input files exist
    if not filtered_metadata_path.exists():
        print(f"Error: Metadata file not found: {filtered_metadata_path}")
        return
    
    if not filtered_audio_dir.exists():
        print(f"Error: Audio directory not found: {filtered_audio_dir}")
        return
    
    print(f"Filtering for monologue content")
    print(f"Reading from: {filtered_dir}")
    print(f"Writing to: {monologue_dir}")
    print()
    
    # Read and filter metadata
    total_count = 0
    monologue_count = 0
    dialogue_count = 0
    skipped_count = 0
    
    with open(filtered_metadata_path, 'r', encoding='utf-8') as infile, \
         open(monologue_metadata_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                total_count += 1
                
                transcript = entry.get('text', '')
                
                # Check if it's a monologue
                if is_monologue(transcript):
                    audio_filename = entry.get('file', '')
                    
                    if not audio_filename:
                        print(f"Warning: Entry {line_num} has no file field, skipping")
                        skipped_count += 1
                        continue
                    
                    # Copy audio file
                    source_audio_path = filtered_audio_dir / audio_filename
                    dest_audio_path = monologue_audio_dir / audio_filename
                    
                    if not source_audio_path.exists():
                        print(f"Warning: Audio file not found: {source_audio_path}, skipping entry")
                        skipped_count += 1
                        continue
                    
                    # Copy audio file (only if it doesn't already exist)
                    if not dest_audio_path.exists():
                        shutil.copy2(source_audio_path, dest_audio_path)
                    
                    # Write monologue metadata entry
                    outfile.write(line + '\n')
                    monologue_count += 1
                    
                    # Progress indicator
                    if monologue_count % 10 == 0:
                        print(f"Processed {total_count} entries, kept {monologue_count} monologues...")
                else:
                    dialogue_count += 1
                
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
    print("=" * 60)
    print("Monologue Filtering Summary:")
    print("=" * 60)
    print(f"Total entries processed: {total_count}")
    print(f"Monologue entries kept: {monologue_count}")
    print(f"Dialogue entries filtered out: {dialogue_count}")
    print(f"Entries skipped (errors): {skipped_count}")
    if total_count > 0:
        print(f"Monologue percentage: {monologue_count/total_count*100:.1f}%")
    print(f"Monologue metadata: {monologue_metadata_path}")
    print(f"Monologue audio directory: {monologue_audio_dir}")


def main():
    """Main filtering function using command-line arguments for input and output directories."""
    parser = argparse.ArgumentParser(
        description="Filter extracted ASR data to keep only monologue content (single-speaker narratives)."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to input filtered directory (should contain wavs/ and metadata.jsonl)."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=str,
        help="Path to output directory where monologue subset will be written."
    )
    args = parser.parse_args()

    filtered_dir = Path(args.input)
    monologue_dir = Path(args.output)

    filter_monologues(filtered_dir, monologue_dir)


if __name__ == "__main__":
    main()
