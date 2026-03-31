#!/usr/bin/env python3
"""
Calculate total duration of ASR data with overall_pass: true
"""

import json
import re
import os
from pathlib import Path

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not available, trying alternative methods...")

def get_audio_duration(audio_path):
    """Get duration of audio file in seconds"""
    if not os.path.exists(audio_path):
        return None
    
    if HAS_LIBROSA:
        try:
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e:
            print(f"Error reading {audio_path} with librosa: {e}")
            return None
    else:
        # Try using soxi or ffprobe
        import subprocess
        # Try soxi first
        try:
            result = subprocess.run(['soxi', '-D', audio_path], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        # Try ffprobe
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-show_entries', 
                'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        return None

def extract_json_from_markdown(content):
    """Extract JSON from markdown code block"""
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # Try to parse as plain JSON
    try:
        return json.loads(content)
    except:
        pass
    
    return None

def main():
    jsonl_file = "/storage/workspace/m00836648/youtube_discovery_uae/batch_69959ab0fcf08190b3fdf0b27aa0e407_merged_file.jsonl"
    base_dir = "/storage/workspace/m00836648/youtube_discovery_uae/data/asr/youtube"
    
    total_duration = 0.0
    pass_count = 0
    fail_count = 0
    missing_files = []
    
    print("Reading JSONL file...")
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # Extract assistant response
                messages = data.get('messages', [])
                assistant_msg = None
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        assistant_msg = msg.get('content', '')
                        break
                
                if not assistant_msg:
                    continue
                
                # Parse JSON from assistant response
                eval_data = extract_json_from_markdown(assistant_msg)
                if not eval_data:
                    continue
                
                # Check overall_pass
                evaluation = eval_data.get('evaluation', {})
                overall_pass = evaluation.get('overall_pass', False)
                
                if overall_pass:
                    pass_count += 1
                    # Get audio file path
                    index = eval_data.get('index', '')
                    if index:
                        # Audio files are in wavs/ subdirectory
                        # Insert wavs/ before the segment filename
                        if '/segment_' in index:
                            # Split path and insert wavs before segment filename
                            parts = index.rsplit('/', 1)  # Split at last /
                            if len(parts) == 2:
                                audio_path = os.path.join(base_dir, parts[0], 'wavs', parts[1])
                            else:
                                audio_path = os.path.join(base_dir, index)
                        else:
                            audio_path = os.path.join(base_dir, index)
                        
                        duration = get_audio_duration(audio_path)
                        
                        if duration is not None:
                            total_duration += duration
                        else:
                            missing_files.append(index)
                            fail_count += 1
                else:
                    fail_count += 1
                
                if line_num % 100 == 0:
                    print(f"Processed {line_num} lines... (Pass: {pass_count}, Fail: {fail_count}, Total duration: {total_duration:.2f}s)")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total entries with overall_pass: true: {pass_count}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Total duration: {total_duration/60:.2f} minutes")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} audio files could not be found or read:")
        for f in missing_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

if __name__ == "__main__":
    main()
