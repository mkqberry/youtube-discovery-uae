#!/usr/bin/env python3
"""
Parse GPT assistant responses from JSONL file and extract JSON content.

This script reads a JSONL file where each line contains:
- custom_id: request identifier
- messages: array of message objects with role and content
  - The assistant response is in the message with role="assistant"
  - The assistant content contains JSON wrapped in markdown code blocks

And extracts the JSON content from the assistant message to create a new JSONL file.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def parse_json_content(content: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON content string, handling potential errors and markdown code fences.
    
    This function uses multiple strategies to extract JSON:
    1. Strip markdown code fences and parse
    2. Find JSON objects embedded in text (even in error messages)
    3. Try to extract JSON using balanced brace matching
    """
    if not content:
        return None
    
    original_content = content
    content = content.strip()
    
    # Strategy 1: Handle markdown code blocks
    if content.startswith('```'):
        # Remove opening fence (```json or ```)
        lines = content.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        # Remove closing fence if present
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        content = '\n'.join(lines).strip()
        
        # Try to parse JSON from code block
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Try parsing the whole content as JSON (might be plain JSON)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Find JSON object boundaries using balanced brace matching
    # This handles cases where JSON is embedded in error messages
    start_idx = content.find('{')
    if start_idx != -1:
        # Use balanced brace matching to find the complete JSON object
        brace_count = 0
        end_idx = -1
        
        for i in range(start_idx, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if end_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    # Strategy 4: Fallback - try simple first { to last } extraction
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = content[start_idx:end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 5: Look for JSON in code blocks anywhere in the text (not just at start)
    # Try to find ```json ... ``` blocks anywhere
    json_block_match = re.search(r'```json\s*\n?(.*?)\n?```', original_content, re.DOTALL)
    if json_block_match:
        json_str = json_block_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to find ``` ... ``` blocks (without json label)
    code_block_match = re.search(r'```\s*\n?(.*?)\n?```', original_content, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # All strategies failed
    return None


def main():
    """Main function to parse GPT responses."""
    # Define paths
    base_dir = Path(__file__).parent
    input_path = base_dir / "batch_69959ab0fcf08190b3fdf0b27aa0e407_merged_file.jsonl"
    output_path = base_dir / "parsed_evaluations_batch_69959ab0fcf08190b3fdf0b27aa0e407_merged_file.jsonl"
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print()
    
    line_count = 0
    success_count = 0
    error_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse the input JSONL line
                entry = json.loads(line)
                
                # Extract assistant response from messages array
                assistant_text = ''
                messages = entry.get('messages', [])
                
                # Find the assistant message (usually the last one with role="assistant")
                for msg in reversed(messages):  # Check from end to start
                    if msg.get('role') == 'assistant':
                        assistant_text = msg.get('content', '')
                        break
                
                # Fallback: try direct 'assistant' field (for backward compatibility)
                if not assistant_text:
                    assistant_text = entry.get('assistant', '')
                
                if not assistant_text:
                    print(f"Warning: Line {line_num} has no assistant field in messages array, skipping", file=sys.stderr)
                    error_count += 1
                    continue
                
                # Extract JSON from assistant response using robust parsing
                parsed_json = parse_json_content(assistant_text)
                
                if parsed_json is None:
                    # Still write an entry with the raw assistant text to avoid data loss
                    # This allows manual inspection/recovery later
                    # Only log warning if it looks like it might have been JSON (contains { or })
                    if '{' in assistant_text or '}' in assistant_text:
                        print(f"Warning: Line {line_num} - could not extract JSON (may be error message or malformed JSON)", file=sys.stderr)
                    
                    fallback_entry = {
                        'raw_assistant': assistant_text,
                        'parse_error': True,
                        'line_number': line_num
                    }
                    outfile.write(json.dumps(fallback_entry, ensure_ascii=False) + '\n')
                    error_count += 1
                    line_count += 1
                    continue
                
                # Write parsed JSON as JSONL
                outfile.write(json.dumps(parsed_json, ensure_ascii=False) + '\n')
                success_count += 1
                line_count += 1
                
                # Progress indicator
                if line_count % 100 == 0:
                    print(f"Processed {line_count} entries...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                # Write a fallback entry to preserve data
                fallback_entry = {
                    'raw_line': line,
                    'parse_error': True,
                    'line_number': line_num,
                    'error': str(e)
                }
                try:
                    outfile.write(json.dumps(fallback_entry, ensure_ascii=False) + '\n')
                    line_count += 1
                except:
                    pass
                error_count += 1
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}", file=sys.stderr)
                # Write a fallback entry to preserve data
                fallback_entry = {
                    'raw_line': line[:500] if len(line) > 500 else line,
                    'parse_error': True,
                    'line_number': line_num,
                    'error': str(e)
                }
                try:
                    outfile.write(json.dumps(fallback_entry, ensure_ascii=False) + '\n')
                    line_count += 1
                except:
                    pass
                error_count += 1
                continue
    
    # Print summary
    print()
    print("=" * 60)
    print("Parsing Summary:")
    print("=" * 60)
    print(f"Total entries processed: {line_count}")
    print(f"Successfully parsed: {success_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output file: {output_path}")
    if output_path.exists():
        print(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
