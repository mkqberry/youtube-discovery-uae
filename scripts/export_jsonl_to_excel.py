#!/usr/bin/env python3
"""
Export JSONL file to Excel format.
"""

import json
import pandas as pd
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def jsonl_to_excel(jsonl_path: Path, excel_path: Path):
    """
    Convert JSONL file to Excel format.
    
    Args:
        jsonl_path: Path to input JSONL file
        excel_path: Path to output Excel file
    """
    logger.info(f"Reading JSONL file: {jsonl_path}")
    
    # Read JSONL file
    entries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                continue
    
    if not entries:
        logger.error("No valid entries found in JSONL file!")
        return
    
    logger.info(f"Found {len(entries)} entries")
    
    # Convert to DataFrame
    df = pd.DataFrame(entries)
    
    # Reorder columns if they exist (put common ones first)
    preferred_order = ['file', 'text', 'duration', 'source_dataset']
    existing_cols = [col for col in preferred_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in preferred_order]
    df = df[existing_cols + other_cols]
    
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"DataFrame shape: {df.shape}")
    
    # Export to Excel
    logger.info(f"Writing to Excel: {excel_path}")
    df.to_excel(excel_path, index=False, engine='openpyxl')
    
    logger.info(f"✓ Successfully exported {len(entries)} entries to {excel_path}")
    
    # Print summary statistics
    if 'duration' in df.columns:
        total_duration = df['duration'].sum()
        logger.info(f"Total duration: {total_duration:.2f}s ({total_duration/3600:.2f} hours)")
    
    if 'source_dataset' in df.columns:
        unique_datasets = df['source_dataset'].nunique()
        logger.info(f"Unique datasets: {unique_datasets}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export JSONL file to Excel format"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output Excel file path (default: input filename with .xlsx extension)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.xlsx')
    
    jsonl_to_excel(input_path, output_path)
