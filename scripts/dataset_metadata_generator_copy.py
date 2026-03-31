#!/usr/bin/env python3
"""
Dataset Metadata Generator

Audio dosyalarından (WAV) metadata bilgileri çıkarır ve dataset_metadata.jsonl dosyası oluşturur.

Özellikler:
- WAV dosyalarının duration bilgisini çıkarır
- Dosya adı, path, sample rate, channels gibi bilgileri toplar
- JSONL formatında metadata dosyası oluşturur
"""

import argparse
import json
import logging
import wave
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_wav_metadata(wav_path: Path) -> Optional[Dict[str, any]]:
    """
    WAV dosyasından metadata bilgilerini çıkarır.

    Args:
        wav_path: WAV dosyasının path'i

    Returns:
        Metadata dictionary veya None (hata durumunda)
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            
            if rate <= 0:
                logger.warning(f"Geçersiz sample rate: {rate} - {wav_path}")
                return None
            
            duration = frames / float(rate)
            
            # Dosya boyutu (bytes)
            file_size = wav_path.stat().st_size
            
            return {
                "duration": round(duration, 3),
                "sample_rate": rate,
                "channels": channels,
                "sample_width": sample_width,
                "frames": frames,
                "file_size_bytes": file_size,
            }
    except Exception as e:
        logger.error(f"WAV metadata çıkarılamadı ({wav_path}): {e}")
        return None


def generate_metadata(
    audio_dir: Path,
    output_file: Optional[Path] = None,
    relative_path: bool = True,
) -> List[Dict[str, any]]:
    """
    Belirtilen dizindeki tüm WAV dosyaları için metadata oluşturur.

    Args:
        audio_dir: Audio dosyalarının bulunduğu dizin
        output_file: Çıktı JSONL dosyası path'i (None ise audio_dir/dataset_metadata.jsonl)
        relative_path: Dosya path'lerini relative olarak kaydet (True) veya absolute (False)

    Returns:
        Metadata entry'lerinin listesi
    """
    if not audio_dir.exists():
        raise FileNotFoundError(f"Dizin bulunamadı: {audio_dir}")
    
    if not audio_dir.is_dir():
        raise ValueError(f"Path bir dizin değil: {audio_dir}")
    
    # WAV dosyalarını bul
    wav_files = sorted(
        [p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"],
        key=lambda p: str(p).lower(),
    )
    
    if not wav_files:
        logger.warning(f"Hiç WAV dosyası bulunamadı: {audio_dir}")
        return []
    
    logger.info(f"{len(wav_files)} WAV dosyası bulundu")
    
    # Metadata oluştur
    metadata_entries: List[Dict[str, any]] = []
    failed_count = 0
    
    for wav_path in tqdm(wav_files, desc="Metadata çıkarılıyor"):
        # Dosya adı (basename)
        filename = wav_path.name
        
        # Relative veya absolute path
        if relative_path:
            file_path = wav_path.relative_to(audio_dir).as_posix()
        else:
            file_path = str(wav_path)
        
        # WAV metadata bilgilerini al
        wav_meta = get_wav_metadata(wav_path)
        
        if wav_meta is None:
            failed_count += 1
            continue
        
        # Entry oluştur
        entry: Dict[str, any] = {
            "file": file_path,
            "filename": filename,
            **wav_meta,
        }
        
        metadata_entries.append(entry)
    
    if failed_count > 0:
        logger.warning(f"{failed_count} dosya işlenemedi")
    
    logger.info(f"{len(metadata_entries)} metadata entry oluşturuldu")
    
    # JSONL dosyasına yaz
    if output_file is None:
        output_file = audio_dir / "dataset_metadata.jsonl"
    else:
        output_file = Path(output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    logger.info(f"Metadata dosyası oluşturuldu: {output_file}")
    
    # Özet istatistikler
    if metadata_entries:
        total_duration = sum(e["duration"] for e in metadata_entries)
        total_size = sum(e["file_size_bytes"] for e in metadata_entries)
        avg_duration = total_duration / len(metadata_entries)
        
        logger.info("=" * 60)
        logger.info("Özet İstatistikler:")
        logger.info(f"  Toplam dosya sayısı: {len(metadata_entries)}")
        logger.info(f"  Toplam süre: {total_duration:.2f} saniye ({total_duration/60:.2f} dakika)")
        logger.info(f"  Ortalama süre: {avg_duration:.2f} saniye")
        logger.info(f"  Toplam dosya boyutu: {total_size / (1024**2):.2f} MB")
        logger.info("=" * 60)
    
    return metadata_entries


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WAV dosyalarından dataset metadata oluşturur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek kullanım:
  python dataset_metadata_generator.py /path/to/audio/dir
  python dataset_metadata_generator.py /path/to/audio/dir --output /path/to/output.jsonl
  python dataset_metadata_generator.py /path/to/audio/dir --absolute-paths
        """
    )
    
    parser.add_argument(
        "audio_dir",
        type=str,
        help="Audio dosyalarının bulunduğu dizin"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Çıktı JSONL dosyası path'i (default: audio_dir/dataset_metadata.jsonl)"
    )
    
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Dosya path'lerini absolute olarak kaydet (default: relative)"
    )
    
    args = parser.parse_args()
    
    audio_dir = Path(args.audio_dir).resolve()
    output_file = Path(args.output).resolve() if args.output else None
    
    try:
        generate_metadata(
            audio_dir=audio_dir,
            output_file=output_file,
            relative_path=not args.absolute_paths,
        )
        logger.info("✓ İşlem tamamlandı")
    except Exception as e:
        logger.error(f"✗ Hata: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
