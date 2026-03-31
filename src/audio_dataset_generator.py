#!/usr/bin/env python3
"""
ASR Dataset Segmenter - Whisper Edition
Özellikler:
- Altyazı-Ses Senkronizasyon Validasyonu (Whisper ile)
- Altyazısız Mod: Otomatik Bölüntüleme + Transkripsiyon (Whisper)
- Facebook Denoiser Entegrasyonu
- Türkçe VAD + Sessizlik Tespiti
"""

import argparse
import json
import os
import re
import logging
import numpy as np
import srt
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import webrtcvad
from pydub import AudioSegment, effects
from pydub.silence import detect_nonsilent
import requests

# Whisper artık remote API kullanıyor, local import gerekmiyor

# Denoiser imports
try:
    import torch
    from denoiser import pretrained
    DENOISER_AVAILABLE = True
except ImportError:
    DENOISER_AVAILABLE = False

# Script'in bulunduğu dizindeki logs klasörünü kullan
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "audio_dataset_generator.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================================
# DENOISER CLASS
# ============================================================================
class FacebookDenoiser:
    def __init__(self, device='cuda', model_path=None):
        if not DENOISER_AVAILABLE:
            raise RuntimeError("Denoiser kütüphanesi yüklü değil")
        
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        logger.info(f"Denoiser başlatılıyor... Cihaz: {self.device}")
        
        self.model = pretrained.dns64(pretrained=False)
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Model yükleniyor: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            else:
                self.model = checkpoint
        else:
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        self.model.to(self.device)
        self.model.eval()

    def process_segment(self, audio_seg: AudioSegment) -> AudioSegment:
        if audio_seg.frame_rate != 16000:
            audio_seg = audio_seg.set_frame_rate(16000)
        audio_seg = audio_seg.set_channels(1)
        
        samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / 32768.0
        wav = torch.from_numpy(samples).view(1, -1).to(self.device)

        with torch.no_grad():
            out = self.model(wav)[0] 

        out_np = out.cpu().numpy().flatten()
        out_np = np.clip(out_np, -1.0, 1.0)
        out_int16 = (out_np * 32767).astype(np.int16)
        
        return AudioSegment(
            out_int16.tobytes(), 
            frame_rate=16000,
            sample_width=2, 
            channels=1
        )


# ============================================================================
# UTILS
# ============================================================================
def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text - remove brackets, normalize whitespace"""
    if not text:
        return ""
    # Remove brackets and annotations
    text = re.sub(r'\[.*?\]|\(.*?\)|{.*?}|<.*?>', '', text)
    # Keep Arabic characters, numbers, and basic punctuation
    # Arabic Unicode range: \u0600-\u06FF
    text = re.sub(r'[^\u0600-\u06FF\w\s.,!?:;\-\']', ' ', text)
    return " ".join(text.split()).strip()


def check_vad_speech(audio_segment: AudioSegment, aggressiveness=3) -> float:
    """Ses segmentinde konuşma oranını hesaplar"""
    check_audio = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    vad = webrtcvad.Vad(aggressiveness)
    frame_ms = 30
    frame_bytes = int(16000 * (frame_ms / 1000.0) * 2)
    raw = check_audio.raw_data
    
    frames = [raw[i:i+frame_bytes] for i in range(0, len(raw), frame_bytes)]
    frames = [f for f in frames if len(f) == frame_bytes]
    
    if not frames:
        return 0.0
    
    speech_count = sum(1 for f in frames if vad.is_speech(f, 16000))
    return speech_count / len(frames)


def text_similarity(text1: str, text2: str) -> float:
    """İki metni karşılaştırır (basit kelime overlap)"""
    words1 = set(normalize_arabic_text(text1).split())
    words2 = set(normalize_arabic_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


# ============================================================================
# WHISPER VALIDATOR (Remote API)
# ============================================================================
class WhisperValidator:
    """Altyazı-Ses eşleşmesini Remote Whisper API ile doğrular"""
    
    def __init__(self, api_url: str, language: str = "ar"):
        self.api_url = api_url.rstrip('/')
        self.language = language
        logger.info(f"Whisper Validator - Remote API: {self.api_url}")
    
    def validate_segment(self, audio_seg: AudioSegment, expected_text: str, threshold=0.5) -> tuple[bool, str, float]:
        """
        Returns: (is_valid, transcribed_text, similarity_score)
        """
        # Geçici dosyaya kaydet
        temp_path = "/tmp/temp_validation.wav"
        audio_seg.export(temp_path, format="wav")
        
        # Remote Whisper API transkripsiyon
        try:
            with open(temp_path, "rb") as f:
                files = {"file": (os.path.basename(temp_path), f, "application/octet-stream")}
                data = {
                    "language": self.language,
                    "task": "transcribe",
                    "return_segments": "true",
                }
                resp = requests.post(
                    f"{self.api_url}/transcribe",
                    files=files,
                    data=data,
                    timeout=600
                )
            
            if resp.status_code != 200:
                raise RuntimeError(f"API error: HTTP {resp.status_code} - {resp.text}")
            
            result = resp.json()
            transcribed = normalize_arabic_text(result.get("text", ""))
            expected = normalize_arabic_text(expected_text)
            
            # Benzerlik hesapla
            similarity = text_similarity(transcribed, expected)
            
            os.remove(temp_path)
            
            return similarity >= threshold, transcribed, similarity
            
        except Exception as e:
            logger.error(f"Whisper API transkripsiyon hatası: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise


# ============================================================================
# SILENCE-BASED SPLITTER (Altyazısız Mod)
# ============================================================================
def split_on_silence_smart(audio: AudioSegment, min_silence_len=500, silence_thresh=-40, 
                           min_duration=22.0, max_duration=45.0) -> List[Dict]:
    """
    Sessizliklere göre sesi böler, akıllı birleştirme yapar
    """
    logger.info("Sessizlik tabanlı bölüntüleme başlatılıyor...")
    
    # Sessiz olmayan bölgeleri bul
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=10
    )
    
    segments = []
    current_segment = None
    
    for start_ms, end_ms in nonsilent_ranges:
        duration = (end_ms - start_ms) / 1000.0
        
        # Çok kısa parçaları atla
        if duration < 0.3:
            continue
        
        # Yeni segment başlat veya mevcut segmente ekle
        if current_segment is None:
            current_segment = {'start': start_ms, 'end': end_ms}
        else:
            # Mevcut segment ile birleştir mi?
            gap = start_ms - current_segment['end']
            combined_duration = (end_ms - current_segment['start']) / 1000.0
            
            if gap < 800 and combined_duration <= max_duration:
                # Birleştir
                current_segment['end'] = end_ms
            else:
                # Mevcut segmenti kaydet
                seg_dur = (current_segment['end'] - current_segment['start']) / 1000.0
                if min_duration <= seg_dur <= max_duration:
                    segments.append(current_segment)
                
                # Yeni segment başlat
                current_segment = {'start': start_ms, 'end': end_ms}
    
    # Son segmenti ekle
    if current_segment:
        seg_dur = (current_segment['end'] - current_segment['start']) / 1000.0
        if min_duration <= seg_dur <= max_duration:
            segments.append(current_segment)
    
    logger.info(f"{len(segments)} otomatik segment oluşturuldu")
    return segments


# ============================================================================
# MAIN SEGMENTER
# ============================================================================
class ProductionSegmenter:
    def __init__(self, args):
        self.args = args
        self.denoiser = None
        self.validator = None
        self.whisper_api_url = args.whisper_api_url
        self.whisper_language = args.whisper_language
        
        # Denoiser
        if args.denoise:
            try:
                self.denoiser = FacebookDenoiser(
                    device=args.device, 
                    model_path=args.model_path
                )
            except Exception as e:
                logger.error(f"Denoiser başlatılamadı: {e}")
                self.denoiser = None
        
        # Whisper Validator (sadece altyazılı modda ve validate açıksa)
        if args.validate and not args.no_subs:
            try:
                self.validator = WhisperValidator(
                    api_url=self.whisper_api_url,
                    language=self.whisper_language
                )
            except Exception as e:
                logger.error(f"Whisper validator başlatılamadı: {e}")
                self.validator = None
    
    def process(self):
        output_dir = Path(self.args.out)
        clips_dir = output_dir / "wavs"
        clips_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "metadata.jsonl"
        rejected_path = output_dir / "rejected.jsonl"
        
        logger.info(f"Ses yükleniyor: {self.args.audio}")
        full_audio = AudioSegment.from_file(self.args.audio).set_frame_rate(16000).set_channels(1)
        
        # MOD SEÇİMİ: Altyazılı vs Altyazısız
        # Eğer altyazı dosyası yoksa otomatik olarak no-subs moduna geç
        if self.args.no_subs or not self.args.subs:
            segments = self._process_no_subs(full_audio)
        else:
            segments = self._process_with_subs()
        
        if not segments:
            logger.error("İşlenecek segment bulunamadı!")
            return
        
        logger.info(f"{len(segments)} segment işlenecek...")
        
        success = 0
        rejected = 0
        
        with open(manifest_path, 'w', encoding='utf-8') as f_out, \
             open(rejected_path, 'w', encoding='utf-8') as f_reject:
            
            for idx, seg in tqdm(enumerate(segments), total=len(segments)):
                start = seg['start']
                end = seg['end']
                expected_text = seg.get('text', '')
                
                # Chunk çıkar
                chunk = full_audio[start:end]
                dur = len(chunk) / 1000.0
                
                # VAD Kontrolü
                speech_ratio = check_vad_speech(chunk)
                if speech_ratio < 0.3:
                    f_reject.write(json.dumps({
                        "idx": idx, "reason": "low_vad", 
                        "speech_ratio": speech_ratio
                    }) + "\n")
                    rejected += 1
                    continue
                
                # Denoise
                if self.denoiser:
                    try:
                        chunk = self.denoiser.process_segment(chunk)
                    except Exception as e:
                        logger.error(f"Denoise hatası (Seg {idx}): {e}")
                
                # Whisper Validation (sadece altyazılı modda)
                if self.validator and expected_text:
                    try:
                        is_valid, transcribed, similarity = self.validator.validate_segment(
                            chunk, expected_text, threshold=self.args.validation_threshold
                        )
                        
                        if not is_valid:
                            f_reject.write(json.dumps({
                                "idx": idx,
                                "reason": "validation_failed",
                                "expected": normalize_arabic_text(expected_text),
                                "transcribed": transcribed,
                                "similarity": similarity
                            }, ensure_ascii=False) + "\n")
                            rejected += 1
                            continue
                    except Exception as e:
                        logger.error(f"Validation hatası (Seg {idx}): {e}")
                
                # Normalize & Export
                chunk = effects.normalize(chunk)
                
                # Fade in/out ekle (çok kısa fade - 10ms)
                # Bu, segment başı/sonundaki ani kesiklikleri yumuşatır
                fade_duration = 10  # ms
                if len(chunk) > fade_duration * 2:
                    chunk = chunk.fade_in(fade_duration).fade_out(fade_duration)
                
                fname = f"segment_{idx:05d}.wav"
                chunk.export(clips_dir / fname, format="wav")
                
                meta = {
                    "file": fname,
                    "text": normalize_arabic_text(expected_text),
                    "duration": dur
                }
                f_out.write(json.dumps(meta, ensure_ascii=False) + "\n")
                success += 1
        
        logger.info(f"✓ Başarılı: {success} dosya")
        logger.info(f"✗ Reddedilen: {rejected} dosya")
        logger.info(f"Çıktı klasörü: {output_dir}")
    
    def _process_with_subs(self) -> List[Dict]:
        """Altyazı dosyasından segmentleri çıkarır"""
        if not self.args.subs or not os.path.exists(self.args.subs):
            logger.error("Altyazı dosyası bulunamadı!")
            return []
        
        with open(self.args.subs, encoding='utf-8') as f:
            subs = list(srt.parse(f.read()))
        
        segments = []
        for sub in subs:
            start_ms = int(sub.start.total_seconds() * 1000)
            end_ms = int(sub.end.total_seconds() * 1000)
            
            # Süre kontrolü (padding dahil)
            dur_with_pad = (end_ms - start_ms + 2 * self.args.pad_ms) / 1000.0
            if not (self.args.min_sec <= dur_with_pad <= self.args.max_sec):
                continue
            
            segments.append({
                'start': max(0, start_ms - self.args.pad_ms),
                'end': end_ms + self.args.pad_ms,
                'text': sub.content
            })
        
        return segments
    
    def _process_no_subs(self, audio: AudioSegment) -> List[Dict]:
        """Altyazısız mod: Sessizlik + Whisper transkripsiyon"""
        logger.info("Altyazısız mod aktif - otomatik bölüntüleme yapılıyor...")
        
        # Sessizlik tabanlı bölüntüle
        raw_segments = split_on_silence_smart(
            audio,
            min_silence_len=self.args.silence_len,
            silence_thresh=self.args.silence_thresh,
            min_duration=self.args.min_sec,
            max_duration=self.args.max_sec
        )
        
        # Padding ekle (kelimelerin kesilmemesi için)
        segments = []
        total_duration_ms = len(audio)
        
        for seg in raw_segments:
            start = max(0, seg['start'] - self.args.pad_ms)
            end = min(total_duration_ms, seg['end'] + self.args.pad_ms)
            
            # OTOMATIK GENIŞLETME: Segment sonunu kontrol et (eğer --aggressive-padding aktifse)
            # Eğer son 200ms'de hala konuşma varsa, biraz daha genişlet
            if self.args.aggressive_padding and end < total_duration_ms - 200:
                tail_check = audio[end:min(total_duration_ms, end + 300)]
                tail_speech_ratio = check_vad_speech(tail_check, aggressiveness=2)
                
                if tail_speech_ratio > 0.2:  # Hala konuşma var!
                    extra_padding = 200
                    end = min(total_duration_ms, end + extra_padding)
                    logger.debug(f"Segment sonu genişletildi (+{extra_padding}ms, speech_ratio={tail_speech_ratio:.2f})")
            
            padded_seg = {
                'start': start,
                'end': end
            }
            segments.append(padded_seg)
        
        logger.info(f"{len(segments)} segment oluşturuldu (padding={self.args.pad_ms}ms)")
        
        # Remote Whisper API ile transkript oluştur
        # Eğer --transcribe bayrağı verilmemişse bile, altyazısız moddaysa transkript oluştur
        if self.whisper_api_url:
            logger.info(f"Remote Whisper API ile transkripsiyon başlatılıyor... ({self.whisper_api_url})")
            
            for seg in tqdm(segments, desc="Transkripsiyon"):
                chunk = audio[seg['start']:seg['end']]
                temp_path = "/tmp/temp_transcribe.wav"
                chunk.export(temp_path, format="wav")
                
                try:
                    with open(temp_path, "rb") as f:
                        files = {"file": (os.path.basename(temp_path), f, "application/octet-stream")}
                        data = {
                            "language": self.whisper_language,
                            "task": "transcribe",
                            "return_segments": "true",
                        }
                        resp = requests.post(
                            f"{self.whisper_api_url}/transcribe",
                            files=files,
                            data=data,
                            timeout=600
                        )
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        print("********************************")
                        print("Response:" )
                        print(result)
                        print("********************************")
                        seg['text'] = result.get("text", "")
                    else:
                        logger.error(f"API error (HTTP {resp.status_code}): {resp.text}")
                        seg['text'] = ""
                except Exception as e:
                    logger.error(f"Transkripsiyon hatası: {e}")
                    seg['text'] = ""
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            # API URL yoksa boş text
            logger.warning("Whisper API URL belirtilmemiş - transkript oluşturulamıyor!")
            for seg in segments:
                seg['text'] = ""
        
        return segments


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Dataset Segmenter with Whisper")
    
    # Ana Parametreler
    parser.add_argument('--audio', required=True, help='Ses/Video dosyası')
    parser.add_argument('--subs', default=None, help='SRT altyazı dosyası (opsiyonel)')
    parser.add_argument('--out', default='dataset_output', help='Çıktı klasörü')
    
    # Mod Seçimi
    parser.add_argument('--no-subs', action='store_true', 
                        help='Altyazısız mod (otomatik bölüntüleme + transkripsiyon)')
    # Validation
    parser.add_argument('--validate', action='store_true',
                        help='Whisper ile altyazı-ses eşleşmesini doğrula')
    parser.add_argument('--validation-threshold', type=float, default=0.5,
                        help='Validation benzerlik eşiği (0-1)')
    parser.add_argument('--whisper-api-url', default='http://10.155.68.58:9000',
                        help='Remote Whisper API URL (default: http://10.155.68.58:9000)')
    parser.add_argument('--whisper-language', default='ar',
                        help='Whisper dil kodu (default: ar)')
    # Denoise
    parser.add_argument('--denoise', action='store_true', help='Facebook Denoiser aktif et')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--model-path', default=None, help='DNS64 model dosyası (.th)')
    
    # Süre Ayarları
    parser.add_argument('--min-sec', type=float, default=22.0, help='Min segment süresi')
    parser.add_argument('--max-sec', type=float, default=45.0, help='Max segment süresi')
    parser.add_argument('--pad-ms', type=int, default=100, 
                        help='Segment sınırlarına eklenecek padding (ms) - kelimelerin kesilmesini önler')
    parser.add_argument('--aggressive-padding', action='store_true',
                        help='Agresif padding modu - segment sonlarını otomatik kontrol edip genişletir')
    
    # Altyazısız Mod Parametreleri
    parser.add_argument('--silence-len', type=int, default=500, 
                        help='Min sessizlik uzunluğu (ms)')
    parser.add_argument('--silence-thresh', type=int, default=-40,
                        help='Sessizlik eşiği (dBFS)')
    parser.add_argument('--transcribe', action='store_true',
                        help='Altyazısız modda Whisper ile transkript oluştur')

    args = parser.parse_args()
    
    # Validasyon kontrolü
    if args.validate and not args.whisper_api_url:
        logger.error("--validate için --whisper-api-url gerekli")
        exit(1)
    
    if args.no_subs and args.subs:
        logger.warning("--no-subs aktif, --subs parametresi görmezden geliniyor")
    
    # Çalıştır
    ProductionSegmenter(args).process()
