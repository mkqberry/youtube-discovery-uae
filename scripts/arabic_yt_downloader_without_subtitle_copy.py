"""
Advanced YouTube Audio Downloader (Enhanced Version)
- JSONL Format Support (Input/Output/Progress/Errors)
- Enhanced Logging with Detailed Progress Tracking
- Dynamic Chrome Profile Rotation for Cookie Errors
- Audio Only Download (WAV Format)
- Human-like Random Delays (3-5 min between downloads)
- Thread-safe & High Performance
- NO PROXY - Uses Chrome profile switching instead
"""

import os
import sys
import time
import logging
import subprocess
import glob
import re
import json
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Set, Optional, Tuple, Dict
from threading import Lock

# ============== CONFIGURATION ==============
@dataclass
class Config:
    # Dosya Yolları (JSONL Format)
    INPUT_FILE: Path = Path("video_urls.jsonl")
    OUTPUT_DIR: Path = Path("data/youtube_download/arabic_without_subtitle_downloads")
    PROGRESS_FILE: Path = Path("arabic_download_without_subtitle_progress.jsonl")
    FAILED_FILE: Path = Path("arabic_download_without_subtitle_failed.jsonl")
    LOG_FILE: Path = Path("arabic_download_without_subtitle_log.txt")
    STATS_FILE: Path = Path("arabic_download_without_subtitle_stats.json")
    
    # Chrome Ayarları
    CHROME_USER_DATA_DIR: str = "/root/.config/google-chrome" 
    
    # Performans
    MAX_WORKERS: int = 1
    
    # Cookie Bekleme Mantığı
    COOKIE_CHECK_INTERVAL: int = 30
    MAX_COOKIE_WAIT_TIME: int = 3600
    
    # Network Retry Ayarları
    MAX_RETRIES: int = 20
    RETRY_DELAY: int = 10
    
    # Verification
    MIN_FILE_SIZE_MB: float = 0.5  # Minimum geçerli dosya boyutu
    
    # Human-like Delay (saniye cinsinden)
    MIN_DELAY_BETWEEN_DOWNLOADS: int = 240  # 4 dakika
    MAX_DELAY_BETWEEN_DOWNLOADS: int = 300  # 5 dakika
    
    # Cookie Error Profil Değiştirme
    MAX_COOKIE_ERRORS_BEFORE_PROFILE_SWITCH: int = 1  # Bu kadar cookie hatası sonrası profil değiştir
    MAX_PROFILE_ATTEMPTS: int = 10  # Maksimum profil deneme sayısı
    
    # Profil Rotasyonu (Başarılı indirme sonrası)
    DOWNLOADS_PER_PROFILE: int = 10  # Bu kadar başarılı indirmeden sonra profil değiştir
    
    # yt-dlp timeout
    YTDLP_TIMEOUT_SEC: int = 300  # 5 dakika

config = Config()



# ============== JSONL HELPER ==============
class JSONLHandler:
    """Thread-safe JSONL okuma/yazma işlemleri"""
    def __init__(self):
        self.lock = Lock()
    
    def read_jsonl(self, filepath: Path) -> List[Dict]:
        """JSONL dosyasını okur"""
        if not filepath.exists():
            return []
        
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSONL parse hatası: {line[:50]}... -> {e}")
        return records
    
    def append_jsonl(self, filepath: Path, data: Dict):
        """Thread-safe JSONL ekleme"""
        with self.lock:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

jsonl_handler = JSONLHandler()

# ============== LOGGING ==============
def setup_logging():
    logger = logging.getLogger('Downloader')
    logger.setLevel(logging.DEBUG)
    
    # Detaylı format
    detailed_formatter = logging.Formatter(
        '%(asctime)s - [%(threadName)-10s] - %(levelname)-8s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File Handler (Detaylı)
    fh = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(detailed_formatter)
    
    # Console Handler (Özet)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(console_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()

# ============== STATISTICS TRACKER ==============
class StatsTracker:
    """İstatistik takip sistemi"""
    def __init__(self):
        self.lock = Lock()
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'cookie_errors': 0,
            'permanent_errors': 0,
            'retry_errors': 0,
            'profile_switches': 0,
            'start_time': datetime.now().isoformat(),
            'last_update': None
        }
        self.load_stats()
    
    def load_stats(self):
        """Önceki istatistikleri yükle"""
        if config.STATS_FILE.exists():
            try:
                with open(config.STATS_FILE, 'r') as f:
                    saved_stats = json.load(f)
                    self.stats.update(saved_stats)
                    logger.info("Önceki istatistikler yüklendi")
            except Exception as e:
                logger.warning(f"İstatistik yükleme hatası: {e}")
    
    def increment(self, key: str):
        with self.lock:
            self.stats[key] = self.stats.get(key, 0) + 1
            self.stats['last_update'] = datetime.now().isoformat()
            self.save_stats()
    
    def save_stats(self):
        """İstatistikleri kaydet"""
        try:
            with open(config.STATS_FILE, 'w') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"İstatistik kaydetme hatası: {e}")
    
    def get_summary(self) -> str:
        """Özet rapor"""
        with self.lock:
            return (
                f"\n{'='*60}\n"
                f"İNDİRME İSTATİSTİKLERİ\n"
                f"{'='*60}\n"
                f"Toplam İşlenen: {self.stats['total_processed']}\n"
                f"Başarılı: {self.stats['successful']}\n"
                f"Başarısız: {self.stats['failed']}\n"
                f"Atlandı: {self.stats['skipped']}\n"
                f"Cookie Hataları: {self.stats['cookie_errors']}\n"
                f"Kalıcı Hatalar: {self.stats['permanent_errors']}\n"
                f"Retry Hataları: {self.stats['retry_errors']}\n"
                f"Profil Değişimleri: {self.stats.get('profile_switches', 0)}\n"
                f"Başlangıç: {self.stats['start_time']}\n"
                f"Son Güncelleme: {self.stats['last_update']}\n"
                f"{'='*60}"
            )

stats_tracker = StatsTracker()

# ============== MANAGER CLASSES ==============
class ProfileManager:
    """Chrome profillerini bulur ve yönetir (Cookie hatası durumunda profil değiştirme destekli)"""
    def __init__(self, user_data_dir: str):
        self.lock = Lock()
        self.base_dir = Path(user_data_dir)
        self.profiles = self._discover_profiles()
        self.current_index = 0
        self.cookie_error_count = 0
        self.successful_download_count = 0  # Başarılı indirme sayacı
        self.current_profile = self.profiles[0] if self.profiles else "Default"
        
        logger.info(f"Chrome Profil Keşfi Tamamlandı: {len(self.profiles)} profil bulundu")
        logger.info(f"Başlangıç profili: {self.current_profile}")
        logger.debug(f"Profiller: {self.profiles}")

    def _discover_profiles(self) -> List[str]:
        """Default ve Profile X klasörlerini bulur"""
        if not self.base_dir.exists():
            logger.warning(f"Chrome data dizini bulunamadı: {self.base_dir}")
            logger.warning("Lütfen CHROME_USER_DATA_DIR ayarını kontrol edin")
            return ["Default"]

        profiles = []
        
        # 'Default' kontrolü
        if (self.base_dir / "Default").exists():
            profiles.append("Default")
            logger.debug("'Default' profil bulundu")
        
        # 'Profile *' kontrolü
        profile_dirs = list(self.base_dir.glob("Profile *"))
        for path in profile_dirs:
            if path.is_dir():
                profiles.append(path.name)
                logger.debug(f"Profil bulundu: {path.name}")
        
        if not profiles:
            logger.warning("Hiçbir profil bulunamadı, 'Default' kullanılıyor")
            return ["Default"]
            
        return sorted(profiles)
    
    def get_current_profile(self) -> str:
        """Aktif profili döndürür"""
        with self.lock:
            return self.current_profile
    
    def get_next_profile(self) -> str:
        """Sıradaki profili döndürür (Round-Robin)"""
        with self.lock:
            self.current_index = (self.current_index + 1) % len(self.profiles)
            self.current_profile = self.profiles[self.current_index]
            logger.debug(f"Profil seçildi: {self.current_profile}")
            return self.current_profile
    
    def increment_cookie_error(self) -> bool:
        """Cookie hatası sayacını artırır, gerekirse profil değiştirir
        
        Returns:
            True: Profil değiştirildi
            False: Profil değiştirilmedi
        """
        with self.lock:
            self.cookie_error_count += 1
            logger.warning(f"Cookie hata sayısı: {self.cookie_error_count}/{config.MAX_COOKIE_ERRORS_BEFORE_PROFILE_SWITCH}")
            
            if self.cookie_error_count >= config.MAX_COOKIE_ERRORS_BEFORE_PROFILE_SWITCH:
                return self._switch_to_next_profile()
            return False
    
    def _switch_to_next_profile(self) -> bool:
        """Sıradaki profile geçer"""
        old_profile = self.current_profile
        self.cookie_error_count = 0
        
        self.current_index = (self.current_index + 1) % len(self.profiles)
        self.current_profile = self.profiles[self.current_index]
        
        logger.info(f"{'*'*60}")
        logger.info(f"🔄 PROFİL DEĞİŞTİRİLİYOR...")
        logger.info(f"Eski: {old_profile}")
        logger.info(f"Yeni: {self.current_profile}")
        logger.info(f"Profil Index: {self.current_index + 1}/{len(self.profiles)}")
        logger.info(f"{'*'*60}")
        
        stats_tracker.increment('profile_switches')
        return True
    
    def reset_error_count(self):
        """Cookie hata sayacını sıfırlar (başarılı indirmede)"""
        with self.lock:
            self.cookie_error_count = 0
    
    def increment_success_count(self) -> bool:
        """Başarılı indirme sayacını artırır, gerekirse profil değiştirir
        
        Returns:
            True: Profil değiştirildi
            False: Profil değiştirilmedi
        """
        with self.lock:
            self.successful_download_count += 1
            logger.info(f"Profil indirme sayısı: {self.successful_download_count}/{config.DOWNLOADS_PER_PROFILE}")
            
            if self.successful_download_count >= config.DOWNLOADS_PER_PROFILE:
                self.successful_download_count = 0  # Sayacı sıfırla
                return self._rotate_profile()
            return False
    
    def _rotate_profile(self) -> bool:
        """Rutin profil rotasyonu (başarılı indirme limiti sonrası)"""
        old_profile = self.current_profile
        
        self.current_index = (self.current_index + 1) % len(self.profiles)
        self.current_profile = self.profiles[self.current_index]
        
        logger.info(f"{'*'*60}")
        logger.info(f"🔄 PROFİL ROTASYONU ({config.DOWNLOADS_PER_PROFILE} indirme tamamlandı)")
        logger.info(f"Eski: {old_profile}")
        logger.info(f"Yeni: {self.current_profile}")
        logger.info(f"Profil Index: {self.current_index + 1}/{len(self.profiles)}")
        logger.info(f"{'*'*60}")
        
        stats_tracker.increment('profile_switches')
        return True
    
    def get_profile_count(self) -> int:
        """Toplam profil sayısını döndürür"""
        return len(self.profiles)

class DownloadManager:
    def __init__(self):
        logger.info("DownloadManager başlatılıyor...")
        
        # Hata kategorileri
        self.PERMANENT_ERRORS = [
            'video is private', 'members-only',
            'video was removed', 'channel has been terminated', 
            'video not found', 'not available in your region',
            'this video is not available', 'video has been removed'
            # 404
        ]
        
        self.COOKIE_ERRORS = [
            'sign in', 'cookie expired', 'login required', 
            'verify your age', 'age-restricted', '403', 
            'account issue', 'cookie error', 'confirm your age',
            'cookies from browser', 'unable to extract', 'cookies are no longer valid'
        ]

        self.profile_manager = ProfileManager(config.CHROME_USER_DATA_DIR)
        
        # JSONL dosyalarından veri yükle
        self.completed_urls = self._load_completed_urls()
        self.permanent_fails = self._load_permanent_fails()
        
        logger.info(f"Daha önce tamamlanan: {len(self.completed_urls)} video")
        logger.info(f"Kalıcı hatalı: {len(self.permanent_fails)} video")
        
        # Output şablonu
        # Klasör ismi sadece video ID olacak şekilde ayarlandı
        self.output_tmpl = str(config.OUTPUT_DIR / '%(id)s' / '%(title)s [%(id)s].%(ext)s')

    def _load_completed_urls(self) -> Set[str]:
        """Başarılı indirmeleri yükle"""
        records = jsonl_handler.read_jsonl(config.PROGRESS_FILE)
        urls = {r['url'] for r in records if r.get('status') == 'success'}
        logger.debug(f"Yüklenen tamamlanmış URL sayısı: {len(urls)}")
        return urls

    def _load_permanent_fails(self) -> Set[str]:
        """Kalıcı hataları yükle"""
        records = jsonl_handler.read_jsonl(config.FAILED_FILE)
        fails = set()
        
        # Kalıcı hata türleri (tekrar denenmeyecek)
        permanent_error_types = ['PERMANENT']
        
        for r in records:
            error_type = r.get('error_type', '').upper()
            error = r.get('error', '').lower()
            
            # Error type kontrolü
            if error_type in permanent_error_types:
                fails.add(r['url'])
                continue
            
            # Error mesajı kontrolü (eski kayıtlar için)
            if any(p in error for p in self.PERMANENT_ERRORS):
                fails.add(r['url'])
        
        logger.debug(f"Yüklenen kalıcı hata sayısı: {len(fails)}")
        return fails

    def _log_success(self, url: str, video_id: str, url_data: Dict, file_size: int):
        """Başarılı indirme kaydı"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'url': url,
            'video_id': video_id,
            'title': url_data.get('title', 'Unknown'),
            'duration': url_data.get('duration'),
            'channel': url_data.get('channel'),
            'asr_status': url_data.get('asr_status'),
            'asr_reason': url_data.get('asr_reason'),
            'file_size_mb': round(file_size / (1024*1024), 2)
        }
        jsonl_handler.append_jsonl(config.PROGRESS_FILE, record)
        logger.info(f"✓ BAŞARILI: {url_data.get('title', 'Unknown')} ({file_size/(1024*1024):.2f} MB)")
        stats_tracker.increment('successful')

    def _log_fail(self, url: str, video_id: str, error: str, error_type: str, url_data: Dict = None):
        """Hata kaydı"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'url': url,
            'video_id': video_id,
            'error': error,
            'error_type': error_type
        }
        # Ek bilgileri ekle
        if url_data:
            record['title'] = url_data.get('title')
            record['duration'] = url_data.get('duration')
            record['channel'] = url_data.get('channel')
            record['asr_status'] = url_data.get('asr_status')
        
        jsonl_handler.append_jsonl(config.FAILED_FILE, record)
        logger.error(f"✗ HATA [{error_type}]: {url} -> {error}")
        
        if error_type == 'PERMANENT':
            stats_tracker.increment('permanent_errors')
        elif error_type == 'COOKIE':
            stats_tracker.increment('cookie_errors')
        else:
            stats_tracker.increment('retry_errors')
        
        stats_tracker.increment('failed')

    def _extract_video_id(self, url: str) -> Optional[str]:
        """URL'den YouTube Video ID'sini çeker"""
        patterns = [
            r'(?:v=|/v/|youtu\.be/|/embed/|/watch\?v=)([a-zA-Z0-9_-]{11})',
            r'shorts/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                logger.debug(f"Video ID çıkarıldı: {video_id} <- {url}")
                return video_id
        
        logger.warning(f"Video ID çıkarılamadı: {url}")
        return None

    def _get_file_info(self, video_id: str) -> Optional[Tuple[str, int]]:
        """
        İndirilmiş dosyanın bilgilerini döndürür (path, size)
        KONTROL: 1 video + 1 altyazı + isteğe bağlı diğer dosyalar (jpg, info.json vb)
        """
        if not video_id:
            return None

        # Çıktı klasörü artık doğrudan video_id adıyla
        target_folder = config.OUTPUT_DIR / video_id

        if not target_folder.exists():
            logger.debug(f"Video klasörü bulunamadı: {video_id}")
            return None

        target_folder = str(target_folder)
        # Desteklenen audio formatları (wav, webm, m4a, opus, mp3)
        valid_audio_extensions = ('.wav', '.webm', '.m4a', '.opus', '.mp3', '.ogg')
        
        if not os.path.exists(target_folder):
            return None
        
        audio_file = None
        audio_size = 0
        part_files = []
        all_files = []
        
        # Klasördeki tüm dosyaları kategorize et
        for filename in os.listdir(target_folder):
            file_path = os.path.join(target_folder, filename)
            all_files.append(filename)
            
            # .part kontrolü (yarım dosya)
            if filename.endswith('.part'):
                part_files.append(filename)
                logger.warning(f"Yarım dosya tespit edildi: {filename}")
                continue
            
            # Audio dosyası kontrolü (MP3)
            if filename.lower().endswith(valid_audio_extensions):
                if audio_file is None:  # İlk audio dosyasını al
                    file_size = os.path.getsize(file_path)
                    min_size = config.MIN_FILE_SIZE_MB * 1024 * 1024
                    
                    if file_size > min_size:
                        audio_file = file_path
                        audio_size = file_size
                        logger.debug(f"Audio dosyası bulundu: {filename} ({file_size/(1024*1024):.2f} MB)")
                    else:
                        logger.warning(f"Audio dosyası çok küçük: {filename} ({file_size} bytes)")
                else:
                    logger.warning(f"Birden fazla audio dosyası var: {filename}")
        
        # ====== DOĞRULAMA KURALLARI ======
        
        # Kural 1: .part dosyası varsa geçersiz
        if part_files:
            logger.warning(f"İndirme tamamlanmamış (part dosyaları): {part_files}")
            return None
        
        # Kural 2: Audio dosyası yoksa geçersiz
        if audio_file is None:
            logger.warning(f"Audio dosyası bulunamadı. Klasördeki dosyalar: {all_files}")
            return None
        
        # ====== TÜM KONTROLLER BAŞARILI ======
        logger.info(f"✓ Dosya doğrulaması BAŞARILI:")
        logger.info(f"  - Audio: {os.path.basename(audio_file)} ({audio_size/(1024*1024):.2f} MB)")
        logger.info(f"  - Ek dosyalar: {len(all_files) - 1} adet")
        
        return audio_file, audio_size

    def _verify_file_exists(self, video_id: str) -> bool:
        """Dosyanın bütünlüğünü doğrular"""
        result = self._get_file_info(video_id)
        is_valid = result is not None
        logger.debug(f"Dosya doğrulama [{video_id}]: {'✓ Geçerli' if is_valid else '✗ Geçersiz'}")
        return is_valid

    def build_command(self, url: str, profile: str) -> List[str]:
        """yt-dlp komutunu oluşturur (proxy run_ytdlp tarafından eklenir)
        
        Args:
            url: YouTube URL
            profile: Chrome profil adı
        """
        full_profile_path = os.path.join(config.CHROME_USER_DATA_DIR, profile)
        browser_arg = f"chrome:{full_profile_path}"
        logger.debug(f"Chrome profil yolu: {full_profile_path}")
        
        # Sadece audio indir (WAV format, en iyi kalite)
        cmd = [
            'yt-dlp',
            '--no-check-certificate',
            '-f', 'bestaudio',  # En iyi ses formatını seç
            '--extract-audio',  # Ses çıkar
            '--audio-format', 'wav',  # WAV formatına dönüştür
            '--audio-quality', '0',  # En yüksek kalite
            '--output', self.output_tmpl,
            '--restrict-filenames',
            '--cookies-from-browser', browser_arg,
            '--socket-timeout', '60',
            '--retries', '5',
            '--verbose',
            '--js-runtimes', 'node',
            url
        ]
        
        logger.debug(f"Komut oluşturuldu: {' '.join(cmd[:5])}... [Profil: {profile}]")
        return cmd
    
    def check_output_for_errors(self, output: str) -> Tuple[str, str]:
        """Çıktıyı analiz eder"""
        out_lower = output.lower()
        
        # Başarı göstergeleri
        success_indicators = ['100%', 'already been downloaded', 'merging formats', 'has already been downloaded']
        if any(s in out_lower for s in success_indicators):
            logger.debug("Başarı göstergesi tespit edildi")
            return "SUCCESS", "İndirme Başarılı"

        # Kalıcı hatalar
        for err in self.PERMANENT_ERRORS:
            if err in out_lower:
                logger.debug(f"Kalıcı hata tespit edildi: {err}")
                return "PERMANENT", f"Kalıcı Hata: {err}"

        # Cookie hataları
        for err in self.COOKIE_ERRORS:
            if err in out_lower:
                logger.debug(f"Cookie hatası tespit edildi: {err}")
                return "COOKIE", f"Cookie Hatası: {err}"

        # Geçici hata
        error_lines = [line for line in output.split('\n') if 'error' in line.lower()]
        error_snippet = '\n'.join(error_lines[-3:]) if error_lines else output[-300:]
        logger.debug(f"Bilinmeyen hata: {error_snippet[:100]}...")
        return "RETRY", f"Geçici Hata: {error_snippet}"

    def download_worker(self, url_data: Dict) -> bool:
        """Tek bir video için indirme süreci (Sadece Audio)"""
        url = url_data.get('url', '')
        custom_title = url_data.get('title', 'Unknown')
        video_id = url_data.get('video_id') or self._extract_video_id(url)
        
        # Aktif profili al
        current_profile = self.profile_manager.get_current_profile()
        
        logger.info(f"{'='*60}")
        logger.info(f"İŞLEM BAŞLIYOR")
        logger.info(f"URL: {url}")
        logger.info(f"Video ID: {video_id}")
        logger.info(f"Başlık: {custom_title}")
        logger.info(f"Süre: {url_data.get('duration', 'N/A')} saniye")
        logger.info(f"Kanal: {url_data.get('channel', 'N/A')}")
        logger.info(f"ASR Durumu: {url_data.get('asr_status', 'N/A')}")
        logger.info(f"Profil: {current_profile}")
        logger.info(f"{'='*60}")

        # ========== AUDIO İNDİR (WAV) ==========
        logger.info("🎵 Audio (WAV) indiriliyor...")
        
        audio_result = self._download_step(url, video_id, url_data)
        
        if not audio_result:
            logger.error("Audio indirme başarısız!")
            return False
        
        # Audio dosya kontrolü
        file_info = self._get_file_info(video_id)
        if file_info is None:
            logger.error("✗ Audio doğrulama BAŞARISIZ")
            self._log_fail(url, video_id, "Audio dosyası bulunamadı veya geçersiz", "AUDIO_INVALID", url_data)
            stats_tracker.increment('total_processed')
            return False
        
        file_path, file_size = file_info
        logger.info(f"✓ İndirme tamamlandı: {file_path}")
        self._log_success(url, video_id, url_data, file_size)
        stats_tracker.increment('total_processed')
        
        # Başarılı indirme sonrası profil rotasyonu kontrolü
        self.profile_manager.increment_success_count()
        
        return True
    
    def _download_step(self, url: str, video_id: str, url_data: Dict) -> bool:
        """
        Audio indirme adımını gerçekleştirir
        Cookie hatası durumunda Chrome profili değiştirerek tekrar dener
        """
        attempt = 0
        profile_attempts = 0  # Bu video için profil deneme sayısı

        while True:
            attempt += 1
            
            # Profil limitini kontrol et
            if profile_attempts > config.MAX_PROFILE_ATTEMPTS:
                logger.error(f"Tüm profiller denendi ({config.MAX_PROFILE_ATTEMPTS} deneme)")
                self._log_fail(url, video_id, "All profiles failed for this video", "PROFILES_EXHAUSTED", url_data)
                stats_tracker.increment('total_processed')
                return False
            
            # Aktif profili al
            current_profile = self.profile_manager.get_current_profile()
            logger.info(f"Deneme {attempt}/{config.MAX_RETRIES} [Profil: {current_profile}]...")
            
            cmd = self.build_command(url, current_profile)
            
            try:
                logger.debug(f"yt-dlp subprocess başlatılıyor...")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config.YTDLP_TIMEOUT_SEC
                )
                
                full_output = result.stdout + result.stderr
                logger.debug(f"yt-dlp çıktısı alındı: {len(full_output)} karakter")
                logger.debug(f"yt-dlp Çıktısı:\n{full_output}")
                status, msg = self.check_output_for_errors(full_output)
                logger.info(f"Durum analizi: {status} - {msg}")

                if status == "SUCCESS":
                    self.profile_manager.reset_error_count()  # Başarılı indirmede hata sayacını sıfırla
                    return True
                
                # Hata yönetimi
                if status == "PERMANENT":
                    logger.error(f"KALICI HATA - İndirme iptal ediliyor")
                    self._log_fail(url, video_id, msg, "PERMANENT", url_data)
                    stats_tracker.increment('total_processed')
                    return False

                elif status == "COOKIE":
                    profile_attempts += 1
                    
                    # Profil değiştirme mantığı
                    profile_switched = self.profile_manager.increment_cookie_error()
                    
                    if profile_switched:
                        # Profil değişti, hemen tekrar dene
                        logger.info("Yeni profil ile tekrar deneniyor...")
                        time.sleep(3)  # Kısa bekleme
                        attempt = 0
                        continue
                    
                    # Profil değişmediyse de tekrar dene (belki geçici hata)
                    logger.warning(f"{'*'*60}")
                    logger.warning(f"COOKIE HATASI TESPIT EDİLDİ")
                    logger.warning(f"Profil: {current_profile}")
                    logger.warning(f"Mesaj: {msg}")
                    logger.warning(f"Profil değiştiriliyor ve tekrar deneniyor...")
                    logger.warning(f"{'*'*60}")
                    
                    # Profili zorla değiştir
                    self.profile_manager.get_next_profile()
                    stats_tracker.increment('profile_switches')
                    time.sleep(5)
                    attempt = 0
                    continue 

                else:  # RETRY
                    if attempt > config.MAX_RETRIES:
                        logger.error(f"TÜM DENEMELER TÜKENDİ ({config.MAX_RETRIES})")
                        self._log_fail(url, video_id, msg, "RETRY", url_data)
                        stats_tracker.increment('total_processed')
                        return False
                    
                    wait_time = config.RETRY_DELAY * attempt
                    logger.warning(f"Geçici hata, {wait_time} saniye sonra tekrar denenecek...")
                    time.sleep(wait_time)
                    continue

            except subprocess.TimeoutExpired:
                logger.warning(f"SUBPROCESS ZAMAN AŞIMI ({config.YTDLP_TIMEOUT_SEC}s)")
                if attempt > config.MAX_RETRIES:
                    self._log_fail(url, video_id, "Process timeout", "TIMEOUT", url_data)
                    stats_tracker.increment('total_processed')
                    return False
                # Timeout'ta profil değiştir ve tekrar dene
                self.profile_manager.get_next_profile()
                profile_attempts += 1
                continue
            except Exception as e:
                logger.exception(f"KRİTİK HATA: {str(e)}")
                self._log_fail(url, video_id, f"Exception: {str(e)}", "EXCEPTION", url_data)
                stats_tracker.increment('total_processed')
                return False

    def start(self):
        """Ana indirme sürecini başlatır"""
        logger.info(f"\n{'#'*60}")
        logger.info("YOUTUBE DOWNLOADER BAŞLATILIYOR")
        logger.info(f"{'#'*60}\n")
        
        # Input kontrolü
        if not config.INPUT_FILE.exists():
            logger.error(f"URLs dosyası bulunamadı: {config.INPUT_FILE}")
            logger.error("Lütfen 'urls_to_download.jsonl' oluşturun")
            logger.error("Format: Her satırda {'url': 'https://...', 'title': '...'}")
            return

        # JSONL input'u oku
        all_url_data = jsonl_handler.read_jsonl(config.INPUT_FILE)
        logger.info(f"Toplam {len(all_url_data)} kayıt yüklendi")

        # Filtreleme
        queue = []
        skipped = 0
        
        for data in all_url_data:
            # video_urls.jsonl yapısına uyum: video_url -> url
            url = data.get('url') or data.get('video_url') or data.get('video')

            if not url:
                logger.warning(f"Geçersiz kayıt atlandı: {data}")
                continue

            # Kanala ait alanı normalize et (isteğe bağlı)
            if 'channel' not in data:
                data['channel'] = data.get('channel_url') or data.get('channel')

            # İleride tutarlı kullanım için 'url' anahtarını garanti et
            data['url'] = url
            
            if url in self.completed_urls:
                logger.debug(f"Atlandı (tamamlanmış): {url}")
                skipped += 1
                continue
            
            if url in self.permanent_fails:
                logger.debug(f"Atlandı (kalıcı hata): {url}")
                skipped += 1
                continue
            
            queue.append(data)

        logger.info(f"\n{'='*60}")
        logger.info("İNDİRME KUYRUK ÖZETİ")
        logger.info(f"{'='*60}")
        logger.info(f"Toplam URL: {len(all_url_data)}")
        logger.info(f"Daha önce inen: {len(self.completed_urls)}")
        logger.info(f"Kalıcı hatalı: {len(self.permanent_fails)}")
        logger.info(f"Atlandı: {skipped}")
        logger.info(f"İndirilecek: {len(queue)}")
        logger.info(f"{'='*60}\n")
        
        if not queue:
            logger.info("İndirilecek yeni video yok.")
            logger.info(stats_tracker.get_summary())
            return

        stats_tracker.stats['skipped'] = skipped
        stats_tracker.save_stats()

        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"İndirme klasörü hazır: {config.OUTPUT_DIR}")

        # Sıralı indirme başlat (multiprocess kaldırıldı)
        logger.info(f"\nSıralı indirme başlatılıyor...\n")
        logger.info(f"Her indirme arası bekleme: {config.MIN_DELAY_BETWEEN_DOWNLOADS//60}-{config.MAX_DELAY_BETWEEN_DOWNLOADS//60} dakika")
        logger.info(f"Kullanılacak profiller: {self.profile_manager.get_profile_count()} adet")
        logger.info(f"Aktif profil: {self.profile_manager.get_current_profile()}")
        
        completed_count = 0
        for data in queue:
            url = data.get('url', 'Unknown')
            completed_count += 1
            
            try:
                logger.info(f"\n[{completed_count}/{len(queue)}] İndirme başlıyor: {url}")
                result = self.download_worker(data)
                logger.info(f"[{completed_count}/{len(queue)}] İşlem tamamlandı: {url}")
                logger.info(f"Sonuç: {'Başarılı' if result else 'Başarısız'}")
            except Exception as exc:
                logger.exception(f"İndirme hatası [{url}]: {exc}")
            
            # Her 5 videoda bir özet
            if completed_count % 5 == 0:
                logger.info(stats_tracker.get_summary())
            
            # İnsancıl bekleme (son video değilse)
            if completed_count < len(queue):
                delay = random.uniform(
                    config.MIN_DELAY_BETWEEN_DOWNLOADS,
                    config.MAX_DELAY_BETWEEN_DOWNLOADS
                )
                logger.info(f"⏳ Sonraki indirme için {delay/60:.1f} dakika bekleniyor...")
                time.sleep(delay)

        # Final rapor
        logger.info(f"\n{'#'*60}")
        logger.info("TÜM İNDİRMELER TAMAMLANDI")
        logger.info(f"{'#'*60}")
        logger.info(stats_tracker.get_summary())

if __name__ == "__main__":
    try:
        logger.info("Program başlatılıyor...")
        manager = DownloadManager()
        manager.start()
        logger.info("Program normal şekilde sonlandı")
    except KeyboardInterrupt:
        logger.warning("\n\nKullanıcı tarafından durduruldu (Ctrl+C)")
        logger.info(stats_tracker.get_summary())
    except Exception as e:
        logger.exception(f"BEKLENMEYEN HATA: {str(e)}")
        sys.exit(1)
