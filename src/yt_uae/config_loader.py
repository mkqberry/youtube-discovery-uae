from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class DiscoveryConfig:
    mode: Literal["channel_first", "video_first", "hybrid"] = "hybrid"
    max_results_per_query: int = 50
    max_pages_per_query: int = 2
    region_code: str = "AE"
    relevance_language: str = "ar"
    include_queries: List[str] = field(default_factory=list)
    seed_channel_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FilterConfig:
    min_duration_seconds: int = 600
    max_duration_seconds: int = 2 * 3600
    drop_if_live: bool = True
    drop_shorts: bool = True


@dataclass(frozen=True)
class ScoringConfig:
    uae_keywords_ar: List[str] = field(default_factory=list)
    uae_keywords_latn: List[str] = field(default_factory=list)
    uae_channel_hints: List[str] = field(default_factory=list)

    single_speaker_positive: List[str] = field(default_factory=list)
    single_speaker_negative: List[str] = field(default_factory=list)

    banned_keywords: List[str] = field(default_factory=list)

    weight_single_speaker: float = 0.45
    weight_uae_relevance: float = 0.45
    weight_caption_quality: float = 0.10


@dataclass(frozen=True)
class CaptionsConfig:
    required: bool = True
    prefer_manual: bool = True
    prefer_languages: List[str] = field(default_factory=lambda: ["ar", "ar-AE", "ar-SA", "ar-EG", "ar-KW", "ar-QA", "ar-BH", "ar-OM"])


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = False
    max_videos_to_validate: int = 200
    segment_seconds: int = 45
    offsets_seconds: List[int] = field(default_factory=lambda: [30, 300, 900])
    min_audio_purity_score: float = 60.0
    whisper_language_check: bool = False
    whisper_model: str = "tiny"


@dataclass(frozen=True)
class RuntimeConfig:
    concurrency: int = 40
    requests_per_second: float = 8.0
    cache_db_path: str = "cache/yt_uae_cache.sqlite"
    output_path: str = "outputs/uae_candidates.jsonl"
    output_format: Literal["jsonl"] = "jsonl"
    log_level: str = "INFO"
    ssl_verify: bool = True
    request_timeout_seconds: float = 60.0  # Timeout for API requests (adjust for slow proxies if needed)
    proxy_list: List[str] = field(default_factory=list)  # List of proxy URLs (e.g., http://user:pass@host:port)


@dataclass(frozen=True)
class PipelineConfig:
    api_keys_env: str = "YOUTUBE_API_KEYS"
    discovery: DiscoveryConfig = DiscoveryConfig()
    filters: FilterConfig = FilterConfig()
    scoring: ScoringConfig = ScoringConfig()
    captions: CaptionsConfig = CaptionsConfig()
    validation: ValidationConfig = ValidationConfig()
    runtime: RuntimeConfig = RuntimeConfig()


def _require_yaml():
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")


def load_config(path: str | Path) -> PipelineConfig:
    _require_yaml()
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/object")
    return parse_config_dict(data)


def parse_config_dict(data: Dict[str, Any]) -> PipelineConfig:
    def get(section: str, default: dict) -> dict:
        v = data.get(section, default)
        return v if isinstance(v, dict) else default

    discovery = DiscoveryConfig(**get("discovery", {}))
    filters = FilterConfig(**get("filters", {}))
    scoring = ScoringConfig(**get("scoring", {}))
    captions = CaptionsConfig(**get("captions", {}))
    validation = ValidationConfig(**get("validation", {}))
    runtime = RuntimeConfig(**get("runtime", {}))

    api_keys_env = data.get("api_keys_env", "YOUTUBE_API_KEYS")
    return PipelineConfig(
        api_keys_env=api_keys_env,
        discovery=discovery,
        filters=filters,
        scoring=scoring,
        captions=captions,
        validation=validation,
        runtime=runtime,
    )

