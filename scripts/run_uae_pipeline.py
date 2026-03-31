#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from yt_uae.cache import SqliteCache
from yt_uae.captions import caption_quality_hint, parse_captions_list_response
from yt_uae.config_loader import PipelineConfig, load_config
from yt_uae.discovery import discover_videos_by_queries, discover_videos_from_channels_uploads
from yt_uae.filters_uae import basic_video_filters
from yt_uae.scoring import build_scoring_keywords, score_video_metadata
from yt_uae.youtube_api_async import YouTubeApiClient, load_api_keys_from_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def _fetch_videos_batched(yt: YouTubeApiClient, cache: SqliteCache, video_ids: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    total_batches = (len(video_ids) + 49) // 50
    logger.info(f"Fetching {len(video_ids)} videos in {total_batches} batches...")
    
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        batch_num = (i // 50) + 1
        cache_key = f"videos:{','.join(batch)}"
        cached = await cache.get_json(cache_key, max_age_seconds=14 * 24 * 3600)
        if cached is None:
            try:
                resp = await yt.videos_list(video_ids=batch)
                await cache.set_json(cache_key, resp)
                if batch_num % 10 == 0 or batch_num == total_batches:
                    logger.info(f"Fetched batch {batch_num}/{total_batches} ({len(out)} videos so far)...")
            except Exception as e:
                logger.warning(f"Failed to fetch batch {batch_num}/{total_batches}: {e} - skipping this batch")
                continue
        else:
            resp = cached
            if batch_num % 10 == 0:
                logger.debug(f"Using cached batch {batch_num}/{total_batches}")
        out.extend(resp.get("items", []) or [])
    
    logger.info(f"Successfully fetched {len(out)} video metadata records")
    return out


async def _fetch_channels_batched(yt: YouTubeApiClient, cache: SqliteCache, channel_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    total_batches = (len(channel_ids) + 49) // 50
    logger.info(f"Fetching {len(channel_ids)} channels in {total_batches} batches...")
    
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i : i + 50]
        batch_num = (i // 50) + 1
        cache_key = f"channels:{','.join(batch)}"
        cached = await cache.get_json(cache_key, max_age_seconds=30 * 24 * 3600)
        if cached is None:
            try:
                resp = await yt.channels_list(channel_ids=batch)
                await cache.set_json(cache_key, resp)
                if batch_num % 5 == 0 or batch_num == total_batches:
                    logger.info(f"Fetched channel batch {batch_num}/{total_batches} ({len(out)} channels so far)...")
            except Exception as e:
                logger.warning(f"Failed to fetch channel batch {batch_num}/{total_batches}: {e} - skipping this batch")
                continue
        else:
            resp = cached
        for ch in resp.get("items", []) or []:
            if "id" in ch:
                out[str(ch["id"])] = ch
    
    logger.info(f"Successfully fetched {len(out)} channel metadata records")
    return out


async def _fetch_captions(yt: YouTubeApiClient, cache: SqliteCache, video_id: str) -> Dict[str, Any]:
    cache_key = f"captions:{video_id}"
    cached = await cache.get_json(cache_key, max_age_seconds=30 * 24 * 3600)
    if cached is None:
        try:
            resp = await yt.captions_list(video_id=video_id)
        except Exception as e:
            resp = {"items": [], "_error": str(e)}
        await cache.set_json(cache_key, resp)
        return resp
    return cached


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


async def run_pipeline(cfg: PipelineConfig) -> None:
    logger.info("=" * 60)
    logger.info("Starting UAE YouTube Discovery Pipeline")
    logger.info("=" * 60)
    
    # Load and validate API keys
    api_keys = load_api_keys_from_env(cfg.api_keys_env)
    if not api_keys:
        logger.error(f"No API keys found in env var {cfg.api_keys_env}")
        logger.error("Please set: export YOUTUBE_API_KEYS='key1,key2,...'")
        raise RuntimeError(f"No API keys found in env var {cfg.api_keys_env}. Provide comma-separated keys.")
    
    logger.info(f"Loaded {len(api_keys)} API key(s)")
    logger.info(f"Config: discovery_mode={cfg.discovery.mode}, ssl_verify={cfg.runtime.ssl_verify}, timeout={cfg.runtime.request_timeout_seconds}s")
    logger.info(f"Output: {cfg.runtime.output_path}")

    cache = SqliteCache(cfg.runtime.cache_db_path)
    await cache.open()
    logger.info(f"Cache DB: {cfg.runtime.cache_db_path}")

    scoring_kw = build_scoring_keywords(cfg.scoring)

    async with YouTubeApiClient(
        api_keys=api_keys,
        concurrency=cfg.runtime.concurrency,
        requests_per_second=cfg.runtime.requests_per_second,
        ssl_verify=cfg.runtime.ssl_verify,
        request_timeout_seconds=cfg.runtime.request_timeout_seconds,
        proxy_list=cfg.runtime.proxy_list,
    ) as yt:
        # --- Discovery ---
        logger.info("=" * 60)
        logger.info("Phase 1: Discovery")
        logger.info("=" * 60)
        
        discovered_video_ids: List[str] = []
        discovered_channel_ids: List[str] = []

        mode = cfg.discovery.mode
        if mode in ("video_first", "hybrid"):
            logger.info(f"Discovering videos by queries (mode: {mode})...")
            logger.info(f"Queries: {len(cfg.discovery.include_queries)} queries")
            try:
                d = await discover_videos_by_queries(
                    yt=yt,
                    cache=cache,
                    queries=cfg.discovery.include_queries,
                    region_code=cfg.discovery.region_code,
                    relevance_language=cfg.discovery.relevance_language,
                    max_results_per_query=cfg.discovery.max_results_per_query,
                    max_pages_per_query=cfg.discovery.max_pages_per_query,
                )
                discovered_video_ids.extend(d.video_ids)
                discovered_channel_ids.extend(d.channel_ids)
                logger.info(f"Query discovery: found {len(d.video_ids)} videos, {len(d.channel_ids)} channels")
            except Exception as e:
                logger.error(f"Query discovery failed: {e}", exc_info=True)
                raise

        seed_channels = list(dict.fromkeys(cfg.discovery.seed_channel_ids + discovered_channel_ids))
        logger.info(f"Total seed channels: {len(seed_channels)}")

        if mode in ("channel_first", "hybrid") and seed_channels:
            logger.info(f"Discovering videos from channels (mode: {mode})...")
            # Limit channel discovery to avoid timeouts with too many channels
            max_channels_to_process = 100  # Process max 100 channels to avoid timeouts
            channels_to_process = seed_channels[:max_channels_to_process]
            if len(seed_channels) > max_channels_to_process:
                logger.info(f"Limiting channel discovery to {max_channels_to_process} channels (out of {len(seed_channels)}) to avoid timeouts")
            
            try:
                d2 = await discover_videos_from_channels_uploads(
                    yt=yt,
                    cache=cache,
                    channel_ids=channels_to_process,
                    per_channel_limit=80,
                )
                discovered_video_ids = list(dict.fromkeys(discovered_video_ids + d2.video_ids))
                discovered_channel_ids = list(dict.fromkeys(discovered_channel_ids + d2.channel_ids))
                logger.info(f"Channel discovery: found {len(d2.video_ids)} additional videos")
            except Exception as e:
                logger.warning(f"Channel discovery failed (continuing with query results only): {e}")
                logger.info("Continuing with videos discovered from queries only...")
                # Don't raise - continue with query-discovered videos only

        logger.info(f"Total discovered: {len(discovered_video_ids)} videos, {len(discovered_channel_ids)} channels")
        
        # Show sample of discovered video IDs
        if discovered_video_ids:
            sample_size = min(10, len(discovered_video_ids))
            logger.info(f"Sample discovered video IDs: {discovered_video_ids[:sample_size]}")

        # Remove already-processed videos for resumability
        remaining_video_ids: List[str] = []
        for vid in discovered_video_ids:
            if not await cache.is_processed("video", vid):
                remaining_video_ids.append(vid)

        logger.info(f"Already processed: {len(discovered_video_ids) - len(remaining_video_ids)}")
        logger.info(f"Remaining to process: {len(remaining_video_ids)}")

        if not remaining_video_ids:
            logger.info("No new videos to process (all processed).")
            # Still create output file with summary
            _ensure_parent(cfg.runtime.output_path)
            out_path = Path(cfg.runtime.output_path)
            with out_path.open("a", encoding="utf-8") as f:
                summary = {
                    "status": "no_new_videos",
                    "message": "All discovered videos were already processed",
                    "total_discovered": len(discovered_video_ids),
                    "already_processed": len(discovered_video_ids),
                }
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            logger.info(f"Summary written to {out_path}")
            await cache.close()
            return

        # --- Fetch video + channel metadata ---
        logger.info("=" * 60)
        logger.info("Phase 2: Fetching metadata")
        logger.info("=" * 60)
        logger.info(f"Fetching metadata for {len(remaining_video_ids)} videos...")
        
        try:
            videos = await _fetch_videos_batched(yt, cache, remaining_video_ids)
            logger.info(f"Fetched {len(videos)} video metadata records")
            
            channel_ids = sorted({(v.get("snippet") or {}).get("channelId") for v in videos if (v.get("snippet") or {}).get("channelId")})
            logger.info(f"Fetching metadata for {len(channel_ids)} channels...")
            
            # Channel metadata is optional - if it fails, continue with empty channel data
            try:
                channels_by_id = await _fetch_channels_batched(yt, cache, [str(c) for c in channel_ids])
                logger.info(f"Fetched {len(channels_by_id)} channel metadata records")
            except Exception as e:
                logger.warning(f"Failed to fetch channel metadata (continuing without it): {e}")
                logger.info("Continuing with video processing - channel metadata will be empty (scoring may be less accurate)")
                channels_by_id = {}  # Empty dict - videos will still be processed
        except Exception as e:
            logger.error(f"Failed to fetch video metadata: {e}", exc_info=True)
            raise

        # --- Process / score (preliminary) ---
        logger.info("=" * 60)
        logger.info("Phase 3: Processing and scoring")
        logger.info("=" * 60)
        
        _ensure_parent(cfg.runtime.output_path)
        out_path = Path(cfg.runtime.output_path)

        # Open output file for real-time writing (create if doesn't exist)
        output_file = out_path.open("a", encoding="utf-8")
        logger.info(f"Output file opened: {out_path.absolute()}")
        
        kept = 0
        dropped = 0

        # Fetch captions concurrently only for likely-eligible videos
        logger.info("Pre-filtering videos and fetching captions...")
        caption_tasks: Dict[str, asyncio.Task] = {}
        pre_filtered = 0
        for v in videos:
            vid = str(v.get("id", ""))
            if not vid:
                continue
            keep_basic, _, _ = basic_video_filters(
                v,
                min_duration_seconds=cfg.filters.min_duration_seconds,
                max_duration_seconds=cfg.filters.max_duration_seconds,
                drop_if_live=cfg.filters.drop_if_live,
                drop_shorts=cfg.filters.drop_shorts,
            )
            if not keep_basic:
                pre_filtered += 1
                continue
            has_caption_flag = ((v.get("contentDetails") or {}).get("caption") == "true")
            if keep_basic and has_caption_flag:
                caption_tasks[vid] = asyncio.create_task(_fetch_captions(yt, cache, vid))
        
        logger.info(f"Pre-filtered out {pre_filtered} videos (duration/live/shorts)")
        logger.info(f"Fetching captions for {len(caption_tasks)} videos...")

        caption_results: Dict[str, Dict[str, Any]] = {}
        if caption_tasks:
            try:
                done = await asyncio.gather(*caption_tasks.values(), return_exceptions=True)
                for k, resp in zip(caption_tasks.keys(), done):
                    if isinstance(resp, Exception):
                        logger.warning(f"Caption fetch failed for {k}: {resp}")
                        caption_results[k] = {"items": [], "_error": str(resp)}
                    else:
                        caption_results[k] = resp
                logger.info(f"Fetched captions for {len([r for r in caption_results.values() if not r.get('_error')])} videos")
            except Exception as e:
                logger.error(f"Error fetching captions: {e}", exc_info=True)

        logger.info("Scoring and filtering videos...")
        processed_count = 0
        for v in videos:
            vid = str(v.get("id", ""))
            if not vid:
                continue

            keep_basic, basic_reasons, dur_s = basic_video_filters(
                v,
                min_duration_seconds=cfg.filters.min_duration_seconds,
                max_duration_seconds=cfg.filters.max_duration_seconds,
                drop_if_live=cfg.filters.drop_if_live,
                drop_shorts=cfg.filters.drop_shorts,
            )

            sn = v.get("snippet", {}) or {}
            title = str(sn.get("title", "") or "")
            desc = str(sn.get("description", "") or "")
            channel_id = str(sn.get("channelId", "") or "")

            ch = channels_by_id.get(channel_id, {}) if channel_id else {}
            ch_sn = ch.get("snippet", {}) or {}
            ch_title = str(ch_sn.get("title", "") or "")
            ch_desc = str(ch_sn.get("description", "") or "")

            captions_resp = caption_results.get(vid, {"items": []}) if keep_basic else {"items": []}
            cap_summary = parse_captions_list_response(captions_resp)
            cap_hint = caption_quality_hint(cap_summary, cfg.captions.prefer_languages, cfg.captions.prefer_manual)

            scores = score_video_metadata(
                title=title,
                description=desc,
                channel_title=ch_title,
                channel_description=ch_desc,
                cfg=scoring_kw,
                caption_quality_hint=cap_hint,
            )

            decision = "keep"
            reason_codes: List[str] = []
            reason_codes.extend(basic_reasons)
            reason_codes.extend(scores.reason_codes)

            if not keep_basic:
                decision = "drop"

            if cfg.captions.required and decision == "keep":
                if not cap_summary.has_arabic:
                    decision = "drop"
                    reason_codes.append("drop:no_arabic_captions")
                elif cfg.captions.prefer_manual and not cap_summary.has_manual_arabic:
                    reason_codes.append("warn:no_manual_arabic_captions")

            record: Dict[str, Any] = {
                "video_id": vid,
                "channel_id": channel_id,
                "title": title,
                "description": desc,
                "publish_date": sn.get("publishedAt"),
                "duration_seconds": dur_s,
                "duration_iso8601": (v.get("contentDetails") or {}).get("duration"),
                "scores": {
                    "single_speaker": scores.single_speaker_score,
                    "uae_relevance": scores.uae_relevance_score,
                    "caption_quality_hint": round(cap_hint, 4),
                    "overall": scores.overall_score,
                },
                "caption_availability": {
                    "has_any": cap_summary.has_any,
                    "has_arabic": cap_summary.has_arabic,
                    "has_manual_arabic": cap_summary.has_manual_arabic,
                    "languages": cap_summary.languages,
                    "preferred_track": asdict(cap_summary.preferred_track) if cap_summary.preferred_track else None,
                    "tracks": [asdict(t) for t in cap_summary.tracks[:20]],
                },
                "decision": decision,
                "reason_codes": sorted(set(reason_codes)),
                "matches": {
                    "uae_terms": scores.matched_uae_terms,
                    "single_speaker_pos": scores.matched_single_speaker_pos,
                    "single_speaker_neg": scores.matched_single_speaker_neg,
                    "banned": scores.matched_banned,
                },
            }
            
            # Write record immediately (real-time output)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()  # Ensure it's written to disk immediately
            
            await cache.mark_processed("video", record["video_id"], record["decision"])
            
            # Log kept videos in real-time
            if record["decision"] == "keep":
                kept += 1
                logger.info(f"✓ KEPT: {title[:60]}... (score: {record['scores']['overall']:.2f}, video_id: {vid})")
            else:
                dropped += 1
                if processed_count % 10 == 0:  # Log dropped videos less frequently
                    logger.debug(f"✗ DROPPED: {title[:60]}... (reasons: {', '.join(record['reason_codes'][:3])})")
            
            processed_count += 1
            if processed_count % 50 == 0:
                logger.info(f"Progress: {processed_count}/{len(videos)} videos processed (kept: {kept}, dropped: {dropped})...")

        output_file.close()
        total_processed = len(videos)
        logger.info(f"Completed processing {total_processed} videos (kept: {kept}, dropped: {dropped})")
        logger.info(f"Output file saved: {out_path.absolute()}")
        
        # Verify file was created and has content
        if out_path.exists():
            file_size = out_path.stat().st_size
            logger.info(f"Output file size: {file_size} bytes")
            if file_size == 0:
                logger.warning("⚠️  Output file is empty - no videos were written!")
        else:
            logger.error(f"⚠️  Output file was not created at {out_path.absolute()}")

        # --- Optional fast audio validation on top-ranked kept candidates ---
        if cfg.validation.enabled:
            logger.info("=" * 60)
            logger.info("Phase 4: Audio validation")
            logger.info("=" * 60)
            
            from yt_uae.audio_validation import WhisperLanguageDetector, validate_video_audio

            whisper_detector = None
            if cfg.validation.whisper_language_check:
                logger.info("Initializing Whisper language detector...")
                whisper_detector = WhisperLanguageDetector(model_name=cfg.validation.whisper_model)

            # Re-read kept videos from output file for validation
            # Note: This is a simplified approach - in production you might want to keep them in memory
            kept_videos = []
            try:
                with out_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line.strip())
                        if rec.get("decision") == "keep":
                            kept_videos.append(rec)
            except FileNotFoundError:
                kept_videos = []
            
            candidates = sorted(kept_videos, key=lambda r: float(r["scores"]["overall"]), reverse=True)
            candidates = candidates[: cfg.validation.max_videos_to_validate]
            logger.info(f"Validating audio for top {len(candidates)} candidates...")

            # Re-open output file in append mode to update validated records
            output_file = out_path.open("a", encoding="utf-8")
            
            # Limit concurrency for ffmpeg/yt-dlp
            sem = asyncio.Semaphore(6)

            async def _validate_one(rec: Dict[str, Any]) -> None:
                async with sem:
                    vr = await validate_video_audio(
                        video_id=rec["video_id"],
                        offsets_seconds=cfg.validation.offsets_seconds,
                        segment_seconds=cfg.validation.segment_seconds,
                        min_audio_purity_score=cfg.validation.min_audio_purity_score,
                        whisper_detector=whisper_detector,
                    )
                rec["audio_validation"] = {
                    "ok": vr.ok,
                    "audio_purity_score": vr.audio_purity_score,
                    "content_type": vr.content_type,
                    "language": vr.language,
                    "reasons": vr.reasons,
                }
                if not vr.ok:
                    rec["decision"] = "drop"
                    rec["reason_codes"] = sorted(set(rec["reason_codes"] + vr.reasons + ["drop:validation_failed"]))
                    # Write updated record
                    output_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    output_file.flush()

            try:
                await asyncio.gather(*[_validate_one(r) for r in candidates])
                output_file.close()
                logger.info(f"Completed audio validation for {len(candidates)} videos")
            except Exception as e:
                output_file.close()
                logger.error(f"Audio validation error: {e}", exc_info=True)

        # Output already written in real-time during processing
        logger.info("=" * 60)
        logger.info("Phase 5: Summary")
        logger.info("=" * 60)

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 60)
        total_processed = kept + dropped
        logger.info(f"Results: processed={total_processed} kept={kept} dropped={dropped}")
        logger.info(f"API metrics: requests={yt.metrics.requests} retries={yt.metrics.retries} rate_limited={yt.metrics.rate_limited} errors={yt.metrics.errors}")
        logger.info(f"Cache stats: hits={cache.stats.hits} misses={cache.stats.misses} writes={cache.stats.writes}")
        
        if yt.metrics.rate_limited > 0:
            logger.warning(f"⚠️  Rate limited {yt.metrics.rate_limited} times - API quota may be exhausted!")
        if yt.metrics.errors > 0:
            logger.warning(f"⚠️  {yt.metrics.errors} API errors occurred")

    await cache.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="UAE Arabic YouTube discovery pipeline (async + cached)")
    ap.add_argument("--config", required=True, help="Path to YAML config (see configs/uae_example.yaml)")
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (DEBUG level)")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    try:
        cfg = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
        asyncio.run(run_pipeline(cfg))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

