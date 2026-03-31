from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from yt_uae.cache import SqliteCache
from yt_uae.youtube_api_async import YouTubeApiClient

# Initialize logger - ensure it's always available
try:
    logger = logging.getLogger(__name__)
except Exception:
    # Fallback if logging setup fails
    import sys
    class FallbackLogger:
        def info(self, msg): print(f"[INFO] {msg}", file=sys.stderr)
        def error(self, msg): print(f"[ERROR] {msg}", file=sys.stderr)
        def warning(self, msg): print(f"[WARNING] {msg}", file=sys.stderr)
        def debug(self, msg): pass
    logger = FallbackLogger()


@dataclass(frozen=True)
class DiscoveryResult:
    video_ids: List[str]
    channel_ids: List[str]


async def discover_videos_by_queries(
    *,
    yt: YouTubeApiClient,
    cache: SqliteCache,
    queries: List[str],
    region_code: str,
    relevance_language: str,
    max_results_per_query: int,
    max_pages_per_query: int,
    video_duration: str = "medium",
) -> DiscoveryResult:
    video_ids: Set[str] = set()
    channel_ids: Set[str] = set()

    for idx, q in enumerate(queries, 1):
        logger.info(f"Query {idx}/{len(queries)}: '{q[:50]}...'")
        page_token: Optional[str] = None
        pages_fetched = 0
        for page_num in range(max_pages_per_query):
            cache_key = f"search:{region_code}:{relevance_language}:{video_duration}:{max_results_per_query}:{q}:{page_token or ''}"
            cached = await cache.get_json(cache_key, max_age_seconds=7 * 24 * 3600)
            if cached is None:
                try:
                    resp = await yt.search_videos(
                        q=q,
                        region_code=region_code,
                        relevance_language=relevance_language,
                        max_results=max_results_per_query,
                        page_token=page_token,
                        video_duration=video_duration,
                    )
                    await cache.set_json(cache_key, resp)
                    pages_fetched += 1
                except Exception as e:
                    logger.error(f"Error searching for query '{q}': {e}")
                    break
            else:
                resp = cached
                logger.debug(f"Using cached results for query '{q}' page {page_num + 1}")

            items = resp.get("items", []) or []
            logger.debug(f"Query '{q}' page {page_num + 1}: found {len(items)} items")
            
            for it in items:
                vid = ((it.get("id") or {}).get("videoId")) if isinstance(it.get("id"), dict) else None
                sn = it.get("snippet", {}) or {}
                cid = sn.get("channelId")
                if vid:
                    video_ids.add(str(vid))
                if cid:
                    channel_ids.add(str(cid))

            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        
        logger.info(f"Query '{q}': {len(video_ids)} unique videos found so far")

    logger.info(f"Query discovery complete: {len(video_ids)} videos, {len(channel_ids)} channels")
    return DiscoveryResult(video_ids=sorted(video_ids), channel_ids=sorted(channel_ids))


async def discover_videos_from_channels_uploads(
    *,
    yt: YouTubeApiClient,
    cache: SqliteCache,
    channel_ids: List[str],
    per_channel_limit: int = 80,
) -> DiscoveryResult:
    """
    Channel-first: fetch channels -> uploads playlist -> recent videoIds.
    """
    all_video_ids: Set[str] = set()
    if not channel_ids:
        logger.info("No channel IDs provided for channel discovery")
        return DiscoveryResult(video_ids=[], channel_ids=[])

    logger.info(f"Discovering videos from {len(channel_ids)} channels...")
    
    # Channels list is batched 50
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i : i + 50]
        batch_num = (i // 50) + 1
        total_batches = (len(channel_ids) + 49) // 50
        logger.info(f"Processing channel batch {batch_num}/{total_batches} ({len(batch)} channels)")
        
        cache_key = f"channels:{','.join(batch)}"
        cached = await cache.get_json(cache_key, max_age_seconds=14 * 24 * 3600)
        if cached is None:
            try:
                resp = await yt.channels_list(channel_ids=batch)
                await cache.set_json(cache_key, resp)
            except Exception as e:
                logger.error(f"Error fetching channels batch: {e}")
                continue
        else:
            resp = cached
            logger.debug(f"Using cached channel data for batch {batch_num}")

        channels_with_uploads = 0
        for ch in resp.get("items", []) or []:
            uploads = (((ch.get("contentDetails") or {}).get("relatedPlaylists") or {}).get("uploads"))
            if not uploads:
                continue
            
            channels_with_uploads += 1

            # paginate playlist items until limit
            got = 0
            page_token: Optional[str] = None
            while got < per_channel_limit:
                max_results = min(50, per_channel_limit - got)
                pl_key = f"playlistItems:{uploads}:{max_results}:{page_token or ''}"
                pl_cached = await cache.get_json(pl_key, max_age_seconds=14 * 24 * 3600)
                if pl_cached is None:
                    try:
                        pl_resp = await yt.playlist_items_list(playlist_id=uploads, max_results=max_results, page_token=page_token)
                        await cache.set_json(pl_key, pl_resp)
                    except Exception as e:
                        logger.warning(f"Error fetching playlist items for channel: {e}")
                        break
                else:
                    pl_resp = pl_cached

                items = pl_resp.get("items", []) or []
                for it in items:
                    vid = ((it.get("contentDetails") or {}).get("videoId"))
                    if vid:
                        all_video_ids.add(str(vid))
                        got += 1
                        if got >= per_channel_limit:
                            break

                page_token = pl_resp.get("nextPageToken")
                if not page_token or not items:
                    break
        
        logger.debug(f"Batch {batch_num}: {channels_with_uploads} channels with uploads, {len(all_video_ids)} total videos so far")

    logger.info(f"Channel discovery complete: {len(all_video_ids)} videos from {len(channel_ids)} channels")
    return DiscoveryResult(video_ids=sorted(all_video_ids), channel_ids=sorted(set(channel_ids)))

