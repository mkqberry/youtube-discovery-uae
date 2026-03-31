from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging
import ssl

import aiohttp


YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

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


class YouTubeApiError(RuntimeError):
    pass


@dataclass
class ApiMetrics:
    requests: int = 0
    errors: int = 0
    retries: int = 0
    rate_limited: int = 0


class KeyRotator:
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("No API keys provided")
        self._keys = api_keys
        self._idx = 0
        self._lock = asyncio.Lock()

    async def next_key(self) -> str:
        async with self._lock:
            k = self._keys[self._idx % len(self._keys)]
            self._idx += 1
            return k


class ProxyRotator:
    def __init__(self, proxy_list: List[str]):
        self._proxies = [p.strip() for p in proxy_list if p.strip()]
        self._idx = 0
        self._lock = asyncio.Lock()

    async def next_proxy(self) -> Optional[str]:
        if not self._proxies:
            return None
        async with self._lock:
            p = self._proxies[self._idx % len(self._proxies)]
            self._idx += 1
            return p


class RateLimiter:
    """Token-bucket-ish limiter (simple and fast)."""

    def __init__(self, rps: float):
        self._rps = max(0.1, float(rps))
        self._min_interval = 1.0 / self._rps
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
            self._next_time = max(self._next_time + self._min_interval, time.monotonic())


class YouTubeApiClient:
    def __init__(
        self,
        *,
        api_keys: List[str],
        concurrency: int = 40,
        requests_per_second: float = 8.0,
        session: Optional[aiohttp.ClientSession] = None,
        user_agent: str = "yt-uae-pipeline/1.0",
        ssl_verify: bool = True,
        request_timeout_seconds: float = 60.0,
        proxy_list: Optional[List[str]] = None,
    ):
        self.metrics = ApiMetrics()
        self._rotator = KeyRotator(api_keys)
        self._proxy_rotator = ProxyRotator(proxy_list or [])
        self._limiter = RateLimiter(requests_per_second)
        self._sem = asyncio.Semaphore(max(1, int(concurrency)))
        self._concurrency = max(1, int(concurrency))
        self._session = session
        self._user_agent = user_agent
        self._ssl_verify = ssl_verify
        self._timeout_seconds = request_timeout_seconds

    async def __aenter__(self):
        if self._session is None:
            # For proxies, we need longer connect timeout - use 30s or 50% of total, whichever is smaller
            connect_timeout = min(30.0, self._timeout_seconds * 0.5)
            timeout = aiohttp.ClientTimeout(
                total=self._timeout_seconds,
                connect=connect_timeout,
                sock_connect=connect_timeout,  # Socket connection timeout
            )
            connector = None
            ssl_context = None
            if not self._ssl_verify:
                # Disable SSL verification for proxy environments
                # Create an SSL context that doesn't verify certificates
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                logger.debug("SSL verification disabled - using unverified SSL context")
            
            # Add limit_per_host to allow more connections per host
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,  # Use None for default SSL, or our custom context if ssl_verify=False
                limit_per_host=self._concurrency,  # Match concurrency limit
                ttl_dns_cache=300,  # Cache DNS for 5 minutes
                use_dns_cache=True,
                force_close=False,  # Keep connections alive for reuse
            )
            
            # Check for proxies
            has_proxies = len(self._proxy_rotator._proxies) > 0
            if has_proxies:
                logger.info(f"Using {len(self._proxy_rotator._proxies)} proxy(ies) from config")
            else:
                # Check for proxy in environment variables
                proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
                if proxy_url:
                    logger.info(f"Using proxy from environment: {proxy_url[:50]}...")
                else:
                    logger.debug("No proxy configured - using direct connection")
            
            # aiohttp automatically uses HTTP_PROXY/HTTPS_PROXY env vars if trust_env=True (default)
            # But we'll pass proxies explicitly per-request if we have a proxy list
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                trust_env=not has_proxies,  # Only trust env if we don't have explicit proxy list
            )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _request_json(
        self,
        path: str,
        params: Dict[str, Any],
        *,
        max_retries: int = 5,
    ) -> Dict[str, Any]:
        assert self._session is not None

        async with self._sem:
            for attempt in range(max_retries + 1):
                await self._limiter.wait()
                key = await self._rotator.next_key()
                proxy_url = await self._proxy_rotator.next_proxy()
                
                q = dict(params)
                q["key"] = key

                headers = {"User-Agent": self._user_agent}
                url = f"{YOUTUBE_API_BASE}/{path}"
                self.metrics.requests += 1
                try:
                    # Use proxy if available, otherwise let aiohttp use env vars or direct connection
                    async with self._session.get(url, params=q, headers=headers, proxy=proxy_url) as resp:
                        if resp.status in (403, 429):
                            self.metrics.rate_limited += 1
                            text = await resp.text()
                            error_msg = f"Rate/quota limited ({resp.status}): {text[:500]}"
                            # Check for specific quota errors
                            if "quota" in text.lower() or "quotaExceeded" in text:
                                error_msg = f"⚠️  QUOTA EXCEEDED: {text[:500]}"
                                # Don't retry on quota exceeded - fail immediately
                                logger.error(error_msg)
                                raise YouTubeApiError(error_msg)
                            # For rate limiting (429), retry with backoff
                            if attempt < max_retries:
                                self.metrics.retries += 1
                                logger.warning(f"Rate limited ({resp.status}) - retrying ({attempt + 1}/{max_retries + 1})...")
                                base = 0.5 * (2 ** attempt)
                                await asyncio.sleep(base + random.uniform(0.0, 0.25))
                                continue
                            raise YouTubeApiError(error_msg)
                        if resp.status >= 400:
                            text = await resp.text()
                            error_msg = f"HTTP {resp.status}: {text[:500]}"
                            if resp.status == 403 and ("quota" in text.lower() or "quotaExceeded" in text):
                                error_msg = f"⚠️  QUOTA EXCEEDED (403): {text[:500]}"
                            raise YouTubeApiError(error_msg)
                        return await resp.json()
                except asyncio.TimeoutError as e:
                    self.metrics.errors += 1
                    error_str = str(e)
                    # Distinguish between connection timeout and read timeout
                    if "Connection timeout" in error_str or "sock_connect" in error_str:
                        timeout_type = "connection"
                        timeout_msg = f"Connection timeout (could not establish TCP connection to API)"
                        likely_cause = "proxy/network issue or API unreachable"
                    else:
                        timeout_type = "read"
                        timeout_msg = f"Request timeout after {self._timeout_seconds}s"
                        likely_cause = "slow response or API quota exhaustion"
                    
                    # Connection timeouts are very unlikely to succeed on retry - only retry once
                    # Read timeouts might succeed, so retry twice
                    max_timeout_retries = 1 if timeout_type == "connection" else min(2, max_retries)
                    
                    if attempt < max_timeout_retries:
                        self.metrics.retries += 1
                        logger.warning(f"{timeout_msg} - retrying ({attempt + 1}/{max_timeout_retries + 1})...")
                        # Shorter backoff for timeouts
                        await asyncio.sleep(1.0 + random.uniform(0.0, 0.5))
                        continue
                    # If we've exhausted timeout retries
                    error_detail = f"{timeout_msg}. Likely cause: {likely_cause}."
                    raise YouTubeApiError(f"{error_detail} Original error: {e}") from e
                except (aiohttp.ClientError, Exception) as e:
                    self.metrics.errors += 1
                    error_type = type(e).__name__
                    if attempt < max_retries:
                        self.metrics.retries += 1
                        logger.warning(f"Network error ({error_type}): {e} - retrying ({attempt + 1}/{max_retries + 1})...")
                        base = 0.5 * (2 ** attempt)
                        await asyncio.sleep(base + random.uniform(0.0, 0.25))
                        continue
                    raise YouTubeApiError(f"Network error ({error_type}): {e}") from e

        raise YouTubeApiError("Unexpected request state")

    async def search_videos(
        self,
        *,
        q: str,
        region_code: str,
        relevance_language: str,
        max_results: int = 50,
        page_token: Optional[str] = None,
        video_duration: str = "medium",
        safe_search: str = "none",
    ) -> Dict[str, Any]:
        return await self._request_json(
            "search",
            {
                "part": "snippet",
                "type": "video",
                "q": q,
                "regionCode": region_code,
                "relevanceLanguage": relevance_language,
                "maxResults": max_results,
                "pageToken": page_token or "",
                "videoDuration": video_duration,
                "safeSearch": safe_search,
                "fields": "nextPageToken,items(id/videoId,snippet(channelId,title,description,publishedAt,channelTitle))",
            },
        )

    async def channels_list(
        self,
        *,
        channel_ids: List[str],
    ) -> Dict[str, Any]:
        return await self._request_json(
            "channels",
            {
                "part": "snippet,statistics,brandingSettings,contentDetails",
                "id": ",".join(channel_ids),
                "maxResults": 50,
                "fields": "items(id,snippet(title,description,country),statistics(subscriberCount,videoCount),brandingSettings(channel/keywords),contentDetails(relatedPlaylists/uploads))",
            },
        )

    async def videos_list(
        self,
        *,
        video_ids: List[str],
    ) -> Dict[str, Any]:
        return await self._request_json(
            "videos",
            {
                "part": "snippet,contentDetails,statistics,status",
                "id": ",".join(video_ids),
                "maxResults": 50,
                "fields": "items(id,snippet(title,description,channelId,channelTitle,publishedAt,liveBroadcastContent,defaultAudioLanguage),contentDetails(duration,caption),status(privacyStatus,embeddable,madeForKids),statistics(viewCount,likeCount,commentCount))",
            },
        )

    async def captions_list(self, *, video_id: str) -> Dict[str, Any]:
        # Note: captions.list requires OAuth for downloading, but listing works for public captions metadata.
        return await self._request_json(
            "captions",
            {
                "part": "snippet",
                "videoId": video_id,
                "fields": "items(id,snippet(language,name,trackKind,isCC,isLarge,isEasyReader,lastUpdated))",
            },
        )

    async def playlist_items_list(
        self,
        *,
        playlist_id: str,
        max_results: int = 50,
        page_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._request_json(
            "playlistItems",
            {
                "part": "contentDetails",
                "playlistId": playlist_id,
                "maxResults": max_results,
                "pageToken": page_token or "",
                "fields": "nextPageToken,items(contentDetails/videoId)",
            },
        )


def load_api_keys_from_env(env_name: str) -> List[str]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return []
    # support comma/space/newline separated
    parts = [p.strip() for p in raw.replace("\n", ",").replace(" ", ",").split(",")]
    return [p for p in parts if p]

