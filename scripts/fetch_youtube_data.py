#!/usr/bin/env python3
"""Fetch YouTube video transcripts and metadata using Memories.ai v2 API.

This script fetches metadata and transcripts for a list of YouTube videos
using the Memories.ai v2 API client, saving results to a JSON file.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

# Add src to path for imports
sys.path.insert(0, str(_project_root / "src"))

from video_sourcing_agent.api.memories_v2_client import MemoriesV2Client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=viZ1l3PjzME",
    "https://www.youtube.com/watch?v=Ze9BqBgAWGg",
    "https://www.youtube.com/watch?v=tJZDzJdJfYU",
    "https://www.youtube.com/watch?v=TButjqrUDqY",
    "https://www.youtube.com/watch?v=vgE2gJ9tjr4",
    "https://www.youtube.com/watch?v=S9tj1zDhXCQ",
    "https://www.youtube.com/watch?v=AO8CjuGxH5U",
    "https://www.youtube.com/watch?v=UcN9AJIE2Po",
    "https://www.youtube.com/watch?v=oUMIYtN0gNg",
    "https://www.youtube.com/watch?v=lKotO0wP8oY",
    "https://www.youtube.com/watch?v=mxOmqjAvZ24",
    "https://www.youtube.com/watch?v=zj4s532wTxE",
    "https://www.youtube.com/watch?v=drDEZA51Dy8",
    "https://www.youtube.com/watch?v=kkUtR4ZLuMo",
]


def extract_video_id(url: str) -> str:
    """Extract video ID from a YouTube URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url.rsplit("/", 1)[-1]


def normalize_youtube_transcript(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw YouTube transcript response from Memories.ai v2.

    API format: {code, msg, data: {lang, content: [{lang, text, offset, duration}]}}
    Normalized to: {text: "full text", segments: [{start, end, text}]}
    """
    segments: list[dict[str, Any]] = []
    text_parts: list[str] = []

    data = raw.get("data", {})
    content = data.get("content", [])
    for seg in content:
        if not isinstance(seg, dict):
            continue
        text = seg.get("text", "").replace("\n", " ").strip()
        if not text:
            continue
        offset_ms = seg.get("offset", 0)
        duration_ms = seg.get("duration", 0)
        start = offset_ms / 1000.0
        end = (offset_ms + duration_ms) / 1000.0
        segments.append({"start": start, "end": end, "text": text})
        text_parts.append(text)

    return {"text": " ".join(text_parts), "segments": segments}


async def fetch_video(memories_v2: MemoriesV2Client, url: str) -> dict[str, Any]:
    """Fetch metadata and transcript for a single YouTube video."""
    video_id = extract_video_id(url)
    result: dict[str, Any] = {
        "url": url,
        "video_id": video_id,
        "metadata": None,
        "transcript": None,
        "error": None,
    }

    try:
        metadata_resp = await memories_v2.get_youtube_metadata(url)
        result["metadata"] = metadata_resp
    except Exception as e:
        logger.warning(f"  [{video_id}] Metadata failed: {e}")
        result["error"] = f"metadata: {e}"

    try:
        raw_transcript = await memories_v2.get_youtube_transcript(url)
        normalized = normalize_youtube_transcript(raw_transcript)
        result["transcript"] = normalized["text"]
    except Exception as e:
        logger.warning(f"  [{video_id}] Transcript failed: {e}")
        error_msg = f"transcript: {e}"
        result["error"] = f"{result['error']}; {error_msg}" if result["error"] else error_msg

    return result


def get_title_from_result(result: dict[str, Any]) -> str:
    """Extract video title from metadata result."""
    meta = result.get("metadata")
    if not meta:
        return "(no metadata)"
    items = meta.get("data", {}).get("items", [])
    if items and isinstance(items[0], dict):
        title = items[0].get("snippet", {}).get("title", "")
        if title:
            return title
    return "(unknown title)"


async def main() -> None:
    logger.info(f"Fetching data for {len(YOUTUBE_URLS)} YouTube videos")

    memories_v2 = MemoriesV2Client()
    results: list[dict[str, Any]] = []
    successful = 0
    failed = 0

    for i, url in enumerate(YOUTUBE_URLS, 1):
        video_id = extract_video_id(url)
        logger.info(f"[{i}/{len(YOUTUBE_URLS)}] Fetching {video_id}...")

        result = await fetch_video(memories_v2, url)
        results.append(result)

        has_meta = result["metadata"] is not None
        has_transcript = bool(result["transcript"])
        if has_meta or has_transcript:
            successful += 1
            transcript_len = len(result["transcript"]) if has_transcript else 0
            logger.info(f"  OK - metadata: {has_meta}, transcript: {transcript_len} chars")
        else:
            failed += 1
            logger.error(f"  FAILED - {result['error']}")

        if i < len(YOUTUBE_URLS):
            await asyncio.sleep(1.0)

    output_path = _project_root / "docs" / "youtube_video_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total_videos": len(YOUTUBE_URLS),
        "successful": successful,
        "failed": failed,
        "videos": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_path}")

    print("\n" + "=" * 100)
    print("YOUTUBE VIDEO FETCH RESULTS")
    print("=" * 100)
    print(f"{'#':<3} | {'Video ID':<13} | {'Title':<40} | {'Transcript':<12} | {'Status':<10}")
    print("-" * 100)

    for i, r in enumerate(results, 1):
        title = get_title_from_result(r)[:40]
        has_transcript = bool(r["transcript"])
        transcript_info = f"{len(r['transcript'])} chars" if has_transcript else "N/A"
        status = "OK" if not r["error"] else "PARTIAL" if (r["metadata"] or r["transcript"]) else "FAIL"
        print(f"{i:<3} | {r['video_id']:<13} | {title:<40} | {transcript_info:<12} | {status:<10}")

    print("-" * 100)
    print(f"Total: {successful}/{len(YOUTUBE_URLS)} successful, {failed} failed")
    print(f"Output: {output_path}")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
