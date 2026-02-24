#!/usr/bin/env python3
"""Demo script: Analyze poorly performing Instagram creators.

Generates optimization recommendations as a markdown report.

This script:
1. Parses test_m.md to identify 10 poorly performing Instagram creators
2. Scrapes engagement data using InstagramApifyCreatorTool
3. Calculates engagement rates and compares against benchmarks
4. Uses Gemini to generate optimization recommendations
5. Finds reference videos from excellent creators
6. Generates shooting scripts for reference videos
7. Outputs a structured markdown report to docs/demo_output.md
"""

import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root (ensures API keys are loaded regardless of working directory)
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

# Add src to path for imports
sys.path.insert(0, str(_project_root / "src"))

from video_sourcing_agent.agent.core import VideoSourcingAgent
from video_sourcing_agent.api.apify_client import ApifyClient
from video_sourcing_agent.api.gemini_client import GeminiClient
from video_sourcing_agent.api.memories_v2_client import MemoriesV2Client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _extract_mai_transcripts(mai_result: dict[str, Any]) -> tuple[str, str]:
    """Extract visual/audio transcript strings from MAI result payload."""
    data = mai_result.get("data", {})
    if not isinstance(data, dict):
        data = {}

    visual = data.get("videoTranscript") or mai_result.get("videoTranscript") or ""
    audio = data.get("audioTranscript") or mai_result.get("audioTranscript") or ""
    return str(visual), str(audio)


def _combine_mai_transcript(visual: str, audio: str) -> str:
    """Combine MAI audio and visual transcripts into one string."""
    return f"{audio}\n\n{visual}".strip()


async def upload_and_analyze_video(
    memories_v2_client: MemoriesV2Client,
    gemini: GeminiClient,
    video_url: str,
    prompt: str,
    max_wait_seconds: int = 120,
    mai_result: dict[str, Any] | None = None,
) -> str:
    """Analyze a social video with MAI transcript + Gemini reasoning.

    Args:
        memories_v2_client: Memories.ai v2 client instance.
        gemini: Gemini client for prompt-specific synthesis.
        video_url: Social media video URL
        prompt: Analysis question/prompt
        max_wait_seconds: Max time to wait for processing.
        mai_result: Optional pre-fetched MAI result to reuse.

    Returns:
        LLM analysis response text
    """
    from google.genai import types

    if mai_result is None:
        logger.info("    Requesting MAI transcript from Memories.ai v2...")
        mai_result = await memories_v2_client.get_mai_transcript(
            video_url=video_url,
            wait=True,
            max_wait=float(max_wait_seconds),
        )

    visual_context, audio_context = _extract_mai_transcripts(mai_result)

    context_prompt = f"""Use the transcription context below to answer the task.

TASK:
{prompt}

VIDEO TRANSCRIPT (visual descriptions):
{visual_context}

AUDIO TRANSCRIPT (speech):
{audio_context}
"""
    messages = [types.Content(role="user", parts=[types.Part(text=context_prompt)])]
    response = gemini.create_message(messages=messages, max_tokens=2048)
    answer = gemini.get_text_response(response)
    if not answer:
        raise RuntimeError("Gemini returned an empty response for MAI transcript analysis")
    return answer


async def get_mai_transcript_text(
    memories_v2_client: MemoriesV2Client,
    video_url: str,
    max_wait_seconds: int = 120,
    mai_result: dict[str, Any] | None = None,
) -> str:
    """Get combined transcript text from Memories.ai v2 MAI endpoint."""
    result = mai_result
    if result is None:
        result = await memories_v2_client.get_mai_transcript(
            video_url=video_url,
            wait=True,
            max_wait=float(max_wait_seconds),
        )

    visual, audio = _extract_mai_transcripts(result)
    return _combine_mai_transcript(visual, audio)


# Engagement rate benchmarks from demo.md
BENCHMARKS = {
    "like_rate": {
        "unhealthy": (0, 1),
        "normal": (1, 3),
        "good": (3, 6),
        "excellent": (6, float("inf")),
    },
    "comment_rate": {
        "unhealthy": (0, 0.05),
        "normal": (0.05, 0.2),
        "good": (0.2, 0.5),
        "excellent": (0.5, float("inf")),
    },
    "share_rate": {
        "unhealthy": (0, 0.2),
        "normal": (0.2, 0.6),
        "good": (0.6, 1),
        "excellent": (1, float("inf")),
    },
    "save_rate": {
        "unhealthy": (0, 0.3),
        "normal": (0.3, 1),
        "good": (1, 3),
        "excellent": (3, float("inf")),
    },
}


@dataclass
class CreatorPerformance:
    """Stores performance data for a creator."""

    creator_name: str
    video_url: str
    video_id: str
    roi: float
    ad_spend: float
    username: str | None = None
    followers: int | None = None
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    saves: int | None = None
    shares: int | None = None
    like_rate: float | None = None
    comment_rate: float | None = None
    save_rate: float | None = None
    share_rate: float | None = None
    # Niche-related fields for finding same-type reference creators
    niche: str | None = None
    target_audience: str | None = None
    product_category: str | None = None


@dataclass
class ReferenceVideo:
    """Stores data for a reference video from an excellent creator."""

    video_url: str
    messaging: str  # Key hook/message from the video (caption excerpt)
    creator_username: str
    followers: int | None
    views: int
    likes: int
    comments: int
    shares: int | None
    saves: int | None
    transcript: str | None = None
    vlm_analysis: str | None = None
    shooting_script: str | None = None


def get_rating(metric_name: str, value: float) -> str:
    """Get rating category for a metric value."""
    if metric_name not in BENCHMARKS:
        return "Unknown"

    benchmarks = BENCHMARKS[metric_name]
    for rating, (low, high) in benchmarks.items():
        if low <= value < high:
            return rating.capitalize()
    return "Unknown"


def parse_test_data(file_path: Path, limit: int = 10, buffer: int = 10) -> list[CreatorPerformance]:
    """Parse test_m.md and extract poorly performing Instagram creators.

    Criteria for poor performance:
    - ROI < 0.7 is definitely bad
    - OR (ROI < 1 AND Ad Spend > $500)

    We focus on Instagram links only (skip Facebook).

    Args:
        file_path: Path to the test_m.md file
        limit: Number of primary creators to return (default: 10)
        buffer: Extra creators for fallback pool (default: 10)

    Returns:
        List of CreatorPerformance objects (limit + buffer for fallback pool)
    """
    content = file_path.read_text()

    # Find the table rows - looking for markdown table format
    lines = content.split("\n")
    creators: list[CreatorPerformance] = []

    for line in lines:
        # Skip header and separator lines
        if not line.startswith("|") or "---" in line or "Row Labels" in line:
            continue

        # Parse table row:
        # | URL | Creator | Ad Spend | Clicks | Earnings | Impressions | ROI |
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 8:
            continue

        url = parts[1]
        creator_name = parts[2]

        # Skip non-Instagram URLs
        if "instagram.com" not in url.lower():
            continue

        # Skip non-reel URLs (static posts don't have view counts)
        if not is_reel_url(url):
            continue

        # Clean URL to remove tracking parameters
        url = clean_instagram_url(url)

        try:
            ad_spend = float(parts[3].replace(",", ""))
            roi = float(parts[7])
        except (ValueError, IndexError):
            continue

        # Check if this is a poorly performing creator
        # ROI < 0.7 is definitely bad OR ROI < 1 with significant spend
        # Per plan: prioritize creators with ad spend > $500 and poor returns
        is_poor = roi < 0.7 or (roi < 1.0 and ad_spend > 500)

        if is_poor:
            # Extract video ID from URL
            video_id = extract_video_id(url)
            creators.append(
                CreatorPerformance(
                    creator_name=creator_name,
                    video_url=url,
                    video_id=video_id,
                    roi=roi,
                    ad_spend=ad_spend,
                )
            )

    # Sort by: prioritize lower ROI with higher ad spend (worse outcome)
    # Sort primarily by ROI ascending, then by ad_spend descending
    creators.sort(key=lambda x: (x.roi, -x.ad_spend))

    # Filter to ensure we only get the truly poor performers
    # Take creators with ad_spend > 500 first, then fill with remaining
    high_spend_poor = [c for c in creators if c.ad_spend > 500]
    low_spend_poor = [c for c in creators if c.ad_spend <= 500]

    # Prefer high-spend poor performers, fill with low-spend if needed
    # Return limit + buffer creators for fallback pool
    total_needed = limit + buffer
    result = high_spend_poor[:total_needed]
    if len(result) < total_needed:
        result.extend(low_spend_poor[: total_needed - len(result)])

    return result[:total_needed]


def extract_video_id(url: str) -> str:
    """Extract video ID from Instagram URL."""
    # Match patterns like /reel/ABC123/ or /p/ABC123/
    patterns = [
        r"/reel/([A-Za-z0-9_-]+)",
        r"/p/([A-Za-z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return ""


def clean_instagram_url(url: str) -> str:
    """Remove tracking parameters from Instagram URL.

    Tracking parameters like ?igsh=... can interfere with scraping.
    """
    if "?" in url:
        url = url.split("?")[0]
    return url


def is_reel_url(url: str) -> bool:
    """Check if URL is an Instagram Reel (not a static post).

    Static posts (/p/) don't have view counts on Instagram,
    so we can only analyze reels (/reel/) properly.
    """
    return "/reel/" in url.lower()


def extract_username_from_video(video_data: dict[str, Any]) -> str | None:
    """Extract username from scraped video data."""
    # Try different field locations
    owner = video_data.get("owner") or video_data.get("ownerUsername") or {}
    if isinstance(owner, str):
        return owner
    if isinstance(owner, dict):
        return owner.get("username") or owner.get("ownerUsername")
    return video_data.get("ownerUsername")


async def scrape_video_metrics(
    client: ApifyClient, url: str
) -> dict[str, Any] | None:
    """Scrape metrics for a specific Instagram video URL (single attempt).

    Args:
        client: ApifyClient instance
        url: Instagram video URL to scrape

    Returns:
        Scraped video data dict or None if scraping fails
    """
    try:
        results = await client.scrape_instagram(urls=[url], max_results=1)
        if results:
            result = results[0]
            logger.debug(f"Scraped data keys: {list(result.keys())}")
            return result
    except Exception as e:
        logger.warning(f"Apify scrape failed for {url}: {e}")
    return None


async def scrape_video_metrics_memories_v2(
    client: MemoriesV2Client, url: str
) -> dict[str, Any] | None:
    """Scrape metrics using Memories.ai v2 API as fallback.

    Args:
        client: MemoriesV2Client instance
        url: Instagram video URL to scrape

    Returns:
        Scraped video data dict or None if scraping fails
    """
    try:
        result = await client.get_instagram_metadata(url)
        # v2 returns {"success": false, "data": null} on some errors (rate limit, etc.)
        # Check for actual success and valid data
        if result and result.get("success") is not False and result.get("data") is not None:
            # Extract the actual data payload if wrapped
            data = result.get("data", result)
            if isinstance(data, dict) and data:
                logger.info(f"  Memories.ai v2 succeeded for {url}")
                return data
            elif isinstance(data, list) and data:
                # Some endpoints return list of results
                logger.info(f"  Memories.ai v2 succeeded for {url}")
                return data[0]
        # Log the error if it's a known error response
        if result and result.get("failed"):
            logger.warning(
                f"Memories.ai v2 API error for {url}: {result.get('msg', 'Unknown error')}"
            )
    except Exception as e:
        logger.warning(f"Memories.ai v2 scrape failed for {url}: {e}")
    return None


def has_valid_metrics(video_data: dict[str, Any] | None) -> bool:
    """Check if scraped data has valid view count (not 0 or missing).

    Args:
        video_data: Scraped video data dict

    Returns:
        True if the data contains valid view metrics
    """
    if not video_data:
        return False

    # Check both Apify and Memories.ai v2 field names
    views = (
        video_data.get("videoViewCount")
        or video_data.get("video_view_count")
        or video_data.get("playCount")
        or video_data.get("viewCount")
        or video_data.get("views")  # Memories.ai v2 format
        or 0
    )
    return views > 0


async def scrape_creator_profile(
    client: ApifyClient, username: str, max_posts: int = 10
) -> dict[str, Any] | None:
    """Scrape creator profile and recent posts."""
    try:
        results = await client.scrape_instagram(
            username=username, max_results=max_posts
        )
        if results:
            # Get profile info from first result's owner data
            first_post = results[0]
            owner = first_post.get("owner") or {}

            followers = owner.get("followersCount") or owner.get(
                "edge_followed_by", {}
            ).get("count")

            return {
                "username": username,
                "followers": followers,
                "following": owner.get("followingCount"),
                "posts_count": owner.get("postsCount"),
                "bio": owner.get("biography"),
                "verified": owner.get("isVerified", False),
                "recent_posts": results,
            }
    except Exception as e:
        logger.warning(f"Failed to scrape creator @{username}: {e}")
    return None


def calculate_engagement_rates(video_data: dict[str, Any]) -> dict[str, float | None]:
    """Calculate engagement rates from video data (supports Apify and Memories.ai v2)."""
    # Try multiple field names for video views (Apify + Memories.ai v2)
    views = (
        video_data.get("videoViewCount")
        or video_data.get("video_view_count")
        or video_data.get("playCount")
        or video_data.get("viewCount")
        or video_data.get("views")  # Memories.ai v2
        or 0
    )

    # Likes (Apify + Memories.ai v2)
    edge_liked = video_data.get("edge_liked_by") or {}
    likes = (
        video_data.get("likesCount")
        or (edge_liked.get("count") if isinstance(edge_liked, dict) else 0)
        or video_data.get("likeCount")
        or video_data.get("likes")  # Memories.ai v2
        or 0
    )

    # Comments (Apify + Memories.ai v2)
    edge_comments = video_data.get("edge_media_to_comment") or {}
    comments = (
        video_data.get("commentsCount")
        or (edge_comments.get("count") if isinstance(edge_comments, dict) else 0)
        or video_data.get("commentCount")
        or video_data.get("comments")  # Memories.ai v2
        or 0
    )

    # Saves and shares (Apify + Memories.ai v2)
    saves = (
        video_data.get("savesCount")
        or video_data.get("saveCount")
        or video_data.get("saves")  # Memories.ai v2
        or 0
    )
    shares = (
        video_data.get("sharesCount")
        or video_data.get("shareCount")
        or video_data.get("shares")  # Memories.ai v2
        or 0
    )

    # Ensure all values are non-negative integers
    views = max(0, int(views)) if views else 0
    likes = max(0, int(likes)) if likes else 0
    comments = max(0, int(comments)) if comments else 0
    saves = max(0, int(saves)) if saves else 0
    shares = max(0, int(shares)) if shares else 0

    rates: dict[str, float | None] = {
        "views": views,
        "likes": likes,
        "comments": comments,
        "saves": saves,
        "shares": shares,
        "like_rate": None,
        "comment_rate": None,
        "save_rate": None,
        "share_rate": None,
    }

    if views and views > 0:
        rates["like_rate"] = (likes / views) * 100
        rates["comment_rate"] = (comments / views) * 100
        rates["save_rate"] = (saves / views) * 100 if saves else None
        rates["share_rate"] = (shares / views) * 100 if shares else None

    return rates


async def generate_recommendations(
    gemini: GeminiClient,
    creator: CreatorPerformance,
    problems: list[str],
) -> str:
    """Use Gemini to generate personalized recommendations."""
    # Format metrics safely
    if creator.like_rate is not None:
        like_rating = get_rating("like_rate", creator.like_rate)
        like_rate_str = f"{creator.like_rate:.2f}% ({like_rating})"
    else:
        like_rate_str = "Unknown"

    if creator.comment_rate is not None:
        comment_rating = get_rating("comment_rate", creator.comment_rate)
        comment_rate_str = f"{creator.comment_rate:.3f}% ({comment_rating})"
    else:
        comment_rate_str = "Unknown"

    if creator.save_rate is not None:
        save_rating = get_rating("save_rate", creator.save_rate)
        save_rate_str = f"{creator.save_rate:.2f}% ({save_rating})"
    else:
        save_rate_str = "Unknown"

    prompt = f"""You are an expert social media strategist. Based on the following
data about an Instagram creator's poorly performing video, provide specific,
actionable optimization recommendations.

Creator: {creator.creator_name}
Video URL: {creator.video_url}
ROI: {creator.roi:.2f}
Ad Spend: ${creator.ad_spend:.2f}

Performance Metrics:
- Views: {creator.views or 'Unknown'}
- Like Rate: {like_rate_str}
- Comment Rate: {comment_rate_str}
- Save Rate: {save_rate_str}

Problems Identified:
{chr(10).join(f'- {p}' for p in problems)}

Provide 3-5 specific, actionable recommendations to improve this creator's
video performance. Focus on:
1. Hook improvement (first 1-3 seconds)
2. Content structure
3. Engagement triggers
4. Call-to-action optimization

Keep recommendations concise and practical."""

    from google.genai import types

    messages = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    response = gemini.create_message(messages=messages, max_tokens=2048)
    return gemini.get_text_response(response) or "Unable to generate recommendations."


async def detect_creator_niche(
    memories_v2_client: MemoriesV2Client,
    gemini: GeminiClient,
    video_url: str,
) -> dict[str, str]:
    """Detect niche/category from video using Memories.ai v2 + Gemini.

    Args:
        memories_v2_client: Memories.ai v2 client for MAI transcript extraction.
        gemini: Gemini client for niche extraction.
        video_url: URL of the video to analyze.

    Returns:
        Dict with niche, audience, and product_category.
    """
    prompt = """Analyze this video and identify:
1. Primary content niche (e.g., fitness, beauty, home decor, tech gadgets, cooking, parenting, lifestyle, travel, finance, fashion, skincare, home organization, kitchen gadgets, pet care)
2. Target audience (e.g., "young mothers", "fitness enthusiasts", "home DIY-ers", "budget shoppers", "beauty beginners")
3. Product category if applicable (e.g., "kitchen gadgets", "skincare", "home organization", "fitness equipment", "cleaning products")

    Return ONLY valid JSON with no extra text: {"niche": "...", "audience": "...", "product_category": "..."}"""

    try:
        response = await upload_and_analyze_video(memories_v2_client, gemini, video_url, prompt)
        # Try to extract JSON from response
        if "{" in response and "}" in response:
            json_start = response.index("{")
            json_end = response.rindex("}") + 1
            json_str = response[json_start:json_end]
            return json.loads(json_str)
    except Exception as e:
        logger.warning(f"Failed to detect niche for {video_url}: {e}")

    # Default fallback
    return {
        "niche": "lifestyle",
        "audience": "general consumers",
        "product_category": "lifestyle products",
    }


def extract_creator_info(metadata: dict[str, Any]) -> tuple[str, int | None]:
    """Extract creator username and followers from metadata.

    Handles both flat Apify format and nested Instagram owner structure.

    Args:
        metadata: Video metadata dict from Apify or Memories.ai v2.

    Returns:
        Tuple of (username, followers).
    """
    # Handle nested Instagram owner structure
    owner = metadata.get("owner", {})
    if isinstance(owner, dict):
        owner_username = owner.get("username")
        owner_followers = (
            owner.get("followersCount")
            or owner.get("edge_followed_by", {}).get("count")
        )
    else:
        owner_username = None
        owner_followers = None

    # Try flat Apify/Memories.ai v2 format
    creator_data = metadata.get("creator", {})
    if isinstance(creator_data, str):
        creator_username = creator_data
        creator_followers = None
    elif isinstance(creator_data, dict):
        creator_username = creator_data.get("username")
        creator_followers = creator_data.get("followers") or creator_data.get("followersCount")
    else:
        creator_username = None
        creator_followers = None

    # Combine sources with priority: owner > ownerUsername > creator
    username = (
        owner_username
        or metadata.get("ownerUsername")
        or creator_username
        or "Unknown"
    )
    followers = owner_followers or creator_followers

    return username, followers


async def find_reference_videos(
    agent: VideoSourcingAgent,
    memories_v2_client: MemoriesV2Client,
    apify_client: ApifyClient,
    niche: str,
    product_category: str | None,
) -> list[ReferenceVideo]:
    """Find 3 excellent same-niche creators with high engagement.

    Args:
        agent: VideoSourcingAgent for searching videos.
        memories_v2_client: Memories.ai v2 client for metadata fallback.
        apify_client: Apify client for fallback scraping.
        niche: Content niche to search for.
        product_category: Specific product category if applicable.

    Returns:
        List of ReferenceVideo objects with full metrics.
    """
    # Build explicit Instagram-only query
    product_part = f"about {product_category}" if product_category else ""
    query = f"""Find 3 top-performing Instagram Reels in the {niche} niche {product_part}
with excellent engagement rates (like rate >6%, comment rate >0.5%).
Focus on viral videos with clear hooks and high saves.

IMPORTANT: Only return Instagram Reels (instagram.com/reel/ URLs).
Do NOT return YouTube, TikTok, or other platforms.
Use the instagram_search tool to find Instagram content."""

    try:
        response = await agent.query(query)

        # Safety filter: only keep Instagram URLs
        instagram_refs = [
            ref for ref in response.video_references
            if "instagram.com" in ref.url.lower()
        ]

        reference_videos: list[ReferenceVideo] = []

        for ref in instagram_refs[:3]:
            try:
                # Try Apify first (no rate limit issues)
                # Memories.ai v2 has tighter rate limits, so use it as fallback only
                metadata = await scrape_video_metrics(apify_client, ref.url)

                # If Apify returned empty/invalid data, try Memories.ai v2 as fallback
                if not has_valid_metrics(metadata):
                    logger.info(f"Apify returned empty data for {ref.url}, trying Memories.ai v2...")
                    metadata = await scrape_video_metrics_memories_v2(memories_v2_client, ref.url)

                # Skip if neither source returned valid metrics
                if not has_valid_metrics(metadata):
                    logger.warning(f"Skipping reference video with invalid metrics: {ref.url}")
                    continue

                # metadata is guaranteed to be non-None here (has_valid_metrics checks it)
                assert metadata is not None

                # Use calculate_engagement_rates() to normalize field names
                rates = calculate_engagement_rates(metadata)
                views = int(rates.get("views") or 0)
                likes = int(rates.get("likes") or 0)
                comments = int(rates.get("comments") or 0)
                shares = rates.get("shares")
                saves = rates.get("saves")

                # Extract creator info using helper (handles nested owner structure)
                creator_username, creator_followers = extract_creator_info(metadata)

                # Extract caption/messaging - try multiple field names
                caption = (
                    metadata.get("caption")
                    or metadata.get("description")
                    or metadata.get("text")
                    or ""
                )
                messaging = caption[:150] + "..." if len(caption) > 150 else caption

                reference_videos.append(
                    ReferenceVideo(
                        video_url=ref.url,
                        messaging=messaging,
                        creator_username=creator_username,
                        followers=creator_followers,
                        views=views,
                        likes=likes,
                        comments=comments,
                        shares=int(shares) if shares else None,
                        saves=int(saves) if saves else None,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to get metadata for reference video {ref.url}: {e}")
                # Still add with basic info from agent response if available
                if ref.views and ref.views > 0:
                    reference_videos.append(
                        ReferenceVideo(
                            video_url=ref.url,
                            messaging=ref.title or "",
                            creator_username=ref.creator or "Unknown",
                            followers=None,
                            views=ref.views or 0,
                            likes=ref.likes or 0,
                            comments=ref.comments or 0,
                            shares=None,
                            saves=None,
                        )
                    )

        return reference_videos
    except Exception as e:
        logger.warning(f"Failed to find reference videos: {e}")
        return []


async def analyze_reference_video(
    memories_v2_client: MemoriesV2Client,
    gemini: GeminiClient,
    video_url: str,
) -> tuple[str, str]:
    """Analyze video content using Memories.ai v2 MAI transcript + Gemini.

    Args:
        memories_v2_client: Memories.ai v2 client for transcript + visual context.
        gemini: Gemini client for prompt-specific analysis.
        video_url: URL of the video to analyze.

    Returns:
        Tuple of (vlm_analysis, transcript).
    """
    transcript = ""
    vlm_analysis = ""
    mai_result: dict[str, Any] | None = None

    # Get transcript via Memories.ai v2 MAI
    try:
        mai_result = await memories_v2_client.get_mai_transcript(
            video_url=video_url,
            wait=True,
        )
        transcript = await get_mai_transcript_text(
            memories_v2_client,
            video_url,
            mai_result=mai_result,
        )
    except Exception as e:
        logger.debug(f"Failed to get transcript for {video_url}: {e}")

    # Prompt-specific analysis via Memories.ai v2 MAI + Gemini
    prompt = """Analyze this video's success factors in detail:

1. Hook (first 3 seconds) - What grabs attention? Describe the specific visual or audio hook.
2. Content structure - How is the video organized? (problem-solution, tutorial, story, etc.)
3. Pacing - Fast, medium, or slow? How does it maintain viewer interest?
4. Call-to-action - What does it ask viewers to do? How explicit is it?
5. Visual style - Production quality, lighting, framing, text overlays, transitions.
6. Key selling points - What are the main value propositions shown?
7. Emotional triggers - What emotions does it evoke? (curiosity, urgency, satisfaction, etc.)

    Be specific and actionable. Describe what makes this video perform well."""

    try:
        vlm_analysis = await upload_and_analyze_video(
            memories_v2_client,
            gemini,
            video_url,
            prompt,
            mai_result=mai_result,
        )
    except Exception as e:
        logger.warning(f"Failed Memories.ai v2 analysis for {video_url}: {e}")
        vlm_analysis = "Analysis unavailable"

    return vlm_analysis, transcript


async def generate_shooting_script(
    gemini: GeminiClient,
    reference_video: ReferenceVideo,
    poor_performer_issues: list[str],
) -> str:
    """Generate shooting script from analyzed reference video.

    Args:
        gemini: GeminiClient for text generation.
        reference_video: Reference video with VLM analysis and transcript.
        poor_performer_issues: List of problems from the poor performer to address.

    Returns:
        Generated 9-shot shooting script.
    """
    from google.genai import types

    # Build context from actual video analysis
    if reference_video.transcript:
        transcript_section = f"\nTRANSCRIPT:\n{reference_video.transcript}"
    else:
        transcript_section = ""

    if reference_video.vlm_analysis:
        vlm_section = (
            f"\nVLM ANALYSIS OF WHAT MAKES THIS VIDEO SUCCESSFUL:\n"
            f"{reference_video.vlm_analysis}"
        )
    else:
        vlm_section = ""

    prompt = f"""Based on this high-performing reference video analysis, create a shooting script.

REFERENCE VIDEO URL: {reference_video.video_url}
CREATOR: @{reference_video.creator_username}
PERFORMANCE: {reference_video.views:,} views, {reference_video.likes:,} likes, {reference_video.comments:,} comments
{vlm_section}
{transcript_section}

ISSUES TO ADDRESS FROM POOR PERFORMER:
{chr(10).join(f'- {issue}' for issue in poor_performer_issues)}

Create a 9-shot shooting script that:
1. Replicates the successful elements from this reference video's analysis
2. Specifically addresses the poor performer's issues listed above
3. Uses concrete, actionable descriptions (not generic placeholders)

Use this EXACT markdown table format:

| Shot | Duration | Shot Role | Visual Content | Voiceover | Subtitle |
|------|----------|-----------|----------------|-----------|----------|
| 1 | 2-3s | Product Reveal | [specific visual based on reference] | [specific voiceover] | [keyword] |
| 2 | 3-5s | Scene Intro | [specific visual] | [specific voiceover] | [keyword] |
| 3 | 4-6s | Pain Point | [specific visual] | [specific voiceover] | [keyword] |
| 4 | 3-5s | Product Intro | [specific visual] | [specific voiceover] | [keyword] |
| 5 | 6-8s | Demo | [specific visual] | [specific voiceover] | [keyword] |
| 6 | 5-7s | Selling Point | [specific visual] | [specific voiceover] | [keyword] |
| 7 | 4-6s | Comparison | [specific visual] | [specific voiceover] | [keyword] |
| 8 | 3-5s | Offer Info | [specific visual] | [specific voiceover] | [keyword] |
| 9 | 3-5s | Call to Action | [specific visual] | [specific voiceover] | [keyword] |

Make the Visual Content and Voiceover columns SPECIFIC based on the actual reference video analysis, not generic templates."""

    messages = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    response = gemini.create_message(messages=messages, max_tokens=2048)
    return gemini.get_text_response(response) or "Unable to generate shooting script."


def generate_markdown_report(
    creators: list[CreatorPerformance],
    recommendations: dict[str, str],
    reference_videos: dict[str, list[ReferenceVideo]],
) -> str:
    """Generate the final markdown report with two-table reference video format."""
    lines = [
        "# Creator Optimization Report",
        "",
        f"Generated for {len(creators)} poorly performing Instagram creators.",
        "",
        "---",
        "",
    ]

    for i, creator in enumerate(creators, 1):
        lines.extend(
            [
                f"## Creator {i}: {creator.creator_name}",
                "",
                f"**Video URL:** [{creator.video_id}]({creator.video_url})",
                f"**ROI:** {creator.roi:.2f}",
                f"**Ad Spend:** ${creator.ad_spend:,.2f}",
            ]
        )

        # Show detected niche if available
        if creator.niche:
            lines.append(f"**Detected Niche:** {creator.niche}")
        if creator.product_category:
            lines.append(f"**Product Category:** {creator.product_category}")

        lines.extend(
            [
                "",
                "### Current Performance",
                "",
                "| Metric | Value | Rating |",
                "|--------|-------|--------|",
            ]
        )

        # Always show views row, even if 0 or missing
        if creator.views:
            lines.append(f"| Views | {creator.views:,} | - |")
        else:
            lines.append("| Views | N/A (scraping failed) | - |")

        # Show rates or N/A when data is unavailable
        if creator.like_rate is not None:
            rating = get_rating("like_rate", creator.like_rate)
            lines.append(f"| Like Rate | {creator.like_rate:.2f}% | {rating} |")
        else:
            lines.append("| Like Rate | N/A | - |")

        if creator.comment_rate is not None:
            rating = get_rating("comment_rate", creator.comment_rate)
            lines.append(f"| Comment Rate | {creator.comment_rate:.3f}% | {rating} |")
        else:
            lines.append("| Comment Rate | N/A | - |")

        if creator.save_rate is not None:
            rating = get_rating("save_rate", creator.save_rate)
            lines.append(f"| Save Rate | {creator.save_rate:.2f}% | {rating} |")
        else:
            lines.append("| Save Rate | N/A | - |")

        lines.append("")

        # Problems identified
        problems = []
        if creator.like_rate is not None and creator.like_rate < 1:
            problems.append("Low like rate indicates content lacks emotional resonance")
        if creator.comment_rate is not None and creator.comment_rate < 0.05:
            problems.append(
                "Low comment rate suggests no controversial or discussion points"
            )
        if creator.save_rate is not None and creator.save_rate < 0.3:
            problems.append(
                "Low save rate means content lacks long-term value or structure"
            )

        if problems:
            lines.extend(
                [
                    "### Problems Identified",
                    "",
                ]
            )
            for problem in problems:
                lines.append(f"- {problem}")
            lines.append("")

        # Recommendations
        if creator.video_id in recommendations:
            lines.extend(
                [
                    "### Optimization Recommendations",
                    "",
                    recommendations[creator.video_id],
                    "",
                ]
            )

        # Reference videos with two-table format (from docs/demo.md)
        if creator.video_id in reference_videos and reference_videos[creator.video_id]:
            refs = reference_videos[creator.video_id]
            lines.extend(
                [
                    "### Reference Videos",
                    "",
                    "| Video Links | Messaging | Creator | Followers |",
                    "|-------------|-----------|---------|-----------|",
                ]
            )

            for ref in refs:
                # Format URL for display
                if len(ref.video_url) > 40:
                    short_url = ref.video_url[:40] + "..."
                else:
                    short_url = ref.video_url
                # Format followers
                if ref.followers:
                    if ref.followers >= 1_000_000:
                        followers_str = f"{ref.followers / 1_000_000:.1f}M"
                    elif ref.followers >= 1_000:
                        followers_str = f"{ref.followers / 1_000:.1f}K"
                    else:
                        followers_str = str(ref.followers)
                else:
                    followers_str = "N/A"
                # Format messaging (truncate if needed)
                messaging = ref.messaging.replace("|", "-").replace("\n", " ")[:80]
                if len(ref.messaging) > 80:
                    messaging += "..."

                lines.append(
                    f"| [{short_url}]({ref.video_url}) | "
                    f'"{messaging}" | @{ref.creator_username} | {followers_str} |'
                )

            lines.append("")

            # Second table: engagement metrics
            lines.extend(
                [
                    "| Views | Likes | Comments | Shares | Saves |",
                    "|-------|-------|----------|--------|-------|",
                ]
            )

            def format_num(n: int | None) -> str:
                """Format numbers with K/M suffixes for readability."""
                if n is None:
                    return "N/A"
                if n >= 1_000_000:
                    return f"{n / 1_000_000:.1f}M"
                elif n >= 1_000:
                    return f"{n / 1_000:.1f}K"
                return str(n)

            for ref in refs:
                row = (
                    f"| {format_num(ref.views)} | {format_num(ref.likes)} | "
                    f"{format_num(ref.comments)} | {format_num(ref.shares)} | "
                    f"{format_num(ref.saves)} |"
                )
                lines.append(row)

            lines.append("")

            # Shooting scripts for each reference video
            for j, ref in enumerate(refs, 1):
                if ref.shooting_script:
                    lines.extend(
                        [
                            f"#### Shooting Script (Based on Reference Video {j})",
                            "",
                            ref.shooting_script,
                            "",
                        ]
                    )

        lines.extend(["---", ""])

    return "\n".join(lines)


async def main():
    """Main entry point for the demo script."""
    logger.info("Starting Creator Optimization Analysis Demo")

    # Paths
    project_root = Path(__file__).parent.parent
    test_data_path = project_root / "docs" / "test_m.md"
    output_path = project_root / "docs" / "demo_output.md"

    # Step 1: Parse test data to identify poor performers (with larger buffer for fallbacks)
    logger.info("Step 1: Parsing test data...")
    all_creators = parse_test_data(test_data_path, limit=10, buffer=10)
    logger.info(
        f"Found {len(all_creators)} poorly performing Instagram creators "
        "(including fallback pool)"
    )

    for c in all_creators[:10]:
        logger.info(
            f"  - {c.creator_name}: ROI={c.roi:.3f}, Ad Spend=${c.ad_spend:,.2f}"
        )

    # Initialize clients
    apify_client = ApifyClient()
    memories_v2_client = MemoriesV2Client()
    gemini = GeminiClient()
    agent = VideoSourcingAgent()

    # Step 2: Scrape engagement data with Apify -> Memories.ai v2 fallback
    logger.info("\nStep 2: Scraping engagement data...")
    creators: list[CreatorPerformance] = []

    for creator in all_creators:
        if len(creators) >= 10:
            break  # We have enough valid creators

        # Clean URL before scraping to remove tracking parameters
        clean_url = clean_instagram_url(creator.video_url)
        logger.info(f"  Scraping {clean_url}...")

        # Try Apify first (single attempt, no retry)
        video_data = await scrape_video_metrics(apify_client, clean_url)
        source = "Apify"

        # If Apify failed or returned bad data, try Memories.ai v2
        if not has_valid_metrics(video_data):
            logger.info("    Apify returned no valid data, trying Memories.ai v2...")
            video_data = await scrape_video_metrics_memories_v2(memories_v2_client, clean_url)
            await asyncio.sleep(1.0)  # Rate limit: Memories.ai v2 has tighter QPS limits
            source = "Memories.ai v2"

        # If both failed, skip this creator
        if not has_valid_metrics(video_data):
            logger.warning(
                f"    Skipping {creator.creator_name} - no valid metrics from either source"
            )
            continue

        # Valid data obtained - populate creator metrics
        logger.info(f"    Success via {source}")

        # video_data is guaranteed to be non-None at this point (has_valid_metrics checks it)
        assert video_data is not None

        # Extract username
        creator.username = extract_username_from_video(video_data)

        # Calculate engagement rates
        rates = calculate_engagement_rates(video_data)
        creator.views = rates.get("views")
        creator.likes = rates.get("likes")
        creator.comments = rates.get("comments")
        creator.saves = rates.get("saves")
        creator.shares = rates.get("shares")
        creator.like_rate = rates.get("like_rate")
        creator.comment_rate = rates.get("comment_rate")
        creator.save_rate = rates.get("save_rate")
        creator.share_rate = rates.get("share_rate")

        if creator.like_rate:
            logger.info(
                f"    Views: {creator.views}, "
                f"Like Rate: {creator.like_rate:.2f}%"
            )
        else:
            logger.info(f"    Views: {creator.views}")

        creators.append(creator)

    if len(creators) < 10:
        logger.warning(f"Only found {len(creators)} creators with valid metrics (target: 10)")

    # Step 3: Generate recommendations (parallelized)
    logger.info("\nStep 3: Generating recommendations...")
    recommendations: dict[str, str] = {}
    problems_by_creator: dict[str, list[str]] = {}

    # First pass: identify problems for each creator (sync, fast)
    creators_with_problems: list[tuple[CreatorPerformance, list[str]]] = []
    for creator in creators:
        if not creator.video_id:
            continue

        # Identify problems based on metrics
        problems = []
        if creator.like_rate is not None and creator.like_rate < 1:
            problems.append("Low like rate (<1%) - content lacks emotional resonance")
        if creator.comment_rate is not None and creator.comment_rate < 0.05:
            problems.append(
                "Low comment rate (<0.05%) - no controversial or discussion points"
            )
        if creator.save_rate is not None and creator.save_rate < 0.3:
            problems.append("Low save rate (<0.3%) - content lacks long-term value")
        if not problems:
            problems.append("Low ROI despite ad spend - overall performance issues")

        problems_by_creator[creator.video_id] = problems
        creators_with_problems.append((creator, problems))

    # Generate recommendations in parallel
    async def gen_recommendation(
        creator: CreatorPerformance, problems: list[str]
    ) -> tuple[str, str]:
        logger.info(f"  [{creator.creator_name}] Generating recommendations...")
        rec = await generate_recommendations(gemini, creator, problems)
        return creator.video_id, rec

    rec_results = await asyncio.gather(
        *[gen_recommendation(c, p) for c, p in creators_with_problems]
    )
    recommendations = dict(rec_results)

    # Step 4: Detect niche and find reference videos for ALL creators (parallelized)
    logger.info("\nStep 4: Detecting niches and finding reference videos...")

    # Helper function to process a single creator
    async def process_single_creator(
        creator: CreatorPerformance, sem: asyncio.Semaphore
    ) -> tuple[str, list[ReferenceVideo]]:
        """Process a single creator: detect niche, find refs, analyze, generate scripts."""
        async with sem:
            logger.info(f"  [{creator.creator_name}] Starting processing...")

            # Detect niche using Memories.ai v2
            logger.info(f"  [{creator.creator_name}] Detecting niche...")
            niche_data = await detect_creator_niche(memories_v2_client, gemini, creator.video_url)
            creator.niche = niche_data.get("niche", "lifestyle")
            creator.target_audience = niche_data.get("audience")
            creator.product_category = niche_data.get("product_category")
            logger.info(
                f"  [{creator.creator_name}] Niche: {creator.niche}, "
                f"Product: {creator.product_category}"
            )

            # Find reference videos in the same niche
            logger.info(
                f"  [{creator.creator_name}] Finding reference videos in "
                f"{creator.niche} niche..."
            )
            refs = await find_reference_videos(
                agent, memories_v2_client, apify_client, creator.niche, creator.product_category
            )
            logger.info(f"  [{creator.creator_name}] Found {len(refs)} reference videos")

            # Analyze reference videos in parallel (within this creator)
            async def analyze_single_ref(ref: ReferenceVideo) -> ReferenceVideo:
                logger.info(
                    f"  [{creator.creator_name}] Analyzing reference: "
                    f"{ref.video_url[:50]}..."
                )

                # Get MAI+Gemini analysis and transcript from Memories.ai v2
                vlm_analysis, transcript = await analyze_reference_video(
                    memories_v2_client, gemini, ref.video_url
                )
                ref.vlm_analysis = vlm_analysis
                ref.transcript = transcript

                # Generate shooting script from analyzed video
                logger.info(
                    f"  [{creator.creator_name}] Generating shooting script for "
                    f"{ref.video_url[:50]}..."
                )
                problems = problems_by_creator.get(creator.video_id, ["Low engagement"])
                ref.shooting_script = await generate_shooting_script(
                    gemini, ref, problems
                )
                return ref

            # Run reference video analyses sequentially with delays (v2 rate limits)
            analyzed_refs = []
            for r in refs:
                analyzed_refs.append(await analyze_single_ref(r))
                await asyncio.sleep(1.0)  # Rate limit: Memories.ai v2 has tighter QPS limits

            logger.info(f"  [{creator.creator_name}] Completed processing")
            return creator.video_id, list(analyzed_refs)

    # Create semaphore to limit concurrent creator processing (v2 rate limits)
    sem = asyncio.Semaphore(1)  # Sequential processing to respect rate limit

    # Process all creators in parallel (semaphore limits concurrency)
    results = await asyncio.gather(
        *[process_single_creator(c, sem) for c in creators]
    )
    reference_videos: dict[str, list[ReferenceVideo]] = dict(results)

    # Step 5: Generate final report
    logger.info("\nStep 5: Generating markdown report...")
    report = generate_markdown_report(
        creators,
        recommendations,
        reference_videos,
    )

    # Write output
    output_path.write_text(report)
    logger.info(f"\nReport written to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nAnalyzed {len(creators)} poorly performing creators:")
    for i, c in enumerate(creators, 1):
        print(f"  {i}. {c.creator_name} - ROI: {c.roi:.3f}")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
