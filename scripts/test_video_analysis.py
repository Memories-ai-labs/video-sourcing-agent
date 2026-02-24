#!/usr/bin/env python3
"""Test script: Analyze TikTok videos using Memories.ai v2 + Gemini APIs.

This script:
1. Parses Product_Related_Videos_Rosabella.csv
2. Selects top 10 videos by view count
3. Gets transcript and metadata via Memories.ai v2 API
4. Analyzes using Gemini with a structured e-commerce analysis prompt
5. Tracks time and token usage per video
6. Outputs results to JSON and prints a summary table

Analysis modes:
- transcript (default): Uses Memories.ai v2 transcript/metadata + Gemini text analysis
- vlm: Uses Memories.ai v2 VLM with CDN download URL (bypasses robots.txt blocking)
- mai: Enhanced transcript mode with visual inference prompt

Note: Memories.ai v2 VLM cannot directly access TikTok page URLs (blocked by robots.txt).
The VLM mode works around this by fetching the CDN download URL from metadata first,
but often fails due to CDN authentication requirements.

The MAI mode uses the regular transcript endpoint with an enhanced Gemini prompt
that requests visual element inference. True MAI visual analysis (videoTranscript)
requires a webhook setup - see MemoriesV2Client.request_mai_transcript() docs.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inlined API clients (self-contained — no video_sourcing_agent dependency)
# ---------------------------------------------------------------------------


class MemoriesV2Client:
    """Lightweight Memories.ai v2 API client (inlined for standalone use)."""

    BASE_URL = "https://mavi-backend.memories.ai/serve/api/v2"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.api_key = (api_key or os.getenv("MEMORIES_API_KEY", "")).strip()
        normalized_base_url = (
            base_url or os.getenv("MEMORIES_BASE_URL", self.BASE_URL)
        ).strip()
        self.base_url = (normalized_base_url or self.BASE_URL).rstrip("/")
        self.timeout = int(os.getenv("API_TIMEOUT_SECONDS", "30"))
        self.default_channel = (
            os.getenv("MEMORIES_DEFAULT_CHANNEL", "memories.ai").strip()
            or "memories.ai"
        )
        self.vlm_model = (
            os.getenv("MEMORIES_VLM_MODEL", "gemini:gemini-3-flash-preview").strip()
            or "gemini:gemini-3-flash-preview"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }

    # -- TikTok endpoints ---------------------------------------------------

    async def get_tiktok_metadata(
        self, video_url: str, channel: str | None = None
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/tiktok/video/metadata",
                headers=self._headers(),
                json={
                    "video_url": video_url,
                    "channel": channel or self.default_channel,
                },
            )
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

    async def get_tiktok_transcript(
        self, video_url: str, channel: str | None = None
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/tiktok/video/transcript",
                headers=self._headers(),
                json={
                    "video_url": video_url,
                    "channel": channel or self.default_channel,
                },
            )
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

    # -- VLM endpoints ------------------------------------------------------

    def _get_mime_type(self, url: str) -> str:
        url_lower = url.lower()
        if any(ext in url_lower for ext in [".mp4", ".m4v"]):
            return "video/mp4"
        elif ".webm" in url_lower:
            return "video/webm"
        elif ".mov" in url_lower:
            return "video/quicktime"
        elif ".avi" in url_lower:
            return "video/x-msvideo"
        elif ".mkv" in url_lower:
            return "video/x-matroska"
        return "video/mp4"

    async def vlm_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/vu/chat/completions",
                headers=self._headers(),
                json={
                    "model": model or self.vlm_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

    async def analyze_video_with_vlm(
        self,
        video_url: str,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 1000,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_uri": video_url,
                        "mime_type": self._get_mime_type(video_url),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return await self.vlm_chat(
            messages=messages, model=model, max_tokens=max_tokens
        )

    async def get_tiktok_video_download_url(self, video_url: str) -> str | None:
        try:
            metadata = await self.get_tiktok_metadata(video_url)
            data = metadata.get("data")
            candidates: list[dict[str, Any]] = []
            if isinstance(data, dict):
                candidates.append(data)
            elif isinstance(data, list):
                candidates.extend([item for item in data if isinstance(item, dict)])

            for candidate in candidates:
                item_info = candidate.get("itemInfo")
                item_struct = (
                    item_info.get("itemStruct")
                    if isinstance(item_info, dict)
                    else None
                )

                possible_sources = [candidate]
                if isinstance(item_info, dict):
                    possible_sources.append(item_info)
                if isinstance(item_struct, dict):
                    possible_sources.append(item_struct)

                for source in possible_sources:
                    video = source.get("video")
                    if not isinstance(video, dict):
                        continue
                    download_url = video.get("downloadAddr") or video.get("playAddr")
                    if isinstance(download_url, str) and download_url:
                        return download_url
        except Exception:
            pass
        return None

    async def analyze_tiktok_video_with_vlm(
        self,
        video_url: str,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 1000,
    ) -> dict[str, Any]:
        download_url = await self.get_tiktok_video_download_url(video_url)
        if not download_url:
            raise ValueError(
                f"Could not get download URL for TikTok video: {video_url}. "
                "Consider using transcript-based analysis instead."
            )
        return await self.analyze_video_with_vlm(
            video_url=download_url, prompt=prompt, model=model, max_tokens=max_tokens
        )


class GeminiClient:
    """Lightweight Gemini API client (inlined for standalone use)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def create_message(
        self,
        messages: list[types.Content],
        system: str | None = None,
        tools: list[types.Tool] | None = None,
        max_tokens: int = 4096,
    ) -> types.GenerateContentResponse:
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system,
        )
        if tools:
            config.tools = tools  # type: ignore[assignment]
            config.automatic_function_calling = types.AutomaticFunctionCallingConfig(
                disable=True
            )
        return self.client.models.generate_content(
            model=self.model,
            contents=messages,  # type: ignore[arg-type]
            config=config,
        )

    def get_text_response(
        self, response: types.GenerateContentResponse
    ) -> str | None:
        if not response.candidates or not response.candidates[0].content:
            return None
        text_parts = []
        parts = response.candidates[0].content.parts
        if parts:
            for part in parts:
                if part.text:
                    text_parts.append(part.text)
        return "\n".join(text_parts) if text_parts else None


# ---------------------------------------------------------------------------
# Analysis modes & prompts
# ---------------------------------------------------------------------------


class AnalysisMode(Enum):
    """Analysis mode for video processing."""

    TRANSCRIPT = "transcript"  # Memories.ai v2 transcript/metadata + Gemini text analysis
    VLM = "vlm"  # Memories.ai v2 VLM with CDN download URL
    MAI = "mai"  # Enhanced transcript mode with visual inference prompt


# TikTok e-commerce quantitative analysis prompt
# Note: Double braces {{ }} are escaped for .format() - they become single braces in output
ANALYSIS_PROMPT = """Analyze this TikTok video based on its transcript and metadata for e-commerce performance. Return your analysis as valid JSON with the following structure:

{{
  "video_summary": "Brief 1-2 sentence description of the video content",
  "product_shown": "Name/type of product featured",
  "hook_analysis": {{
    "hook_type": "question|statement|visual_surprise|problem_reveal|transformation",
    "hook_text": "Opening words/hook from the transcript",
    "hook_effectiveness_score": 1-10
  }},
  "content_structure": {{
    "format": "tutorial|review|unboxing|testimonial|demonstration|lifestyle|before_after",
    "pacing": "fast|medium|slow",
    "duration_estimate_seconds": number
  }},
  "selling_points": [
    "Key benefit 1",
    "Key benefit 2",
    "Key benefit 3"
  ],
  "call_to_action": {{
    "cta_present": true/false,
    "cta_type": "shop_now|link_in_bio|comment|follow|none",
    "cta_text": "CTA text if present in transcript"
  }},
  "emotional_triggers": ["curiosity", "urgency", "social_proof", "fear_of_missing_out", "satisfaction", "trust"],
  "engagement_metrics": {{
    "views": number,
    "likes": number,
    "engagement_rate_percent": number
  }},
  "virality_factors": {{
    "shareability_score": 1-10,
    "relatability_score": 1-10,
    "uniqueness_score": 1-10
  }},
  "improvement_suggestions": [
    "Suggestion 1",
    "Suggestion 2"
  ]
}}

VIDEO METADATA:
{metadata}

VIDEO TRANSCRIPT:
{transcript}

Return ONLY valid JSON, no additional text or markdown formatting."""

# MAI analysis prompt (for visual + audio transcript analysis)
# Note: Double braces {{ }} are escaped for .format() - they become single braces in output
MAI_ANALYSIS_PROMPT = """Analyze this TikTok video based on its visual scene descriptions and audio transcript for e-commerce performance. Return your analysis as valid JSON with the following structure:

{{
  "video_summary": "Brief 1-2 sentence description of the video content",
  "product_shown": "Name/type of product featured",
  "hook_analysis": {{
    "hook_type": "question|statement|visual_surprise|problem_reveal|transformation",
    "hook_description": "What grabs attention in the first 1-3 seconds (visual + audio)",
    "hook_effectiveness_score": 1-10
  }},
  "content_structure": {{
    "format": "tutorial|review|unboxing|testimonial|demonstration|lifestyle|before_after",
    "pacing": "fast|medium|slow",
    "visual_quality": "Based on scene descriptions"
  }},
  "selling_points": [
    "Key benefit 1 shown or mentioned",
    "Key benefit 2 shown or mentioned",
    "Key benefit 3 shown or mentioned"
  ],
  "call_to_action": {{
    "cta_present": true/false,
    "cta_type": "shop_now|link_in_bio|comment|follow|none",
    "cta_description": "CTA detected from visual or audio"
  }},
  "emotional_triggers": ["curiosity", "urgency", "social_proof", "fear_of_missing_out", "satisfaction", "trust"],
  "visual_elements": {{
    "text_overlays": true/false,
    "product_close_ups": true/false,
    "face_shown": true/false,
    "before_after_comparison": true/false
  }},
  "virality_factors": {{
    "shareability_score": 1-10,
    "relatability_score": 1-10,
    "uniqueness_score": 1-10
  }},
  "improvement_suggestions": [
    "Suggestion 1",
    "Suggestion 2"
  ]
}}

VIDEO METADATA:
{metadata}

VISUAL SCENE DESCRIPTIONS (from AI video analysis):
{video_transcript}

AUDIO TRANSCRIPT (speech-to-text):
{audio_transcript}

Return ONLY valid JSON, no additional text or markdown formatting."""

# VLM-specific prompt (for direct video analysis)
VLM_ANALYSIS_PROMPT = """Watch this TikTok video and analyze it for e-commerce performance. Return your analysis as valid JSON with the following structure:

{
  "video_summary": "Brief 1-2 sentence description of what happens in the video",
  "product_shown": "Name/type of product featured",
  "hook_analysis": {
    "hook_type": "question|statement|visual_surprise|problem_reveal|transformation",
    "hook_description": "What grabs attention in the first 1-3 seconds",
    "hook_effectiveness_score": 1-10
  },
  "content_structure": {
    "format": "tutorial|review|unboxing|testimonial|demonstration|lifestyle|before_after",
    "pacing": "fast|medium|slow",
    "visual_quality": "low|medium|high|professional"
  },
  "selling_points": [
    "Key benefit 1 shown or mentioned",
    "Key benefit 2 shown or mentioned",
    "Key benefit 3 shown or mentioned"
  ],
  "call_to_action": {
    "cta_present": true/false,
    "cta_type": "shop_now|link_in_bio|comment|follow|none",
    "cta_description": "Description of the CTA if present"
  },
  "emotional_triggers": ["curiosity", "urgency", "social_proof", "fear_of_missing_out", "satisfaction", "trust"],
  "visual_elements": {
    "text_overlays": true/false,
    "product_close_ups": true/false,
    "face_shown": true/false,
    "before_after_comparison": true/false
  },
  "virality_factors": {
    "shareability_score": 1-10,
    "relatability_score": 1-10,
    "uniqueness_score": 1-10
  },
  "improvement_suggestions": [
    "Suggestion 1",
    "Suggestion 2"
  ]
}

Return ONLY valid JSON, no additional text or markdown formatting."""


@dataclass
class VideoData:
    """Data for a single video from CSV."""

    url: str
    title: str
    creator: str
    views: int
    likes: int
    duration: int  # seconds


@dataclass
class AnalysisResult:
    """Result of analyzing a single video."""

    video: VideoData
    analysis: dict[str, Any] | None
    error: str | None
    duration_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    transcript: str | None
    metadata: dict[str, Any] | None


def parse_csv(file_path: Path, limit: int = 10) -> list[VideoData]:
    """Parse CSV and return top videos by view count.

    Args:
        file_path: Path to the CSV file
        limit: Number of videos to return

    Returns:
        List of VideoData objects sorted by views (descending)
    """
    videos: list[VideoData] = []

    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            url = row.get("TikTok官网视频主页", "").strip()

            # Skip invalid URLs
            if not url or "tiktok.com" not in url.lower():
                continue

            # Parse view count (may be formatted with commas or as plain number)
            views_str = row.get("播放量", "0").strip()
            try:
                views = int(views_str.replace(",", ""))
            except ValueError:
                views = 0

            # Parse likes
            likes_str = row.get("点赞数", "0").strip()
            try:
                likes = int(likes_str.replace(",", ""))
            except ValueError:
                likes = 0

            # Parse duration (seconds)
            duration_str = row.get("视频时长", "0").strip()
            try:
                duration = int(duration_str)
            except ValueError:
                duration = 0

            videos.append(
                VideoData(
                    url=url,
                    title=row.get("视频标题", "")[:100],  # Truncate long titles
                    creator=row.get("达人昵称", "Unknown"),
                    views=views,
                    likes=likes,
                    duration=duration,
                )
            )

    # Sort by views descending and take top N
    videos.sort(key=lambda v: v.views, reverse=True)
    return videos[:limit]


def format_number(n: int) -> str:
    """Format number with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


async def get_video_data(
    memories_v2: MemoriesV2Client,
    video_url: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Get transcript and metadata for a video.

    Args:
        memories_v2: Memories.ai v2 client
        video_url: TikTok video URL

    Returns:
        Tuple of (metadata dict, transcript text)
    """
    metadata = None
    transcript_text = None

    # Get metadata
    try:
        metadata_result = await memories_v2.get_tiktok_metadata(video_url)
        if metadata_result.get("code") == "0000":
            metadata = metadata_result.get("data", {})
    except Exception as e:
        logger.warning(f"Failed to get metadata: {e}")

    # Get transcript
    try:
        transcript_result = await memories_v2.get_tiktok_transcript(video_url)
        if transcript_result.get("code") == "0000":
            data = transcript_result.get("data", {})
            # Extract transcript text - may be in different formats
            if isinstance(data, dict):
                inner_data = data.get("data", data)
                if isinstance(inner_data, dict):
                    transcript_text = inner_data.get("transcript", "")
                    # Also check for normalized format
                    if not transcript_text:
                        transcript_text = inner_data.get("text", "")
    except Exception as e:
        logger.warning(f"Failed to get transcript: {e}")

    return metadata, transcript_text


async def analyze_video(
    memories_v2: MemoriesV2Client,
    gemini: GeminiClient,
    video: VideoData,
) -> AnalysisResult:
    """Analyze a single video using Memories.ai v2 + Gemini.

    Args:
        memories_v2: Memories.ai v2 client for transcript/metadata
        gemini: Gemini client for analysis
        video: Video data to analyze

    Returns:
        AnalysisResult with analysis data and metrics
    """
    start_time = time.time()
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    transcript = None
    metadata = None

    try:
        logger.info(f"  Analyzing: {video.url[:60]}...")

        # Step 1: Get transcript and metadata from Memories.ai v2
        logger.info("    Getting transcript and metadata...")
        metadata, transcript = await get_video_data(memories_v2, video.url)

        if not transcript and not metadata:
            raise ValueError("Failed to get both transcript and metadata")

        # Step 2: Format metadata for analysis
        metadata_str = ""
        if metadata:
            item_info = metadata.get("itemInfo", {}).get("itemStruct", {})
            stats = item_info.get("stats", {})
            author = item_info.get("author", {})

            metadata_str = f"""Title: {item_info.get('desc', video.title)}
Creator: {author.get('uniqueId', video.creator)}
Duration: {item_info.get('video', {}).get('duration', video.duration)} seconds
Views: {stats.get('playCount', video.views)}
Likes: {stats.get('diggCount', video.likes)}
Comments: {stats.get('commentCount', 0)}
Shares: {stats.get('shareCount', 0)}"""
        else:
            metadata_str = f"""Title: {video.title}
Creator: {video.creator}
Duration: {video.duration} seconds
Views: {video.views}
Likes: {video.likes}"""

        # Step 3: Analyze with Gemini
        logger.info("    Analyzing with Gemini...")
        prompt = ANALYSIS_PROMPT.format(
            metadata=metadata_str,
            transcript=transcript or "(No transcript available)"
        )

        messages = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        response = gemini.create_message(messages=messages, max_tokens=4096)

        # Extract token usage
        usage = response.usage_metadata
        if usage:
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            total_tokens = usage.total_token_count or (input_tokens + output_tokens)

        # Extract response text
        content = gemini.get_text_response(response) or ""

        duration = time.time() - start_time

        # Try to parse JSON from content
        analysis = None
        if content:
            # Try to extract JSON if wrapped in markdown code block
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end > json_start:
                    content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                if json_end > json_start:
                    content = content[json_start:json_end].strip()

            # Try to find JSON object in content
            if "{" in content:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_end > json_start:
                    try:
                        analysis = json.loads(content[json_start:json_end])
                    except json.JSONDecodeError:
                        # Store raw content if not valid JSON
                        analysis = {"raw_response": content}
            else:
                analysis = {"raw_response": content}

        logger.info(
            f"    Completed in {duration:.1f}s - "
            f"Tokens: {input_tokens:,} in / {output_tokens:,} out"
        )

        return AnalysisResult(
            video=video,
            analysis=analysis,
            error=None,
            duration_seconds=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            transcript=transcript,
            metadata=metadata,
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"    Failed after {duration:.1f}s: {e}")

        return AnalysisResult(
            video=video,
            analysis=None,
            error=str(e),
            duration_seconds=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            transcript=transcript,
            metadata=metadata,
        )


async def analyze_video_vlm(
    memories_v2: MemoriesV2Client,
    video: VideoData,
) -> AnalysisResult:
    """Analyze a video using Memories.ai v2 VLM (direct video analysis).

    This method attempts to bypass robots.txt restrictions by using the CDN
    download URL instead of the TikTok page URL.

    Note: This approach often fails because TikTok CDN URLs require
    authentication headers that Vertex AI cannot provide.

    Args:
        memories_v2: Memories.ai v2 client for VLM analysis
        video: Video data to analyze

    Returns:
        AnalysisResult with analysis data and metrics
    """
    start_time = time.time()
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    try:
        logger.info(f"  Analyzing with VLM: {video.url[:60]}...")

        # Step 1: Get download URL and analyze with VLM
        logger.info("    Getting CDN download URL and analyzing...")
        result = await memories_v2.analyze_tiktok_video_with_vlm(
            video_url=video.url,
            prompt=VLM_ANALYSIS_PROMPT,
            model="gemini:gemini-2.5-flash",  # Use valid model
            max_tokens=4096,
        )

        duration = time.time() - start_time

        # Check for VLM errors (Vertex AI URL access issues)
        if result.get("status") == "errored":
            error_info = result.get("error", {})
            error_msg = error_info.get("message", "Unknown VLM error")
            logger.error(f"    VLM error: {error_msg[:100]}...")
            return AnalysisResult(
                video=video,
                analysis=None,
                error=f"VLM: {error_msg}",
                duration_seconds=duration,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                transcript=None,
                metadata=None,
            )

        # Extract token usage from VLM response
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        # Extract response content
        content = ""
        choices = result.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")

        # Try to parse JSON from content
        analysis = None
        if content:
            # Try to extract JSON if wrapped in markdown code block
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end > json_start:
                    content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                if json_end > json_start:
                    content = content[json_start:json_end].strip()

            # Try to find JSON object in content
            if "{" in content:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_end > json_start:
                    try:
                        analysis = json.loads(content[json_start:json_end])
                    except json.JSONDecodeError:
                        analysis = {"raw_response": content}
            else:
                analysis = {"raw_response": content}

        logger.info(
            f"    Completed in {duration:.1f}s - "
            f"Tokens: {input_tokens:,} in / {output_tokens:,} out"
        )

        return AnalysisResult(
            video=video,
            analysis=analysis,
            error=None,
            duration_seconds=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            transcript=None,  # VLM mode doesn't use transcript
            metadata=None,  # Metadata used internally for download URL
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"    Failed after {duration:.1f}s: {e}")

        return AnalysisResult(
            video=video,
            analysis=None,
            error=str(e),
            duration_seconds=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            transcript=None,
            metadata=None,
        )


async def analyze_video_mai(
    memories_v2: MemoriesV2Client,
    gemini: GeminiClient,
    video: VideoData,
) -> AnalysisResult:
    """Analyze a video using enhanced transcript analysis with visual context prompt.

    Note: The true MAI endpoint (which provides visual scene descriptions from
    Gemini VLM) requires a webhook URL and doesn't support polling. This mode
    uses the regular transcript endpoint with an enhanced prompt that instructs
    Gemini to infer visual elements from the audio/text content.

    For full MAI visual analysis, you would need to:
    1. Set up a webhook receiver (ngrok, cloud function, etc.)
    2. Call memories_v2.request_mai_transcript(video_url, webhook_url="...")
    3. Receive results via the webhook

    This mode provides:
    - Metadata: Video info from TikTok
    - Audio transcript: Speech-to-text from regular transcript endpoint
    - Enhanced analysis: Gemini prompt that requests visual element inference

    Args:
        memories_v2: Memories.ai v2 client for transcript
        gemini: Gemini client for analysis
        video: Video data to analyze

    Returns:
        AnalysisResult with analysis data and metrics
    """
    start_time = time.time()
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    transcript_text = None
    metadata = None

    try:
        logger.info(f"  Analyzing with MAI mode: {video.url[:60]}...")

        # Step 1: Get metadata (for context in analysis)
        logger.info("    Getting metadata...")
        try:
            metadata_result = await memories_v2.get_tiktok_metadata(video.url)
            if metadata_result.get("code") == "0000":
                metadata = metadata_result.get("data", {})
        except Exception as e:
            logger.warning(f"    Failed to get metadata: {e}")

        # Step 2: Get transcript using the regular endpoint
        logger.info("    Getting transcript...")
        transcript_result = await memories_v2.get_tiktok_transcript(video.url)

        # Parse transcript from response
        transcript_segments = []
        if transcript_result.get("code") == "0000":
            data = transcript_result.get("data", {})
            if isinstance(data, dict):
                inner_data = data.get("data", {})
                if isinstance(inner_data, dict):
                    segments = inner_data.get("transcript", [])
                    if isinstance(segments, list):
                        for seg in segments:
                            if isinstance(seg, dict):
                                start = seg.get("start", 0)
                                end = seg.get("end", 0)
                                text = seg.get("text", "")
                                if text:
                                    transcript_segments.append(f"[{start:.1f}s-{end:.1f}s] {text}")

        transcript_text = "\n".join(transcript_segments) if transcript_segments else "(No transcript available)"
        logger.info(f"    Got {len(transcript_segments)} transcript segments")

        # Step 3: Format metadata for analysis
        metadata_str = ""
        if metadata:
            item_info = metadata.get("itemInfo", {}).get("itemStruct", {})
            stats = item_info.get("stats", {})
            author = item_info.get("author", {})

            metadata_str = f"""Title: {item_info.get('desc', video.title)}
Creator: {author.get('uniqueId', video.creator)}
Duration: {item_info.get('video', {}).get('duration', video.duration)} seconds
Views: {stats.get('playCount', video.views)}
Likes: {stats.get('diggCount', video.likes)}
Comments: {stats.get('commentCount', 0)}
Shares: {stats.get('shareCount', 0)}"""
        else:
            metadata_str = f"""Title: {video.title}
Creator: {video.creator}
Duration: {video.duration} seconds
Views: {video.views}
Likes: {video.likes}"""

        # Step 4: Analyze with Gemini using the enhanced MAI-style prompt
        # Since we don't have visual descriptions, we'll ask Gemini to infer
        # visual elements based on the audio content and common TikTok patterns
        logger.info("    Analyzing with Gemini (enhanced prompt)...")
        prompt = MAI_ANALYSIS_PROMPT.format(
            metadata=metadata_str,
            video_transcript="(Visual analysis not available - infer from audio context)",
            audio_transcript=transcript_text,
        )

        messages = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        response = gemini.create_message(messages=messages, max_tokens=4096)

        # Extract token usage
        usage = response.usage_metadata
        if usage:
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            total_tokens = usage.total_token_count or (input_tokens + output_tokens)

        # Extract response text
        content = gemini.get_text_response(response) or ""

        duration = time.time() - start_time

        # Try to parse JSON from content
        analysis = None
        if content:
            # Try to extract JSON if wrapped in markdown code block
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end > json_start:
                    content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                if json_end > json_start:
                    content = content[json_start:json_end].strip()

            # Try to find JSON object in content
            if "{" in content:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_end > json_start:
                    try:
                        analysis = json.loads(content[json_start:json_end])
                    except json.JSONDecodeError:
                        analysis = {"raw_response": content}
            else:
                analysis = {"raw_response": content}

        logger.info(
            f"    Completed in {duration:.1f}s - "
            f"Tokens: {input_tokens:,} in / {output_tokens:,} out"
        )

        return AnalysisResult(
            video=video,
            analysis=analysis,
            error=None,
            duration_seconds=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            transcript=transcript_text,
            metadata=metadata,
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"    Failed after {duration:.1f}s: {e}")

        return AnalysisResult(
            video=video,
            analysis=None,
            error=str(e),
            duration_seconds=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            transcript=transcript_text,
            metadata=metadata,
        )


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost based on Gemini Flash pricing.

    Gemini 2.5 Flash pricing (approximate):
    - Input: $0.075 per 1M tokens
    - Output: $0.30 per 1M tokens
    """
    input_cost = (input_tokens / 1_000_000) * 0.075
    output_cost = (output_tokens / 1_000_000) * 0.30
    return input_cost + output_cost


def print_summary_table(results: list[AnalysisResult]) -> None:
    """Print a formatted summary table of results."""
    print("\n" + "=" * 120)
    print("VIDEO ANALYSIS RESULTS")
    print("=" * 120)

    # Header
    print(
        f"{'#':<3} | {'Creator':<20} | {'Views':<8} | "
        f"{'Time (s)':<9} | {'Input Tok':<10} | {'Output Tok':<11} | "
        f"{'Cost ($)':<8} | {'Status':<10}"
    )
    print("-" * 120)

    total_time = 0.0
    total_input = 0
    total_output = 0
    success_count = 0

    for i, r in enumerate(results, 1):
        status = "OK" if r.analysis and not r.error else f"ERR: {r.error[:20]}..." if r.error else "No data"
        cost = calculate_cost(r.input_tokens, r.output_tokens)

        print(
            f"{i:<3} | {r.video.creator[:20]:<20} | {format_number(r.video.views):<8} | "
            f"{r.duration_seconds:<9.1f} | {r.input_tokens:<10,} | {r.output_tokens:<11,} | "
            f"${cost:<7.4f} | {status:<10}"
        )

        total_time += r.duration_seconds
        total_input += r.input_tokens
        total_output += r.output_tokens
        if r.analysis and not r.error:
            success_count += 1

    # Totals
    print("-" * 120)
    total_cost = calculate_cost(total_input, total_output)
    print(
        f"{'TOTAL':<3} | {'':<20} | {'':<8} | "
        f"{total_time:<9.1f} | {total_input:<10,} | {total_output:<11,} | "
        f"${total_cost:<7.4f} | {success_count}/{len(results)} OK"
    )
    print("=" * 120)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze TikTok videos using Memories.ai v2 + Gemini APIs"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["transcript", "vlm", "mai"],
        default="transcript",
        help="Analysis mode: 'transcript' uses text, 'vlm' uses direct video, 'mai' uses MAI transcript (visual + audio)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of videos to analyze (default: 10)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file (default: docs/Product_Related_Videos_Rosabella.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (default: docs/video_analysis_results.json)",
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    mode = AnalysisMode(args.mode)

    logger.info(f"Starting TikTok Video Analysis Test (mode: {mode.value})")

    # Paths
    project_root = Path(__file__).parent.parent
    default_csv = project_root / "docs" / "Product_Related_Videos_Rosabella.csv"
    default_output = project_root / "docs" / "video_analysis_results.json"
    csv_path = Path(args.csv) if args.csv else default_csv
    output_path = Path(args.output) if args.output else default_output

    # Check CSV exists
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return

    # Step 1: Parse CSV and select top videos
    logger.info("Step 1: Parsing CSV...")
    videos = parse_csv(csv_path, limit=args.limit)
    logger.info(f"Selected {len(videos)} videos (top by view count)")

    for i, v in enumerate(videos, 1):
        logger.info(f"  {i}. {v.creator} - {format_number(v.views)} views")

    # Initialize clients
    memories_v2 = MemoriesV2Client()
    gemini = GeminiClient() if mode in (AnalysisMode.TRANSCRIPT, AnalysisMode.MAI) else None

    # Step 2: Analyze each video
    if mode == AnalysisMode.TRANSCRIPT:
        logger.info("\nStep 2: Analyzing videos with Memories.ai v2 transcript + Gemini...")
    elif mode == AnalysisMode.VLM:
        logger.info("\nStep 2: Analyzing videos with Memories.ai v2 VLM (CDN download URL)...")
    else:  # MAI
        logger.info("\nStep 2: Analyzing videos with MAI transcript (visual + audio)...")

    results: list[AnalysisResult] = []

    for i, video in enumerate(videos, 1):
        logger.info(f"\n[{i}/{len(videos)}] Processing video...")

        if mode == AnalysisMode.TRANSCRIPT:
            result = await analyze_video(memories_v2, gemini, video)
        elif mode == AnalysisMode.VLM:
            result = await analyze_video_vlm(memories_v2, video)
        else:  # MAI
            result = await analyze_video_mai(memories_v2, gemini, video)

        results.append(result)

        # Small delay between requests to avoid rate limiting
        if i < len(videos):
            await asyncio.sleep(1.0)

    # Step 3: Print summary table
    print_summary_table(results)

    # Step 4: Save detailed results to JSON
    logger.info(f"\nSaving detailed results to {output_path}...")

    output_data = {
        "summary": {
            "analysis_mode": mode.value,
            "total_videos": len(results),
            "successful": sum(1 for r in results if r.analysis and not r.error),
            "failed": sum(1 for r in results if r.error),
            "total_time_seconds": sum(r.duration_seconds for r in results),
            "total_input_tokens": sum(r.input_tokens for r in results),
            "total_output_tokens": sum(r.output_tokens for r in results),
            "total_cost_usd": calculate_cost(
                sum(r.input_tokens for r in results),
                sum(r.output_tokens for r in results),
            ),
        },
        "videos": [
            {
                "url": r.video.url,
                "creator": r.video.creator,
                "title": r.video.title,
                "views": r.video.views,
                "likes": r.video.likes,
                "duration_seconds": r.video.duration,
                "analysis": r.analysis,
                "error": r.error,
                "metrics": {
                    "time_seconds": r.duration_seconds,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "total_tokens": r.total_tokens,
                    "cost_usd": calculate_cost(r.input_tokens, r.output_tokens),
                },
                "transcript_available": r.transcript is not None,
                "metadata_available": r.metadata is not None,
            }
            for r in results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Analysis mode: {mode.value}")
    print(f"Videos analyzed: {output_data['summary']['successful']}/{output_data['summary']['total_videos']}")
    print(f"Total time: {output_data['summary']['total_time_seconds']:.1f} seconds")
    print(f"Total tokens: {output_data['summary']['total_input_tokens']:,} input / {output_data['summary']['total_output_tokens']:,} output")
    print(f"Total cost: ${output_data['summary']['total_cost_usd']:.4f}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
