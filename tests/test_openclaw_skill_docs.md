# OpenClaw skill smoke test scenarios

## Scope

Manual validation checklist for `video-sourcing` OpenClaw skill integration.

## Scenarios

1. `/video_sourcing <query>` with managed bootstrap runtime
   - Preconditions:
     - `VIDEO_SOURCING_AGENT_ROOT` unset
     - `git` and `uv` available
     - required API keys set
   - Send: `/video_sourcing Find trending UGC videos for SaaS this week`
   - Expect:
     - Runner bootstraps `~/.openclaw/data/video-sourcing-agent/v0.2.0` on first run.
     - Query executes successfully.

2. `/video_sourcing <query>` with local override runtime
   - Preconditions:
     - `skills.entries["video-sourcing"].env.VIDEO_SOURCING_AGENT_ROOT` points to valid local repo.
   - Send: `/video_sourcing Find trending UGC videos for SaaS this week`
   - Expect:
     - Override path is used.
     - Managed bootstrap path is not required.

3. Deterministic message count matrix
   - Fast run (<6s):
     - Exactly 2 user-visible messages.
   - Slow run (>=6s):
     - 3+ user-visible messages with throttled middle progress.

4. Missing prerequisites behavior
   - Missing `git` or `uv`:
     - Clear actionable failure message.
   - Missing `GOOGLE_API_KEY` or `YOUTUBE_API_KEY`:
     - Clear actionable failure message.

5. Free-form non-strict behavior
   - Free-form query still routes correctly and emits natural progress/final response.

## Channel matrix

1. Telegram DM with `/video_sourcing`.
2. Telegram free-form.
3. Non-Telegram channel (for example Discord) with block/partial streaming fallback.
