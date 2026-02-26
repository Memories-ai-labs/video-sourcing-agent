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
     - Runner bootstraps `~/.openclaw/data/video-sourcing-agent/v0.2.3` on first run.
     - Query executes successfully.

2. `/video_sourcing <query>` with local override runtime
   - Preconditions:
     - `skills.entries["video-sourcing"].env.VIDEO_SOURCING_AGENT_ROOT` points to valid local repo.
   - Send: `/video_sourcing Find trending UGC videos for SaaS this week`
   - Expect:
     - Override path is used.
     - Managed bootstrap path is not required.

3. Deterministic message count matrix
   - Fast run (<5s):
     - Exactly 2 user-visible messages.
   - Slow run (>=5s):
     - 3+ user-visible messages with throttled middle progress.

4. Missing prerequisites behavior
   - Missing `git` or `uv`:
     - Clear actionable failure message.
   - Missing `GOOGLE_API_KEY` or `YOUTUBE_API_KEY`:
     - Clear actionable failure message.

5. Free-form non-strict behavior
   - Free-form query still routes correctly and emits natural progress/final response.

6. Telegram typing indicator clears after terminal response
   - Preconditions:
     - `agents.defaults.typingMode` set to `"message"`
     - `agents.defaults.typingIntervalSeconds` set to `6`
   - Send: `/video_sourcing top trending videos about AI`
   - Expect:
     - Start/progress/final messages still arrive.
     - Typing indicator clears promptly after terminal message (typically within 5-8 seconds).

## Channel matrix

1. Telegram DM with `/video_sourcing`.
2. Telegram free-form.
3. Non-Telegram channel (for example Discord) with block/partial streaming fallback.
