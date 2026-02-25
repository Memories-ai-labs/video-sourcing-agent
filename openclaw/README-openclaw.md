# OpenClaw Skill Integration

This integration installs the `video-sourcing` skill for OpenClaw and runs a pinned self-bootstrap runtime.

Default runtime source:

- `https://github.com/Memories-ai-labs/video-sourcing-agent`
- pinned tag: `v0.2.3`
- managed path: `~/.openclaw/data/video-sourcing-agent/v0.2.3`

`VIDEO_SOURCING_AGENT_ROOT` is optional and only needed when you want to override the managed runtime with a local checkout.

## Prerequisites

1. OpenClaw installed and channel(s) configured.
2. Host binaries: `git` and `uv`.
3. API keys in OpenClaw global `env.vars`:
   - `GOOGLE_API_KEY` (required)
   - `YOUTUBE_API_KEY` (required)
   - `MEMORIES_API_KEY` / `EXA_API_KEY` / `APIFY_API_TOKEN` (optional)
4. Host runtime execution enabled:

```json5
{
  agents: {
    defaults: {
      sandbox: {
        mode: "off",
      },
    },
  },
}
```

## Install from this repository

1. Install the skill:

```bash
bash /Users/samuelzhang/Documents/GitHub/video-sourcing-agent/openclaw/install_skill.sh
```

Use `--copy` to copy instead of symlink:

```bash
bash /Users/samuelzhang/Documents/GitHub/video-sourcing-agent/openclaw/install_skill.sh --copy
```

2. Add env vars and enable the skill in `~/.openclaw/openclaw.json`:

```json5
{
  env: {
    vars: {
      GOOGLE_API_KEY: "your_google_api_key",
      YOUTUBE_API_KEY: "your_youtube_data_api_key",
      MEMORIES_API_KEY: "optional_memories_key",
      EXA_API_KEY: "optional_exa_key",
      APIFY_API_TOKEN: "optional_apify_token",
    },
  },
  skills: {
    entries: {
      "video-sourcing": {
        enabled: true,
      },
    },
  },
}
```

3. Merge the rest of settings from [openclaw_config.example.json5](/Users/samuelzhang/Documents/GitHub/video-sourcing-agent/openclaw/openclaw_config.example.json5).

4. Restart OpenClaw Gateway.

## Optional local override

If you want to run against a local development checkout instead of the pinned managed runtime, set:

```json5
{
  skills: {
    entries: {
      "video-sourcing": {
        enabled: true,
        env: {
          VIDEO_SOURCING_AGENT_ROOT: "/absolute/path/to/local/video-sourcing-agent",
        },
      },
    },
  },
}
```

## Usage

Slash command:

`/video_sourcing Find the fastest-growing tech videos in the past 48 hours`

Free-form:

`Show me trending TikTok videos about mindful movement this week`

`/video_sourcing` deterministic UX behavior:

1. Immediate start message: `Starting video sourcing...`
2. Throttled middle progress messages when runtime reaches 5+ seconds.
3. Final terminal message (`complete`, `clarification_needed`, or `error`).

## Troubleshooting

1. Missing `git` or `uv`
   - Install both on the OpenClaw host.
2. Missing required API keys
   - Set `GOOGLE_API_KEY` and `YOUTUBE_API_KEY` in global `env.vars`.
3. Bootstrap clone/sync failures
   - Verify host network access to GitHub and that tag `v0.2.3` exists.
4. `/video_sourcing` returns only final response without progress
   - Restart OpenClaw Gateway after skill updates.
5. Override path invalid
   - Ensure `VIDEO_SOURCING_AGENT_ROOT` points to a valid repository directory.
