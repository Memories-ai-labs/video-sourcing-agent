# video-sourcing-openclaw-skill

Standalone OpenClaw skill package for deterministic self-bootstrap execution.

Default runtime source:

- `https://github.com/Memories-ai-labs/video-sourcing-agent`
- pinned tag: `v0.2.0`
- managed path: `~/.openclaw/data/video-sourcing-agent/v0.2.0`

`VIDEO_SOURCING_AGENT_ROOT` is optional and only for local development override.

## Requirements

1. Host binaries on OpenClaw machine:
   - `git`
   - `uv`
2. Required API keys in OpenClaw global env:
   - `GOOGLE_API_KEY`
   - `YOUTUBE_API_KEY`
3. Sandbox off for host runtime flow.

## Install

1. Link or copy the skill folder to OpenClaw:

```bash
mkdir -p ~/.openclaw/skills
ln -sfn \
  /Users/samuelzhang/Documents/GitHub/video-sourcing-agent/openclaw/packages/video-sourcing-openclaw-skill/skills/video-sourcing \
  ~/.openclaw/skills/video-sourcing
```

2. Add env vars and enable skill in `~/.openclaw/openclaw.json`:

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

3. Keep runtime tools allowlist compatible with this workflow:

```json5
{
  tools: {
    allow: ["exec", "process", "read", "memory_search"],
    deny: ["write", "edit", "apply_patch", "browser", "canvas", "cron", "gateway"],
  },
}
```

4. Ensure sandbox is disabled:

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

5. Restart OpenClaw Gateway and use:

`/video_sourcing Find trending UGC videos for SaaS this week`

## Optional local override

Use this only if you want to run a local checkout instead of managed pinned runtime:

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

## Runtime behavior

`/video_sourcing` deterministic runner mode:

1. Immediate start message.
2. Throttled progress updates after 6 seconds.
3. Final terminal result (`complete`, `clarification_needed`, `error`).

Free-form behavior remains non-strict.

## Troubleshooting

1. `Required binary not found on PATH: git` or `uv`
   - Install missing binaries on host.
2. `Required environment variable is not set: GOOGLE_API_KEY` / `YOUTUBE_API_KEY`
   - Add required keys in global env vars.
3. Clone/sync errors during first run
   - Confirm host can access GitHub and fetch tag `v0.2.0`.
4. No progress updates shown
   - Restart gateway and verify channel stream mode settings.
