# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 1.1.x (`main`) | ✅ |
| < 1.1 | ❌ |

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's private reporting instead:
**Security** tab → **Report a vulnerability**.

You can expect an acknowledgement within a few days. Useful details: affected component (API, UI, a specific vector-store adapter), steps to reproduce, and impact.

## Notes for deployers

- The FastAPI service has no authentication; put it behind a gateway before exposing it publicly.
- `GEMINI_API_KEY` and database credentials are read from environment variables. Never commit them (`.env` is git-ignored).
- The demo downloads its data bundle from this repository's `release-assets` branch and verifies the SHA-256 before loading it.
