# Security Policy

## Reporting A Vulnerability

Please report vulnerabilities privately to the maintainers before opening a public issue.

- Preferred: GitHub Security Advisories (private report)
- Include: impact, reproduction steps, affected version/commit, and any proof of concept

Maintainers will acknowledge reports as quickly as possible and provide status updates during triage.

## Supported Versions

The latest `main` branch is supported for security updates.

## Hardening Notes

- Rust gRPC server supports API-key auth and per-workspace rate limiting.
- Python embedding HTTP/gRPC endpoints support API-key auth and request limits.
- Full API keys and raw vectors are intentionally not logged.
