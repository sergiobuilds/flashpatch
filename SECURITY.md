# Security policy

## 1 Supported version

The latest commit on the default branch is the supported development version. Release tags identify immutable release snapshots.

## 2 Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Include the affected command, a minimal reproducer, expected impact, and whether crafted media or project files are required. Do not open a public issue for an unpatched vulnerability.

Reports involving path traversal, receipt verification bypass, command execution, unsafe archive handling, or a false `PASS` receive the highest priority. You can expect acknowledgement within 72 hours and a status update after reproduction.

## 3 Trust boundary

Godot projects, videos, frame archives, traces, contracts, and receipts are untrusted inputs. Run third-party game projects in an isolated environment. A FlashPatch receipt is engineering evidence about the declared inputs; it is not a digital signature or clinical certification.

## 4 Change history

- 2026-08-26: published the initial vulnerability reporting and trust-boundary policy.
