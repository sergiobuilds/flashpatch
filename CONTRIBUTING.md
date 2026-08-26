# Contributing to FlashPatch

## 1 Scope

FlashPatch accepts changes that improve reproducible visual-risk detection, source attribution, fail-closed patch validation, public fixtures, or release integrity. A change must not broaden an engine claim beyond the evidence exercised by the repository.

## 2 Development setup

```bash
git clone https://github.com/sergiobuilds/flashpatch-public.git
cd flashpatch-public
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/check_public_release.py
```

Godot renderer tests additionally require Godot 4 and a display. On headless Linux, run the public proof with `xvfb-run -a flashpatch godot-demo`.

## 3 Change requirements

- Add or update a test for behavior changes.
- Keep `PASS`, `SAFE`, `FAIL`, and `INCONCLUSIVE` semantics fail-closed.
- Never convert missing, ambiguous, or malformed evidence into success.
- Use redistributable inputs and record their provenance.
- Keep receipts machine-readable and bind public artifacts by SHA-256.
- Update the README when a command, supported scope, or evidence claim changes.

## 4 Pull requests

Describe the user-visible outcome, the evidence used, and the exact verification commands. Keep unrelated refactors separate. A pull request is ready when the full public suite and release audit pass from a clean checkout.

## 5 Change history

- 2026-08-26: added the initial public contribution contract.
