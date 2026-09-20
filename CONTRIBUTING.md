# Contributing to FlashPatch

## 1 Scope

FlashPatch accepts changes that improve reproducible visual-risk detection, source attribution, fail-closed patch validation, public fixtures, or release integrity. A change must not broaden an engine claim beyond the evidence exercised by the repository.

## 2 Development setup

```bash
git clone https://github.com/sergiobuilds/flashpatch.git
cd flashpatch
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

## 5 Product decisions and roadmap

The default branch of this public repository owns FlashPatch's product design, development, validation and release contracts. The [README](README.md) owns the current product contract; [engine coverage](proof/engine-coverage.json) owns the machine-readable evidence boundaries. Neither a private plan nor an unpublished result changes those contracts.

Propose roadmap work in a [public issue](https://github.com/sergiobuilds/flashpatch/issues) with the problem, intended scope and observable acceptance criteria. An issue is a proposal until a maintainer accepts it; work is complete only when its implementation, relevant tests and contract updates are merged. Pull requests record the decision and evidence. Changes to engine support require redistributable evidence that contributors can inspect and reproduce. Current evidence limits remain in the README and coverage file rather than a second roadmap summary.

## 6 Maintainer access and publication

Maintainer access applies only to this repository. Contributing does not require organization membership, a private checkout or access to company strategy and customer records. Repository write or maintain access does not grant access to other repositories; maintainers request any permission changes from the repository owner.

Builds, tests and release validation must run from a public checkout with public dependencies. Do not add tokens, deploy keys, private reusable workflows or runners that expose private company repositories or workspaces. Review imported material for redistribution rights and confidential content before publishing it in commits, issues, pull requests, logs or artifacts. Keep private history out of public branches; bring only reviewed changes into this repository.

Use the [security policy](SECURITY.md) for vulnerability reports. Keep private evidence out of public reports and provide a redistributable reproducer when possible.
