## Outcome

Describe the user-visible result.

## Evidence

List the fixture, receipt, or source used to justify the change.

## Verification

```bash
python -m pytest -q
python scripts/check_public_release.py
```

- [ ] Behavior changes include tests.
- [ ] Public claims match checked-in evidence.
- [ ] Hazardous media is labeled and does not autoplay.
- [ ] No private paths, credentials, or non-redistributable inputs are included.
