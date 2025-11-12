# Contributing

Follow this checklist before shipping any change:

* [ ] `SECURITY.md` present; flags default secure.
* [ ] `ingest_policy.yaml` enforced; quarantine path exists.
* [ ] `merkle/manifest.jsonl` written on ingest; `--verify-provenance` enabled.
* [ ] Dual indices configured; `τ ≥ 3`; refusal message wired.
* [ ] MIA defenses on: subsampled top-K, jitter, snippet-only, QPS throttle.
* [ ] Red-team tests green; CI required for merge.
* [ ] `--audit` prints flags, Merkle root, and KB stats.
* [ ] `--reveal-sources` not used in production images.
