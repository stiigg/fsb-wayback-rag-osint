# RAG Security Literature

This living document collates research and operational guidance informing the FSB Wayback OSINT security posture.

---

* **GDPR/NIS2 evidence**: provenance manifest, refusal logs, secure defaults, rate-limiting.
* **Controls → Risks**:

  * Stochastic retrieval → MIAs (black-box leakage).
  * Consensus/Refusal → poisoning/hijack steering.
  * Manifest verification → supply-chain tampering.
  * Quarantine + policy → ingestion hijack.
  * No logs + airgap → prompt/trace exfil.

Audit artifacts:

* `merkle/manifest.jsonl` (append-only),
* `audit_report.json` (`--audit`),
* CI red-team history.
