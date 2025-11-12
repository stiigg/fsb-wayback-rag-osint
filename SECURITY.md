## Security Contract

- Determinism == vulnerability. Retrieval is stochastic.
- Ingest is the trust boundary; retrieval is the leak boundary.
- “Guardrails” do not count as controls. Only structural defenses do.
- Unsafe flags require a SECURITY_RISK_JUSTIFICATION.md in the PR.

### Runtime Defaults
- Read-only, no network egress (except allowlisted LLMs), no prompt/chunk logs.
- Stochastic top-K + snippet-only citations; doc IDs/paths hidden by default.
- Dual-index consensus (FAISS ∩ HNSW). If overlap < τ: refuse.
- Provenance verified (hashes + Merkle root) pre-answer; on fail: refuse.

### Merge Gates
- All red-team tests must pass.
- Any change to ingest policy, retriever params, or provenance code requires 2 approvers (one security owner).

Owners:

* **Security**: `@maintainer-sec`
* **Data/Provenance**: `@maintainer-prov`
* **Retriever/Index**: `@maintainer-ret`
