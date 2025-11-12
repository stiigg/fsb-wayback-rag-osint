# FSB Wayback RAG (No Bullshit Edition)

This repo is now a blunt OSINT helper: load a metadata JSONL, clip overlapping hits, spit out snippets, and move on.

## Layout
- `fsb_wayback/utils.py` – runtime throttle + audit dump in one angry file.
- `fsb_wayback/retrieval.py` – top-k, snippet trimming, overlap math. Nothing else.
- `fsb_wayback/config.yaml` – default paths for people who forget flags.
- `fsb_rag_cli.py` – interactive shell that refuses nonsense instead of role-playing a SOC.
- `tests/` – sanity checks proving the slimmed logic still works.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run It
```bash
python fsb_rag_cli.py --meta data/meta.jsonl --manifest data/manifest.jsonl --mock-llm
```
- `--mock-llm` prints stitched snippets so you can run offline.
- Drop the flag when you wire your own model. Until then the CLI just refuses with `LLM disabled`.
- Defaults for `--meta`, `--manifest`, and `--top-k` live in `fsb_wayback/config.yaml` and can be overridden with `FSB_METADATA`, `FSB_MANIFEST`, or `FSB_TOP_K` environment variables.

The prompt loop prints answers, masks sources unless you ask for them, and refuses if provenance fails or the two toy indexes disagree.

## Audits
Need proof you ran it? Add `--audit` and we dump `audit_report.json` with the active knobs, the latest manifest entry, and the raw overlap trace. No dashboards, no jargon.
