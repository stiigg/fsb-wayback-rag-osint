# FSB RAG CLI

This is a minimal Retrieval-Augmented Generation (RAG) stack on top of your
Wayback/FSB archive JSONL (e.g. the output of your `fsb_wayback_scraper.py`
and optional diff layer).

## Files

- `fsb_build_index.py` – builds a FAISS vector index + metadata from a JSONL archive.
- `fsb_rag_cli.py` – interactive CLI to query the archive using an LLM with RAG.
- `requirements.txt` – Python dependencies.

## Prerequisites

- Python 3.9+
- A JSONL file with one snapshot per line, containing at least:
  - `extracted.title`
  - `extracted.headings` (list of {tag, text})
  - `extracted.text`
  - Optional: `fear_score`, `is_change`, `date_iso`, `original_url`, `wayback_url`

This matches the structure emitted by your upgraded `fsb_wayback_scraper.py`
and (optionally) your narrative diff script.

## Install dependencies

```bash
pip install -r requirements.txt
```

## 1. Build the index

Example:

```bash
python fsb_build_index.py \
  --input-jsonl data/fsb_snapshots_diff.jsonl \
  --index-out data/fsb_faiss.index \
  --meta-out data/fsb_meta.jsonl \
  --model-name sentence-transformers/all-mpnet-base-v2
```

## 2. Run the RAG CLI

The CLI expects an OpenAI-compatible API. Set your key:

```bash
export OPENAI_API_KEY="sk-..."
```

Then:

```bash
python fsb_rag_cli.py \
  --index data/fsb_faiss.index \
  --meta data/fsb_meta.jsonl \
  --model-name sentence-transformers/all-mpnet-base-v2 \
  --openai-model gpt-4o-mini
```

You’ll drop into an interactive shell. Example queries:

- `How did FSB's language about extremism and terrorism change between 2000 and 2014?`
- `Summarise changes to counter-terrorism narratives around the 2014 Crimea annexation.`
- `Which entities are repeatedly framed as external enemies in these snapshots?`

Type `:q` or `:quit` to exit.

## Notes

- This is deliberately simple and transparent:
  - FAISS for ANN search
  - SentenceTransformers for embeddings
  - OpenAI (or any OpenAI-compatible endpoint) for the LLM
- You can swap the embedding or LLM backend by editing the scripts,
  as long as you preserve the interface:
  - `embed(text: str) -> np.ndarray`
  - `llm_qa(context: str, question: str) -> str`

Use this as a baseline and harden / expand it as you need.
