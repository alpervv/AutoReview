# AutoReview

A 3-stage AI review pipeline that analyses any codebase using **Claude Opus 4.6** and **Codex 5.3** and produces a single, arbitrated "supreme report".

## How it works

```
Your query
  │
  ├── Stage 1 (parallel)   Claude + Codex independently analyse the codebase
  ├── Stage 2 (parallel)   Each model cross-reviews the other's findings
  └── Stage 3              Codex arbitrates both reports → supreme report
```

The tool is **read-only** — it never modifies the codebase being analysed.

## Requirements

- Python 3.10+
- [`claude`](https://claude.ai/code) CLI installed and authenticated
- [`codex`](https://github.com/openai/codex) CLI installed and authenticated
- `pip install rich`

## Usage

```bash
pip install rich
cd /path/to/any/codebase
python /path/to/AutoReview/autoreview.py
```

Enter a query at the prompt, e.g.:

```
Enter your query: summarize the main purpose of this codebase
```

The report is saved as `review_YYYYMMDD_HHMMSS.md` in the current working directory.
