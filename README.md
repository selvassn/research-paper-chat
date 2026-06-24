# research-paper-chat

A small RAG system for asking questions over research papers. I built it to learn how to *evaluate* retrieval and answer quality properly — the chat loop is the easy part; the eval suite is the point.

## What it does

Ingests papers from the QASPER dataset into a vector store, retrieves the relevant chunks for a question, reranks them, and answers from that context with Claude.

## Stack

- **Voyage AI** — embeddings (`voyage-4-large`) and reranking (`rerank-2`)
- **Postgres + pgvector** — vector store
- **Anthropic Claude** (Haiku) — answer generation and the LLM-as-judge
- **QASPER** (`allenai/qasper`) — the question/answer dataset

## Evals

Retrieval and answer quality are measured separately, because a good answer score means nothing if retrieval never surfaced the evidence.

- **Retrieval** — hit-rate@k: does the right paper's chunk land in the top results.
- **Answer quality**, two ways:
  - **Token-F1** (SQuAD/QASPER style) — deterministic and cheap, but blind to paraphrasing and citation tokens.
  - **LLM-as-judge** — semantic correct/incorrect with a justification, to catch what token overlap misses.

The mean token-F1 comes out low (~0.1) here: the answers are verbose prose while the gold answers are short citation spans, so word overlap can't see that they mean the same thing. That gap is exactly why the judge exists.

## Running

Needs a Postgres instance with the `pgvector` extension and a `chunks` table, plus `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` in `.env`. DB connection is currently hardcoded to `postgresql://postgres:postgres@localhost:5432/papers`.

```bash
uv sync
uv run main.py        # ingest + embed the papers
uv run retrival.py    # ask a sample question
uv run eval.py        # run the eval suite
```