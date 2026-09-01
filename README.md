# Who's Grading the Analysts?

A multi-agent RAG pipeline that fact-checks Wall Street analyst reports against a company's actual SEC filings — using an LLM as the judge.

## The problem

Analyst reports are full of specific, checkable numbers: revenue figures, YoY growth rates, margins, cash flow. Those numbers are supposed to trace back to a company's official SEC filings (10-K, 10-Q), but in practice they don't always:

- Analysts misquote or misremember a figure from a prior report
- A number gets carried forward after a restatement
- "Services revenue" in one report means something slightly different in another
- Sloppy rounding or unit errors (billions vs millions) slip through

Nobody manually re-derives every number in every report against the underlying filing — it's too slow to do at scale, and by the time a discrepancy is caught (if it ever is), the report has already shaped a trading or investment decision. There's no automated "grader" checking analysts' claims against the primary source.

## The approach

This project builds that grader as a small pipeline of specialized agents, backed by retrieval over real SEC filings rather than an LLM's own (unreliable) memory of financial data:

1. **Ingest** ([ingest.py](ingest.py), [mcps/sec_edgar.py](mcps/sec_edgar.py)) — pull filings for a company directly from SEC EDGAR, chunk the text, and embed it into a local Chroma vector store.
2. **Claim Extractor** ([agents/claim_extractor.py](agents/claim_extractor.py)) — an LLM (GLM) reads a raw analyst report and pulls out every discrete, verifiable factual claim (e.g. splits "revenue was $85.3B, up from $69.1B" into two separate claims — current period and prior period — so each can be checked independently).
3. **Retriever** ([agents/rag_retriever.py](agents/rag_retriever.py)) — for each claim, hybrid search (semantic via Chroma + keyword via BM25, merged with Reciprocal Rank Fusion) pulls the most relevant chunks from the ingested filings. Claims with no sufficiently relevant filing content are marked `OUT_OF_SCOPE` instead of forcing a verdict.
4. **Judge** ([agents/judge.py](agents/judge.py)) — a *different* model family (Gemini, not GLM) compares the claim against the retrieved evidence and renders a calibrated verdict: `SUPPORTED`, `CONTRADICTED`, or `INSUFFICIENT_EVIDENCE`, with a confidence score and cited evidence. Using a different model family from the extractor reduces self-preference bias.
5. **Pipeline** ([pipeline.py](pipeline.py)) — orchestrates all of the above, fanning claims out across threads (I/O-bound LLM calls) and printing a verdict summary table.
6. **Eval** ([eval/run_eval.py](eval/run_eval.py)) — runs the pipeline against a hand-labeled claim set with known-correct verdicts and reports accuracy plus per-class precision/recall/F1, so changes to prompts, chunking, or retrieval can be measured rather than eyeballed.

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your OpenRouter key + EDGAR user agent

python ingest.py            # pull + embed SEC filings for a ticker
python pipeline.py "Apple Services revenue grew 12% YoY in Q3 2026"
python pipeline.py --file report.txt

python eval/run_eval.py     # measure accuracy against labeled claims
```

## Stack

- **LLMs**: GLM (claim extraction) + Gemini (judging), via OpenRouter
- **Retrieval**: Chroma (semantic) + BM25 (keyword), merged with Reciprocal Rank Fusion
- **Embeddings**: `sentence-transformers` (local, no API key)
- **Data source**: SEC EDGAR (10-K / 10-Q filings)
