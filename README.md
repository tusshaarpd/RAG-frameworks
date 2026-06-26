# 🏟️ Framework Arena — One Task, Five Frameworks

A learning and demo tool that makes the differences between agentic
frameworks **visible, not theoretical**. The exact same task — *upload a
PDF, ask a question, get a grounded answer* — runs through five
implementations, and you compare the traces side by side:

| Arena | Framework | The mental model |
|---|---|---|
| 0 | **Raw** (no framework) | RAG is just five function calls |
| 1 | **LangChain** | Everything is a chain of piped components |
| 2 | **LlamaIndex** | The data and retrieval quality are the product |
| 3 | **LangGraph** | Real apps are stateful graphs that branch and loop |
| 4 | **CrewAI** | Work is done by a team of role-played agents |
| 5 | Google ADK | Stub — see `arenas/arena_adk.py` |

For every arena the app shows the final answer, a step-by-step trace
timeline (with retrieved chunk previews and per-call latency), an
architecture diagram, lines of code, and LLM call count — so each
framework answers the only question that matters: **what did I gain and
what did I pay?**

## Setup

```bash
pip install -r requirements.txt

# one env var, OpenAI is the only paid service used
export OPENAI_API_KEY=sk-...        # mac/Linux
setx OPENAI_API_KEY sk-...          # Windows (then open a new terminal)

streamlit run app.py
```

Everything else runs locally on CPU: embeddings are
`sentence-transformers/all-MiniLM-L6-v2`, the vector store is in-memory
FAISS. No GPU, no other keys. "Cheap mode" (default on) uses
`gpt-4o-mini`; switching it off uses `gpt-4o`.

A 3-page sample PDF about crop protection is bundled
(`sample_data/sample.pdf`), so the app works out of the box. Regenerate it
with `python sample_data/generate_sample.py`.

## Three suggested demo questions

1. **"What is integrated pest management?"** — the happy path. All five
   arenas answer well; compare latency, step count, and LLM calls.
2. **"What is the capital of France?"** — unanswerable from the PDF.
   Watch the LangGraph tab: the grader says NO, the query gets rewritten,
   and the retrieve→grade→rewrite **loop fires visibly** in the trace.
3. **"When should a farmer spray for cereal aphids?"** — a detail
   question. Compare retrieval quality: LlamaIndex shows per-node
   similarity scores; check whether the threshold (5 aphids per tiller)
   makes it into each answer.

## Reading the code

Each `arenas/arena_*.py` module is under 150 lines and starts with the
same three-part comment block: *what this framework believes, what it
gave us for free, what it cost us*. Those comments are the lesson — the
code styles are deliberately **not** homogenized, because the differences
in idiom are the point.

```
framework-arena/
  app.py                  # Streamlit UI: sidebar, scorecard, tabs, compare
  core/
    trace.py              # TraceCollector / TraceStep / ArenaResult
    ingest.py             # pypdf extraction + naive chunking
    embeddings.py         # MiniLM + FAISS helpers (shared, CPU)
    llm.py                # the single place the OpenAI model is chosen
  arenas/
    arena_raw.py          # the baseline everything else is wrapping
    arena_langchain.py    # LCEL pipe + callbacks
    arena_llamaindex.py   # nodes, scores, tree_summarize
    arena_langgraph.py    # the grade/rewrite cycle
    arena_crewai.py       # Researcher -> Writer handoff
    arena_adk.py          # stub
  sample_data/sample.pdf
```

Notes: arenas run sequentially and one arena failing never kills the run —
the failure renders in that arena's tab. Missing optional dependencies
disable that arena's checkbox instead of crashing the app. PDFs are capped
at 20 pages to keep cost and latency sane.
