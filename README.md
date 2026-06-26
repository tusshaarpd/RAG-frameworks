# Framework Arena

**One task. Five frameworks. Same PDF. Same question. Same model.**

Framework Arena is a learning and demo tool that makes the differences between AI agent frameworks *visible*, not theoretical. Instead of reading blog posts that each crown a different winner, you run the identical RAG task through five different architectures simultaneously and compare what each one actually did: every step taken, every LLM call made, every millisecond spent, and every line of code it cost.

---

## What problem does this solve?

Every "which AI framework should I use?" post picks one and declares it the best. That answer is useless because the right framework depends entirely on what you are building.

Framework Arena shows you the trade-off directly:

- **Raw Python** proves that RAG is just five function calls — no framework needed
- **LangChain** shows what you gain (composability) and lose (debuggability) from a pipe abstraction
- **LlamaIndex** shows what happens when retrieval quality is the first-class concern
- **LangGraph** shows what becomes possible when your pipeline can loop and branch
- **CrewAI** shows what it feels like to describe *who* is doing the work instead of *what* the code does

The lesson is in the side-by-side trace: what each framework did, what it gave you, and what it cost.

---

## How it works

```
Upload a PDF  →  Type a question  →  Hit "Run the arena"
                                           ↓
         ┌─────────┬──────────┬───────────┬──────────┬─────────┐
         │  Raw    │LangChain │ LlamaIndex │LangGraph │ CrewAI  │
         │         │          │            │          │         │
         │ extract │ loader   │ reader     │ retrieve │ Researcher
         │ chunk   │ splitter │ nodes      │ grade    │    ↓    │
         │ embed   │ embedder │ index      │ rewrite? │ Writer  │
         │ retrieve│ retriever│ query      │ generate │         │
         │ generate│ llm      │ synthesize │          │         │
         └─────────┴──────────┴────────────┴──────────┴─────────┘
                                           ↓
         Scorecard: latency · LLM calls · lines of code · answer
         Trace timeline: every step, duration, input, output
         Architecture diagram: the actual data flow for each framework
```

Each arena runs in its own isolated try/except — one framework failing never kills the run. The failure renders in that arena's tab and the others continue.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/tusshaarpd/RAG-frameworks.git
cd RAG-frameworks
pip install -r requirements.txt
```

> All embeddings run locally on CPU using `all-MiniLM-L6-v2`. No GPU required. The only paid service is OpenAI.

### 2. Set your OpenAI API key

```bash
# macOS / Linux
export OPENAI_API_KEY=sk-...

# Windows (Command Prompt)
setx OPENAI_API_KEY sk-...
# Then open a new terminal window for it to take effect

# Or create a .env file in the project root (gitignored)
echo OPENAI_API_KEY=sk-... > .env
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. A bundled sample PDF about crop protection is included, so the app works immediately without uploading anything.

---

## Using the app

### Sidebar controls

| Control | Description |
|---|---|
| PDF uploader | Upload any PDF (first 20 pages are used) |
| Use bundled sample PDF | Tick this to use the included sample without uploading |
| Question | The question all frameworks will answer |
| Arena checkboxes | Select which frameworks to run (missing packages auto-disable) |
| Cheap mode | On = `gpt-4o-mini`, Off = `gpt-4o` |
| Run the arena | Executes all selected frameworks sequentially |

### Results area

**Scorecard** — at-a-glance comparison table showing status, answer length, latency, step count, LLM call count, and lines of code for each framework.

**Per-arena tabs** — for each framework:
- The final answer
- Architecture diagram (the actual data flow)
- Trace timeline — every step with duration, input preview, and output preview. LangGraph retries are indented so loop iterations are visually obvious.

**Compare tab** — answers side by side, plus bar charts for latency and LLM calls across all frameworks.

---

## The five frameworks

### Arena 0 · Raw Python

> *"RAG is just five function calls."*

No framework. Just `pypdf` → fixed-size chunks → `all-MiniLM-L6-v2` embeddings → flat FAISS index → one OpenAI call. ~60 lines of code. Total visibility — every byte that moves is a line you can read. This is what every other framework is wrapping.

**Use it to:** understand what is actually happening before you add abstraction.

---

### Arena 1 · LangChain

> *"Every LLM app is a chain of swappable components."*

LangChain's signature is the LCEL `|` pipe operator: `loader | splitter | embedder | retriever | prompt | llm | parser`. Each component is a Lego brick you can swap with one import change. A callback system observes every component without modifying it.

**What it gives you:** pre-built loaders for dozens of file types, composable chains, easy component swapping.  
**What it costs you:** an abstraction tax — when something breaks, you debug LangChain's wrappers, not your own code. A straight chain cannot loop or branch.

**Use it for:** linear pipelines where composability and component swapping matter.

---

### Arena 2 · LlamaIndex

> *"The hard part of RAG is the data and retrieval quality — the LLM step is almost incidental."*

Documents become `Nodes` — chunks that carry metadata, relationships, and per-node similarity scores. The `tree_summarize` response mode recursively summarizes retrieved chunks up a tree instead of stuffing them all into one prompt. Retrieval quality is the first-class design goal.

**What it gives you:** sentence-aware chunking, per-node similarity scores surfaced out of the box, advanced synthesis modes, a single reader call for any file type.  
**What it costs you:** a parallel universe of nouns (`Node`, `Settings`, `StorageContext`) and global singleton config that can surprise you in multi-model apps.

**Use it for:** long or complex documents where retrieval quality is the bottleneck.

---

### Arena 3 · LangGraph

> *"Real LLM apps are stateful graphs that branch and loop — not straight lines."*

LangGraph adds what LangChain cannot express: cycles. You declare a `StateGraph` with typed state, node functions, and conditional edges. The graph runs a state machine: retrieve → grade relevance → if bad, rewrite the query and loop back → if good (or attempts exhausted), generate. A straight LCEL chain literally cannot do this.

**What it gives you:** legal cycles, conditional routing, explicit state you can inspect between every node, self-correcting retrieval without custom loop code.  
**What it costs you:** ceremony — define a state schema, node functions, and edge wiring before any RAG happens. Roughly double the lines of code for the same happy path, and 2–4 extra LLM calls per run (grading + optional rewrite).

**Use it for:** agentic workflows with branching logic, retries, or validation steps before generation.

---

### Arena 4 · CrewAI

> *"Work gets done by a team. Describe who the agents are, not what the code does."*

You cast roles instead of writing pipelines. A `Researcher` agent (with a PDF search tool) finds relevant passages, then hands its findings to a `Writer` agent who composes the answer. The role, goal, and backstory *become* the system prompt. The handoff is automatic — the Writer receives the Researcher's output as context with no wiring code.

**What it gives you:** agent-to-agent handoff, tool-use loops where the agent decides how many times to search, natural language task description.  
**What it costs you:** control and tokens. The agents decide how many LLM calls to make — the same question typically costs 4–8x the raw arena's single call. The prompts are CrewAI's, not yours.

**Use it for:** tasks that genuinely split into specialist roles (research + writing, analysis + review).

---

## Three suggested demo questions

| Question | What to watch |
|---|---|
| `What is integrated pest management?` | The happy path. All five arenas answer well. Compare latency, step count, and LLM calls in the scorecard. |
| `What is the capital of France?` | Unanswerable from the PDF. In the LangGraph tab, watch the grader say NO, the query get rewritten, and the retrieve → grade → rewrite **loop fire visibly** in the trace timeline. |
| `When should a farmer spray for cereal aphids?` | A detail question. Compare retrieval: LlamaIndex surfaces per-node similarity scores. Check whether the threshold (5 aphids per tiller) makes it into each framework's answer. |

---

## Project structure

```
RAG-frameworks/
├── app.py                      # Streamlit UI: sidebar, scorecard, tabs, compare
├── requirements.txt
├── .env.example                # Copy to .env and add your key
├── FRAMEWORKS.md               # Deep-dive written comparison of all frameworks
│
├── core/
│   ├── ingest.py               # pypdf extraction + fixed-size chunking
│   ├── embeddings.py           # all-MiniLM-L6-v2 + FAISS helpers (shared, CPU)
│   ├── llm.py                  # single place where the OpenAI model is chosen
│   └── trace.py                # TraceCollector, TraceStep, ArenaResult
│
├── arenas/
│   ├── arena_raw.py            # Arena 0: plain Python baseline
│   ├── arena_langchain.py      # Arena 1: LCEL pipe + BaseCallbackHandler
│   ├── arena_llamaindex.py     # Arena 2: Nodes, VectorStoreIndex, tree_summarize
│   ├── arena_langgraph.py      # Arena 3: StateGraph with grade/rewrite cycle
│   ├── arena_crewai.py         # Arena 4: Researcher → Writer crew
│   └── arena_adk.py            # Stub — Google ADK (not yet implemented)
│
└── sample_data/
    ├── sample.pdf              # Bundled 3-page PDF on crop protection
    └── generate_sample.py      # Regenerate the sample PDF with fpdf2
```

### Reading the arena code

Each `arenas/arena_*.py` file is under 150 lines and opens with the same three-part comment block:

```python
# What this framework believes: ...
# What it gave us for free: ...
# What it cost us: ...
```

Those comments are the lesson. The code styles are deliberately not homogenized — the differences in idiom between `chain.invoke(question)` and `graph.compile().invoke({...})` and `crew.kickoff()` are the point.

---

## Design decisions

**Why only OpenAI?** One API key keeps setup friction low. The embeddings are local (MiniLM), so you only pay for generation calls. Swapping the LLM in any arena is one line change in `core/llm.py`.

**Why FAISS?** It runs in-memory on CPU with no server. Every arena uses the same FAISS index (or LangChain/LlamaIndex wrappers over the same MiniLM model), so retrieval differences between arenas are about framework behavior, not index quality.

**Why cap at 20 pages?** To keep latency and cost sane during demos. The limit is in `core/ingest.py` and trivially removable.

**Why sequential execution?** Arenas run one after another (not in parallel) so the progress bar is meaningful and the Streamlit event loop stays simple. For a production benchmark you would parallelize.

---

## Extending the project

**Add a new arena:**
1. Create `arenas/arena_myframework.py` with a `run(pdf_path, question, trace) -> ArenaResult` function
2. Add a dict entry to the `ARENAS` list in `app.py`
3. Specify the `requires` dict so missing packages auto-disable the checkbox

**Swap the embedding model:**
Edit `core/embeddings.py`. The model name is defined once and shared across all arenas that use the raw FAISS path.

**Use a different LLM provider:**
Edit `core/llm.py`. Arenas that use LangChain or LlamaIndex wrappers will need their own model swap in the arena file.

---

## Requirements

- Python 3.10+
- OpenAI API key (`OPENAI_API_KEY`)
- CPU only — no GPU required
- ~2 GB disk for model weights (downloaded on first run by `sentence-transformers`)

Install everything with:

```bash
pip install -r requirements.txt
```

Key version pins: `numpy<2` (required by PyTorch wheels), `langchain 0.3.x` (community package not yet on 1.x).
