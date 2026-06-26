# RAG Framework Comparison

Same task. Same PDF. Same question. Same model. Five completely different architectures.

This document explains what each framework in Framework Arena believes, what it gives you for free, and what it costs you.

---

## The Mental Model That Matters

Every framework below is wrapping the same five operations:

```
extract → chunk → embed → retrieve → generate
```

The difference is not *what* they do — it is *how much opinion they bring* about how those steps should be structured, composed, and extended.

---

## Arena 0 · Raw Python (no framework)

**Core belief:** RAG is just function calls. There is no framework.

### Architecture
```
extract → chunk → embed → retrieve → generate
```

### What it does
- Reads the PDF with `pypdf`
- Splits text into fixed 800-char windows with 100-char overlap
- Embeds chunks with `all-MiniLM-L6-v2` (local, no API cost)
- Builds a flat FAISS index for cosine similarity search
- Retrieves top-4 chunks and passes them to one OpenAI call

### What you get for free
Nothing — and that is the point. Every byte that moves is a line of code you can read. Total observability, zero mystery.

### What it costs you
You hand-rolled chunking, retrieval, and prompt assembly. For this task that is ~60 lines. Every other framework below is some opinion about how to hide these same 60 lines.

### When to use it
- Learning what RAG actually does under the hood
- Performance-critical paths where you need full control
- Situations where framework overhead is not justified

---

## Arena 1 · LangChain

**Core belief:** Every LLM app is a chain — components piped left to right, each one a swappable Lego brick.

### Architecture
```
PyPDFLoader → RecursiveCharacterTextSplitter → HuggingFaceEmbeddings
→ FAISS retriever | prompt | ChatOpenAI | StrOutputParser
```

### What it does
- Uses `PyPDFLoader` to load the PDF (one `Document` per page)
- Splits with `RecursiveCharacterTextSplitter` — tries `\n\n`, then `\n`, then spaces before cutting
- Builds a `FAISS` vectorstore through LangChain's wrapper
- Wires everything into an LCEL chain using the `|` pipe operator
- Observes retriever and LLM activity through a `BaseCallbackHandler`

### What you get for free
- Pre-built loaders and splitters for dozens of file types
- The `|` pipe syntax for composing components declaratively
- A callback system that observes every component without modifying them
- Easy swap of any component (change one import to switch the LLM, embedder, or vectorstore)

### What it costs you
- An abstraction tax: five imports from four packages to do what Raw does with none
- When something misbehaves, you debug LangChain's wrappers, not your own code
- A straight chain cannot loop or branch — it is a line, not a graph

### When to use it
- Rapid prototyping with many swappable components
- Apps that are genuinely linear: load → retrieve → generate
- Teams that want a shared vocabulary for LLM pipelines

---

## Arena 2 · LlamaIndex

**Core belief:** The hard part of RAG is the *data*, not the chain. The query engine is almost an afterthought bolted onto a great index.

### Architecture
```
SimpleDirectoryReader → SentenceSplitter → Nodes
→ VectorStoreIndex → query engine (tree_summarize)
```

### What it does
- Loads the PDF with `SimpleDirectoryReader` — one call handles any file type
- Splits into `Nodes` using sentence-aware boundaries (not just character count)
- Builds a `VectorStoreIndex` from the nodes
- Queries with `tree_summarize` response mode: recursively summarizes retrieved chunks up a tree instead of stuffing them into one prompt

### What you get for free
- `Nodes` as first-class objects: chunks that carry metadata, relationships, and per-node similarity scores surfaced automatically
- `tree_summarize` for long-document synthesis that LangChain requires you to assemble by hand
- Native observability through `LlamaDebugHandler` — LLM call counts are honest
- One reader call for any file type

### What it costs you
- A parallel universe of nouns: `Node`, `Settings`, `ServiceContext`, `StorageContext`
- Global singleton config (`Settings.llm`, `Settings.embed_model`) that can surprise you in multi-model apps
- Steeper learning curve if you are coming from LangChain's mental model

### When to use it
- Long or complex documents where retrieval quality is the bottleneck
- Apps where you need per-chunk metadata and similarity scores
- Scenarios requiring advanced synthesis (multi-doc, hierarchical summarization)

---

## Arena 3 · LangGraph

**Core belief:** Real LLM apps are not lines — they are *graphs* with state. They branch, loop, and retry. Declare nodes, edges, and a state schema; the framework runs the machine.

### Architecture
```
retrieve → grade relevance
    ↓ relevant          ↓ not relevant (max 2 attempts)
  generate          rewrite query → retrieve (loop)
```

### What it does
- Ingests and indexes the PDF once, outside the graph
- Defines four node functions: `retrieve`, `grade`, `rewrite`, `generate`
- Wires them into a `StateGraph` with a `TypedDict` state schema
- After retrieval, a grader LLM call decides if the chunks are relevant
- If not relevant: rewrites the query and loops back to retrieve (up to 2 attempts)
- If relevant (or attempts exhausted): proceeds to generate

### What you get for free
- Legal cycles — a `retrieve → rewrite → retrieve` loop that LCEL literally cannot express
- Explicit state you can inspect between every node
- Conditional routing (`add_conditional_edges`) with named branches
- Self-correcting retrieval without any custom loop logic

### What it costs you
- Ceremony: define a state schema, node functions, and edge wiring before any RAG happens
- Roughly double the lines of code of Raw for the same happy path
- More LLM calls (grading + optional rewrite = 2–4 extra calls per run)

### When to use it
- Agentic RAG where retrieval quality must be validated before generating
- Workflows with branching logic, retries, or human-in-the-loop steps
- Any pipeline where "what happens next depends on what just happened"

---

## Arena 4 · CrewAI

**Core belief:** Work gets done by a *team*. You do not write a pipeline — you cast characters. Each agent has a role, a goal, and a backstory, and the framework turns that into a system prompt.

### Architecture
```
Researcher (role + goal + backstory + Search PDF tool)
    ↓ handoff: research findings
Writer (role + goal + backstory)
    ↓
  answer
```

### What it does
- Ingests and indexes the PDF, then wraps the search in a `@tool`
- Casts a `Researcher` agent: goal is to find relevant passages, has the search tool
- Casts a `Writer` agent: goal is to compose a grounded answer from the Researcher's notes
- Runs a sequential `Crew`: Researcher task → Writer task, with automatic context handoff

### What you get for free
- Agent-to-agent handoff: Writer automatically receives Researcher's output as context
- Tool-use loops: the Researcher decides how many times to call the search tool
- Prompt engineering through role/goal/backstory — you describe *who* the agent is, not *what* to say
- Natural language task description instead of code wiring

### What it costs you
- Control and tokens: the agents decide how many LLM calls to make, not you
- The same question costs several times Raw's single call
- The prompts driving the agents are CrewAI's, not yours — harder to tune precisely
- Non-deterministic call count makes cost estimation difficult

### When to use it
- Tasks that genuinely benefit from role specialization (research + writing, analysis + review)
- Rapid prototyping of multi-agent workflows without custom orchestration code
- Situations where you want agents to reason about *how* to use tools, not just *when*

---

## Side-by-Side Summary

| | Raw | LangChain | LlamaIndex | LangGraph | CrewAI |
|---|---|---|---|---|---|
| **Mental model** | Function calls | Chain / pipe | Data + index | State machine | Team of agents |
| **LLM calls (typical)** | 1 | 1 | 1–2 | 3–5 | 4–8+ |
| **Control** | Total | High | High | High | Low |
| **Can loop / retry** | Manual | No | No | Yes (built-in) | Yes (agent decides) |
| **Best at** | Visibility | Composability | Retrieval quality | Conditional workflows | Role-based tasks |
| **Hardest thing to debug** | Nothing (it's your code) | LangChain wrappers | Global config | State schema | Agent reasoning |
| **Lines of code (this task)** | ~60 | ~90 | ~80 | ~120 | ~90 |

---

## The Trade Every Abstraction Makes

```
Less code    ←————————————————→    Less control
More features ←————————————————→    More tokens
```

Raw Python sits at one extreme: maximum control, maximum visibility, minimum magic.
CrewAI sits at the other: minimum code to describe a task, minimum visibility into execution.

LangChain, LlamaIndex, and LangGraph occupy the middle — each optimizing for a different axis:
- LangChain optimizes for **composability**
- LlamaIndex optimizes for **retrieval quality**
- LangGraph optimizes for **workflow complexity**

**The right framework is the one whose trade matches your problem.**
If your retrieval is the bottleneck, reach for LlamaIndex. If your workflow has branches and loops, reach for LangGraph. If you are still figuring out what your pipeline should even do, start with Raw.
