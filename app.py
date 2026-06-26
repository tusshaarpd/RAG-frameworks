"""Framework Arena - one task, five frameworks.

Upload a PDF, ask a question, and watch the same RAG task run through five
different architectures. The lesson is in the side-by-side traces: what each
framework did, what it gave you, and what it cost.

Run with:  streamlit run app.py   (needs OPENAI_API_KEY in the environment)
"""
import importlib
import importlib.util
import tempfile
import traceback
from pathlib import Path
from time import perf_counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load OPENAI_API_KEY (and any other vars) from a local .env file if present.
# Real environment variables always win over .env values.
load_dotenv(Path(__file__).parent / ".env")

from core import llm
from core.trace import ArenaResult, TraceCollector

st.set_page_config(page_title="Framework Arena", page_icon="🏟️",
                   layout="wide")

# ---------------------------------------------------------------------------
# Arena registry: module + required pip packages + architecture diagram.
# Availability is checked at startup so a missing dependency disables one
# checkbox instead of crashing the whole app.
# ---------------------------------------------------------------------------
ARENAS = [
    dict(key="raw", label="Arena 0 · Raw (no framework)",
         module="arenas.arena_raw",
         requires={},
         diagram="""digraph { rankdir=LR; node [shape=box, style=rounded];
            extract -> chunk -> embed -> retrieve -> generate }"""),
    dict(key="langchain", label="Arena 1 · LangChain",
         module="arenas.arena_langchain",
         requires={"langchain_openai": "langchain-openai",
                   "langchain_huggingface": "langchain-huggingface",
                   "langchain_community": "langchain-community"},
         diagram="""digraph { rankdir=LR; node [shape=box, style=rounded];
            loader -> splitter -> embeddings -> retriever -> prompt
            -> llm -> parser }"""),
    dict(key="llamaindex", label="Arena 2 · LlamaIndex",
         module="arenas.arena_llamaindex",
         requires={"llama_index.core": "llama-index-core",
                   "llama_index.llms.openai": "llama-index-llms-openai",
                   "llama_index.embeddings.huggingface":
                       "llama-index-embeddings-huggingface"},
         diagram="""digraph { rankdir=LR; node [shape=box, style=rounded];
            reader -> nodes -> index -> retriever -> tree_summarize }"""),
    dict(key="langgraph", label="Arena 3 · LangGraph",
         module="arenas.arena_langgraph",
         requires={"langgraph": "langgraph",
                   "langchain_openai": "langchain-openai"},
         diagram="""digraph { rankdir=LR; node [shape=box, style=rounded];
            retrieve -> grade; grade -> generate [label="relevant"];
            grade -> rewrite [label="not relevant"];
            rewrite -> retrieve [label="loop", style=dashed, color=red] }"""),
    dict(key="crewai", label="Arena 4 · CrewAI",
         module="arenas.arena_crewai",
         requires={"crewai": "crewai"},
         diagram="""digraph { rankdir=LR; node [shape=box, style=rounded];
            Researcher [label="Researcher\\n(+ PDF search tool)"];
            Writer [label="Writer"];
            Researcher -> Writer [label="handoff: findings"] }"""),
]

KIND_ICONS = {"ingest": "📄", "retrieve": "🔍", "llm": "🤖",
              "decision": "🔀", "handoff": "🤝"}


def missing_packages(arena) -> list[str]:
    return [pip_name for mod, pip_name in arena["requires"].items()
            if importlib.util.find_spec(mod) is None]


# ---------------------------------------------------------------------------
# Sidebar: inputs and run controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏟️ Framework Arena")
    st.caption("One task, five frameworks. Same PDF, same question - "
               "compare what each framework actually does.")

    if llm.api_key():
        st.success("OPENAI_API_KEY detected", icon="🔑")
    else:
        st.error("No API key. Set the env var and restart:\n\n"
                 "`setx OPENAI_API_KEY sk-...` (Windows)\n\n"
                 "`export OPENAI_API_KEY=sk-...` (mac/Linux)", icon="🔑")

    uploaded = st.file_uploader("PDF (max 20 pages used)", type="pdf")
    sample_path = Path(__file__).parent / "sample_data" / "sample.pdf"
    use_sample = st.checkbox("Use bundled sample PDF", value=uploaded is None,
                             disabled=not sample_path.exists())

    question = st.text_input("Question",
                             "What is integrated pest management?")

    st.divider()
    selected = []
    for arena in ARENAS:
        missing = missing_packages(arena)
        checked = st.checkbox(
            arena["label"], value=not missing, disabled=bool(missing),
            help=("Missing packages: " + ", ".join(missing)) if missing
            else None)
        if checked and not missing:
            selected.append(arena)

    cheap = st.toggle("Cheap mode (gpt-4o-mini instead of gpt-4o)",
                      value=True)
    run_clicked = st.button("▶ Run the arena", type="primary",
                            width="stretch",
                            disabled=not llm.api_key())

# ---------------------------------------------------------------------------
# Execute the selected arenas. Each runs in its own try/except: one arena
# failing must not kill the run.
# ---------------------------------------------------------------------------
if run_clicked:
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(uploaded.getvalue())
        tmp.close()
        pdf_path = tmp.name
    elif use_sample and sample_path.exists():
        pdf_path = str(sample_path)
    else:
        st.warning("Upload a PDF or tick the sample PDF box.")
        st.stop()

    llm.set_cheap_mode(cheap)
    results = {}
    progress = st.progress(0.0, text="Starting…")
    for i, arena in enumerate(selected):
        progress.progress(i / max(len(selected), 1),
                          text=f"Running {arena['label']}…")
        trace = TraceCollector(arena=arena["key"])
        t0 = perf_counter()
        try:
            module = importlib.import_module(arena["module"])
            results[arena["key"]] = module.run(pdf_path, question, trace)
        except Exception:
            results[arena["key"]] = ArenaResult(
                answer="", steps=trace.steps,
                latency_ms=int((perf_counter() - t0) * 1000),
                framework_loc=0, architecture_note="",
                llm_calls=trace.llm_calls, error=traceback.format_exc())
    progress.empty()
    st.session_state["results"] = results
    st.session_state["question"] = question

# ---------------------------------------------------------------------------
# Render results: scorecard, per-arena tabs, compare tab
# ---------------------------------------------------------------------------
results = st.session_state.get("results")
if not results:
    st.markdown(
        "## How this works\n"
        "1. Pick a PDF (or use the bundled sample) and type a question.\n"
        "2. Hit **Run the arena** - the same task executes through every "
        "selected framework.\n"
        "3. Compare the traces: steps taken, LLM calls made, latency paid, "
        "and lines of code each framework cost.\n\n"
        "Try asking *'What is the capital of France?'* to watch LangGraph's "
        "rewrite loop fire when retrieval fails.")
    st.stop()

ran = [a for a in ARENAS if a["key"] in results]

# --- Scorecard: the at-a-glance verdict -----------------------------------
st.subheader("Scorecard")
rows = []
for arena in ran:
    r = results[arena["key"]]
    rows.append({
        "Framework": arena["label"],
        "Status": "❌ failed" if r.error else "✅ ok",
        "Answer chars": len(r.answer),
        "Latency (ms)": r.latency_ms,
        "Steps": len(r.steps),
        "LLM calls": r.llm_calls,
        "Lines of code": r.framework_loc,
        "Architecture": r.architecture_note,
    })
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# --- Tabs: one per arena + Compare -----------------------------------------
tabs = st.tabs([a["label"] for a in ran] + ["⚖️ Compare"])

for tab, arena in zip(tabs, ran):
    r = results[arena["key"]]
    with tab:
        if r.error:
            st.error("This arena failed - the others kept running.")
            st.code(r.error)
        else:
            st.success(r.answer)

        st.markdown("**Architecture**")
        st.graphviz_chart(arena["diagram"])

        st.markdown("**Trace timeline**")
        for step in r.steps:
            indent = "&nbsp;" * 6 * step.indent
            retry = " 🔁 retry" if step.indent > 0 else ""
            label = (f"{KIND_ICONS.get(step.kind, '▫️')} {step.order}. "
                     f"{step.name} · {step.duration_ms} ms{retry}")
            with st.expander(label):
                if step.indent:
                    st.caption(f"loop iteration {step.indent + 1}")
                st.markdown(f"{indent}**in:** {step.input_preview or '—'}",
                            unsafe_allow_html=True)
                st.markdown(f"{indent}**out:** {step.output_preview or '—'}",
                            unsafe_allow_html=True)

# --- Compare tab: answers side by side + cost charts -----------------------
with tabs[-1]:
    st.markdown(f"**Question:** {st.session_state.get('question', '')}")
    cols = st.columns(len(ran))
    for col, arena in zip(cols, ran):
        r = results[arena["key"]]
        with col:
            st.markdown(f"**{arena['label']}**")
            if r.error:
                st.error("failed")
            else:
                st.info(r.answer)

    chart_df = pd.DataFrame({
        "arena": [a["label"].split("·")[1].strip() for a in ran],
        "latency_ms": [results[a["key"]].latency_ms for a in ran],
        "llm_calls": [results[a["key"]].llm_calls for a in ran],
    }).set_index("arena")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Latency per arena (ms)**")
        st.bar_chart(chart_df["latency_ms"])
    with c2:
        st.markdown("**LLM calls per arena**")
        st.bar_chart(chart_df["llm_calls"])
    st.caption("The real question every chart answers: what did I gain and "
               "what did I pay by using this framework?")
