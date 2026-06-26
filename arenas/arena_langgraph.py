# ============================================================================
# ARENA 3: LANGGRAPH (the workflow controller)
# What this framework believes: real LLM apps are not lines, they are GRAPHS
#   with state - they branch, they loop, they retry. So you declare nodes,
#   edges, and a state object, and the framework runs the machine.
# What it gave us for free: a legal CYCLE (retrieve -> grade -> rewrite ->
#   retrieve again), conditional routing, and explicit state you can inspect
#   between every node. A straight LCEL chain literally cannot loop.
# What it cost us: ceremony. We define a State schema, node functions, and
#   edge wiring before any RAG happens - roughly double the code of the raw
#   arena for the same happy path.
# ============================================================================
from time import perf_counter
from typing import TypedDict

from core import embeddings, ingest, llm
from core.trace import ArenaResult, TraceCollector, count_loc

ARCHITECTURE_NOTE = (
    "Explicit state machine: retrieve -> grade -> (rewrite -> retrieve loop "
    "if chunks judged irrelevant, max 2 attempts) -> generate. The cycle and "
    "the grading decision are things a straight chain cannot express."
)

MAX_ATTEMPTS = 2


class GraphState(TypedDict):
    question: str            # the user's original question
    rewritten_question: str  # current working query (may be rewritten)
    chunks: list             # retrieved chunk texts
    grade: str               # 'yes'/'no' relevance verdict
    attempts: int            # retrieval attempts so far
    answer: str


def run(pdf_path: str, question: str, trace: TraceCollector) -> ArenaResult:
    from langchain_openai import ChatOpenAI  # LangGraph pairs with LangChain
    from langgraph.graph import END, StateGraph

    t0 = perf_counter()
    chat = ChatOpenAI(model=llm.model_name(), temperature=0)

    # Ingestion happens once, outside the graph - the graph is about control
    # flow over an already-indexed document.
    with trace.timed("ingest + index", "ingest", pdf_path) as t:
        text, pages, _ = ingest.extract_text(pdf_path)
        chunks = ingest.chunk_text(text)
        index = embeddings.build_index(chunks)
        t.output = f"{pages} pages -> {len(chunks)} chunks -> FAISS"

    # --- Node functions. Each gets state in, returns a state delta. -------
    def retrieve(state: GraphState):
        q = state["rewritten_question"]
        with trace.timed(f"node: retrieve (attempt {state['attempts'] + 1})",
                         "retrieve", q) as t:
            t.indent = state["attempts"]  # indent retries in the timeline
            hits = embeddings.search(index, chunks, q, k=4)
            t.output = " | ".join(f"[{s:.2f}] {c[:60]}" for c, s in hits)
        return {"chunks": [c for c, _ in hits],
                "attempts": state["attempts"] + 1}

    def grade(state: GraphState):
        prompt = (
            "Do these passages contain information that answers the "
            f"question? Reply only 'yes' or 'no'.\nQuestion: "
            f"{state['question']}\nPassages:\n" + "\n".join(state["chunks"]))
        with trace.timed("node: grade", "llm", prompt) as t:
            t.indent = state["attempts"] - 1
            verdict = chat.invoke(prompt).content.strip().lower()
            t.output = f"grade = {verdict}"
        return {"grade": "yes" if "yes" in verdict else "no"}

    def rewrite(state: GraphState):
        prompt = ("Rewrite this question to retrieve better passages from a "
                  f"document. Reply with the rewritten question only.\n"
                  f"Question: {state['rewritten_question']}")
        with trace.timed("node: rewrite", "llm", prompt) as t:
            t.indent = state["attempts"] - 1
            new_q = chat.invoke(prompt).content.strip()
            t.output = new_q
        return {"rewritten_question": new_q}

    def generate(state: GraphState):
        prompt = ("Answer using ONLY this context. If it lacks the answer, "
                  "say so.\nContext:\n" + "\n\n".join(state["chunks"]) +
                  f"\n\nQuestion: {state['question']}")
        with trace.timed("node: generate", "llm", prompt) as t:
            answer = chat.invoke(prompt).content
            t.output = answer
        return {"answer": answer}

    # The conditional edge - where LangGraph earns its keep.
    def route_after_grade(state: GraphState) -> str:
        if state["grade"] == "no" and state["attempts"] < MAX_ATTEMPTS:
            trace.add("edge decision", "decision", f"grade={state['grade']}",
                      f"grade said NO, rewriting query, attempt "
                      f"{state['attempts'] + 1}", indent=state["attempts"] - 1)
            return "rewrite"
        trace.add("edge decision", "decision", f"grade={state['grade']}",
                  "proceeding to generate"
                  if state["grade"] == "yes" else
                  "attempts exhausted, generating with best chunks")
        return "generate"

    # --- Wire the graph: nodes, edges, and the cycle. ----------------------
    graph = StateGraph(GraphState)
    # Node names must not collide with state keys, hence "grade_chunks".
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_chunks", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_chunks")
    graph.add_conditional_edges("grade_chunks", route_after_grade,
                                {"rewrite": "rewrite", "generate": "generate"})
    graph.add_edge("rewrite", "retrieve")  # <- the cycle
    graph.add_edge("generate", END)

    final = graph.compile().invoke({
        "question": question, "rewritten_question": question,
        "chunks": [], "grade": "", "attempts": 0, "answer": ""})

    return ArenaResult(
        answer=final["answer"],
        steps=trace.steps,
        latency_ms=int((perf_counter() - t0) * 1000),
        framework_loc=count_loc(__file__),
        architecture_note=ARCHITECTURE_NOTE,
        llm_calls=trace.llm_calls,
    )
