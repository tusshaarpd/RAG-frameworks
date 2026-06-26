# ============================================================================
# ARENA 4: CREWAI (the team of roles)
# What this framework believes: work gets done by a TEAM. You don't write a
#   pipeline, you cast characters - each agent has a role, a goal, and a
#   backstory - then hand them tasks and let them talk to each other.
# What it gave us for free: agent-to-agent handoff (Writer automatically
#   receives Researcher's output as context), tool-use loops, and prompts we
#   never wrote - the role/goal/backstory becomes the system prompt.
# What it cost us: control and tokens. The agents decide how many LLM calls
#   to make; the same question costs several times the raw arena's single
#   call, and the prompts driving it are CrewAI's, not ours.
# ============================================================================
from time import perf_counter

from core import embeddings, ingest, llm
from core.trace import ArenaResult, TraceCollector, count_loc

ARCHITECTURE_NOTE = (
    "Two role-played agents in sequence: a Researcher with a FAISS search "
    "tool finds passages, then hands its findings to a Writer who composes "
    "the answer. The handoff is the framework's core abstraction."
)

# The custom tool needs access to the run's index + trace; CrewAI tools are
# plain functions, so we pass these through module globals set in run().
_search_state = {}


def run(pdf_path: str, question: str, trace: TraceCollector) -> ArenaResult:
    from crewai import LLM, Agent, Crew, Process, Task
    from crewai.tools import tool

    t0 = perf_counter()

    # Same shared index as the raw arena - the tool wraps it.
    with trace.timed("ingest + index", "ingest", pdf_path) as t:
        text, pages, _ = ingest.extract_text(pdf_path)
        chunks = ingest.chunk_text(text)
        index = embeddings.build_index(chunks)
        t.output = f"{pages} pages -> {len(chunks)} chunks -> FAISS"
    _search_state.update(index=index, chunks=chunks, trace=trace)

    @tool("Search PDF")
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF for passages relevant to the query."""
        s = _search_state
        with s["trace"].timed("tool: Search PDF", "retrieve", query) as t:
            hits = embeddings.search(s["index"], s["chunks"], query, k=4)
            result = "\n\n".join(c for c, _ in hits)
            t.output = result
        return result

    # Every agent "thought" is one LLM call - count them via the callback.
    def on_step(step):
        trace.add("agent step", "llm",
                  getattr(step, "thought", "") or "(agent reasoning)",
                  getattr(step, "output", None) or getattr(step, "text", ""))

    crew_llm = LLM(model=llm.model_name(), temperature=0)

    # The casting call: role / goal / backstory IS the prompt engineering.
    researcher = Agent(
        role="Researcher",
        goal="Find the passages in the document that answer the question.",
        backstory="A meticulous analyst who only quotes the source document.",
        tools=[search_pdf], llm=crew_llm, verbose=False, max_iter=4)
    writer = Agent(
        role="Writer",
        goal="Compose a clear, grounded answer from the researcher's notes.",
        backstory="A technical writer who never adds facts beyond the notes.",
        llm=crew_llm, verbose=False, max_iter=3)

    research_task = Task(
        description=f"Find passages relevant to: {question}",
        expected_output="The most relevant passages, quoted.",
        agent=researcher,
        callback=lambda out: trace.add(
            "handoff: Researcher -> Writer", "handoff",
            "research findings passed as context to Writer", out.raw))
    write_task = Task(
        description=f"Answer using only the research notes: {question}",
        expected_output="A grounded answer to the question.",
        agent=writer,
        context=[research_task])  # <- the handoff wiring

    trace.add("crew kickoff", "decision",
              f"question: {question}",
              "sequential process: Researcher task then Writer task")
    crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task],
                process=Process.sequential, step_callback=on_step,
                verbose=False)
    answer = str(crew.kickoff())

    return ArenaResult(
        answer=answer,
        steps=trace.steps,
        latency_ms=int((perf_counter() - t0) * 1000),
        framework_loc=count_loc(__file__),
        architecture_note=ARCHITECTURE_NOTE,
        llm_calls=trace.llm_calls,
    )
