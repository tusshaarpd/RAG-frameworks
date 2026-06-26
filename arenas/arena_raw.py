# ============================================================================
# ARENA 0: NO FRAMEWORK
# What this framework believes: nothing. There is no framework. RAG is just
#   five function calls in a row: extract, chunk, embed, retrieve, generate.
# What it gave us for free: nothing - and therefore total visibility. Every
#   byte that moves is a line of code you can read on this page.
# What it cost us: we hand-rolled chunking, retrieval, and prompt assembly.
#   For this simple task that cost ~60 lines. Every other arena is some
#   framework's opinion about how to hide these same 60 lines.
# ============================================================================
from time import perf_counter

from core import embeddings, ingest, llm
from core.trace import ArenaResult, TraceCollector, count_loc

ARCHITECTURE_NOTE = (
    "Plain Python: pypdf -> fixed-size chunks -> MiniLM embeddings -> flat "
    "FAISS -> one OpenAI call. Zero abstraction; this is what every framework "
    "below is wrapping."
)


def run(pdf_path: str, question: str, trace: TraceCollector) -> ArenaResult:
    t0 = perf_counter()

    # 1. Extract raw text from the PDF.
    with trace.timed("extract", "ingest", pdf_path) as t:
        text, pages, truncated = ingest.extract_text(pdf_path)
        t.output = f"{pages} pages, {len(text)} chars" + (
            " (truncated to 20 pages)" if truncated else "")

    # 2. Chunk it: 800-char windows with 100-char overlap. Naive on purpose.
    with trace.timed("chunk", "ingest", f"{len(text)} chars") as t:
        chunks = ingest.chunk_text(text)
        t.output = f"{len(chunks)} chunks of ~{ingest.CHUNK_SIZE} chars"

    # 3 + 4. Embed every chunk and put the vectors in a flat FAISS index.
    with trace.timed("embed + index", "ingest", f"{len(chunks)} chunks") as t:
        index = embeddings.build_index(chunks)
        t.output = f"FAISS IndexFlatIP, {index.ntotal} vectors, dim {index.d}"

    # 5. Retrieve: cosine top-k. Show the actual chunks - no mystery.
    with trace.timed("retrieve", "retrieve", question) as t:
        hits = embeddings.search(index, chunks, question, k=4)
        t.output = " | ".join(f"[{s:.2f}] {c[:60]}" for c, s in hits)

    # 6. Generate: paste the chunks into a prompt, one API call, done.
    context = "\n\n---\n\n".join(c for c, _ in hits)
    prompt = (
        f"Answer the question using ONLY the context below. "
        f"If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    with trace.timed("generate", "llm", prompt) as t:
        answer = llm.complete(prompt)
        t.output = answer

    return ArenaResult(
        answer=answer,
        steps=trace.steps,
        latency_ms=int((perf_counter() - t0) * 1000),
        framework_loc=count_loc(__file__),
        architecture_note=ARCHITECTURE_NOTE,
        llm_calls=trace.llm_calls,
    )
