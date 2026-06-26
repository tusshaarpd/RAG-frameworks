# ============================================================================
# ARENA 2: LLAMAINDEX (the data specialist)
# What this framework believes: the hard part of RAG is the DATA, not the
#   chain. Documents become Nodes (chunks with metadata + relationships),
#   and retrieval quality is the product. The query engine is almost an
#   afterthought bolted onto a great index.
# What it gave us for free: SimpleDirectoryReader (one line per file type),
#   per-node similarity scores, and response synthesizers like
#   tree_summarize that LangChain makes you assemble by hand.
# What it cost us: a parallel universe of nouns (Node, ServiceContext ->
#   Settings, StorageContext) and global singleton config that can surprise
#   you in multi-model apps.
# ============================================================================
from time import perf_counter

from core import llm
from core.trace import ArenaResult, TraceCollector, count_loc

ARCHITECTURE_NOTE = (
    "SimpleDirectoryReader -> Nodes -> VectorStoreIndex -> query engine with "
    "tree_summarize synthesis. Retrieval-first design: per-node similarity "
    "scores are surfaced, the LLM step is almost incidental."
)


def run(pdf_path: str, question: str, trace: TraceCollector) -> ArenaResult:
    # Lazy imports so a missing llama-index only disables this arena.
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from datetime import datetime

    from llama_index.core.callbacks import (CallbackManager, CBEventType,
                                            LlamaDebugHandler)
    from llama_index.core.callbacks.schema import TIMESTAMP_FORMAT
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai import OpenAI

    t0 = perf_counter()

    # Global Settings: LlamaIndex's "configure once, used everywhere" idiom.
    debug = LlamaDebugHandler()  # native observability hook
    Settings.llm = OpenAI(model=llm.model_name(), temperature=0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2",
                                                device="cpu")
    Settings.callback_manager = CallbackManager([debug])

    # 1. Load: one reader, any file type. LlamaIndex's ingestion calling card.
    with trace.timed("SimpleDirectoryReader", "ingest", pdf_path) as t:
        docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()[:20]
        t.output = f"{len(docs)} Documents (one per page)"

    # 2. Parse into Nodes - the framework's first-class chunk object.
    with trace.timed("SentenceSplitter -> Nodes", "ingest",
                     f"{len(docs)} documents") as t:
        nodes = SentenceSplitter(chunk_size=800, chunk_overlap=100
                                 ).get_nodes_from_documents(docs)
        t.output = f"{len(nodes)} Nodes (sentence-aware boundaries)"

    # 3. Index the nodes.
    with trace.timed("VectorStoreIndex", "ingest", f"{len(nodes)} nodes") as t:
        index = VectorStoreIndex(nodes)
        t.output = "in-memory vector index built"

    # 4 + 5. Retrieve and synthesize. tree_summarize is the depth feature
    # LangChain does not foreground: it recursively summarizes retrieved
    # chunks up a tree instead of stuffing them into one prompt.
    engine = index.as_query_engine(response_mode="tree_summarize",
                                   similarity_top_k=4)
    with trace.timed("query (retrieve)", "retrieve", question) as t:
        response = engine.query(question)
        # Surface per-node similarity scores - LlamaIndex hands these out
        # natively; in LangChain you would dig for them.
        t.output = " | ".join(
            f"[score {n.score:.3f}] {n.node.get_content()[:60]}"
            for n in response.source_nodes)

    # Replay the LLM calls the debug handler observed inside the engine,
    # so tree_summarize's call count is honest in the trace.
    for start, end in debug.get_event_pairs(CBEventType.LLM):
        # CBEvent.time is a formatted string, not a datetime, so parse it.
        t_start = datetime.strptime(start.time, TIMESTAMP_FORMAT)
        t_end = datetime.strptime(end.time, TIMESTAMP_FORMAT)
        ms = (t_end - t_start).total_seconds() * 1000
        trace.add("tree_summarize LLM call", "llm",
                  start.payload.get("messages", ""),
                  end.payload.get("response", ""), ms)

    answer = str(response)
    return ArenaResult(
        answer=answer,
        steps=trace.steps,
        latency_ms=int((perf_counter() - t0) * 1000),
        framework_loc=count_loc(__file__),
        architecture_note=ARCHITECTURE_NOTE,
        llm_calls=trace.llm_calls,
    )
