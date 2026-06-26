# ============================================================================
# ARENA 1: LANGCHAIN (the toolkit)
# What this framework believes: every LLM app is a CHAIN - components piped
#   left to right, each one a swappable Lego brick (loader | splitter |
#   embedder | retriever | prompt | llm | parser).
# What it gave us for free: pre-built loaders/splitters/vectorstore glue, the
#   `|` pipe syntax, and a callback system that observes every component.
# What it cost us: an abstraction tax. Five imports from four packages to do
#   what arena_raw does with none, and when something misbehaves you debug
#   LangChain's wrappers, not your own code.
# ============================================================================
from time import perf_counter
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from core import llm
from core.trace import ArenaResult, TraceCollector, count_loc

ARCHITECTURE_NOTE = (
    "Classic LCEL pipe: PyPDFLoader -> RecursiveCharacterTextSplitter -> "
    "HuggingFace embeddings -> FAISS retriever | prompt | ChatOpenAI | "
    "parser. A straight line - no branching, no state."
)


class _TraceCallback(BaseCallbackHandler):
    """LangChain callback handler that mirrors retriever and LLM activity
    into our TraceCollector - this is LangChain's native observability."""

    def __init__(self, trace: TraceCollector):
        self.trace, self._t0 = trace, {}

    # LangChain calls these hooks on every retriever / chat-model run.
    def on_retriever_start(self, serialized, query, *, run_id: UUID, **kw):
        self._t0[run_id] = (perf_counter(), query)

    def on_retriever_end(self, documents, *, run_id: UUID, **kw):
        t0, query = self._t0.pop(run_id, (perf_counter(), ""))
        docs = " | ".join(d.page_content[:60] for d in documents)
        self.trace.add("retriever (FAISS)", "retrieve", query, docs,
                       (perf_counter() - t0) * 1000)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        text = " ".join(m.content for batch in messages for m in batch)
        self._t0[run_id] = (perf_counter(), text)

    def on_llm_end(self, response, *, run_id: UUID, **kw):
        t0, prompt = self._t0.pop(run_id, (perf_counter(), ""))
        out = response.generations[0][0].text
        self.trace.add("ChatOpenAI", "llm", prompt, out,
                       (perf_counter() - t0) * 1000)


def run(pdf_path: str, question: str, trace: TraceCollector) -> ArenaResult:
    # Lazy imports: if LangChain isn't installed only THIS arena dies.
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    t0 = perf_counter()

    # Ingestion half of the chain: LangChain's own loader and splitter.
    with trace.timed("PyPDFLoader", "ingest", pdf_path) as t:
        docs = PyPDFLoader(pdf_path).load()[:20]  # one Document per page
        t.output = f"{len(docs)} page-documents"

    with trace.timed("RecursiveCharacterTextSplitter", "ingest",
                     f"{len(docs)} docs") as t:
        splits = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100).split_documents(docs)
        t.output = f"{len(splits)} chunks (splits on \\n\\n, \\n, ' ' first)"

    with trace.timed("FAISS.from_documents", "ingest",
                     f"{len(splits)} chunks") as t:
        embedder = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"})
        vectorstore = FAISS.from_documents(splits, embedder)
        t.output = "FAISS vectorstore built via LangChain wrapper"

    # Query half: the LCEL pipe. THIS is LangChain's signature move.
    prompt = ChatPromptTemplate.from_template(
        "Answer using ONLY this context. If it lacks the answer, say so.\n\n"
        "Context:\n{context}\n\nQuestion: {question}")
    chain = (
        {"context": vectorstore.as_retriever(search_kwargs={"k": 4}),
         "question": RunnablePassthrough()}
        | prompt
        | ChatOpenAI(model=llm.model_name(), temperature=0)
        | StrOutputParser()
    )
    answer = chain.invoke(question,
                          config={"callbacks": [_TraceCallback(trace)]})

    return ArenaResult(
        answer=answer,
        steps=trace.steps,
        latency_ms=int((perf_counter() - t0) * 1000),
        framework_loc=count_loc(__file__),
        architecture_note=ARCHITECTURE_NOTE,
        llm_calls=trace.llm_calls,
    )
