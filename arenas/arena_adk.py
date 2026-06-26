# ============================================================================
# ARENA 5: GOOGLE ADK - STUB
# TODO: implement a single ADK agent with a retrieval tool over the shared
# FAISS index, mirroring the other arenas' run() contract. Skipped for now:
# google-adk is Gemini-first (wants Google auth / Vertex or AI Studio keys),
# which conflicts with this demo's "one API key only" constraint.
# To implement: pip install google-adk, build an Agent with a function tool
# wrapping core.embeddings.search, and adapt its event stream into TraceSteps.
# ============================================================================
from core.trace import ArenaResult, TraceCollector, count_loc

ARCHITECTURE_NOTE = "Not implemented - stub for Google's Agent Development Kit."


def run(pdf_path: str, question: str, trace: TraceCollector) -> ArenaResult:
    raise NotImplementedError(
        "Google ADK arena is a stub. See the TODO at the top of arena_adk.py.")
