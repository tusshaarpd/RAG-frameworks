"""Trace primitives shared by every arena.

A TraceStep is the atom of comparison: every arena, no matter how different
its internals, must report what it did as a flat list of steps. This is what
makes five wildly different frameworks comparable on one screen.
"""
from dataclasses import dataclass, field
from time import perf_counter

PREVIEW_LIMIT = 300  # chars - keep the UI readable


def _preview(text) -> str:
    """Truncate any value to a short, display-safe string."""
    s = str(text) if text is not None else ""
    s = s.replace("\n", " ").strip()
    return s[:PREVIEW_LIMIT] + ("…" if len(s) > PREVIEW_LIMIT else "")


@dataclass
class TraceStep:
    order: int
    arena: str
    name: str
    kind: str            # ingest | retrieve | llm | decision | handoff
    input_preview: str
    output_preview: str
    duration_ms: int
    indent: int = 0      # used to visually nest LangGraph retry iterations


@dataclass
class TraceCollector:
    """Collects steps for one arena run. Arenas call add(); the UI reads steps."""
    arena: str
    steps: list = field(default_factory=list)

    def add(self, name: str, kind: str, input_val="", output_val="",
            duration_ms: int = 0, indent: int = 0) -> TraceStep:
        step = TraceStep(
            order=len(self.steps) + 1,
            arena=self.arena,
            name=name,
            kind=kind,
            input_preview=_preview(input_val),
            output_preview=_preview(output_val),
            duration_ms=int(duration_ms),
            indent=indent,
        )
        self.steps.append(step)
        return step

    def timed(self, name: str, kind: str, input_val=""):
        """Context manager: times a block and records it as one step.

        Usage:
            with trace.timed("embed", "ingest", f"{n} chunks") as t:
                vectors = model.encode(chunks)
                t.output = f"{vectors.shape}"
        """
        return _TimedStep(self, name, kind, input_val)

    @property
    def llm_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "llm")


class _TimedStep:
    def __init__(self, collector, name, kind, input_val):
        self.collector, self.name, self.kind = collector, name, kind
        self.input_val, self.output = input_val, ""
        self.indent = 0

    def __enter__(self):
        self._t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        ms = (perf_counter() - self._t0) * 1000
        out = self.output if exc is None else f"ERROR: {exc}"
        self.collector.add(self.name, self.kind, self.input_val, out, ms,
                           indent=self.indent)
        return False  # never swallow exceptions


@dataclass
class ArenaResult:
    """What every arena returns. The scorecard is built from these."""
    answer: str
    steps: list
    latency_ms: int
    framework_loc: int
    architecture_note: str
    llm_calls: int = 0
    error: str = ""      # non-empty means the arena failed; UI renders it


def count_loc(module_file: str) -> int:
    """Lines of code in an arena module - a fairness metric on the scorecard.
    Counts non-blank, non-comment lines so the big lesson-comments are free."""
    try:
        with open(module_file, encoding="utf-8") as f:
            return sum(
                1 for line in f
                if line.strip() and not line.strip().startswith("#")
            )
    except OSError:
        return 0
