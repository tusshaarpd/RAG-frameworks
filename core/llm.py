"""Single place to choose the LLM. Every arena routes through here.

Provider: OpenAI. Key from OPENAI_API_KEY env var.
'Cheap mode' swaps gpt-4o -> gpt-4o-mini; the toggle lives in the sidebar.
Arenas use framework-native OpenAI integrations (langchain-openai,
llama-index-llms-openai, CrewAI's litellm) but always ask THIS module
which model name to use, so the comparison stays apples-to-apples.
"""
import os

SMART_MODEL = "gpt-4o"
CHEAP_MODEL = "gpt-4o-mini"

# Module-level flag set once by the UI before a run.
_cheap_mode = True


def set_cheap_mode(cheap: bool):
    global _cheap_mode
    _cheap_mode = cheap


def model_name() -> str:
    return CHEAP_MODEL if _cheap_mode else SMART_MODEL


def api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def get_client():
    """Raw OpenAI client - used by the no-framework arena."""
    from openai import OpenAI
    return OpenAI()


def complete(prompt: str, system: str = "", max_tokens: int = 600) -> str:
    """One plain chat completion. The raw arena and helpers use this."""
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model_name(),
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content or ""
