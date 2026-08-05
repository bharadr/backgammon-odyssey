"""B4: the real model behind the injected `(system, user) -> str` seam.

`anthropic` is imported lazily so that importing this module (and the CLI) does
not require the SDK or a key -- only constructing `AnthropicLLM` does.
"""
import os

DEFAULT_MODEL = os.environ.get("COACH_MODEL", "claude-sonnet-5")


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        from anthropic import Anthropic
        self._client = Anthropic()          # reads ANTHROPIC_API_KEY from env
        self._model = model or DEFAULT_MODEL
        self._max_tokens = max_tokens

    def __call__(self, system: str, user: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return _text_of(message).strip()


def _text_of(message) -> str:
    """The response's text: the `.text` of each text block, concatenated.

    A response's `content` is a list of typed blocks; a plain answer is one text
    block, but we keep and join all of them and skip any non-text blocks.
    """
    text_blocks = [block.text for block in message.content if block.type == "text"]
    return "".join(text_blocks)


def _offline_llm(system: str, user: str) -> str:
    return "[No ANTHROPIC_API_KEY set -- narrated explanation skipped.]"


def make_llm():
    """The real Anthropic LLM if a key is set, else an offline stand-in so the
    binaries still run (printing the engine evidence without narration)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _offline_llm
    return AnthropicLLM()
