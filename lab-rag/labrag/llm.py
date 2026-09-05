"""Answer generation. One small interface, three providers, and 'none'.

* `anthropic` - Claude via the official SDK. Default when ANTHROPIC_API_KEY is set.
* `ollama`    - a local model (llama3.1, gemma, ...). Nothing leaves the machine.
* `openai`    - OpenAI or any /v1/chat/completions-compatible server.
* none        - LabRAG still works as a search engine: it shows the passages
                and where they came from, it just does not write a summary.
"""

from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger(__name__)


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 8000) -> str: ...


class LLMError(RuntimeError):
    """A human-readable failure (bad key, server down, model missing)."""


class AnthropicLLM:
    def __init__(self, model: str = "claude-opus-5", api_key: str | None = None, effort: str = "medium", timeout: float = 300.0):
        import anthropic

        self.name = f"anthropic:{model}"
        self.model = model
        self.effort = effort
        kwargs = {"timeout": timeout, "max_retries": 2}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**kwargs)
        self._anthropic = anthropic
        self._fallbacks_supported = True

    def complete(self, system: str, user: str, max_tokens: int = 8000) -> str:
        messages = [{"role": "user", "content": user}]
        try:
            if self._fallbacks_supported:
                try:
                    # Server-side refusal fallbacks: if a safety classifier declines, the API
                    # re-runs the request on a fallback model inside the same call.
                    response = self._client.beta.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        system=system,
                        messages=messages,
                        output_config={"effort": self.effort},
                        betas=["server-side-fallback-2026-07-01"],
                        fallbacks="default",
                    )
                except (self._anthropic.BadRequestError, TypeError) as exc:
                    log.info("Fallbacks not accepted (%s); using the plain Messages API", exc)
                    self._fallbacks_supported = False
                    response = self._plain(system, messages, max_tokens)
            else:
                response = self._plain(system, messages, max_tokens)
        except self._anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.") from exc
        except self._anthropic.NotFoundError as exc:
            raise LLMError(f"Anthropic does not know the model '{self.model}'. Check LABRAG_LLM_MODEL.") from exc
        except self._anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit hit. Wait a minute and try again.") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach api.anthropic.com. Check the network connection.") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc

        if response.stop_reason == "refusal":
            return "The model declined to answer this question."
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if response.stop_reason == "max_tokens":
            text += "\n\n[answer cut short: increase LABRAG_MAX_TOKENS]"
        return text

    def _plain(self, system: str, messages: list[dict], max_tokens: int):
        return self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": self.effort},
        )


class OllamaLLM:
    def __init__(self, model: str = "llama3.1", url: str = "http://localhost:11434", timeout: float = 600.0):
        import httpx

        self.name = f"ollama:{model}"
        self.model = model
        self.url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._httpx = httpx

    def complete(self, system: str, user: str, max_tokens: int = 8000) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }
        try:
            r = self._client.post(f"{self.url}/api/chat", json=payload)
        except self._httpx.HTTPError as exc:
            raise LLMError(f"Could not reach Ollama at {self.url}. Is `ollama serve` running?") from exc
        if r.status_code == 404:
            raise LLMError(f"Ollama does not have the model '{self.model}'. Run: ollama pull {self.model}")
        if r.status_code >= 400:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r.json().get("message", {}).get("content", "").strip()


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = "", base_url: str = "https://api.openai.com/v1", timeout: float = 300.0):
        import httpx

        self.name = f"openai:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)
        self._httpx = httpx

    def complete(self, system: str, user: str, max_tokens: int = 8000) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        try:
            r = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        except self._httpx.HTTPError as exc:
            raise LLMError(f"Could not reach {self.base_url}.") from exc
        if r.status_code == 401:
            raise LLMError("The OpenAI-compatible server rejected the API key.")
        if r.status_code >= 400:
            raise LLMError(f"LLM server error {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"].strip()
