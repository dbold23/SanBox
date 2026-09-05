"""Build the embedder and the LLM from Settings, with sensible auto-detection."""

from __future__ import annotations

import logging

from .config import Settings
from .embed import Embedder, FastEmbedEmbedder, HashEmbedder, OllamaEmbedder, OpenAIEmbedder
from .llm import LLM, AnthropicLLM, OllamaLLM, OpenAILLM

log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}
DEFAULT_EMBED_MODELS = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
}


class ProviderError(RuntimeError):
    pass


def ollama_available(url: str, timeout: float = 1.5) -> bool:
    try:
        import httpx

        r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def make_embedder(settings: Settings) -> Embedder:
    kind = settings.embed
    model = settings.embed_model or DEFAULT_EMBED_MODELS.get(kind)
    try:
        if kind == "fastembed":
            settings.models_cache_dir.mkdir(parents=True, exist_ok=True)
            return FastEmbedEmbedder(model=model, cache_dir=str(settings.models_cache_dir))
        if kind == "ollama":
            return OllamaEmbedder(model=model, url=settings.ollama_url)
        if kind == "openai":
            return OpenAIEmbedder(model=model, api_key=settings.openai_api_key or "", base_url=settings.openai_base_url)
        if kind == "hash":
            return HashEmbedder()
    except ImportError as exc:
        raise ProviderError(f"The '{kind}' embedder needs a package that is not installed: {exc}") from exc
    except Exception as exc:
        hint = ""
        if kind == "fastembed":
            hint = " (first run downloads the model from huggingface.co; check the network, or set LABRAG_EMBED=hash to get going without it)"
        elif kind == "ollama":
            hint = f" (is Ollama running at {settings.ollama_url}? did you `ollama pull {model}`?)"
        raise ProviderError(f"Could not start the '{kind}' embedder: {exc}{hint}") from exc
    raise ProviderError(f"Unknown LABRAG_EMBED value {kind!r}. Use fastembed, ollama, openai or hash.")


def resolve_llm_kind(settings: Settings) -> str:
    kind = settings.llm
    if kind != "auto":
        return kind
    if settings.anthropic_api_key:
        return "anthropic"
    if settings.openai_api_key:
        return "openai"
    if ollama_available(settings.ollama_url):
        return "ollama"
    return "none"


def make_llm(settings: Settings) -> LLM | None:
    kind = resolve_llm_kind(settings)
    model = settings.llm_model or DEFAULT_MODELS.get(kind)
    if kind == "none":
        return None
    if kind == "anthropic":
        if not settings.anthropic_api_key:
            raise ProviderError("LABRAG_LLM=anthropic but ANTHROPIC_API_KEY is not set.")
        return AnthropicLLM(model=model, api_key=settings.anthropic_api_key, effort=settings.llm_effort)
    if kind == "openai":
        return OpenAILLM(model=model, api_key=settings.openai_api_key or "", base_url=settings.openai_base_url)
    if kind == "ollama":
        return OllamaLLM(model=model, url=settings.ollama_url)
    raise ProviderError(f"Unknown LABRAG_LLM value {kind!r}. Use anthropic, openai, ollama or none.")
