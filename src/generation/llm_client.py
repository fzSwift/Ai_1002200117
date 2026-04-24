from __future__ import annotations

import os
import json
from collections.abc import Iterator
from urllib import request

from dotenv import load_dotenv
from openai import OpenAI

from src.generation.answer_composer import compose_structured_answer
from src.generation.offline_answer import compose_offline_fallback, try_sentence_answer
from src.generation.prompt_builder import UNKNOWN_FROM_DOCUMENTS

load_dotenv()


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


class LLMClient:
    """
    Zero-cost default: if OPENAI_API_KEY is unset or still the placeholder, the app runs
    in offline mode (extractive natural-language answers when possible).

    To use the OpenAI API (paid), set a real OPENAI_API_KEY and do not set OFFLINE_MODE.
    """

    def __init__(self) -> None:
        self.provider = "offline"
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        force_offline = _truthy_env("OFFLINE_MODE")
        # Default to Ollama unless explicitly disabled (USE_OLLAMA=0/false/off).
        use_ollama = os.getenv("USE_OLLAMA", "1").strip().lower() in ("1", "true", "yes", "on")
        placeholder = api_key.lower() in ("", "your_openai_api_key_here", "sk-placeholder")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.ollama_api_key = (os.getenv("OLLAMA_API_KEY") or "").strip()
        self.response_mode = os.getenv("RESPONSE_MODE", "detailed")
        self.offline = force_offline

        if self.offline:
            self.model = "offline"
            self.client = None
            return

        if use_ollama:
            self.provider = "ollama"
            self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            self.client = None
            return

        self.offline = placeholder
        if self.offline:
            self.model = "offline"
            self.client = None
            return

        self.provider = "openai"
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(api_key=api_key)

    def _ollama_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"
        return headers

    def _ollama_generate(self, prompt: str, *, stream: bool) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        req = request.Request(
            f"{self.ollama_base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._ollama_headers(),
            method="POST",
        )
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        return str((data.get("message") or {}).get("content", "")).strip()

    def _offline_from_chunks(self, query: str, chunks: list[dict]) -> str:
        """Return only a natural-language answer; chunks stay in the pipeline for the UI evidence panel."""
        if not chunks:
            return UNKNOWN_FROM_DOCUMENTS

        composed = compose_structured_answer(query, chunks, mode=self.response_mode)
        if composed.text and UNKNOWN_FROM_DOCUMENTS not in composed.text:
            return composed.text

        sentence = try_sentence_answer(query, chunks)
        if sentence:
            return sentence.strip()

        fallback = compose_offline_fallback(query, chunks)
        if fallback:
            return fallback.strip()

        return UNKNOWN_FROM_DOCUMENTS

    def generate(self, prompt: str, query: str = "", chunks: list[dict] | None = None) -> str:
        if self.offline:
            return self._offline_from_chunks(query, chunks or [])
        if self.provider == "ollama":
            try:
                return self._ollama_generate(prompt, stream=False)
            except Exception:
                return self._offline_from_chunks(query, chunks or [])

        try:
            assert self.client is not None
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception:
            if os.getenv("USE_OLLAMA_FALLBACK", "1").strip().lower() in ("1", "true", "yes", "on"):
                try:
                    return self._ollama_generate(prompt, stream=False)
                except Exception:
                    return self._offline_from_chunks(query, chunks or [])
            return self._offline_from_chunks(query, chunks or [])

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Yield decoded token deltas from the chat completions API (API mode only)."""
        if self.offline:
            raise RuntimeError("Streaming requires a configured OpenAI API key.")
        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
            req = request.Request(
                f"{self.ollama_base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers=self._ollama_headers(),
                method="POST",
            )
            with request.urlopen(req, timeout=300) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    part = json.loads(line)
                    token = (part.get("message") or {}).get("content")
                    if token:
                        yield token
            return
        assert self.client is not None
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                yield choice.delta.content

    def pure_llm_answer(self, query: str) -> str:
        if self.offline:
            return (
                f"Question: {query}"
            )
        prompt = (
            "Answer the following user question directly. "
            "Do not use retrieval.\n\n"
            f"Question: {query}"
        )
        return self.generate(prompt)
