"""Shared Gemini client wrapper: retries, rate limiting, disk-backed response cache.

Cache key = hash(model + prompt_version + rendered_input). Re-runs are free unless
the input or the prompt version changes. Bumping config.TAGGING_PROMPT_VERSION
(or editing the prompt file, whose content is folded into the hash) invalidates
only the affected entries.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from astro_analysis import config

load_dotenv(config.PROJECT_ROOT / ".env")


class RateLimiter:
    """Simple token-bucket-ish limiter: at most N calls per rolling 60s window."""

    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def wait(self):
        with self._lock:
            now = time.time()
            self._timestamps = [t for t in self._timestamps if now - t < 60]
            if len(self._timestamps) >= self.calls_per_minute:
                sleep_for = 60 - (now - self._timestamps[0]) + 0.05
                time.sleep(max(sleep_for, 0))
                now = time.time()
                self._timestamps = [t for t in self._timestamps if now - t < 60]
            self._timestamps.append(time.time())


class GeminiCachedClient:
    def __init__(
        self,
        model: str = config.GEMINI_MODEL,
        prompt_version: str = config.TAGGING_PROMPT_VERSION,
        cache_dir: Path = config.CACHE_DIR,
        requests_per_minute: int = config.GEMINI_REQUESTS_PER_MINUTE,
    ):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment/.env")
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.prompt_version = prompt_version
        self.cache_dir = Path(cache_dir) / "gemini" / model / prompt_version
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = RateLimiter(requests_per_minute)
        self.stats = {"cache_hits": 0, "api_calls": 0, "errors": 0}
        self._stats_lock = threading.Lock()
        self._cache_write_lock = threading.Lock()

    def _cache_key(self, prompt: str, system_instruction: str, response_schema_repr: str) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode())
        h.update(self.prompt_version.encode())
        h.update(system_instruction.encode())
        h.update(response_schema_repr.encode())
        h.update(prompt.encode())
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    @retry(
        stop=stop_after_attempt(config.GEMINI_MAX_RETRIES),
        wait=wait_exponential(multiplier=config.GEMINI_BASE_DELAY_SECONDS, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_api(self, prompt: str, system_instruction: str, response_schema: Optional[dict]):
        self._limiter.wait()
        gen_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json" if response_schema else None,
            response_schema=response_schema,
            temperature=0.0,
            max_output_tokens=65536,
        )
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=gen_config,
        )
        return resp.text

    def generate_json(
        self,
        prompt: str,
        system_instruction: str = "",
        response_schema: Optional[dict] = None,
        use_cache: bool = True,
    ) -> dict:
        """Call Gemini expecting a JSON response; cache the raw text on disk."""
        schema_repr = json.dumps(response_schema, sort_keys=True) if response_schema else ""
        key = self._cache_key(prompt, system_instruction, schema_repr)
        cache_path = self._cache_path(key)

        if use_cache and cache_path.exists():
            with self._stats_lock:
                self.stats["cache_hits"] += 1
            raw = json.loads(cache_path.read_text())["response_text"]
        else:
            try:
                raw = self._call_api(prompt, system_instruction, response_schema)
                with self._stats_lock:
                    self.stats["api_calls"] += 1
            except Exception:
                with self._stats_lock:
                    self.stats["errors"] += 1
                raise
            with self._cache_write_lock:
                cache_path.write_text(json.dumps({
                    "prompt": prompt,
                    "system_instruction": system_instruction,
                    "response_text": raw,
                }))

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Gemini response was not valid JSON: {raw[:500]!r}") from e
