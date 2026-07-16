"""Unified LLM client for distillation, labeling, and judging.

All providers are OpenAI-compatible, so one code path serves GLM (Zhipu), DeepSeek, OpenAI,
or a local Ollama server. Configure via the [llm] block in params.yaml and store your key in
the env var named by llm.api_key_env (provider: local needs no key).

    provider: glm       -> https://api.z.ai/api/paas/v4/     key env: GLM_API_KEY       model: glm-4.5-flash
    provider: deepseek  -> https://api.deepseek.com          key env: DEEPSEEK_API_KEY  model: deepseek-chat
    provider: openai    -> (default OpenAI)                   key env: OPENAI_API_KEY    model: gpt-5.4-nano
    provider: local     -> http://localhost:11434/v1 (Ollama) key: none                 model: gemma3:4b

Hardened for long batch loops:
  * the client is created ONCE and reused (no per-call TCP/TLS handshake),
  * real exponential backoff with jitter on 429/5xx/timeouts (up to ~6 retries),
  * temperature-compatibility fallback: GPT-5-family models reject `temperature`
    unless reasoning effort is 'none'; if the API returns "unsupported", the call
    is retried once without the parameter instead of failing the whole stage.
"""
from __future__ import annotations
import os, time, random
from book_atlas.utils import get_logger

log = get_logger("llm")

_BASE_URLS = {
    "glm": "https://api.z.ai/api/paas/v4/",
    "deepseek": "https://api.deepseek.com",
    "openai": None,                        # OpenAI default base
    "local": "http://localhost:11434/v1",  # Ollama OpenAI-compatible endpoint
}

_CLIENT_CACHE: dict = {}
_RETRIABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _client(cfg_llm: dict):
    from openai import OpenAI
    provider = cfg_llm["provider"]
    base = _BASE_URLS.get(provider)
    if provider == "local":
        key = "ollama"
    else:
        key = os.environ.get(cfg_llm.get("api_key_env", "") or "", "")
        if not key and provider == "openai":
            key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError(
                f"No API key in env var '{cfg_llm.get('api_key_env')}' for provider '{provider}'. "
                f"Add a line to the .env file in the project root:  {cfg_llm.get('api_key_env')}=your-key")
    cache_key = (provider, base, key)
    if cache_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[cache_key] = OpenAI(base_url=base, api_key=key) if base else OpenAI(api_key=key)
    return _CLIENT_CACHE[cache_key]


def _is_retriable(exc: Exception) -> bool:
    msg = str(exc).lower()
    status = getattr(exc, "status_code", None)
    return (status in _RETRIABLE_STATUS
            or "rate limit" in msg or "429" in msg or "overloaded" in msg
            or "timeout" in msg or "timed out" in msg or "connection" in msg)


def chat(cfg_llm: dict, prompt: str, system: str | None = None, max_retries: int = 6) -> str:
    client = _client(cfg_llm)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    temp = cfg_llm.get("temperature", 0.3)
    send_temp = temp is not None
    attempt = 0
    while True:
        try:
            kw = {"temperature": temp} if send_temp else {}
            r = client.chat.completions.create(model=cfg_llm["model"], messages=msgs, **kw)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if send_temp and "temperature" in msg and ("unsupported" in msg or "does not support" in msg):
                log.warning("Model rejected `temperature`; retrying without it "
                            "(set  llm.temperature: null  in params.yaml to silence this).")
                send_temp = False
                continue
            if attempt < max_retries and _is_retriable(e):
                delay = min(60.0, (2.0 ** attempt) + random.random())
                log.warning(f"LLM call failed ({e}); retry {attempt + 1}/{max_retries} in {delay:.1f}s")
                time.sleep(delay)
                attempt += 1
                continue
            raise


def available(cfg_llm: dict) -> bool:
    try:
        chat(cfg_llm, "Reply with OK.", max_retries=2)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning(f"LLM provider '{cfg_llm.get('provider')}' unavailable ({e}). Steps will use fallbacks.")
        return False
