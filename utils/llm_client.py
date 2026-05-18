"""
LLM client factory.
Supports OpenAI / Anthropic / Gemini / local Ollama.
"""
import os
import re
import time
from langchain_openai import ChatOpenAI


def _clean_key(v: str | None) -> str | None:
    if not v:
        return None
    s = v.strip()
    if not s:
        return None
    low = s.lower()
    if "your-key" in low or "your key" in low:
        return None
    if low in ("changeme", "replace-me", "replace_with_your_key", "todo"):
        return None
    return s


def _normalize_gemini_model(model: str | None) -> str | None:
    if model is None:
        return None
    s = (model or "").strip()
    if not s:
        return None
    if s.startswith("models/"):
        s = s[len("models/") :]
    if s.endswith("-latest"):
        s = s[: -len("-latest")]
    return s


def _deepseek_openai_model_id(model: str | None, base_url: str) -> str:
    """
    OpenAI-compatible gateways (e.g. ofox) often expect vendor-prefixed ids, matching research_lab inference.
    Official api.deepseek.com typically accepts ids without the deepseek/ prefix.
    """
    m = (model or "").strip() or "deepseek-chat"
    b = (base_url or "").lower()
    if "ofox.ai" in b:
        if m == "deepseek-v3.2":
            return "deepseek/deepseek-v3.2"
        if m == "deepseek-chat":
            return "deepseek-chat"
        if m.startswith("deepseek-"):
            return f"deepseek/{m}"
    return m


def _normalize_deepseek_openai_base_url(url: str | None) -> str:
    """
    DeepSeek documents the host as https://api.deepseek.com; the OpenAI-compatible API lives under /v1.
    """
    raw = (url or "").strip().rstrip("/")
    if not raw:
        return "https://api.deepseek.com/v1"
    if raw in ("https://api.deepseek.com", "http://api.deepseek.com"):
        return "https://api.deepseek.com/v1"
    return raw


class _RetryingLLM:
    def __init__(self, inner, provider: str, temperature: float, model: str | None):
        self._inner = inner
        self._provider = provider
        self._temperature = temperature
        self._model = model

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _parse_retry_seconds(self, msg: str) -> float:
        m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", msg, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return 0.0
        m = re.search(r"retryDelay['\"]?\s*:\s*['\"]([0-9]+)s['\"]", msg)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return 0.0
        return 0.0

    def _is_rate_limited(self, msg: str) -> bool:
        m = msg.lower()
        return ("resource_exhausted" in m) or ("quota exceeded" in m) or ("too many requests" in m) or (" 429 " in f" {m} ")

    def _is_model_not_found(self, msg: str) -> bool:
        m = msg.lower()
        if "not_found" in m or "not found" in m:
            return True
        if "models/" in m and "is not found for api version" in m:
            return True
        if " 404 " in f" {m} " and "model" in m:
            return True
        return False

    def _is_invalid_api_key(self, msg: str) -> bool:
        m = msg.lower()
        if ("api key not valid" in m) or ("api_key_invalid" in m) or ("invalid api key" in m):
            return True
        if "invalid or expired api key" in m:
            return True
        if ("authenticationerror" in m or "401" in m) and ("invalid" in m or "unauthorized" in m):
            return True
        return False

    def _is_transient_network_error(self, err: Exception, msg: str) -> bool:
        m = (msg or "").lower()
        if (
            "server disconnected without sending a response" in m
            or "remoteprotocolerror" in m
            or "connection reset by peer" in m
            or "connection aborted" in m
            or "readtimeout" in m
            or "connecttimeout" in m
            or "timed out" in m
            or "temporary failure in name resolution" in m
            or "name or service not known" in m
            or "network is unreachable" in m
            or "connection refused" in m
        ):
            return True
        try:
            import httpcore
            import httpx
        except Exception:
            httpcore = None
            httpx = None
        if httpx is not None:
            try:
                if isinstance(err, (httpx.TimeoutException, httpx.TransportError)):
                    return True
            except Exception:
                pass
        if httpcore is not None:
            try:
                if isinstance(err, (httpcore.TimeoutException, httpcore.NetworkError, httpcore.RemoteProtocolError)):
                    return True
            except Exception:
                pass
        return False

    def _fallback_models(self) -> list[str]:
        raw = (os.environ.get("RESEARCH_AGENT_GEMINI_FALLBACK_MODELS") or "").strip()
        if raw:
            return [x.strip() for x in raw.split(",") if x.strip()]
        return [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]

    def _allow_model_fallback(self) -> bool:
        v = (os.environ.get("RESEARCH_AGENT_GEMINI_ALLOW_MODEL_FALLBACK") or "").strip().lower()
        return v in ("1", "true", "yes", "y", "on")

    def _max_retries(self) -> int:
        try:
            return max(0, int(os.environ.get("RESEARCH_AGENT_LLM_RETRY_MAX", "2")))
        except Exception:
            return 2

    def _make_gemini(self, model: str):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=_normalize_gemini_model(model), temperature=self._temperature)

    def _is_ollama_not_running(self, msg: str) -> bool:
        m = (msg or "").lower()
        return ("connection refused" in m or "failed to connect" in m) and ("11434" in m or "ollama" in m or "localhost" in m)

    def invoke(self, *args, **kwargs):
        attempts = 0
        fallback_queue = []
        if self._provider in ("gemini", "google") and self._allow_model_fallback():
            current = _normalize_gemini_model(getattr(self._inner, "model", None) or self._model or os.environ.get("GEMINI_MODEL"))
            seen = set()
            for m in self._fallback_models():
                nm = _normalize_gemini_model(m)
                if not nm:
                    continue
                if nm == current:
                    continue
                if nm in seen:
                    continue
                seen.add(nm)
                fallback_queue.append(nm)

        last_err = None
        while attempts <= self._max_retries():
            try:
                return self._inner.invoke(*args, **kwargs)
            except Exception as e:
                last_err = e
                msg = str(e)
                if self._provider in ("gemini", "google") and self._is_invalid_api_key(msg):
                    raise RuntimeError(
                        "Invalid GEMINI_API_KEY. Set a valid key or switch provider "
                        "(RESEARCH_AGENT_LLM_PROVIDER=openai/anthropic) and configure the corresponding API key."
                    ) from e
                if self._provider == "openai" and self._is_invalid_api_key(msg):
                    prov = (os.environ.get("RESEARCH_AGENT_LLM_PROVIDER") or "").strip().lower()
                    if prov in ("deepseek", "deep-seek"):
                        raise RuntimeError(
                            "Invalid DEEPSEEK_API_KEY or OPENAI_API_KEY for the configured base URL (401). "
                            "Default host is https://api.deepseek.com/v1 (set DEEPSEEK_BASE_URL=https://api.deepseek.com if you prefer; it is normalized to …/v1). "
                            "Third-party gateways (e.g. ofox) require DEEPSEEK_BASE_URL=https://api.ofox.ai/v1 with a key from that host. "
                            "If both DEEPSEEK_API_KEY and OPENAI_API_KEY are set, DEEPSEEK_API_KEY is used first—remove a stale value. "
                            "On ofox, OPENAI_MODEL is rewritten to deepseek/… when the base URL is ofox."
                        ) from e
                    raise RuntimeError("Invalid OPENAI_API_KEY. Update your OpenAI key and retry.") from e
                if self._provider in ("anthropic", "claude") and self._is_invalid_api_key(msg):
                    raise RuntimeError("Invalid ANTHROPIC_API_KEY. Update your Anthropic key and retry.") from e
                if self._provider == "ollama" and self._is_ollama_not_running(msg):
                    raise RuntimeError(
                        "Ollama is not reachable (http://localhost:11434). Start Ollama (ollama serve) "
                        "or set RESEARCH_AGENT_LLM_PROVIDER to gemini/openai/anthropic."
                    ) from e
                if self._is_transient_network_error(e, msg):
                    if attempts >= self._max_retries():
                        raise RuntimeError(
                            f"Network error while calling LLM provider ({self._provider}). Please retry."
                        ) from e
                    backoff = min(60.0, (2.0 ** attempts) + 0.25)
                    print(f"LLM: transient network error, retrying in {backoff:.1f}s")
                    time.sleep(backoff)
                    attempts += 1
                    continue
                if self._provider in ("gemini", "google") and self._is_model_not_found(msg):
                    if fallback_queue:
                        nxt = fallback_queue.pop(0)
                        try:
                            cur = _normalize_gemini_model(getattr(self._inner, "model", None) or self._model or os.environ.get("GEMINI_MODEL"))
                        except Exception:
                            cur = None
                        print(f"LLM: model not found, falling back from {cur or '(unknown)'} to {nxt}")
                        self._inner = self._make_gemini(nxt)
                        attempts += 1
                        continue
                    current = _normalize_gemini_model(getattr(self._inner, "model", None) or self._model or os.environ.get("GEMINI_MODEL"))
                    api_version_hint = ""
                    if "is not found for api version" in (msg or "").lower():
                        api_version_hint = (
                            " This error text indicates an API version mismatch "
                            "(the SDK may be calling v1beta). "
                            "Upgrading google-genai/langchain-google-genai or switching to a model available on that API version usually fixes it."
                        )
                    raise RuntimeError(
                        "LLM model not found / not supported by this API. "
                        f"Current model={current or '(unknown)'}. "
                        f"Original error: {msg}. "
                        "Fix by setting GEMINI_MODEL to a supported model for your key/API. "
                        "If you explicitly want automatic downgrade attempts, set RESEARCH_AGENT_GEMINI_ALLOW_MODEL_FALLBACK=1 "
                        "(and optionally RESEARCH_AGENT_GEMINI_FALLBACK_MODELS)."
                        + api_version_hint
                    ) from e
                if not self._is_rate_limited(msg):
                    raise
                if attempts >= self._max_retries():
                    try:
                        cur = _normalize_gemini_model(getattr(self._inner, "model", None) or self._model or os.environ.get("GEMINI_MODEL"))
                    except Exception:
                        cur = None
                    raise RuntimeError(
                        "LLM rate-limited/quota exceeded. "
                        f"Provider={self._provider} model={cur or '(unknown)'}. "
                        f"Original error: {msg}"
                    ) from e
                retry_s = self._parse_retry_seconds(msg)
                backoff = retry_s if retry_s > 0 else min(60.0, (2.0 ** attempts) + 0.25)
                try:
                    cur = _normalize_gemini_model(getattr(self._inner, "model", None) or self._model or os.environ.get("GEMINI_MODEL"))
                except Exception:
                    cur = None
                print(f"LLM: rate limited, waiting {min(120.0, backoff + 0.5):.1f}s then retrying (model={cur or '(unknown)'})")
                time.sleep(min(120.0, backoff + 0.5))
                attempts += 1

        raise last_err

def get_llm(
    temperature: float = 0.5,
    model: str | None = None,
    provider: str | None = None,
):
    provider = (provider or os.environ.get("RESEARCH_AGENT_LLM_PROVIDER") or os.environ.get("LLM_PROVIDER") or "").strip().lower()
    gemini_key = _clean_key(os.environ.get("GEMINI_API_KEY"))
    openai_key = _clean_key(os.environ.get("OPENAI_API_KEY"))
    anthropic_key_env = _clean_key(os.environ.get("ANTHROPIC_API_KEY"))
    ofox_key_env = _clean_key(os.environ.get("OFOX_API_KEY"))
    anthropic_base_url = (
        os.environ.get("ANTHROPIC_BASE_URL")
        or os.environ.get("OFOX_BASE_URL")
        or os.environ.get("ANTHROPIC_API_URL")
        or None
    )
    is_ofox_anthropic = bool(anthropic_base_url and "api.ofox.ai" in anthropic_base_url)
    if anthropic_base_url and "api.ofox.ai" in anthropic_base_url and ofox_key_env:
        anthropic_key = ofox_key_env
    else:
        anthropic_key = anthropic_key_env or ofox_key_env
    if gemini_key is None and os.environ.get("GEMINI_API_KEY"):
        os.environ.pop("GEMINI_API_KEY", None)
    if openai_key is None and os.environ.get("OPENAI_API_KEY"):
        os.environ.pop("OPENAI_API_KEY", None)
    if anthropic_key is None and os.environ.get("ANTHROPIC_API_KEY"):
        os.environ.pop("ANTHROPIC_API_KEY", None)

    if provider in ("gemini", "google"):
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY is required when provider=gemini")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise RuntimeError("Missing dependency: langchain-google-genai") from e
        gm = _normalize_gemini_model(model or os.environ.get("GEMINI_MODEL") or "gemini-1.5-flash")
        inner = ChatGoogleGenerativeAI(
            model=gm,
            temperature=temperature,
        )
        return _RetryingLLM(inner, provider="gemini", temperature=temperature, model=gm)

    if provider in ("openai",):
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is required when provider=openai")
        inner = ChatOpenAI(
            model=model or os.environ.get("OPENAI_MODEL", "gpt-4o"),
            temperature=temperature,
        )
        return _RetryingLLM(inner, provider="openai", temperature=temperature, model=model)

    if provider in ("anthropic", "claude"):
        if not anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY (or OFOX_API_KEY) is required when provider=anthropic")
        if is_ofox_anthropic:
            try:
                from langchain_core.messages import AIMessage
                from experiment.ofox_client import OfoxAnthropicClient, extract_batch_item_text_and_usage
            except Exception as e:
                raise RuntimeError("Missing dependency: experiment/ofox_client for OFOX Anthropic gateway") from e

            class _OfoxAnthropicShim:
                def __init__(self, *, api_key: str, base_url: str, model_name: str, temperature: float):
                    self._client = OfoxAnthropicClient(api_key=api_key, base_url=base_url)
                    self._model = model_name
                    self._temperature = float(temperature)

                def invoke(self, messages, **_kwargs):
                    system_parts = []
                    user_parts = []
                    for m in messages or []:
                        t = getattr(m, "type", "") or ""
                        c = getattr(m, "content", "")
                        if t == "system":
                            system_parts.append(str(c or ""))
                        else:
                            user_parts.append(str(c or ""))
                    system_text = "\n\n".join([x for x in system_parts if x.strip()])
                    user_text = "\n\n".join([x for x in user_parts if x.strip()]) or "(no user content)"
                    resp = self._client.messages.create(
                        model=self._model,
                        max_tokens=1800,
                        temperature=self._temperature,
                        system=system_text,
                        messages=[{"role": "user", "content": user_text}],
                    )
                    text, _usage = extract_batch_item_text_and_usage({"result": {"type": "succeeded", "message": resp}})
                    return AIMessage(content=text or "")

            mname = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            inner = _OfoxAnthropicShim(api_key=anthropic_key, base_url=anthropic_base_url, model_name=mname, temperature=temperature)
            return _RetryingLLM(inner, provider="anthropic", temperature=temperature, model=mname)

        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise RuntimeError("Missing dependency: langchain-anthropic") from e
        inner = ChatAnthropic(
            model=model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            temperature=temperature,
            base_url=anthropic_base_url,
            api_key=anthropic_key,
        )
        return _RetryingLLM(inner, provider="anthropic", temperature=temperature, model=model)

    if provider in ("deepseek", "deep-seek"):
        dsk = _clean_key(os.environ.get("DEEPSEEK_API_KEY"))
        oai = _clean_key(os.environ.get("OPENAI_API_KEY"))
        api_key = dsk or oai
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY (or OPENAI_API_KEY) is required when provider=deepseek")
        base_url = _normalize_deepseek_openai_base_url(
            os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        )
        raw_m = model or os.environ.get("OPENAI_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
        m = _deepseek_openai_model_id(raw_m, base_url)
        inner = ChatOpenAI(
            model=m,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )
        return _RetryingLLM(inner, provider="openai", temperature=temperature, model=m)

    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            gm = _normalize_gemini_model(model or os.environ.get("GEMINI_MODEL") or "gemini-1.5-flash")
            inner = ChatGoogleGenerativeAI(
                model=gm,
                temperature=temperature,
            )
            return _RetryingLLM(inner, provider="gemini", temperature=temperature, model=gm)
        except ImportError:
            pass

    if openai_key:
        inner = ChatOpenAI(
            model=model or os.environ.get("OPENAI_MODEL", "gpt-4o"),
            temperature=temperature,
        )
        return _RetryingLLM(inner, provider="openai", temperature=temperature, model=model)

    try:
        from langchain_anthropic import ChatAnthropic
        if anthropic_key:
            default_headers = {"authorization": f"Bearer {anthropic_key}"} if is_ofox_anthropic else None
            inner = ChatAnthropic(
                model=model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                temperature=temperature,
                base_url=anthropic_base_url,
                api_key=anthropic_key,
                default_headers=default_headers,
            )
            return _RetryingLLM(inner, provider="anthropic", temperature=temperature, model=model)
    except ImportError:
        pass

    dsk = _clean_key(os.environ.get("DEEPSEEK_API_KEY"))
    if dsk:
        base_url = _normalize_deepseek_openai_base_url(
            os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        )
        raw_m = model or os.environ.get("OPENAI_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
        m = _deepseek_openai_model_id(raw_m, base_url)
        inner = ChatOpenAI(
            model=m,
            temperature=temperature,
            api_key=dsk,
            base_url=base_url,
        )
        return _RetryingLLM(inner, provider="openai", temperature=temperature, model=m)

    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError as e:
        raise RuntimeError("No LLM provider available. Set GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY, or install Ollama chat support.") from e

    inner = ChatOllama(
        model=model or os.environ.get("OLLAMA_MODEL", "llama3"),
        temperature=temperature,
    )
    return _RetryingLLM(inner, provider="ollama", temperature=temperature, model=model)
