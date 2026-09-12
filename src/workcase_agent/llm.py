"""LLM 어댑터 — 답변 문장 생성을 어느 회사 모델로도 할 수 있게 한다.

검색·판정·담당자 라우팅은 LLM 없이 파이썬이 처리하고, 이 모듈은 '맞는 사례가 있을 때'
사례 필드를 문장으로 정리하는 데만 쓰인다. 어떤 키가 환경변수에 있느냐로 제공자가 정해진다.

  환경변수                         제공자     기본 모델
  ANTHROPIC_API_KEY                anthropic  claude-sonnet-5
  OPENAI_API_KEY                   openai     gpt-5-mini
  GEMINI_API_KEY (또는 GOOGLE_API_KEY)  gemini     gemini-3.8-flash  (무료 티어 가능)
  GROQ_API_KEY                     groq       llama-3.3-70b-versatile (무료 티어 가능)

  LLM_PROVIDER=anthropic|openai|gemini|groq  로 강제 지정, LLM_MODEL 로 모델 지정 가능.

SDK 없이 requests 만 사용한다(HTTP 형식은 세 가지: Anthropic Messages / OpenAI Chat Completions / Gemini generateContent).
"""
import os
from typing import Optional

import requests

TIMEOUT = 90

PROVIDERS = [
    # (이름, 키 환경변수들, 기본 모델, 모델 후보(앞 것이 실패하면 순서대로 시도))
    ("anthropic", ("ANTHROPIC_API_KEY",), "claude-sonnet-5", ["claude-sonnet-5"]),
    ("openai", ("OPENAI_API_KEY",), "gpt-5-mini", ["gpt-5-mini", "gpt-4.1-mini"]),
    ("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "gemini-3.8-flash",
     ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash"]),
    ("groq", ("GROQ_API_KEY",), "llama-3.3-70b-versatile",
     ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]),
]


class LLMError(RuntimeError):
    pass


def _key_for(env_names):
    for n in env_names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return None


def detect() -> Optional[dict]:
    """사용할 제공자/모델/키를 결정. 키가 하나도 없으면 None (LLM 없이 동작)."""
    forced = os.environ.get("LLM_PROVIDER", "").strip().lower()
    for name, envs, default, candidates in PROVIDERS:
        if forced and name != forced:
            continue
        key = _key_for(envs)
        if not key:
            continue
        model = os.environ.get("LLM_MODEL", "").strip() or default
        cands = [model] + [c for c in candidates if c != model]
        return {"provider": name, "model": model, "candidates": cands, "key": key}
    return None


def available() -> Optional[dict]:
    """키를 뺀 상태 정보(제공자·모델). 화면 표시용."""
    d = detect()
    return {"provider": d["provider"], "model": d["model"]} if d else None


# ── 제공자별 호출 ────────────────────────────────────────────────

def _anthropic(key, model, system, prompt, max_tokens, temperature):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": max_tokens, "temperature": temperature,
              **({"system": system} if system else {}),
              "messages": [{"role": "user", "content": prompt}]},
        timeout=TIMEOUT)
    _raise(r)
    return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")


def _openai_compat(base, key, model, system, prompt, max_tokens, temperature):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    r = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature},
        timeout=TIMEOUT)
    _raise(r)
    return r.json()["choices"][0]["message"]["content"]


def _gemini(key, model, system, prompt, max_tokens, temperature):
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json=body, timeout=TIMEOUT)
    _raise(r)
    cands = r.json().get("candidates") or []
    if not cands:
        raise LLMError("gemini: 응답에 candidates 가 없습니다 (안전 필터 등)")
    return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))


def _raise(r):
    if r.status_code >= 400:
        raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")


def generate(prompt: str, system: Optional[str] = None,
             max_tokens: int = 1800, temperature: float = 0.2) -> str:
    """프롬프트 → 텍스트. 키가 없으면 LLMError. 모델 후보를 순서대로 시도한다."""
    d = detect()
    if not d:
        raise LLMError("LLM API 키가 설정되어 있지 않습니다.")
    last = None
    for model in d["candidates"]:
        try:
            if d["provider"] == "anthropic":
                return _anthropic(d["key"], model, system, prompt, max_tokens, temperature)
            if d["provider"] == "openai":
                return _openai_compat("https://api.openai.com/v1", d["key"], model, system, prompt, max_tokens, temperature)
            if d["provider"] == "groq":
                return _openai_compat("https://api.groq.com/openai/v1", d["key"], model, system, prompt, max_tokens, temperature)
            if d["provider"] == "gemini":
                return _gemini(d["key"], model, system, prompt, max_tokens, temperature)
        except LLMError as e:
            last = e
            # 모델 이름 오류(404/400)나 한도 초과(429)면 다음 후보 시도, 그 외는 즉시 실패
            if not any(code in str(e) for code in ("HTTP 404", "HTTP 400", "HTTP 429")):
                raise
        except requests.RequestException as e:
            raise LLMError(f"네트워크 오류: {e}") from e
    raise LLMError(f"{d['provider']}: 모든 모델 후보 실패 — {last}")
