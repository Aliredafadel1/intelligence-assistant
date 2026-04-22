from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_env_path() -> Path:
    """
    Resolve .env robustly:
    1) project root next to src/
    2) current working directory and its parents
    """
    candidate = PROJECT_ROOT / ".env"
    if candidate.exists():
        return candidate

    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        p = base / ".env"
        if p.exists():
            return p
    return candidate


ENV_PATH = resolve_env_path()

MANAGED_KEYS = {
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "LLM_MODEL",
}


def load_env_file(env_path: Path) -> None:
    """
    Load selected environment variables from a .env file into os.environ.
    Existing non-empty values in os.environ are preserved unless the key is managed.
    """
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if not key:
            continue

        if key in MANAGED_KEYS:
            os.environ[key] = value
            continue

        current = os.environ.get(key)
        if current is None or not str(current).strip() or "YOUR_" in str(current):
            os.environ[key] = value


# Load .env as soon as module imports
load_env_file(ENV_PATH)


def get_llm_settings(debug: bool = False) -> tuple[str, str]:
    """
    Return (provider, api_key).

    Supported providers:
    - openai
    - openrouter
    - groq
    """
    provider = (os.environ.get("LLM_PROVIDER") or "groq").strip().lower()
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()

    if debug:
        print(f"ENV_PATH: {ENV_PATH}")
        print(f"ENV exists: {ENV_PATH.exists()}")
        print(f"provider: {provider!r}")
        print(f"openai exists: {bool(openai_key)}")
        print(f"openai len: {len(openai_key)}")
        print(f"groq exists: {bool(groq_key)}")
        print(f"groq prefix: {groq_key[:4]!r}")
        print(f"groq len: {len(groq_key)}")
        print(f"openrouter exists: {bool(openrouter_key)}")
        print(f"openrouter len: {len(openrouter_key)}")

    # Auto-fallback provider selection if chosen one is not configured.
    if provider == "openrouter" and not openrouter_key:
        provider = "groq" if groq_key else ("openai" if openai_key else "openrouter")
    if provider == "groq" and not groq_key:
        provider = "openai" if openai_key else ("openrouter" if openrouter_key else "groq")
    if provider == "openai" and not openai_key:
        provider = "groq" if groq_key else ("openrouter" if openrouter_key else "openai")

    if provider == "openai":
        if not openai_key or "YOUR_OPENAI_API_KEY" in openai_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is missing or invalid. Add a real key to .env or environment."
            )
        return provider, openai_key

    if provider == "openrouter":
        if not openrouter_key or "YOUR_OPENROUTER_KEY" in openrouter_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is missing or invalid. Add a real key to .env or environment."
            )
        return provider, openrouter_key

    if provider == "groq":
        if not groq_key:
            raise EnvironmentError(
                "GROQ_API_KEY is missing or invalid. Add a real Groq key to .env or environment."
            )
        if "YOUR_GROQ_KEY" in groq_key or "REPLACE_WITH_REAL_GROQ_KEY" in groq_key:
            raise EnvironmentError(
                "GROQ_API_KEY is still placeholder text. Replace it with your real Groq key."
            )
        return provider, groq_key

    raise ValueError("LLM_PROVIDER must be 'openai', 'openrouter', or 'groq'.")


def build_client(debug: bool = False) -> OpenAI:
    provider, api_key = get_llm_settings(debug=debug)

    if provider == "openai":
        return OpenAI(api_key=api_key)

    if provider == "openrouter":
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )


def default_model() -> str:
    provider = (os.environ.get("LLM_PROVIDER") or "groq").strip().lower()
    configured = (os.environ.get("LLM_MODEL") or "").strip()

    if provider == "openai":
        if not configured:
            return "gpt-4o-mini"
        return configured

    if provider == "groq":
        if not configured or "/" in configured or configured.endswith(":free"):
            return "llama-3.1-8b-instant"
        return configured

    if configured:
        return configured

    return "meta-llama/llama-3.1-8b-instruct:free"


def local_fallback_response(prompt: str) -> str:
    text = prompt.lower()
    mode = "rag" if "retrieved context" in text else "non_rag"

    if any(k in text for k in ("down", "outage", "cannot login", "can't login", "payment failed", "failed payment")):
        priority = "P1"
        confidence = 0.76
    elif any(k in text for k in ("slow", "delay", "issue", "bug", "error")):
        priority = "P2"
        confidence = 0.62
    else:
        priority = "P3"
        confidence = 0.55

    return json.dumps(
        {
            "priority": priority,
            "confidence": confidence,
            "rationale": f"Local fallback ({mode}) used because hosted LLM key is missing/invalid.",
            "next_action": "Collect account details, verify scope, and route to support owner.",
        }
    )


def generate_text(
    prompt: str,
    *,
    system_prompt: str = "You are a helpful assistant for support-ticket triage.",
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 300,
    allow_fallback: bool = True,
    debug: bool = False,
) -> str:
    if not prompt.strip():
        raise ValueError("Prompt is empty.")

    try:
        client = build_client(debug=debug)
    except Exception:
        if allow_fallback:
            return local_fallback_response(prompt)
        raise

    target_model = model or default_model()

    try:
        resp = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception:
        if allow_fallback:
            return local_fallback_response(prompt)
        raise

    content = resp.choices[0].message.content
    return "" if content is None else content.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text via Groq or OpenRouter."
    )
    parser.add_argument("--prompt", type=str, required=True, help="User prompt text.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant for support-ticket triage.",
        help="System instruction.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--no-fallback", action="store_true", help="Disable local fallback.")
    parser.add_argument("--json", action="store_true", help="Return result as JSON.")
    parser.add_argument("--debug", action="store_true", help="Print safe env debug info.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    text = generate_text(
        args.prompt,
        system_prompt=args.system_prompt,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        allow_fallback=not args.no_fallback,
        debug=args.debug,
    )

    if args.json:
        print(json.dumps({"model": args.model or default_model(), "text": text}, indent=2))
    else:
        print(text.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()