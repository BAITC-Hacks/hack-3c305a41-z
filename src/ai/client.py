"""Lazy OpenAI access. Import failures and missing keys degrade, never crash."""
import json
import os
from typing import Any


class LLMUnavailable(RuntimeError):
    """Raised when the narrative layer cannot run; callers keep offline text."""


def build_client(config: dict[str, Any]):
    if not os.environ.get(config["api_key_env"]):
        raise LLMUnavailable(f"Переменная {config['api_key_env']} не задана")
    try:
        from openai import OpenAI
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise LLMUnavailable("Пакет openai не установлен: pip install -r requirements-llm.txt") from error
    return OpenAI(api_key=os.environ[config["api_key_env"]], timeout=config["timeout_seconds"])


def ask_json(client, config: dict[str, Any], model: str, system: str, user: str) -> dict[str, Any]:
    """One structured call. Retries once, then gives up so the caller can fall back."""
    last_error: Exception | None = None
    for _ in range(config["retries"] + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=config["temperature"],
                max_tokens=config["max_output_tokens"],
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return json.loads(response.choices[0].message.content)
        except Exception as error:  # noqa: BLE001 - any API or parsing failure is recoverable
            last_error = error
    raise LLMUnavailable(f"Вызов модели не удался: {last_error}")
