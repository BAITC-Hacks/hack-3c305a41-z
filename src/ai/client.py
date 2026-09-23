"""Lazy OpenAI access. Import failures and missing keys degrade, never crash."""
import json
import os
import re
from typing import Any

UNSUPPORTED = re.compile(r"[Uu]nsupported (?:parameter|value): '([a-z_]+)'")


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


def create_chat(client, config: dict[str, Any], model: str, messages: list[dict], **extra):
    """One request, adapted to whatever the chosen model accepts.

    Model families disagree on parameters: newer ones reject `max_tokens` and
    want `max_completion_tokens`, and some refuse a custom temperature. Rather
    than pinning the code to one family, an unsupported parameter named in the
    error is renamed or dropped and the call is repeated. A jury changing the
    model in config.yaml should not have to change the code.
    """
    params = {"model": model, "messages": messages, "temperature": config["temperature"],
              "max_tokens": config["max_output_tokens"], **extra}
    for _ in range(len(params)):
        try:
            return client.chat.completions.create(**params)
        except Exception as error:  # noqa: BLE001 - the message is the only signal we have
            match = UNSUPPORTED.search(str(error))
            name = match.group(1) if match else None
            if name not in params:
                raise
            if name == "max_tokens":
                params["max_completion_tokens"] = params.pop("max_tokens")
            else:
                params.pop(name)
    raise LLMUnavailable(f"Модель {model} отклонила все поддерживаемые наборы параметров")


def ask_json(client, config: dict[str, Any], model: str, system: str, user: str) -> dict[str, Any]:
    """One structured call. Retries once, then gives up so the caller can fall back."""
    last_error: Exception | None = None
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for _ in range(config["retries"] + 1):
        try:
            response = create_chat(client, config, model, messages, response_format={"type": "json_object"})
            return json.loads(response.choices[0].message.content)
        except Exception as error:  # noqa: BLE001 - any API or parsing failure is recoverable
            last_error = error
    raise LLMUnavailable(f"Вызов модели не удался: {last_error}")
