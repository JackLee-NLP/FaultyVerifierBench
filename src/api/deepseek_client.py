from __future__ import annotations

import os
import random
import time
from typing import Any, Optional

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from src.api.base_client import BaseLLMClient

load_dotenv()


class DeepSeekClient(BaseLLMClient):
    """DeepSeek chat-completions client with robust extraction and retries."""

    _NON_RETRYABLE_ERRORS = (
        AuthenticationError,
        PermissionDeniedError,
        BadRequestError,
        UnprocessableEntityError,
    )

    _RETRYABLE_API_ERRORS = (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
    )

    def __init__(
        self,
        *args,
        thinking_enabled: bool = False,
        reasoning_effort: str = "high",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.thinking_enabled = bool(thinking_enabled)
        self.reasoning_effort = str(reasoning_effort).strip().lower()

    def _key(self) -> str:
        key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing")
        return key

    @staticmethod
    def _normalize_content(content: Any) -> str:
        """Normalize string or structured message content into text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, (list, tuple)):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _usage_summary(response: Any) -> str:
        usage = getattr(response, "usage", None)
        if usage is None:
            return "none"
        fields = (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "completion_tokens_details",
        )
        values = []
        for field in fields:
            value = getattr(usage, field, None)
            if value is not None:
                values.append(f"{field}={value!r}")
        return ", ".join(values) if values else repr(usage)

    def _extract_content(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("DeepSeek returned no choices")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise RuntimeError("DeepSeek returned a choice without a message")

        content = self._normalize_content(getattr(message, "content", None))
        reasoning = self._normalize_content(
            getattr(message, "reasoning_content", None)
        )
        finish_reason = getattr(choice, "finish_reason", None)

        if not content:
            print(
                "Empty content | "
                f"finish_reason={finish_reason!r} | "
                f"content_length={len(content)} | "
                f"reasoning_present={bool(reasoning)} | "
                f"reasoning_length={len(reasoning)} | "
                f"usage={self._usage_summary(response)}",
                flush=True,
            )
            raise RuntimeError("DeepSeek returned empty content")

        return content

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        # Empty/no-content responses are transient response failures and should
        # consume the configured retry budget. Other RuntimeErrors, such as a
        # missing API key, remain non-retryable.
        if isinstance(error, RuntimeError):
            return str(error) in {
                "DeepSeek returned empty content",
                "DeepSeek returned no choices",
                "DeepSeek returned a choice without a message",
            }
        return isinstance(error, DeepSeekClient._RETRYABLE_API_ERRORS)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        last_error: Optional[Exception] = None
        request_max_tokens = self.max_output_tokens

        for attempt in range(1, self.max_retries + 1):
            client = None
            started = time.time()
            print(
                f"Request attempt {attempt}/{self.max_retries} | "
                f"model={self.model_name} | "
                f"max_tokens={request_max_tokens} | "
                f"thinking={'enabled' if self.thinking_enabled else 'disabled'}",
                flush=True,
            )

            try:
                client = OpenAI(
                    api_key=self._key(),
                    base_url=os.getenv(
                        "DEEPSEEK_BASE_URL",
                        "https://api.deepseek.com",
                    ),
                    timeout=float(self.read_timeout),
                    max_retries=0,
                )

                request = {
		    "model": self.model_name,
		    "messages": [
			{
			    "role": "system",
			    "content": system_prompt,
			},
			{
			    "role": "user",
			    "content": user_prompt,
			},
		    ],
		    "stream": False,
		    "max_tokens": request_max_tokens,
		    "extra_body": {
			"thinking": {
			    "type": (
				"enabled"
				if self.thinking_enabled
				else "disabled"
			    )
			}
		    },
		}

                if self.thinking_enabled:
                    request["reasoning_effort"] = (
                        self.reasoning_effort
                    )

                response = client.chat.completions.create(**request)
                return self._extract_content(response)

            except Exception as error:
                last_error = error
                elapsed = time.time() - started
                print(
                    "Request failed | "
                    f"attempt={attempt}/{self.max_retries} | "
                    f"elapsed={elapsed:.2f}s | error={error!r}",
                    flush=True,
                )

                if isinstance(error, self._NON_RETRYABLE_ERRORS):
                    raise RuntimeError(
                        f"DeepSeek non-retryable error: {error}"
                    ) from error

                if not self._is_retryable(error):
                    raise

                # DeepSeek's max_tokens budget includes hidden reasoning
                # tokens. A high-effort response can therefore finish with no
                # visible JSON; give the next attempt a larger budget.
                if (
                    str(error) == "DeepSeek returned empty content"
                    and attempt < self.max_retries
                ):
                    request_max_tokens = min(request_max_tokens * 2, 32768)
                    print(
                        "retry: empty visible content; "
                        f"increasing max_tokens to {request_max_tokens}",
                        flush=True,
                    )

                if attempt < self.max_retries:
                    delay = (
                        float(self.retry_base_delay) * (2 ** (attempt - 1))
                        + random.uniform(0.0, float(self.retry_base_delay))
                    )
                    print(
                        "retry: "
                        f"next_attempt={attempt + 1}/{self.max_retries} | "
                        f"delay={delay:.2f}s",
                        flush=True,
                    )
                    time.sleep(delay)

            finally:
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass

        raise RuntimeError(
            f"DeepSeek failed after {self.max_retries} attempts: "
            f"{last_error!r}"
        ) from last_error
