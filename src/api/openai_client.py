from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .base_client import BaseLLMClient

load_dotenv()


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible client configured for the OpenRouter endpoint."""

    def __init__(self, *args: Any, reasoning_enabled: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reasoning_enabled = bool(reasoning_enabled)

    def _key(self) -> str:
        # This client defaults to OpenRouter, so require its matching key rather
        # than accidentally sending OPENAI_API_KEY to the OpenRouter endpoint.
        key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return key

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = OpenAI(
            api_key=self._key(),
            base_url=os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
            timeout=float(self.read_timeout),
        )

        try:
            headers = {}
            if os.getenv("OPENROUTER_REFERER"):
                headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERER")
            if os.getenv("OPENROUTER_TITLE"):
                headers["X-OpenRouter-Title"] = os.getenv("OPENROUTER_TITLE")

            reasoning = {"enabled": self.reasoning_enabled}
            print(
                "Request | "
                f"model={self.model_name} | "
                f"max_tokens={self.max_output_tokens} | "
                f"temperature={self.temperature} | "
                f"reasoning_enabled={self.reasoning_enabled}",
                flush=True,
            )

            response = client.chat.completions.create(
                model=self.model_name,
                temperature=float(self.temperature),
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # reasoning is OpenRouter-specific. The OpenAI Python SDK passes
                # non-OpenAI request fields through extra_body.
                extra_body={"reasoning": reasoning},
                extra_headers=headers or None,
            )

            if not response.choices:
                raise RuntimeError(
                    "OpenRouter returned no choices "
                    f"(model={self.model_name})"
                )

            choice = response.choices[0]
            message = choice.message
            content = str(message.content or "").strip()
            if not content:
                refusal = str(getattr(message, "refusal", "") or "").strip()
                reasoning = str(getattr(message, "reasoning", "") or "").strip()
                details = [
                    f"model={self.model_name}",
                    f"finish_reason={getattr(choice, 'finish_reason', None)}",
                ]
                if refusal:
                    details.append(f"refusal={refusal[:200]}")
                if reasoning:
                    details.append(
                        "reasoning_present=true; increase max_tokens or "
                        "reduce reasoning effort"
                    )
                raise RuntimeError(
                    "OpenRouter returned empty visible content ("
                    + ", ".join(details)
                    + ")"
                )
            return content
        finally:
            client.close()
