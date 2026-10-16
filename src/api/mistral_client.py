import os
import random
import time
from typing import Any, List, Optional

from mistralai.client import Mistral


class MistralClient:
    """
    Mistral API client for FaultyVerifierBench.

    The public interface matches the existing evaluation pipeline:

        generate(system_prompt, user_prompt) -> str
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        connect_timeout: float = 20.0,
        read_timeout: float = 240.0,
        max_retries: int = 3,
        retry_base_delay: float = 5.0,
    ) -> None:
        api_key = os.environ.get(
            "MISTRAL_API_KEY"
        )

        if not api_key:
            raise EnvironmentError(
                "MISTRAL_API_KEY is not set."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        self.client = Mistral(
            api_key=api_key
        )

    @staticmethod
    def _extract_content(
        content: Any,
    ) -> str:
        """
        Extract plain text from Mistral message content.

        Mistral normally returns a string, but the SDK may also
        represent content as a list of content chunks.
        """

        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: List[str] = []

            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                    continue

                if isinstance(part, dict):
                    text = part.get("text")

                    if text is not None:
                        text_parts.append(
                            str(text)
                        )

                    continue

                text = getattr(
                    part,
                    "text",
                    None,
                )

                if text is not None:
                    text_parts.append(
                        str(text)
                    )

            return "".join(
                text_parts
            ).strip()

        return str(content).strip()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Call the Mistral chat-completion API.

        The shared Step 08 pipeline supplies the system prompt and
        user prompt separately. They are converted into Mistral's
        required messages list.
        """

        if not isinstance(
            system_prompt,
            str,
        ):
            raise TypeError(
                "system_prompt must be a string, "
                f"not {type(system_prompt).__name__}."
            )

        if not isinstance(
            user_prompt,
            str,
        ):
            raise TypeError(
                "user_prompt must be a string, "
                f"not {type(user_prompt).__name__}."
            )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        last_error: Optional[Exception] = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            try:
                chat_response = (
                    self.client.chat.complete(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        n=1,
                    )
                )

                if not chat_response.choices:
                    raise RuntimeError(
                        "Mistral returned no "
                        "completion choices."
                    )

                choice = (
                    chat_response.choices[0]
                )

                content = (
                    choice.message.content
                )

                output_text = (
                    self._extract_content(
                        content
                    )
                )

                if not output_text:
                    raise RuntimeError(
                        "Mistral returned an "
                        "empty completion."
                    )

                return output_text

            except Exception as exc:
                last_error = exc

                if attempt >= self.max_retries:
                    break

                delay = (
                    self.retry_base_delay
                    * (2 ** (attempt - 1))
                )

                jitter = random.uniform(
                    0.0,
                    0.5,
                )

                total_delay = delay + jitter

                print(
                    "[MISTRAL] Request failed "
                    f"on attempt {attempt}/"
                    f"{self.max_retries}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    "[MISTRAL] Retrying after "
                    f"{total_delay:.2f} seconds."
                )

                time.sleep(
                    total_delay
                )

        if last_error is None:
            raise RuntimeError(
                "Mistral API call failed without "
                "a captured exception."
            )

        raise RuntimeError(
            "Mistral API call failed after "
            f"{self.max_retries} attempts: "
            f"{type(last_error).__name__}: "
            f"{last_error}"
        ) from last_error