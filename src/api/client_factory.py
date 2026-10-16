from src.api.deepseek_client import DeepSeekClient
from src.api.gemini_client import GeminiClient
from src.api.huggingface_client import HuggingFaceClient
from src.api.mistral_client import MistralClient
from src.api.openai_client import OpenAIClient


def get_client(
    provider: str,
    model_name: str,
    temperature: float = 0.7,
    connect_timeout: float = 20.0,
    read_timeout: float = 240.0,
    max_retries: int = 3,
    retry_base_delay: float = 5.0,
    max_output_tokens: int = 2048,
    gemini_thinking_level: str = "high",
    gemini_include_thoughts: bool = False,
    deepseek_thinking_enabled: bool = False,
    deepseek_reasoning_effort: str = "high",
    local_do_sample: bool = False,
    local_enable_thinking: bool = False,
    local_trust_remote_code: bool = True,
    local_device: str = "cuda:0",
    local_dtype: str = "auto",
    local_attn_implementation: str | None = "sdpa",
    local_batch_size: int = 4,
    local_max_batch_tokens: int = 16384,
    local_max_input_tokens: int | None = 8192,
    local_oom_split_batch: bool = True,
):
    provider = provider.strip().lower()
    max_output_tokens = int(max_output_tokens)
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be greater than zero")

    common = dict(
        model_name=model_name,
        temperature=temperature,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        max_output_tokens=max_output_tokens,
    )

    if provider == "openai":
        # In this project, provider=openai is the OpenAI-compatible OpenRouter
        # transport. Reuse the existing evaluation-level reasoning switch so
        # OpenRouter models run under the same explicit-reasoning condition as
        # DeepSeek.
        return OpenAIClient(
            **common,
            reasoning_enabled=deepseek_thinking_enabled,
        )

    if provider == "deepseek":
        return DeepSeekClient(
            **common,
            thinking_enabled=deepseek_thinking_enabled,
            reasoning_effort=deepseek_reasoning_effort,
        )

    if provider == "gemini":
        return GeminiClient(
            **common,
            thinking_level=gemini_thinking_level,
            include_thoughts=gemini_include_thoughts,
        )

    if provider == "mistral":
        return MistralClient(**common)

    if provider == "huggingface":
        local = dict(common)
        local.pop("max_output_tokens")
        return HuggingFaceClient(
            **local,
            max_new_tokens=max_output_tokens,
            do_sample=local_do_sample,
            enable_thinking=local_enable_thinking,
            trust_remote_code=local_trust_remote_code,
            device=local_device,
            dtype=local_dtype,
            attn_implementation=local_attn_implementation,
            batch_size=local_batch_size,
            max_batch_tokens=local_max_batch_tokens,
            max_input_tokens=local_max_input_tokens,
            oom_split_batch=local_oom_split_batch,
        )

    raise ValueError(f"Unsupported provider: {provider}")
