from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        connect_timeout: float = 20.0,
        read_timeout: float = 240.0,
        max_retries: int = 3,
        retry_base_delay: float = 5.0,
        max_output_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_base_delay = retry_base_delay
        self.max_output_tokens = int(max_output_tokens)
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError
