import gc
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingFaceClient:
    """One loaded local causal LM with static, length-aware batch generation."""

    supports_batch_generation = True

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        connect_timeout: float = 20.0,
        read_timeout: float = 240.0,
        max_retries: int = 3,
        retry_base_delay: float = 5.0,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        enable_thinking: bool = False,
        trust_remote_code: bool = True,
        device: str = "cuda:0",
        dtype: str = "auto",
        allow_cpu_fallback: bool = False,
        attn_implementation: Optional[str] = "sdpa",
        batch_size: int = 4,
        max_batch_tokens: int = 16384,
        max_input_tokens: Optional[int] = 8192,
        oom_split_batch: bool = True,
    ) -> None:
        del connect_timeout, read_timeout, max_retries, retry_base_delay
        self.model_name = model_name
        self.model_path = Path(model_name).expanduser().resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local model directory does not exist: {self.model_path}")

        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.enable_thinking = bool(enable_thinking)
        self.trust_remote_code = bool(trust_remote_code)
        self.requested_device = str(device).strip().lower()
        self.device = self.requested_device
        self.dtype_name = str(dtype).strip().lower()
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.attn_implementation = (
            str(attn_implementation).strip() if attn_implementation else None
        )
        self.batch_size = int(batch_size)
        self.max_batch_tokens = int(max_batch_tokens)
        self.max_input_tokens = (
            int(max_input_tokens) if max_input_tokens is not None else None
        )
        self.oom_split_batch = bool(oom_split_batch)
        self.model: Optional[torch.nn.Module] = None

        if self.max_new_tokens <= 0 or self.batch_size <= 0:
            raise ValueError("max_new_tokens and batch_size must be positive")
        if self.max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")
        if self.max_input_tokens is not None and self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive or null")
        if self.do_sample and self.temperature <= 0:
            raise ValueError("temperature must be positive when sampling")

        self._validate_requested_device()
        self.torch_dtype = self._resolve_torch_dtype()
        self._print_runtime_diagnostics()

        print(f"[HUGGINGFACE] Loading tokenizer from: {self.model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
            padding_side="left",
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"[HUGGINGFACE] Loading model from: {self.model_path}", flush=True)
        kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": str(self.model_path),
            "dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": True,
            "low_cpu_mem_usage": False,
            "output_loading_info": True,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        try:
            loaded_model, loading_info = AutoModelForCausalLM.from_pretrained(**kwargs)
            self.model = loaded_model
            self._validate_loading_info(loading_info)
            self.model.to(torch.device(self.device))
            self.model.eval()
        except Exception:
            self._release_model()
            raise

        self._validate_model_device_placement()
        self.input_device = self._get_input_device()
        self._generation_lock = threading.Lock()
        print(
            "Ready | "
            f"device={self.input_device} | dtype={self.torch_dtype} | "
            f"batch_size={self.batch_size} | max_batch_tokens={self.max_batch_tokens} | "
            f"max_input_tokens={self.max_input_tokens} | attention={self.attn_implementation}",
            flush=True,
        )

    def _validate_loading_info(self, loading_info: Any) -> None:
        info = loading_info if isinstance(loading_info, dict) else {}
        missing = set(info.get("missing_keys", []) or [])
        mismatched = info.get("mismatched_keys", []) or []
        errors = info.get("error_msgs", []) or []
        tied = bool(getattr(self.model.config, "tie_word_embeddings", False))
        if "lm_head.weight" in missing and not tied:
            raise RuntimeError("lm_head.weight is missing while tie_word_embeddings=False")
        if mismatched:
            raise RuntimeError(f"Checkpoint shape mismatch: {mismatched}")
        if errors:
            raise RuntimeError(f"Checkpoint loading errors: {errors}")
        print(
            f"Loading info | missing={sorted(missing)} | "
            f"unexpected={sorted(info.get('unexpected_keys', []) or [])}",
            flush=True,
        )

    def _validate_requested_device(self) -> None:
        if self.device == "cuda":
            self.device = "cuda:0"
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                if not self.allow_cpu_fallback:
                    raise RuntimeError("CUDA requested but unavailable")
                self.device = "cpu"
                return
            index = self._get_cuda_device_index(self.device)
            if index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device does not exist: {self.device}")
            torch.cuda.set_device(index)
        elif self.device != "cpu":
            raise ValueError("device must be cpu, cuda, or cuda:N")

    @staticmethod
    def _get_cuda_device_index(device: str) -> int:
        return int(device.split(":", 1)[1]) if ":" in device else 0

    def _resolve_torch_dtype(self):
        mapping = {
            "float32": torch.float32, "fp32": torch.float32,
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        }
        if self.dtype_name in mapping:
            selected = mapping[self.dtype_name]
        elif self.dtype_name == "auto":
            selected = (
                torch.bfloat16
                if self.device.startswith("cuda") and torch.cuda.is_bf16_supported()
                else torch.float16 if self.device.startswith("cuda") else torch.float32
            )
        else:
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")
        if self.device == "cpu" and selected == torch.float16:
            raise ValueError("float16 CPU inference is unsupported")
        return selected

    def _print_runtime_diagnostics(self) -> None:
        print(
            f"torch={torch.__version__} | cuda={torch.version.cuda} | "
            f"available={torch.cuda.is_available()} | device={self.device} | "
            f"dtype={self.torch_dtype}",
            flush=True,
        )
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(self._get_cuda_device_index(self.device))
            print(
                f"GPU={p.name} | memory_gb={p.total_memory / 1024**3:.2f}",
                flush=True,
            )

    def _validate_model_device_placement(self) -> None:
        counts: Dict[str, int] = {}
        for parameter in self.model.parameters():
            counts[str(parameter.device)] = counts.get(str(parameter.device), 0) + parameter.numel()
        print(f"Parameter placement={counts}", flush=True)
        if self.device.startswith("cuda") and (counts.get("cpu", 0) or counts.get("meta", 0)):
            raise RuntimeError("GPU-only execution requested but parameters remain off GPU")

    def _get_input_device(self) -> torch.device:
        embeddings = self.model.get_input_embeddings()
        parameter = next(embeddings.parameters(), None) if embeddings is not None else None
        return parameter.device if parameter is not None else torch.device(self.device)

    def _release_model(self) -> None:
        if getattr(self, "model", None) is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
            self.model = None
        gc.collect()
        self._clear_cuda_cache()

    @staticmethod
    def _clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt.strip():
            messages.append({"role": "user", "content": user_prompt})
        if not messages:
            raise ValueError("At least one prompt must be non-empty")
        kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": self.enable_thinking}
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError as error:
            if "enable_thinking" not in str(error):
                raise
            kwargs.pop("enable_thinking")
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def estimate_prompt_tokens(self, system_prompt: str, user_prompt: str) -> int:
        prompt = self._build_prompt(system_prompt, user_prompt)
        return len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    @staticmethod
    def _extract_json_object(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            parsed = json.loads(cleaned)
            return json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else cleaned
        except json.JSONDecodeError:
            pass
        start = cleaned.find("{")
        if start < 0:
            return cleaned
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if escaped:
                escaped = False
            elif char == "\\" and in_string:
                escaped = True
            elif char == '"':
                in_string = not in_string
            elif not in_string:
                if char == "{": depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start:index + 1]
                        try:
                            parsed = json.loads(candidate)
                            return json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else cleaned
                        except json.JSONDecodeError:
                            return cleaned
        return cleaned

    def _tokenize_prompts(self, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=self.max_input_tokens is not None,
            max_length=self.max_input_tokens,
            add_special_tokens=False,
        )
        return {
            key: value.to(self.input_device, non_blocking=self.input_device.type == "cuda")
            for key, value in encoded.items()
        }

    def _generate_prompt_batch(self, prompts: Sequence[str]) -> list[str]:
        model_inputs = self._tokenize_prompts(prompts)
        input_width = int(model_inputs["input_ids"].shape[1])
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
        with self._generation_lock, torch.inference_mode():
            generated = self.model.generate(**model_inputs, **kwargs)
        outputs = []
        for row in generated:
            text = self.tokenizer.decode(row[input_width:], skip_special_tokens=True).strip()
            if not text:
                raise RuntimeError("Local model returned an empty response")
            outputs.append(self._extract_json_object(text))
        del generated, model_inputs
        return outputs

    def generate_batch(
        self,
        requests: Sequence[tuple[str, str]],
        _allow_split: bool = True,
    ) -> list[str]:
        if not requests:
            return []
        prompts = [self._build_prompt(system, user) for system, user in requests]
        try:
            return self._generate_prompt_batch(prompts)
        except torch.cuda.OutOfMemoryError as error:
            self._clear_cuda_cache()
            if self.oom_split_batch and _allow_split and len(requests) > 1:
                midpoint = len(requests) // 2
                print(
                    f"OOM for batch={len(requests)}; splitting into "
                    f"{midpoint}+{len(requests) - midpoint}",
                    flush=True,
                )
                return self.generate_batch(requests[:midpoint]) + self.generate_batch(requests[midpoint:])
            raise RuntimeError(
                "CUDA OOM during local generation. Reduce local_batch_size, "
            ) from error

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.generate_batch([(system_prompt, user_prompt)])[0]
