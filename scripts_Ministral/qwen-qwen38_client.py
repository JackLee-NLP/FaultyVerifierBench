"""Linux-only Qwen3.8-27B client for the isolated scripts_qwen38 launcher."""
from __future__ import annotations

import gc
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping

import psutil
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, TextStreamer

GIB = 1024 ** 3


class Qwen38LinuxClient:
    """Raw BF16 Qwen3.8-27B with static CUDA+CPU placement and RAM guards."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        connect_timeout: float = 30.0,
        read_timeout: float = 240.0,
        max_retries: int = 3,
        retry_base_delay: float = 5.0,
        max_new_tokens: int = 2048,
        max_output_tokens: int | None = None,
        do_sample: bool = False,
        enable_thinking: bool = False,
        trust_remote_code: bool = True,
        **_: Any,
    ) -> None:
        del connect_timeout, read_timeout, max_retries, retry_base_delay
        self.model_path = Path(model_name).expanduser().resolve()
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_output_tokens or max_new_tokens)
        self.do_sample = bool(do_sample)
        self.enable_thinking = bool(enable_thinking)
        self.trust_remote_code = bool(trust_remote_code)
        self.stream_output = os.getenv("QWEN38_STREAM_OUTPUT", "1") != "0"
        self.reserve_ram_gib = float(os.getenv("QWEN38_RAM_RESERVE_GIB", "3"))
        self.hard_floor_gib = float(os.getenv("QWEN38_RAM_HARD_FLOOR_GIB", "2"))
        self.gpu_reserve_gib = float(os.getenv("QWEN38_GPU_RESERVE_GIB", "3"))
        self._generation_lock = threading.Lock()
        self.last_generation_metadata: dict[str, Any] = {}
        self.processor: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.input_device = torch.device("cpu")

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Qwen3.8 model directory not found: {self.model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3.8 Linux client requires a working CUDA GPU.")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        self._assert_ram_available("before model loading")
        self.max_memory = self._resolve_max_memory()
        self._print_runtime()
        self._load()

    def _assert_ram_available(self, stage: str) -> None:
        available = psutil.virtual_memory().available / GIB
        if available <= self.hard_floor_gib:
            raise MemoryError(
                f"RAM safety floor reached {stage}: available={available:.2f}GiB, "
                f"required_above={self.hard_floor_gib:.2f}GiB"
            )

    def _resolve_max_memory(self) -> dict[int | str, str]:
        available_ram = psutil.virtual_memory().available / GIB
        # Accelerate's CPU allocation ceiling is below current MemAvailable by the
        # configured reserve. The watchdog independently enforces the 2 GiB floor.
        cpu_limit = int(available_ram - self.reserve_ram_gib)
        if cpu_limit < 1:
            raise MemoryError(
                f"Insufficient free RAM: available={available_ram:.2f}GiB, "
                f"reserve={self.reserve_ram_gib:.2f}GiB"
            )

        props = torch.cuda.get_device_properties(0)
        total_gpu = props.total_memory / GIB
        free_gpu, _ = torch.cuda.mem_get_info(0)
        free_gpu_gib = free_gpu / GIB
        gpu_limit = int(min(total_gpu, free_gpu_gib) - self.gpu_reserve_gib)
        if gpu_limit < 1:
            raise MemoryError(
                f"Insufficient free VRAM: free={free_gpu_gib:.2f}GiB, "
                f"reserve={self.gpu_reserve_gib:.2f}GiB"
            )
        return {0: f"{gpu_limit}GiB", "cpu": f"{cpu_limit}GiB"}

    def _print_runtime(self) -> None:
        vm = psutil.virtual_memory()
        print(
            "[QWEN3.8] Linux runtime | "
            f"torch={torch.__version__} | cuda_build={torch.version.cuda} | "
            f"gpu={torch.cuda.get_device_name(0)} | "
            f"available_ram_gib={vm.available / GIB:.2f} | "
            f"hard_floor_gib={self.hard_floor_gib:.2f} | "
            f"max_memory={self.max_memory} | max_new_tokens={self.max_new_tokens}",
            flush=True,
        )

    def _load(self) -> None:
        os.environ.setdefault(
            "PYTORCH_ALLOC_CONF",
            "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7",
        )
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[QWEN3.8] Loading processor: {self.model_path}", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._assert_ram_available("immediately before from_pretrained")
        print("[QWEN3.8] Loading BF16 model with static CUDA+CPU placement", flush=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            str(self.model_path),
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory=self.max_memory,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            offload_buffers=True,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
            use_safetensors=True,
        )
        self.model.eval()
        self._validate_placement()
        self.input_device = self._resolve_input_device()
        self._assert_ram_available("after model loading")
        print(
            f"[QWEN3.8] Model ready | input_device={self.input_device} | "
            f"available_ram_gib={psutil.virtual_memory().available / GIB:.2f}",
            flush=True,
        )

    def _validate_placement(self) -> None:
        device_map = getattr(self.model, "hf_device_map", None)
        if not isinstance(device_map, Mapping):
            raise RuntimeError("Model did not expose hf_device_map after device_map='auto'.")
        disk = [name for name, place in device_map.items() if str(place) == "disk"]
        explicit_meta = [name for name, place in device_map.items() if str(place) == "meta"]
        gpu = [name for name, place in device_map.items() if str(place) in {"0", "cuda", "cuda:0"}]
        cpu = [name for name, place in device_map.items() if str(place) == "cpu"]
        if disk:
            raise RuntimeError(f"Disk offload is forbidden by RAM safety policy: {disk[:20]}")
        if explicit_meta:
            raise RuntimeError(f"Explicit meta destinations found: {explicit_meta[:20]}")
        if not gpu:
            raise RuntimeError("Automatic placement assigned no modules to CUDA.")
        print(
            f"[QWEN3.8] Placement passed | gpu_modules={len(gpu)} | "
            f"cpu_modules={len(cpu)} | disk_modules=0",
            flush=True,
        )

    def _resolve_input_device(self) -> torch.device:
        embeddings = self.model.get_input_embeddings()
        if embeddings is not None:
            parameter = next(embeddings.parameters(), None)
            if parameter is not None and parameter.device.type != "meta":
                return parameter.device
        for place in self.model.hf_device_map.values():
            text = str(place)
            if text in {"0", "cuda", "cuda:0"}:
                return torch.device("cuda:0")
        return torch.device("cpu")

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        if user_prompt.strip():
            messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})
        if not messages:
            raise ValueError("At least one prompt must be non-empty.")
        return messages

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._assert_ram_available("before generation")
        values = self.processor.apply_chat_template(
            self._messages(system_prompt, user_prompt),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        )
        inputs = {
            key: value.to(self.input_device)
            for key, value in values.items()
            if isinstance(value, torch.Tensor)
        }
        input_length = int(inputs["input_ids"].shape[1])
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            kwargs.update({"temperature": self.temperature, "top_p": 0.8, "top_k": 20})
        if self.stream_output:
            kwargs["streamer"] = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        print(
            f"[QWEN3.8] Generation started | input_tokens={input_length} | "
            f"max_new_tokens={self.max_new_tokens} | stream={self.stream_output}",
            flush=True,
        )
        if self.stream_output:
            print("[QWEN3.8] STREAM BEGIN", flush=True)
        with self._generation_lock, torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        if self.stream_output:
            print("\n[QWEN3.8] STREAM END", flush=True)

        new_ids = output[0, input_length:]
        generated_tokens = int(new_ids.shape[0])
        reached_limit = generated_tokens >= self.max_new_tokens
        text = self.processor.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        self.last_generation_metadata = {
            "generated_tokens": generated_tokens,
            "max_new_tokens": self.max_new_tokens,
            "reached_max_new_tokens": reached_limit,
            "stopped_before_limit": not reached_limit,
            "output_characters": len(text),
        }
        print(
            "[QWEN3.8] Generation completed | "
            f"generated_tokens={generated_tokens} | max_new_tokens={self.max_new_tokens} | "
            f"reached_max_new_tokens={reached_limit} | "
            f"available_ram_gib={psutil.virtual_memory().available / GIB:.2f}",
            flush=True,
        )
        if not text:
            raise RuntimeError("Qwen3.8 returned an empty response.")
        return self._extract_json(text)

    @staticmethod
    def _extract_json(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            value = json.loads(cleaned)
            return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else cleaned
        except json.JSONDecodeError:
            return cleaned

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
