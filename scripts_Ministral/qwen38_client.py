"""Isolated Linux local-model client used by scripts_qwen38.

Despite the historical filename, this client supports both Qwen3.8 and
Ministral 3 multimodal Transformers checkpoints without modifying src/.
"""
from __future__ import annotations

import gc
import json
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import psutil
import torch
from transformers import (
    AutoConfig,
    AutoModelForMultimodalLM,
    AutoProcessor,
    TextStreamer,
)

GIB = 1024 ** 3


class Qwen38LinuxClient:
    """Memory-guarded local multimodal Hugging Face inference client."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
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
        self.gpu_reserve_gib = float(os.getenv("QWEN38_GPU_RESERVE_GIB", "2"))

        self.processor: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.model_config: Any = None
        self.input_device = torch.device("cpu")
        self.last_generation_metadata: dict[str, Any] = {}
        self._generation_lock = threading.Lock()

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {self.model_path}")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("Local Linux client requires a working CUDA GPU")

        self._assert_ram_available("before model loading")
        self.max_memory = self._resolve_max_memory()
        self._print_runtime()

        try:
            self._load()
        except Exception:
            self.close()
            raise

    @property
    def model_label(self) -> str:
        return self.model_path.name

    @property
    def is_qwen(self) -> bool:
        return "qwen" in self.model_label.lower()

    @property
    def is_mistral_family(self) -> bool:
        lowered = self.model_label.lower()
        return "mistral" in lowered or "ministral" in lowered

    def _assert_ram_available(self, stage: str) -> None:
        available = psutil.virtual_memory().available / GIB
        if available <= self.hard_floor_gib:
            raise MemoryError(
                f"RAM safety floor reached {stage}: available={available:.2f}GiB, "
                f"required_above={self.hard_floor_gib:.2f}GiB"
            )

    def _resolve_max_memory(self) -> dict[int | str, str]:
        available_ram = psutil.virtual_memory().available / GIB
        cpu_limit = int(available_ram - self.reserve_ram_gib)
        if cpu_limit < 1:
            raise MemoryError(
                f"Insufficient free RAM: available={available_ram:.2f}GiB, "
                f"reserve={self.reserve_ram_gib:.2f}GiB"
            )

        free_gpu_bytes, total_gpu_bytes = torch.cuda.mem_get_info(0)
        free_gpu_gib = free_gpu_bytes / GIB
        total_gpu_gib = total_gpu_bytes / GIB
        gpu_limit = int(min(free_gpu_gib, total_gpu_gib) - self.gpu_reserve_gib)
        if gpu_limit < 1:
            raise MemoryError(
                f"Insufficient free VRAM: free={free_gpu_gib:.2f}GiB, "
                f"reserve={self.gpu_reserve_gib:.2f}GiB"
            )
        return {0: f"{gpu_limit}GiB", "cpu": f"{cpu_limit}GiB"}

    def _print_runtime(self) -> None:
        print(
            "[LOCAL MODEL] Linux runtime | "
            f"model={self.model_label} | torch={torch.__version__} | "
            f"cuda_build={torch.version.cuda} | gpu={torch.cuda.get_device_name(0)} | "
            f"available_ram_gib={psutil.virtual_memory().available / GIB:.2f} | "
            f"hard_floor_gib={self.hard_floor_gib:.2f} | "
            f"max_memory={self.max_memory} | max_new_tokens={self.max_new_tokens}",
            flush=True,
        )

    def _load(self) -> None:
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7",
        )
        gc.collect()
        torch.cuda.empty_cache()

        self.model_config = AutoConfig.from_pretrained(
            str(self.model_path),
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )
        quant_config = getattr(self.model_config, "quantization_config", None)
        print(
            "[LOCAL MODEL] Checkpoint configuration | "
            f"model_type={getattr(self.model_config, 'model_type', None)} | "
            f"quantized={quant_config is not None} | "
            f"quantization_config={quant_config}",
            flush=True,
        )

        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": True,
        }
        if self.is_mistral_family:
            processor_kwargs["fix_mistral_regex"] = True

        print(f"[LOCAL MODEL] Loading processor: {self.model_path}", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            **processor_kwargs,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
            self.tokenizer.pad_token_id = eos_token_id

        self._assert_ram_available("immediately before from_pretrained")
        print(
            "[LOCAL MODEL] Loading checkpoint with static CUDA+CPU placement | "
            "compute_dtype=torch.bfloat16",
            flush=True,
        )
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
            "[LOCAL MODEL] Model ready | "
            f"input_device={self.input_device} | "
            f"available_ram_gib={psutil.virtual_memory().available / GIB:.2f}",
            flush=True,
        )

    def _validate_placement(self) -> None:
        device_map = getattr(self.model, "hf_device_map", None)

        if isinstance(device_map, Mapping):
            disk_modules = [
                name for name, placement in device_map.items()
                if str(placement) == "disk"
            ]
            explicit_meta = [
                name for name, placement in device_map.items()
                if str(placement) == "meta"
            ]
            gpu_modules = [
                name for name, placement in device_map.items()
                if str(placement) in {"0", "cuda", "cuda:0"}
            ]
            cpu_modules = [
                name for name, placement in device_map.items()
                if str(placement) == "cpu"
            ]
            if disk_modules:
                raise RuntimeError(
                    "Disk offload is forbidden by RAM safety policy: "
                    f"{disk_modules[:20]}"
                )
            if explicit_meta:
                raise RuntimeError(f"Explicit meta destinations found: {explicit_meta[:20]}")
            if not gpu_modules:
                raise RuntimeError("Automatic placement assigned no modules to CUDA")
            print(
                "[LOCAL MODEL] Placement passed via hf_device_map | "
                f"gpu_modules={len(gpu_modules)} | cpu_modules={len(cpu_modules)} | "
                "disk_modules=0",
                flush=True,
            )
            return

        parameter_devices = Counter(str(p.device) for p in self.model.parameters())
        buffer_devices = Counter(str(b.device) for b in self.model.buffers())
        meta_parameters = [
            name for name, parameter in self.model.named_parameters()
            if parameter.device.type == "meta"
        ]
        cuda_parameters = sum(
            count for device, count in parameter_devices.items()
            if device.startswith("cuda")
        )
        cpu_parameters = parameter_devices.get("cpu", 0)

        if meta_parameters:
            raise RuntimeError(
                "Model contains unresolved meta parameters without hf_device_map: "
                f"{meta_parameters[:20]}"
            )
        if cuda_parameters == 0:
            raise RuntimeError(
                "Loaded model has no parameters on CUDA: "
                f"{dict(parameter_devices)}"
            )

        print(
            "[LOCAL MODEL] Placement passed via actual parameter devices | "
            "hf_device_map=absent | "
            f"cuda_parameters={cuda_parameters} | cpu_parameters={cpu_parameters} | "
            f"parameter_devices={dict(parameter_devices)} | "
            f"buffer_devices={dict(buffer_devices)}",
            flush=True,
        )

    def _resolve_input_device(self) -> torch.device:
        embeddings = self.model.get_input_embeddings()
        if embeddings is not None:
            parameter = next(embeddings.parameters(), None)
            if parameter is not None and parameter.device.type != "meta":
                return parameter.device

        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, Mapping):
            for placement in device_map.values():
                if str(placement) in {"0", "cuda", "cuda:0"}:
                    return torch.device("cuda:0")

        for parameter in self.model.parameters():
            if parameter.device.type == "cuda":
                return parameter.device
        raise RuntimeError("Unable to identify a CUDA input device")

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        if user_prompt.strip():
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            )
        if not messages:
            raise ValueError("At least one prompt must be non-empty")
        return messages

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._assert_ram_available("before generation")
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self.is_qwen:
            template_kwargs["enable_thinking"] = self.enable_thinking

        values = self.processor.apply_chat_template(
            self._messages(system_prompt, user_prompt),
            **template_kwargs,
        )
        inputs = {
            key: value.to(self.input_device)
            for key, value in values.items()
            if isinstance(value, torch.Tensor)
        }
        if "input_ids" not in inputs:
            raise RuntimeError("Processor did not return input_ids")
        input_length = int(inputs["input_ids"].shape[1])

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            generation_kwargs.update(
                {"temperature": self.temperature, "top_p": 0.8, "top_k": 20}
            )
        if self.stream_output:
            generation_kwargs["streamer"] = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        print(
            "[LOCAL MODEL] Generation started | "
            f"input_tokens={input_length} | max_new_tokens={self.max_new_tokens} | "
            f"stream={self.stream_output}",
            flush=True,
        )
        if self.stream_output:
            print("[LOCAL MODEL] STREAM BEGIN", flush=True)

        with self._generation_lock, torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)

        if self.stream_output:
            print("\n[LOCAL MODEL] STREAM END", flush=True)

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
            "[LOCAL MODEL] Generation completed | "
            f"generated_tokens={generated_tokens} | "
            f"max_new_tokens={self.max_new_tokens} | "
            f"reached_max_new_tokens={reached_limit} | "
            f"available_ram_gib={psutil.virtual_memory().available / GIB:.2f}",
            flush=True,
        )
        if not text:
            raise RuntimeError("Local model returned an empty response")
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
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        return cleaned

    def get_last_generation_metadata(self) -> dict[str, Any]:
        return dict(self.last_generation_metadata)

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            try:
                from accelerate.hooks import remove_hook_from_module

                remove_hook_from_module(model, recurse=True)
            except Exception:
                pass

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_config = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass
