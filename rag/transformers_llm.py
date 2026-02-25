import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", os.getenv("TRANSFORMERS_VERBOSITY", "error"))

from typing import Any, Iterator, Optional, List
import re
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    GenerationConfig,
)

from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk, LLMResult, Generation


class TransformersLLM(BaseLLM):

    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    model: Any = None
    tokenizer: Any = None

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        quantization: str = "none",
        **kwargs,
    ):
        env_model = os.getenv("TRANSFORMERS_MODEL")
        env_quant = os.getenv("QUANTIZATION")
        env_device = os.getenv("LLM_DEVICE")
        env_max_tokens = os.getenv("MAX_NEW_TOKENS")
        env_temp = os.getenv("TEMPERATURE")

        if env_model:
            model_name = env_model
        if env_quant:
            quantization = env_quant
        if env_device:
            device = env_device
        if env_max_tokens:
            try:
                max_new_tokens = int(env_max_tokens)
            except Exception:
                pass
        if env_temp:
            try:
                temperature = float(env_temp)
            except Exception:
                pass

        super().__init__(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        print(f"Loading model: {model_name}")
        if quantization == "4bit":
            print("Using 4-bit quantization (reduces memory ~75%)")
        elif quantization == "8bit":
            print("Using 8-bit quantization (reduces memory ~50%)")
        else:
            print("This will download ~28GB on first run...")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        dtype_obj = dtype_map.get(dtype, "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model_kwargs = {"device_map": device, "trust_remote_code": True}
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["dtype"] = dtype_obj

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        self._generation_lock = None
        try:
            import threading

            self._generation_lock = threading.Lock()
        except Exception:
            self._generation_lock = None

        print(f"✓ Model loaded successfully on {self.model.device}")

    @property
    def _llm_type(self) -> str:
        return "transformers"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        temp = float(kwargs.get("temperature", self.temperature))
        do_sample = bool(kwargs.get("do_sample", temp > 0))
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", None)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.1,
        )

        try:
            gen_cfg = GenerationConfig(do_sample=do_sample)
            if do_sample:
                gen_cfg.temperature = temp
                if top_p is not None:
                    gen_cfg.top_p = float(top_p)
                if top_k is not None:
                    gen_cfg.top_k = int(top_k)
            gen_kwargs["generation_config"] = gen_cfg
        except Exception:
            gen_kwargs["do_sample"] = do_sample
            if do_sample:
                gen_kwargs["temperature"] = temp
                if top_p is not None:
                    gen_kwargs["top_p"] = float(top_p)
                if top_k is not None:
                    gen_kwargs["top_k"] = int(top_k)

        with torch.no_grad():
            lock = getattr(self, "_generation_lock", None)
            if lock is not None:
                lock.acquire()
            try:
                outputs = self.model.generate(**gen_kwargs)
            finally:
                if lock is not None:
                    lock.release()

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        def _clean_warnings(s: str) -> str:
            if not s:
                return s
            patterns = [
                r"The following generation flags are not valid[\s\S]*?$",
                r"Set `TRANSFORMERS_VERBOSITY=info`[\s\S]*?$",
                r"transformers_verbosity",
            ]
            out = s.replace("\r\n", "\n")
            for p in patterns:
                out = re.sub(p, "", out, flags=re.IGNORECASE)
            out = re.sub(r"\n{3,}", "\n\n", out)
            return out.strip()

        return _clean_warnings(response)

    def _stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        temp = float(kwargs.get("temperature", self.temperature))
        do_sample = bool(kwargs.get("do_sample", temp > 0))
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", None)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
            "use_cache": True,
            "repetition_penalty": 1.1,
        }

        try:
            gen_cfg = GenerationConfig(do_sample=do_sample)
            if do_sample:
                gen_cfg.temperature = temp
                if top_p is not None:
                    gen_cfg.top_p = float(top_p)
                if top_k is not None:
                    gen_cfg.top_k = int(top_k)
            generation_kwargs["generation_config"] = gen_cfg
        except Exception:
            generation_kwargs["do_sample"] = do_sample
            if do_sample:
                generation_kwargs["temperature"] = temp
                if top_p is not None:
                    generation_kwargs["top_p"] = float(top_p)
                if top_k is not None:
                    generation_kwargs["top_k"] = int(top_k)

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        patterns = [
            r"The following generation flags are not valid[\s\S]*?(?:$|\n)",
            r"may be ignored[\s\S]*?(?:$|\n)",
            r"Set `TRANSFORMERS_VERBOSITY=info`[\s\S]*?(?:$|\n)",
            r"transformers_verbosity",
            r"\[.*temperature.*\]",
        ]

        for fragment in streamer:
            chunk_text = fragment if isinstance(fragment, str) else str(fragment)
            buffer += chunk_text
            for p in patterns:
                buffer = re.sub(p, "", buffer, flags=re.IGNORECASE)
            if "\n" in buffer:
                to_emit, buffer = buffer.rsplit("\n", 1)
                to_emit = to_emit + "\n"
                to_emit = re.sub(r"\n{3,}", "\n\n", to_emit)
                if to_emit.strip():
                    yield GenerationChunk(text=to_emit)

        thread.join()
        if buffer:
            for p in patterns:
                buffer = re.sub(p, "", buffer, flags=re.IGNORECASE)
            buffer = re.sub(r"\n{3,}", "\n\n", buffer)
            if buffer.strip():
                if not buffer.endswith("\n"):
                    buffer = buffer + "\n"
                yield GenerationChunk(text=buffer)
