# ASR Concurrent Stream Server v1.0.6
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vllm import TextPrompt

import numpy as np

from server.asr_processor import AsrProcessor
from server.asr_utils import (
    SAMPLE_RATE,
    normalize_language_name,
    validate_language,
    parse_asr_output,
)


@dataclass
class ASRStreamingState:
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    chunk_size_samples: int

    context_before_sec: float
    context_before_samples: int
    context_after_sec: float
    context_after_samples: int

    chunk_id: int
    buffer: np.ndarray
    history: np.ndarray

    prompt_raw: str
    context: str
    force_language: Optional[str]

    language: str
    text: str
    _raw_decoded: str
    partials_list: list


class Qwen3ASRModel:
    def __init__(
        self,
        backend: str,
        engine: Any,
        processor: Any,
        sampling_params: Optional[Any] = None,
        max_new_tokens: int = 512,
    ):
        self.backend = backend
        self.engine = engine
        self.processor = processor
        self.sampling_params = sampling_params
        self.max_new_tokens = max_new_tokens

    @classmethod
    def create_engine(
        cls,
        model: str,
        max_new_tokens: int = 4096,
        **kwargs,
    ):
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

        engine_args = AsyncEngineArgs(
            model=model,
            limit_mm_per_prompt={"audio": 1},
            **kwargs,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        processor = AsrProcessor.from_pretrained(model, fix_mistral_regex=True)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

        return cls(
            backend="vllm",
            engine=engine,
            processor=processor,
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
        )

    def _build_text_prompt(self, context: str, force_language: Optional[str]) -> str:
        system_msg = context or ""
        if getattr(self, "punctuate", False):
            system_msg = (system_msg + " Transcribe the audio with proper punctuation including periods, question marks, and commas.").strip()
        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        base = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if force_language:
            base = base + f"language {force_language}<asr_text>"
        return base

    def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 0.5,
        context_before_sec: float = 5.0,
        context_after_sec: float = 0.5,
    ) -> ASRStreamingState:
        force_language = None
        if language is not None and str(language).strip() != "":
            ln = normalize_language_name(str(language))
            validate_language(ln)
            force_language = ln

        chunk_size_samples = int(round(float(chunk_size_sec) * SAMPLE_RATE))
        chunk_size_samples = max(1, chunk_size_samples)

        context_before_samples = int(round(float(context_before_sec) * SAMPLE_RATE))
        context_after_samples = int(round(float(context_after_sec) * SAMPLE_RATE))

        prompt_raw = self._build_text_prompt(context=context, force_language=force_language)

        return ASRStreamingState(
            unfixed_chunk_num=int(unfixed_chunk_num),
            unfixed_token_num=int(unfixed_token_num),
            chunk_size_sec=float(chunk_size_sec),
            chunk_size_samples=int(chunk_size_samples),
            context_before_sec=float(context_before_sec),
            context_before_samples=int(context_before_samples),
            context_after_sec=float(context_after_sec),
            context_after_samples=int(context_after_samples),
            chunk_id=0,
            buffer=np.zeros((0,), dtype=np.float32),
            history=np.zeros((0,), dtype=np.float32),
            prompt_raw=prompt_raw,
            context=context or "",
            force_language=force_language,
            language="",
            text="",
            _raw_decoded="",
            partials_list=[],
        )

    def _compute_prefix(self, state: ASRStreamingState) -> str:
        if state.chunk_id < state.unfixed_chunk_num:
            return ""
        cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
        k = int(state.unfixed_token_num)
        while True:
            end_idx = max(0, len(cur_ids) - k)
            prefix = self.processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
            if '\ufffd' not in prefix:
                return prefix
            if end_idx == 0:
                return ""
            k += 1

    def _build_window(self, state: ASRStreamingState, chunk: np.ndarray) -> np.ndarray:
        past = state.history
        if past.shape[0] > state.context_before_samples:
            past = past[-state.context_before_samples:]

        future = state.buffer[:state.context_after_samples]

        parts = []
        if past.shape[0] > 0:
            parts.append(past)
        parts.append(chunk)
        if future.shape[0] > 0:
            parts.append(future)

        return np.concatenate(parts, axis=0)

    def prepare_chunk(self, state: ASRStreamingState, pcm: np.ndarray) -> Optional[TextPrompt]:
        x = np.asarray(pcm)
        if x.ndim != 1:
            x = x.reshape(-1)

        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)

        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x], axis=0)

        if state.buffer.shape[0] < state.chunk_size_samples:
            return None

        chunk = state.buffer[:state.chunk_size_samples]
        state.buffer = state.buffer[state.chunk_size_samples:]

        window = self._build_window(state, chunk)

        state.history = np.concatenate([state.history, chunk], axis=0)
        if state.history.shape[0] > state.context_before_samples:
            state.history = state.history[-state.context_before_samples:]

        prefix = self._compute_prefix(state)
        prompt = state.prompt_raw + prefix
        return TextPrompt(prompt=prompt, multi_modal_data={"audio": [window]})

    @staticmethod
    def _normalize_word(w: str) -> str:
        return w.lower().strip(".,!?;:")

    @staticmethod
    def _find_overlap_words(words1: list, words2: str, min_overlap: int = 2) -> int:
        n1 = len(words1)
        words2_list = words2.split() if isinstance(words2, str) else words2
        n2 = len(words2_list)
        best = 0
        for k in range(min(n1, n2), min_overlap - 1, -1):
            match = True
            for j in range(k):
                if Qwen3ASRModel._normalize_word(words1[n1 - k + j]) != Qwen3ASRModel._normalize_word(words2_list[j]):
                    match = False
                    break
            if match:
                best = k
                break
        return best

    @staticmethod
    def _merge_new_partial(accumulated: str, new_partial: str) -> str:
        if not accumulated:
            return new_partial
        acc_words = accumulated.split()
        norm_acc_words = [Qwen3ASRModel._normalize_word(w) for w in acc_words]
        new_words = new_partial.split()
        norm_new_words = [Qwen3ASRModel._normalize_word(w) for w in new_words]
        best_pos = -1
        best_len = 0
        for pos in range(len(norm_acc_words)):
            for k in range(2, min(len(norm_acc_words) - pos, len(norm_new_words)) + 1):
                if norm_acc_words[pos:pos + k] == norm_new_words[:k]:
                    if k > best_len:
                        best_len = k
                        best_pos = pos
        if best_len >= 2:
            prefix = acc_words[:best_pos]
            suffix_start = best_pos + best_len
            if suffix_start >= len(acc_words):
                new_start = best_len
            else:
                remaining_acc = norm_acc_words[suffix_start:]
                new_start = best_len
                for k in range(best_len, len(norm_new_words) - 1):
                    if k + len(remaining_acc) <= len(norm_new_words):
                        if norm_new_words[k:k + len(remaining_acc)] == remaining_acc:
                            new_start = k
                            break
                else:
                    new_start = best_len
            result_words = prefix + new_words[new_start:]
            return " ".join(result_words)
        else:
            new_start = 0
            for k in range(min(len(norm_acc_words), len(norm_new_words)), 0, -1):
                if norm_acc_words[-k:] == norm_new_words[:k]:
                    new_start = k
                    break
            return " ".join(acc_words + new_words[new_start:])

    @staticmethod
    def _recompose_partials(partials: list) -> str:
        if not partials:
            return ""
        result = partials[0]
        for i in range(1, len(partials)):
            result = Qwen3ASRModel._merge_new_partial(result, partials[i])
        return result

    def apply_output(self, state: ASRStreamingState, gen_text: str) -> None:
        prefix = self._compute_prefix(state)
        state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.partials_list.append(txt)
        state.text = self._recompose_partials(state.partials_list)
        state.chunk_id += 1

    def finish_streaming_transcribe(self, state: ASRStreamingState) -> Optional[TextPrompt]:
        if state.buffer is None or state.buffer.shape[0] == 0:
            return None

        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)

        state.history = np.concatenate([state.history, tail], axis=0)
        if state.history.shape[0] > state.context_before_samples:
            state.history = state.history[-state.context_before_samples:]

        if tail.shape[0] < state.chunk_size_samples:
            return None

        window = self._build_window(state, tail)

        prefix = self._compute_prefix(state)
        prompt = state.prompt_raw + prefix
        return TextPrompt(prompt=prompt, multi_modal_data={"audio": [window]})

    def apply_finalize_output(self, state: ASRStreamingState, gen_text: str) -> None:
        prefix = self._compute_prefix(state)
        state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.partials_list.append(txt)
        state.text = self._recompose_partials(state.partials_list)
        state.partials_list = []
        state.chunk_id += 1
