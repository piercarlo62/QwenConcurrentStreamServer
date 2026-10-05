# ASR Concurrent Stream Server v1.0.6
import difflib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

    MATCH_THRESHOLD = 0.75
    MATCH_SCORE = 1.0
    MISMATCH_PENALTY = -0.5
    GAP_PENALTY = -0.7
    UNSTABLE_TAIL_WORDS = 2
    MIN_OVERLAP_WORDS = 2
    OVERLAP_WINDOW = 15

    @staticmethod
    def _normalize_word(w: str) -> str:
        return w.lower().strip(".,!?;:")

    @staticmethod
    def _word_similarity(w1: str, w2: str) -> float:
        return difflib.SequenceMatcher(
            None,
            Qwen3ASRModel._normalize_word(w1),
            Qwen3ASRModel._normalize_word(w2),
        ).ratio()

    @staticmethod
    def _soft_match(w1: str, w2: str) -> Tuple[bool, float]:
        sim = Qwen3ASRModel._word_similarity(w1, w2)
        return (sim >= Qwen3ASRModel.MATCH_THRESHOLD, sim)

    @staticmethod
    def _fuzzy_align(old_words: List[str], new_words: List[str]) -> Tuple[int, int, float]:
        window = Qwen3ASRModel.OVERLAP_WINDOW
        tail_len = min(len(old_words), window)
        tail = old_words[-tail_len:]
        head_len = min(len(new_words), window)
        head = new_words[:head_len]
        best_score = -float('inf')
        best_i = 0
        best_j = 0
        dp = [[0.0] * (len(head) + 1) for _ in range(len(tail) + 1)]
        for i in range(len(tail) + 1):
            dp[i][0] = 0.0
        for j in range(len(head) + 1):
            dp[0][j] = 0.0
        for i in range(1, len(tail) + 1):
            for j in range(1, len(head) + 1):
                matched, sim = Qwen3ASRModel._soft_match(tail[i - 1], head[j - 1])
                match_score = dp[i - 1][j - 1] + (Qwen3ASRModel.MATCH_SCORE * sim if matched else Qwen3ASRModel.MISMATCH_PENALTY)
                gap_old = dp[i - 1][j] + Qwen3ASRModel.GAP_PENALTY
                gap_new = dp[i][j - 1] + Qwen3ASRModel.GAP_PENALTY
                dp[i][j] = max(match_score, gap_old, gap_new)
        j = len(head)
        best_score = -float('inf')
        best_i = len(tail)
        for i in range(len(tail) + 1):
            score = dp[i][j] - (len(tail) - i) * 0.01
            if score > best_score:
                best_score = score
                best_i = i
        i = best_i
        j = len(head)
        while j > 0 and i > 0:
            matched, sim = Qwen3ASRModel._soft_match(tail[i - 1], head[j - 1])
            match_score = dp[i - 1][j - 1] + (Qwen3ASRModel.MATCH_SCORE * sim if matched else Qwen3ASRModel.MISMATCH_PENALTY)
            if abs(dp[i][j] - match_score) < 0.001:
                i -= 1
                j -= 1
            elif abs(dp[i][j] - (dp[i - 1][j] + Qwen3ASRModel.GAP_PENALTY)) < 0.001:
                i -= 1
            else:
                j -= 1
        tail_start_in_old = len(old_words) - tail_len
        align_start = tail_start_in_old + i
        align_end = tail_start_in_old + best_i
        return (align_start, j, best_score)

    @staticmethod
    def _fuzzy_merge(accumulated: str, new_partial: str) -> str:
        if not accumulated:
            return new_partial
        acc_words = accumulated.split()
        new_words = new_partial.split()
        if not new_words:
            return accumulated
        align_start, new_start, score = Qwen3ASRModel._fuzzy_align(acc_words, new_words)
        max_possible = min(
            min(len(acc_words), Qwen3ASRModel.OVERLAP_WINDOW),
            min(len(new_words), Qwen3ASRModel.OVERLAP_WINDOW),
        )
        match_ratio = score / max_possible if max_possible > 0 else 0
        if match_ratio < 0.3 or new_start < Qwen3ASRModel.MIN_OVERLAP_WORDS:
            return " ".join(acc_words + new_words)
        unstable_tail = min(Qwen3ASRModel.UNSTABLE_TAIL_WORDS, len(acc_words) - align_start)
        if align_start < len(acc_words) - unstable_tail:
            result = acc_words[:align_start] + new_words[new_start:]
        else:
            committed_end = max(align_start, len(acc_words) - unstable_tail)
            result = acc_words[:committed_end] + new_words[new_start:]
        return " ".join(result)

    @staticmethod
    def _recompose_partials(partials: list) -> str:
        if not partials:
            return ""
        result = partials[0]
        for i in range(1, len(partials)):
            result = Qwen3ASRModel._fuzzy_merge(result, partials[i])
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
