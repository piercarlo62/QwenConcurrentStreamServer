# ASR Concurrent Stream Server v1.0.6
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from difflib import SequenceMatcher

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
    previous_partial: str
    partials_list: list
    stream_id: str


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
        stream_id: str = "",
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
            previous_partial="",
            partials_list=[],
            stream_id=stream_id,
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
    MIN_OVERLAP_WORDS = 2

    MATCH_THRESHOLD = 0.75
    GAP = -1.5
    MAX_TAIL = 20
    UNSTABLE = 2
    MIN_MATCHES = 2
    MIN_RATIO = 0.5

    @staticmethod
    def _norm(w: str) -> str:
        return re.sub(r"[^\w]", "", w.lower())

    @staticmethod
    def _word_sim(a: str, b: str) -> float:
        a, b = Qwen3ASRModel._norm(a), Qwen3ASRModel._norm(b)
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _score_pair(a: str, b: str) -> float:
        s = Qwen3ASRModel._word_sim(a, b)
        if s >= Qwen3ASRModel.MATCH_THRESHOLD:
            return 2.0 * s
        return -1.0

    @staticmethod
    def _align_tail_head(tail: List[str], head: List[str]):
        n, m = len(tail), len(head)
        H = [[0.0] * (m + 1) for _ in range(n + 1)]
        M = [[0] * (m + 1) for _ in range(n + 1)]
        for j in range(1, m + 1):
            H[0][j] = H[0][j - 1] + Qwen3ASRModel.GAP
        for i in range(1, n + 1):
            H[i][0] = 0.0
            for j in range(1, m + 1):
                sp = Qwen3ASRModel._score_pair(tail[i - 1], head[j - 1])
                cands = [
                    (H[i - 1][j - 1] + sp, M[i - 1][j - 1] + (1 if sp > 0 else 0)),
                    (H[i - 1][j] + Qwen3ASRModel.GAP, M[i - 1][j]),
                    (H[i][j - 1] + Qwen3ASRModel.GAP, M[i][j - 1]),
                ]
                H[i][j], M[i][j] = max(cands, key=lambda c: c[0])
        best_j = max(range(1, m + 1), key=lambda j: H[n][j])
        return H[n][best_j], best_j, M[n][best_j]

    @staticmethod
    def _is_prefix(committed: List[str], new: List[str]) -> bool:
        if len(committed) > len(new):
            return False
        for i in range(len(committed)):
            if Qwen3ASRModel._norm(committed[i]) != Qwen3ASRModel._norm(new[i]):
                return False
        return True

    @staticmethod
    def _fuzzy_prefix_match(committed: List[str], new: List[str]) -> int:
        best = 0
        max_check = min(len(committed), len(new))
        for k in range(max_check, 1, -1):
            match = True
            for i in range(k):
                if Qwen3ASRModel._norm(committed[i]) != Qwen3ASRModel._norm(new[i]):
                    match = False
                    break
            if match:
                best = k
                break
        return best

    @staticmethod
    def _merge_words(committed: List[str], new: List[str]) -> List[str]:
        if not committed:
            return list(new)

        prefix_len = Qwen3ASRModel._fuzzy_prefix_match(committed, new)
        if prefix_len >= min(len(committed), 2):
            return list(new)

        max_tail = Qwen3ASRModel.MAX_TAIL
        unstable = Qwen3ASRModel.UNSTABLE
        min_matches = Qwen3ASRModel.MIN_MATCHES
        min_ratio = Qwen3ASRModel.MIN_RATIO

        tail_start = max(0, len(committed) - max_tail)
        tail = committed[tail_start:]
        head = new[:len(tail) + 5]

        score, j, matches = Qwen3ASRModel._align_tail_head(tail, head)
        min_matched = min(len(tail), j)
        effective_min = min(min_matches, min_matched)
        ok = matches >= effective_min and matches / max(1, min_matched) >= min_ratio

        if not ok:
            return committed + list(new)

        keep = max(0, len(committed) - unstable)
        kept_tail = committed[tail_start:keep]
        if kept_tail:
            _, j_kept, m_kept = Qwen3ASRModel._align_tail_head(kept_tail, head)
            if m_kept >= 1:
                return committed[:keep] + list(new[j_kept:])
        if keep <= tail_start:
            return committed[:tail_start] + list(new)
        return committed + list(new[j:])

    def apply_output(self, state: ASRStreamingState, gen_text: str) -> None:
        prefix = self._compute_prefix(state)
        state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.partials_list.append(txt)
        committed_words = state.text.split() if state.text else []
        new_words = txt.split()
        merged = self._merge_words(committed_words, new_words)
        state.text = " ".join(merged)
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
        committed_words = state.text.split() if state.text else []
        new_words = txt.split()
        merged = self._merge_words(committed_words, new_words)
        state.text = " ".join(merged)
        self._save_partials_debug(state.partials_list, state.stream_id)
        state.previous_partial = ""
        state.partials_list = []
        state.chunk_id += 1

    @staticmethod
    def _save_partials_debug(partials: list, stream_id: str) -> None:
        try:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server", "partials_lists")
            os.makedirs(out_dir, exist_ok=True)
            ts = int(time.time() )
            short_id = stream_id[:8]
            filepath = os.path.join(out_dir, f"{ts}_{short_id}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                for i, p in enumerate(partials):
                    f.write(f"[partial {i}] {p}\n")
        except Exception:
            pass
