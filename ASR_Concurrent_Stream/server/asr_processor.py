# ASR Concurrent Stream Server v1.0.6
import re

import numpy as np


def _get_feat_extract_output_lengths(input_lengths):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return output_lengths


class AsrProcessor:
    def __init__(self, processor):
        self.processor = processor
        self.audio_token = "<|audio_pad|>"
        self.audio_bos_token = "<|audio_start|>"
        self.audio_eos_token = "<|audio_end|>"

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|audio_pad|>"
        if not hasattr(processor, "audio_bos_token"):
            processor.audio_bos_token = "<|audio_start|>"
        if not hasattr(processor, "audio_eos_token"):
            processor.audio_eos_token = "<|audio_end|>"
        return cls(processor)

    def apply_chat_template(self, conversations, **kwargs):
        return self.processor.apply_chat_template(conversations, **kwargs)

    @property
    def tokenizer(self):
        return self.processor.tokenizer
