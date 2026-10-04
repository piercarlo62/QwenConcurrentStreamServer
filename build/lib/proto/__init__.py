"""
Generated protocol buffer code for ASR Concurrent Stream

This module is generated from proto/asr.proto
"""

from proto.asr_pb2 import (
    HealthCheckRequest,
    HealthCheckResponse,
    StreamStartRequest,
    StreamStartResponse,
    StreamFinalizeRequest,
    StreamFinalizeResponse,
    StreamStatusRequest,
    StreamStatusResponse,
    StreamConfig,
    AudioChunkRequest,
    TranscriptionResponse,
    DESCRIPTOR,
)

from proto.asr_pb2_grpc import (
    ASRServiceStub,
    ASRServiceServicer,
    ASRService,
    add_ASRServiceServicer_to_server,
)

__all__ = [
    "HealthCheckRequest",
    "HealthCheckResponse",
    "StreamStartRequest",
    "StreamStartResponse",
    "StreamFinalizeRequest",
    "StreamFinalizeResponse",
    "StreamStatusRequest",
    "StreamStatusResponse",
    "StreamConfig",
    "AudioChunkRequest",
    "TranscriptionResponse",
    "ASRServiceStub",
    "ASRServiceServicer",
    "ASRService",
    "add_ASRServiceServicer_to_server",
    "DESCRIPTOR",
]
