"""
Stream Manager for ASR Concurrent Stream Server

Manages per-stream lifecycle, state tracking, resource limits, and finalization.
Provides async-safe operations for stream creation, access, and cleanup.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class StreamState(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    CLOSED = "closed"
    ERROR = "error"


class StreamLimitsExceededError(Exception):
    """Raised when maximum concurrent streams limit is exceeded"""

    pass


class StreamNotFoundError(Exception):
    """Raised when requested stream does not exist"""

    pass


@dataclass
class StreamMetrics:
    """Metrics for a single stream"""

    chunks_received: int = 0
    chunks_processed: int = 0
    transcripts_sent: int = 0
    last_chunk_time_ms: int = 0
    finalization_request_time_ms: Optional[int] = None
    finalization_complete_time_ms: Optional[int] = None


@dataclass
class StreamConfig:
    """Configuration for streaming behavior"""

    chunk_size_sec: float = 0.5
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5
    max_buffer_chunks: int = 10
    finalization_timeout_ms: int = 5000


@dataclass
class ManagedStream:
    """Stream state managed by StreamManager"""

    stream_id: str
    state: StreamState = StreamState.INITIALIZING
    config: StreamConfig = field(default_factory=StreamConfig)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    context: str = ""
    forced_language: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def age_seconds(self) -> float:
        """Stream age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def inactive_seconds(self) -> float:
        """Inactive time in seconds"""
        return (datetime.now() - self.last_activity).total_seconds()


class StreamManager:
    """
    Manages stream lifecycle and enforces resource limits.

    Features:
    - Max concurrent streams enforcement
    - Stale stream cleanup
    - Per-stream finalization tracking
    - Activity tracking and metrics
    - Async-safe operations
    """

    def __init__(
        self,
        max_concurrent_streams: int = 30,
        stream_timeout: int = 300,
    ):
        self.max_concurrent_streams = max_concurrent_streams
        self.stream_timeout = stream_timeout

        self._streams: Dict[str, ManagedStream] = {}
        self._lock = asyncio.Lock()

        self._cleanup_task: Optional[asyncio.Task] = None

        self._total_streams_created: int = 0
        self._total_streams_closed: int = 0
        self._total_streams_finalized: int = 0

    async def start_cleanup_task(self):
        """Start background stale stream cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_worker())
            logger.info("Started stream cleanup task")

    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped stream cleanup task")

    async def _cleanup_worker(self):
        """Background worker to clean up stale streams"""
        logger.debug("Stream cleanup worker started")
        while True:
            try:
                await asyncio.sleep(60)
                cleaned = await self.cleanup_stale_streams()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} stale streams")
            except asyncio.CancelledError:
                logger.debug("Stream cleanup worker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in stream cleanup worker: {e}", exc_info=True)

    async def create_stream(
        self,
        stream_id: str,
        context: str = "",
        language: Optional[str] = None,
        config: Optional[StreamConfig] = None,
    ) -> ManagedStream:
        """Create a new stream"""
        async with self._lock:
            if len(self._streams) >= self.max_concurrent_streams:
                raise StreamLimitsExceededError(
                    f"Maximum concurrent streams limit reached ({self.max_concurrent_streams})"
                )

            if stream_id in self._streams:
                logger.warning(f"Stream {stream_id} already exists, replacing")
                del self._streams[stream_id]

            if config is None:
                config = StreamConfig()

            stream = ManagedStream(
                stream_id=stream_id,
                state=StreamState.ACTIVE,
                config=config,
                context=context,
                forced_language=language,
            )

            self._streams[stream_id] = stream
            self._total_streams_created += 1

            logger.debug(f"Created stream {stream_id}")
            return stream

    async def get_stream(self, stream_id: str) -> Optional[ManagedStream]:
        """Get stream by ID"""
        async with self._lock:
            stream = self._streams.get(stream_id)
            if stream:
                stream.last_activity = datetime.now()
            return stream

    async def request_finalization(self, stream_id: str) -> bool:
        """Mark stream for finalization (priority processing)"""
        async with self._lock:
            stream = self._streams.get(stream_id)
            if not stream:
                return False

            if stream.state != StreamState.ACTIVE:
                return False

            stream.state = StreamState.FINALIZING
            stream.metrics.finalization_request_time_ms = int(
                datetime.now().timestamp() * 1000
            )
            logger.debug(f"Finalization requested for stream {stream_id}")
            return True

    async def complete_finalization(
        self,
        stream_id: str,
        final_transcript: str = "",
        language: str = "",
    ) -> bool:
        """Mark finalization complete"""
        async with self._lock:
            stream = self._streams.get(stream_id)
            if not stream:
                return False

            stream.state = StreamState.FINALIZED
            stream.metrics.finalization_complete_time_ms = int(
                datetime.now().timestamp() * 1000
            )
            self._total_streams_finalized += 1

            logger.debug(
                f"Finalization complete for stream {stream_id}: "
                f"transcript='{final_transcript[:50]}...', language={language}"
            )
            return True

    async def close_stream(self, stream_id: str) -> bool:
        """Close a stream and release resources"""
        async with self._lock:
            stream = self._streams.pop(stream_id, None)
            if not stream:
                return False

            stream.state = StreamState.CLOSED
            self._total_streams_closed += 1

            logger.debug(
                f"Closed stream {stream_id} (age: {stream.age_seconds:.1f}s, "
                f"chunks: {stream.metrics.chunks_received})"
            )
            return True

    async def update_metrics(
        self,
        stream_id: str,
        chunks_received: int = 0,
        chunks_processed: int = 0,
        transcripts_sent: int = 0,
    ) -> None:
        """Update stream metrics"""
        async with self._lock:
            stream = self._streams.get(stream_id)
            if stream:
                stream.last_activity = datetime.now()
                stream.metrics.chunks_received += chunks_received
                stream.metrics.chunks_processed += chunks_processed
                stream.metrics.transcripts_sent += transcripts_sent
                stream.metrics.last_chunk_time_ms = int(
                    datetime.now().timestamp() * 1000
                )

    async def get_streams_for_finalization(self) -> list[str]:
        """Get list of streams marked for finalization"""
        async with self._lock:
            return [
                sid
                for sid, s in self._streams.items()
                if s.state == StreamState.FINALIZING
            ]

    async def get_active_stream_count(self) -> int:
        """Get count of active streams (not finalizing)"""
        async with self._lock:
            return sum(
                1 for s in self._streams.values() if s.state == StreamState.ACTIVE
            )

    async def cleanup_stale_streams(self) -> int:
        """Clean up streams inactive longer than timeout"""
        async with self._lock:
            now = datetime.now()
            cleaned = 0
            stream_ids = list(self._streams.keys())

            for stream_id in stream_ids:
                stream = self._streams[stream_id]
                inactive_time = (now - stream.last_activity).total_seconds()

                if inactive_time > self.stream_timeout:
                    del self._streams[stream_id]
                    stream.state = StreamState.CLOSED
                    self._total_streams_closed += 1
                    cleaned += 1
                    logger.info(
                        f"Cleaned up stale stream {stream_id} ({inactive_time:.1f}s inactive)"
                    )

            return cleaned

    async def get_metrics(self) -> dict:
        """Get stream manager metrics"""
        async with self._lock:
            return {
                "active_streams": len(self._streams),
                "total_created": self._total_streams_created,
                "total_closed": self._total_streams_closed,
                "total_finalized": self._total_streams_finalized,
                "max_concurrent": self.max_concurrent_streams,
                "stream_timeout": self.stream_timeout,
            }

    async def get_stream_info(self, stream_id: str) -> Optional[dict]:
        """Get stream info for debugging"""
        stream = await self.get_stream(stream_id)
        if not stream:
            return None

        return {
            "stream_id": stream.stream_id,
            "state": stream.state.value,
            "age_seconds": stream.age_seconds,
            "inactive_seconds": stream.inactive_seconds,
            "chunks_received": stream.metrics.chunks_received,
            "chunks_processed": stream.metrics.chunks_processed,
            "context": stream.context,
            "language": stream.forced_language,
            "config": {
                "chunk_size_sec": stream.config.chunk_size_sec,
                "unfixed_chunk_num": stream.config.unfixed_chunk_num,
                "unfixed_token_num": stream.config.unfixed_token_num,
            },
        }

