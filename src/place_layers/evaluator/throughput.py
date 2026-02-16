"""Throughput benchmarking evaluator.

Measures generation throughput (output tokens/sec) using synthetic prompts
with configurable input and output sequence lengths.

Unlike other evaluators, this does not write trace records or use lm-eval
tasks. It sends synthetic generation requests and measures wall-clock
throughput.

Usage:
    config = ThroughputEvaluatorConfig(
        backend=RESTBackendConfig(...),
        input_seq_len=512,
        output_seq_len=128,
        num_requests=100,
    )
    async with config.create_evaluator(output_dir) as evaluator:
        tokens_per_sec = await evaluator.evaluate(placement)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field

from place_layers.backend import AsyncBackend, BackendError, SyncBackend
from place_layers.backend.types import GenerateTextRequest
from place_layers.clients.types import GenerationResult
from place_layers.core.types import LayerPlacement
from place_layers.evaluator.backend_evaluator import BackendEvaluator, BackendEvaluatorConfig
from place_layers.evaluator.base_evaluator import BaseEvaluator
from place_layers.evaluator.progress import create_progress

logger = logging.getLogger(__name__)


class ThroughputEvaluator(BackendEvaluator):
    """Evaluator that benchmarks generation throughput.

    Generates synthetic prompts calibrated to a target token count,
    sends generation requests to the backend, and measures output
    tokens/sec.

    Does not write traces or use lm-eval tasks.
    """

    config: "ThroughputEvaluatorConfig"

    def __init__(
        self, config: "ThroughputEvaluatorConfig", *, output_dir: Path
    ) -> None:
        super().__init__(config, output_dir=output_dir)

    async def _calibrate_prompt(self) -> str:
        """Build a synthetic prompt targeting input_seq_len tokens.

        Uses a single calibration request to measure chars-per-token ratio,
        then adjusts the prompt length accordingly.
        """
        target_tokens = self.config.input_seq_len
        fill_text = self.config.prompt_fill_text

        # Initial estimate: ~4 characters per token
        estimated_chars = int(target_tokens * 4.0)
        repetitions = (estimated_chars // len(fill_text)) + 1
        candidate = (fill_text * repetitions)[:estimated_chars]

        # Calibrate with one request
        cal_request = GenerateTextRequest(
            key="__calibration__",
            prompt=candidate,
            max_tokens=1,
            temperature=0.0,
        )

        actual_tokens = 0
        if isinstance(self._backend, AsyncBackend):
            async for result in self._backend.generate_text([cal_request]):
                if isinstance(result, BackendError):
                    raise RuntimeError(f"Calibration failed: {result.message}")
                actual_tokens = len(result.prompt_token_ids)
        else:
            for result in self._backend.generate_text([cal_request]):
                actual_tokens = len(result.prompt_token_ids)

        if actual_tokens == 0:
            logger.warning(
                "Calibration returned 0 prompt tokens; using character estimate"
            )
            return candidate

        # Scale proportionally
        chars_per_token = len(candidate) / actual_tokens
        adjusted_chars = int(target_tokens * chars_per_token)
        repetitions = (adjusted_chars // len(fill_text)) + 1
        prompt = (fill_text * repetitions)[:adjusted_chars]

        logger.info(
            f"Prompt calibrated: target={target_tokens} tokens, "
            f"{chars_per_token:.2f} chars/token, "
            f"{len(prompt)} chars"
        )
        return prompt

    async def _run_requests(
        self,
        requests: list[GenerateTextRequest],
        phase: str,
    ) -> list[GenerationResult]:
        """Send generation requests, collecting results with progress bar."""
        results: list[GenerationResult] = []
        errors = 0
        total = len(requests)
        log_interval = max(1, total // 10)  # Log every ~10%
        start = time.perf_counter()

        with create_progress(transient=(phase == "warmup")) as progress:
            task_id = progress.add_task(f"[cyan]{phase}", total=total)

            if isinstance(self._backend, AsyncBackend):
                async for result in self._backend.generate_text(requests):
                    if isinstance(result, BackendError):
                        errors += 1
                        logger.warning(
                            f"{phase} error [{result.key}]: {result.message}"
                        )
                    else:
                        results.append(result)
                    progress.advance(task_id)
                    completed = len(results) + errors
                    if completed % log_interval == 0:
                        elapsed = time.perf_counter() - start
                        output_tokens = sum(
                            len(s.token_ids) for r in results for s in r.samples
                        )
                        tps = output_tokens / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"{phase}: {completed}/{total} "
                            f"({elapsed:.0f}s, {tps:,.0f} tok/s)"
                        )
            else:
                for result in self._backend.generate_text(requests):
                    results.append(result)
                    progress.advance(task_id)
                    completed = len(results)
                    if completed % log_interval == 0:
                        elapsed = time.perf_counter() - start
                        output_tokens = sum(
                            len(s.token_ids) for r in results for s in r.samples
                        )
                        tps = output_tokens / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"{phase}: {completed}/{total} "
                            f"({elapsed:.0f}s, {tps:,.0f} tok/s)"
                        )

        if errors > 0:
            logger.warning(f"{phase}: {errors} request(s) failed")

        return results

    async def evaluate(self, parameters: LayerPlacement | None) -> float:
        """Benchmark throughput for a placement.

        Returns output tokens/sec (higher is better).
        """
        if parameters is not None:
            logger.info(
                f"Benchmarking placement: {parameters.assignments[:4]}... "
                f"({len(parameters.assignments)} layers)"
            )
            await self._set_placement(parameters)

        # Calibrate synthetic prompt to target token length
        prompt = await self._calibrate_prompt()

        def make_requests(prefix: str, count: int) -> list[GenerateTextRequest]:
            return [
                GenerateTextRequest(
                    key=f"{prefix}_{i}",
                    prompt=prompt,
                    max_tokens=self.config.output_seq_len,
                    temperature=0.0,
                    ignore_eos=self.config.ignore_eos,
                )
                for i in range(count)
            ]

        # Warmup (not timed)
        if self.config.num_warmup_requests > 0:
            logger.info(
                f"Warming up with {self.config.num_warmup_requests} requests..."
            )
            await self._run_requests(
                make_requests("warmup", self.config.num_warmup_requests),
                phase="warmup",
            )

        # Timed measurement
        logger.info(
            f"Measuring: {self.config.num_requests} requests, "
            f"input~{self.config.input_seq_len} tokens, "
            f"output<={self.config.output_seq_len} tokens"
        )
        bench_requests = make_requests("bench", self.config.num_requests)

        start = time.perf_counter()
        results = await self._run_requests(bench_requests, phase="benchmark")
        elapsed = time.perf_counter() - start

        # Compute metrics
        total_output_tokens = sum(
            len(s.token_ids) for r in results for s in r.samples
        )
        total_input_tokens = sum(len(r.prompt_token_ids) for r in results)
        num_completed = len(results)
        num_errors = self.config.num_requests - num_completed

        output_tps = total_output_tokens / elapsed if elapsed > 0 else 0.0
        total_tps = (
            (total_input_tokens + total_output_tokens) / elapsed
            if elapsed > 0
            else 0.0
        )
        avg_latency_ms = (elapsed / num_completed * 1000) if num_completed else 0.0
        avg_output_per_req = (
            total_output_tokens / num_completed if num_completed else 0.0
        )

        # Report
        logger.info("")
        logger.info("=== Throughput Benchmark Results ===")
        logger.info(f"  Completed: {num_completed} / {self.config.num_requests}")
        if num_errors > 0:
            logger.info(f"  Errors: {num_errors}")
        logger.info(f"  Wall time: {elapsed:.2f}s")
        logger.info(f"  Input tokens: {total_input_tokens:,}")
        logger.info(f"  Output tokens: {total_output_tokens:,}")
        logger.info(f"  Output throughput: {output_tps:,.1f} tok/s")
        logger.info(f"  Total throughput: {total_tps:,.1f} tok/s")
        logger.info(f"  Avg latency: {avg_latency_ms:.1f} ms/request")
        logger.info(f"  Avg output/request: {avg_output_per_req:.1f} tokens")
        logger.info("====================================")

        return output_tps


class ThroughputEvaluatorConfig(BackendEvaluatorConfig):
    """Configuration for throughput benchmarking evaluator."""

    _impl_class: ClassVar[type[BaseEvaluator]] = ThroughputEvaluator

    evaluator_type: Literal["throughput"] = "throughput"

    input_seq_len: int = Field(
        default=512,
        description="Target input prompt length in tokens",
        gt=0,
    )
    output_seq_len: int = Field(
        default=128,
        description="Maximum output tokens per request",
        gt=0,
    )
    num_requests: int = Field(
        default=100,
        description="Number of generation requests for measurement",
        gt=0,
    )
    num_warmup_requests: int = Field(
        default=5,
        description="Warmup requests before measurement (not timed)",
        ge=0,
    )
    ignore_eos: bool = Field(
        default=True,
        description="Ignore EOS token to guarantee exact output_seq_len tokens per request",
    )
    prompt_fill_text: str = Field(
        default="The quick brown fox jumps over the lazy dog. ",
        description="Text repeated to build synthetic prompts of target token length",
    )
