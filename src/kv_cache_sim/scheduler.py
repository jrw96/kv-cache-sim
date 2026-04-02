from collections import deque

from kv_cache_sim.block_allocator import BlockAllocator
from kv_cache_sim.models import (
    GPUConfig,
    ModelConfig,
    Request,
    RequestState,
    SchedulerMetrics,
)
from kv_cache_sim.timing_model import TimingModel


class Scheduler:
    """Continuous batching scheduler with paged KV cache management.

    Implements Orca-style iteration-level scheduling: requests enter
    and leave the running batch independently, without waiting for
    batch boundaries.
    """

    def __init__(
        self,
        gpu: GPUConfig,
        model: ModelConfig,
        block_size: int = 16,
        max_output_length: int = 512,
        *,
        preallocate: bool = False,
    ) -> None:

        self.timing: TimingModel = TimingModel(gpu, model)
        self.block_size: int = block_size
        self.preallocate: bool = preallocate
        self.max_output_length: int = max_output_length

        model_weights_memory_req: int = model.num_params * model.dtype_bytes
        kv_memory_req: int = gpu.total_memory - model_weights_memory_req
        total_blocks: int = kv_memory_req // (
            block_size * self.timing.kv_cache_bytes_per_token()
        )
        self.allocator = BlockAllocator(total_blocks, block_size)

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.completed: list[Request] = []

        self.history: list[SchedulerMetrics] = []
        self.clock: float = 0.0

    def _tokens_to_allocate(self, request: Request) -> int:
        """Tokens to reserve at admission time."""
        if self.preallocate:
            return request.prompt_length + self.max_output_length
        return request.prompt_length

    def _admit_requests(self, max_admitted: int | None = None) -> None:
        """Move requests from waiting to running if memory is available."""
        admitted: int = 0
        while self.waiting:
            if max_admitted is not None and admitted >= max_admitted:
                break
            request: Request = self.waiting[0]
            tokens: int = self._tokens_to_allocate(request)
            if not self.allocator.can_allocate(tokens):
                break
            self.waiting.popleft()
            self.allocator.allocate(request.request_id, tokens)
            request.state = RequestState.PREFILLING
            request.prefill_start_time = self.clock
            self._process_prefill(request)
            admitted += 1

    def _process_prefill(self, request: Request) -> None:
        """Process prefill and move request to running batch."""
        self.clock += self.timing.prefill_time(request.prompt_length)
        request.state = RequestState.RUNNING
        request.first_token_time = self.clock
        self.running.append(request)

    def _decode_step(self) -> None:
        """Run one decode step for the entire running batch."""
        if not self.running:
            return

        self.clock += self.timing.decode_step_time()

        completed_this_step: list[Request] = []
        for request in self.running:
            request.generated_tokens += 1
            if not self.preallocate:
                self.allocator.append(request.request_id, 1)
            if request.generated_tokens >= request.output_length:
                request.state = RequestState.COMPLETED
                request.completion_time = self.clock
                self.allocator.free(request.request_id)
                completed_this_step.append(request)

        for request in completed_this_step:
            self.running.remove(request)
            self.completed.append(request)

    def _record_metrics(self) -> None:
        """Snapshot current state for later analysis."""
        self.history.append(
            SchedulerMetrics(
                timestamp=self.clock,
                batch_size=len(self.running),
                memory_utilisation=self.allocator.get_utilisation(),
            )
        )

    def _arrive_requests(self, pending: deque[Request]) -> None:
        """Move requests that have arrived by current clock to waiting queue."""
        while pending and pending[0].arrival_time <= self.clock:
            self.waiting.append(pending.popleft())

    def _reset(self) -> None:
        """Reset state for a new simulation run."""
        self.waiting = deque()
        self.running = []
        self.completed = []
        self.history = []
        self.clock = 0.0
        self.allocator = BlockAllocator(self.allocator.total_blocks, self.block_size)

    def run(self, requests: list[Request]) -> list[Request]:
        """Run the simulation. Returns completed requests with metrics."""
        self._reset()
        pending: deque[Request] = deque(sorted(requests, key=lambda r: r.arrival_time))

        while pending or self.waiting or self.running:
            # If nothing is happening we can just advance the clock to the next event
            if not self.waiting and not self.running and pending:
                self.clock = pending[0].arrival_time

            self._arrive_requests(pending)
            self._admit_requests()
            self._decode_step()
            self._record_metrics()

        return self.completed

    def run_static(
        self, requests: list[Request], batch_size: int = 32
    ) -> list[Request]:
        """Run the simulation, but with static batching."""
        self._reset()
        pending: deque[Request] = deque(sorted(requests, key=lambda r: r.arrival_time))

        while pending or self.waiting or self.running:
            if not self.waiting and not self.running and pending:
                self.clock = pending[0].arrival_time

            self._arrive_requests(pending)

            if not self.running and self.waiting:
                self._admit_requests(max_admitted=batch_size)

            self._decode_step()
            self._record_metrics()

        return self.completed
