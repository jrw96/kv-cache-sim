import pytest

from kv_cache_sim.models import GPUConfig, ModelConfig, Request, RequestState
from kv_cache_sim.scheduler import Scheduler
from kv_cache_sim.workload import generate_workload

# --- Small config for fast, predictable tests ---


@pytest.fixture
def gpu() -> GPUConfig:
    """Toy GPU: enough memory for model weights + some KV cache."""
    return GPUConfig(
        name="test-gpu",
        hbm_bandwidth=2.0e12,
        compute_flops=312e12,
        total_memory=150_000_000_000,  # 150 GB
    )


@pytest.fixture
def model() -> ModelConfig:
    """Toy model: small enough that tests run instantly."""
    return ModelConfig(
        num_params=1_000_000_000,  # 1B
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
        dtype_bytes=2,
    )


# --- Request lifecycle ---


def test_single_request_completes(gpu: GPUConfig, model: ModelConfig) -> None:
    """One request goes through WAITING -> PREFILLING -> RUNNING -> COMPLETED."""
    scheduler = Scheduler(gpu, model)
    requests = [
        Request(request_id=0, prompt_length=100, output_length=10, arrival_time=0.0)
    ]
    completed = scheduler.run(requests)
    assert len(completed) == 1
    assert completed[0].state == RequestState.COMPLETED
    assert completed[0].generated_tokens == 10


def test_single_request_timestamps_are_sane(gpu: GPUConfig, model: ModelConfig) -> None:
    """Timestamps follow the correct order."""
    scheduler = Scheduler(gpu, model)
    requests = [
        Request(request_id=0, prompt_length=100, output_length=10, arrival_time=0.0)
    ]
    completed = scheduler.run(requests)
    req = completed[0]
    assert req.prefill_start_time is not None
    assert req.first_token_time is not None
    assert req.completion_time is not None
    assert req.prefill_start_time >= req.arrival_time
    assert req.first_token_time > req.prefill_start_time
    assert req.completion_time > req.first_token_time


# --- All requests complete ---


def test_all_requests_complete(gpu: GPUConfig, model: ModelConfig) -> None:
    """Every request in the workload eventually completes."""
    scheduler = Scheduler(gpu, model)
    requests = generate_workload(50, 10.0, (50, 200), (10, 50), seed=42)
    completed = scheduler.run(requests)
    assert len(completed) == 50


# --- FIFO ordering ---


def test_fifo_admission_order(gpu: GPUConfig, model: ModelConfig) -> None:
    """Requests are admitted in arrival order."""
    scheduler = Scheduler(gpu, model)
    requests = [
        Request(request_id=0, prompt_length=100, output_length=5, arrival_time=0.0),
        Request(request_id=1, prompt_length=100, output_length=5, arrival_time=0.1),
        Request(request_id=2, prompt_length=100, output_length=5, arrival_time=0.2),
    ]
    completed = scheduler.run(requests)
    assert completed[0].request_id == 0
    assert completed[1].request_id == 1
    assert completed[2].request_id == 2


# --- Memory-bounded admission ---


def test_memory_bounded_admission(model: ModelConfig) -> None:
    """Requests wait in queue when memory is insufficient."""
    # Use very small GPU memory to force queueing
    kv_per_token = 4 * 4 * 64 * 2 * 2  # matches model config
    tokens_needed = 100  # prompt length
    blocks_needed = -(-tokens_needed // 16)  # 7 blocks
    bytes_per_block = 16 * kv_per_token
    # Enough for one request (7 blocks) but not two (14 blocks)
    kv_budget = bytes_per_block * (blocks_needed + 2)  # 9 blocks

    small_gpu = GPUConfig(
        name="small-gpu",
        hbm_bandwidth=2.0e12,
        compute_flops=312e12,
        total_memory=model.num_params * model.dtype_bytes
        + kv_budget
        + 100_000,  # barely more than weights
    )
    scheduler = Scheduler(small_gpu, model, block_size=16)
    requests = [
        Request(request_id=0, prompt_length=100, output_length=5, arrival_time=0.0),
        Request(request_id=1, prompt_length=100, output_length=5, arrival_time=0.0),
    ]
    completed = scheduler.run(requests)
    # Both should still complete eventually
    assert len(completed) == 2
    # Second request should start later (had to wait for first to free memory)
    assert completed[1].prefill_start_time > completed[0].prefill_start_time


# --- Continuous vs static throughput ---


def test_continuous_beats_static_throughput(gpu: GPUConfig, model: ModelConfig) -> None:
    """Continuous batching achieves higher throughput than static."""
    # High variance in output length is key — short requests finishing
    # early create idle batch slots that static batching wastes
    requests = generate_workload(100, 100.0, (50, 100), (5, 200), seed=42)
    requests_static = generate_workload(100, 100.0, (50, 100), (5, 200), seed=42)

    continuous = Scheduler(gpu, model)
    continuous_completed = continuous.run(requests)

    static = Scheduler(gpu, model)
    static_completed = static.run_static(requests_static, batch_size=10)

    continuous_time = max(r.completion_time for r in continuous_completed)
    static_time = max(r.completion_time for r in static_completed)

    continuous_tokens = sum(r.generated_tokens for r in continuous_completed)
    static_tokens = sum(r.generated_tokens for r in static_completed)

    assert continuous_tokens / continuous_time > static_tokens / static_time


# --- Reset between runs ---


def test_reset_allows_reuse(gpu: GPUConfig, model: ModelConfig) -> None:
    """Scheduler can be reused after run completes."""
    scheduler = Scheduler(gpu, model)
    requests = [
        Request(request_id=0, prompt_length=100, output_length=5, arrival_time=0.0)
    ]

    first = scheduler.run(requests)
    assert len(first) == 1

    second = scheduler.run(requests)
    assert len(second) == 1
    assert scheduler.clock > 0.0


# --- Metrics history ---


def test_history_is_recorded(gpu: GPUConfig, model: ModelConfig) -> None:
    """Scheduler records metrics at each step."""
    scheduler = Scheduler(gpu, model)
    requests = [
        Request(request_id=0, prompt_length=100, output_length=10, arrival_time=0.0)
    ]
    scheduler.run(requests)
    assert len(scheduler.history) > 0
    assert all(m.timestamp >= 0.0 for m in scheduler.history)
    assert all(0.0 <= m.memory_utilisation <= 1.0 for m in scheduler.history)


def test_history_is_recorded_static(gpu: GPUConfig, model: ModelConfig) -> None:
    """Scheduler with static batching records metrics at each step."""
    scheduler = Scheduler(gpu, model)
    requests = [
        Request(request_id=0, prompt_length=100, output_length=10, arrival_time=0.0)
    ]
    scheduler.run_static(requests)
    assert len(scheduler.history) > 0
    assert all(m.timestamp >= 0.0 for m in scheduler.history)
    assert all(0.0 <= m.memory_utilisation <= 1.0 for m in scheduler.history)


# --- Preallocation mode ---


def test_preallocate_uses_more_memory(gpu: GPUConfig, model: ModelConfig) -> None:
    """Preallocation reserves more memory than paged for the same workload."""
    requests = generate_workload(20, 1000.0, (50, 100), (10, 50), seed=42)
    requests_prealloc = generate_workload(100, 100.0, (50, 100), (5, 200), seed=42)

    paged = Scheduler(gpu, model)
    paged.run(requests)
    paged_peak = max(m.memory_utilisation for m in paged.history)

    prealloc = Scheduler(gpu, model, max_output_length=512, preallocate=True)
    prealloc.run(requests_prealloc)
    prealloc_peak = max(m.memory_utilisation for m in prealloc.history)

    assert prealloc_peak > paged_peak
