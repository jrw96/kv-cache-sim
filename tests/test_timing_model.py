import pytest

from kv_cache_sim.models import A100_80GB, LLAMA_70B
from kv_cache_sim.timing_model import TimingModel


@pytest.fixture
def timing() -> TimingModel:
    return TimingModel(gpu=A100_80GB, model=LLAMA_70B)


# --- Sanity checks against hand-calculated values ---


def test_prefill_sanity(timing: TimingModel) -> None:
    """70B model, 1000 tokens: 2 * 70e9 * 1000 / 312e12 ≈ 0.449s"""
    result = timing.prefill_time(prompt_length=1000)
    assert result == pytest.approx(0.449, rel=0.01)


def test_decode_sanity(timing: TimingModel) -> None:
    """70B FP16: 140e9 bytes / 2e12 bytes/sec = 0.07s"""
    result = timing.decode_step_time()
    assert result == pytest.approx(0.07, rel=0.01)


def test_kv_cache_per_token(timing: TimingModel) -> None:
    """80 layers * 8 heads * 128 dim * 2 (K+V) * 2 bytes = 327,680"""
    result = timing.kv_cache_bytes_per_token()
    assert result == 327_680


# --- Scaling behaviour ---


def test_prefill_scales_with_prompt_length(timing: TimingModel) -> None:
    """Double the prompt, double the prefill time."""
    t1 = timing.prefill_time(prompt_length=1000)
    t2 = timing.prefill_time(prompt_length=2000)
    assert t2 == pytest.approx(2.0 * t1)


def test_prefill_scales_with_batch_size(timing: TimingModel) -> None:
    """Double the batch, double the prefill time."""
    t1 = timing.prefill_time(prompt_length=1000, batch_size=1)
    t2 = timing.prefill_time(prompt_length=1000, batch_size=2)
    assert t2 == pytest.approx(2.0 * t1)


def test_max_context_for_memory(timing: TimingModel) -> None:
    """1 GB should hold 1e9 / 327680 ≈ 3051 tokens."""
    result = timing.max_context_for_memory(1_000_000_000)
    assert result == 1_000_000_000 // 327_680
