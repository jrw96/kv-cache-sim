import pytest

from kv_cache_sim.metrics import compute_summary
from kv_cache_sim.models import Request, RequestState


def _make_completed_request(
    request_id: int,
    arrival: float,
    first_token: float,
    completion: float,
    tokens: int,
) -> Request:
    """Helper to build a completed request with timestamps."""
    return Request(
        request_id=request_id,
        prompt_length=100,
        output_length=tokens,
        arrival_time=arrival,
        state=RequestState.COMPLETED,
        generated_tokens=tokens,
        prefill_start_time=arrival,
        first_token_time=first_token,
        completion_time=completion,
    )


def test_compute_summary_throughput() -> None:
    """Throughput = total tokens / total time."""
    completed = [
        _make_completed_request(
            0, arrival=0.0, first_token=0.1, completion=1.0, tokens=100
        ),
        _make_completed_request(
            1, arrival=0.0, first_token=0.1, completion=2.0, tokens=200
        ),
    ]
    summary = compute_summary(completed)
    assert summary.total_tokens == 300
    assert summary.throughput == pytest.approx(300 / 2.0)


def test_compute_summary_ttft() -> None:
    """TTFT = first_token_time - arrival_time."""
    completed = [
        _make_completed_request(
            0, arrival=0.0, first_token=0.5, completion=1.0, tokens=10
        ),
        _make_completed_request(
            1, arrival=1.0, first_token=1.2, completion=2.0, tokens=10
        ),
    ]
    summary = compute_summary(completed)
    assert summary.ttft_values[0] == pytest.approx(0.5)
    assert summary.ttft_values[1] == pytest.approx(0.2)


def test_compute_summary_tpot() -> None:
    """TPOT = (completion - first_token) / generated_tokens."""
    completed = [
        _make_completed_request(
            0, arrival=0.0, first_token=0.0, completion=1.0, tokens=10
        ),
    ]
    summary = compute_summary(completed)
    assert summary.avg_tpot == pytest.approx(0.1)
