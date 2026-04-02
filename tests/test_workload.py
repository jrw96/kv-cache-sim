from kv_cache_sim.workload import generate_workload


def test_generate_workload_arrival_order() -> None:
    """Requests arrive in monotonically increasing order."""
    requests = generate_workload(
        num_requests=100,
        arrival_rate=10.0,
        prompt_length_range=(100, 500),
        output_length_range=(50, 200),
        seed=42,
    )
    for i in range(1, len(requests)):
        assert requests[i].arrival_time > requests[i - 1].arrival_time


def test_generate_workload_lengths_in_range() -> None:
    """Prompt and output lengths stay within specified ranges."""
    requests = generate_workload(
        num_requests=100,
        arrival_rate=10.0,
        prompt_length_range=(100, 500),
        output_length_range=(50, 200),
        seed=42,
    )
    for req in requests:
        assert 100 <= req.prompt_length <= 500
        assert 50 <= req.output_length <= 200


def test_generate_workload_reproducible() -> None:
    """Same seed produces identical workloads."""
    a = generate_workload(10, 10.0, (100, 500), (50, 200), seed=42)
    b = generate_workload(10, 10.0, (100, 500), (50, 200), seed=42)
    for ra, rb in zip(a, b, strict=True):
        assert ra.arrival_time == rb.arrival_time
        assert ra.prompt_length == rb.prompt_length
        assert ra.output_length == rb.output_length
