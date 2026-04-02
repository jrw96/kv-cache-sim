import random

from kv_cache_sim.models import Request


def generate_workload(
    num_requests: int,
    arrival_rate: float,
    prompt_length_range: tuple[int, int],
    output_length_range: tuple[int, int],
    seed: int | None = None,
) -> list[Request]:
    """Generate synthetic request stream with Poisson arrivals."""
    rng = random.Random(seed)
    clock = 0.0

    requests: list[Request] = []
    for i in range(num_requests):
        clock += rng.expovariate(arrival_rate)
        requests.append(
            Request(
                request_id=i,
                arrival_time=clock,
                prompt_length=rng.randint(*prompt_length_range),
                output_length=rng.randint(*output_length_range),
            )
        )

    return requests
