from kv_cache_sim.metrics import (
    compute_summary,
    plot_batch_size_over_time,
    plot_memory_over_time,
    plot_throughput_comparison,
    plot_ttft_distribution,
)
from kv_cache_sim.models import (
    A100_80GB,
    LLAMA_1B,
)
from kv_cache_sim.scheduler import Scheduler
from kv_cache_sim.workload import generate_workload


def run_simulation():
    gpu = A100_80GB
    model = LLAMA_1B

    requests = generate_workload(1000, 100.0, (50, 100), (5, 200), seed=42)
    scheduler = Scheduler(gpu, model)
    completed_continuous = scheduler.run(requests)
    summary_continuous = compute_summary(completed_continuous)

    requests_static = generate_workload(1000, 100.0, (50, 100), (5, 200), seed=42)
    scheduler_static = Scheduler(gpu, model)
    completed_static = scheduler_static.run_static(requests_static)
    summary_static = compute_summary(completed_static)

    summary_dict = {"continuous": summary_continuous, "static": summary_static}
    histories_dict = {
        "continuous": scheduler.history,
        "static": scheduler_static.history,
    }
    plot_throughput_comparison(summary_dict)
    plot_ttft_distribution(summary_dict)
    plot_memory_over_time(histories_dict)
    plot_batch_size_over_time(scheduler.history)


if __name__ == "__main__":
    run_simulation()
