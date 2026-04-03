import statistics

import matplotlib.pyplot as plt

from kv_cache_sim.models import Request, SchedulerMetrics, SimulationSummary


def compute_summary(completed: list[Request]) -> SimulationSummary:
    first_arrival: float = min([r.arrival_time for r in completed])
    last_completion: float = max([r.completion_time for r in completed])
    total_time: float = last_completion - first_arrival

    total_tokens: int = sum([r.generated_tokens for r in completed])
    throughput: float = total_tokens / total_time

    ttft: list[float] = [r.first_token_time - r.arrival_time for r in completed]
    avg_ttft: float = sum(ttft) / len(ttft)
    p50_ttft: float = statistics.median(ttft)
    p99_ttft: float = statistics.quantiles(ttft, n=100)[98]

    tpot: list[float] = [
        (r.completion_time - r.first_token_time) / r.generated_tokens for r in completed
    ]
    avg_tpot: float = sum(tpot) / len(tpot)
    p50_tpot: float = statistics.median(tpot)
    p99_tpot: float = statistics.quantiles(tpot, n=100)[98]

    return SimulationSummary(
        total_time=total_time,
        total_tokens=total_tokens,
        throughput=throughput,
        ttft_values=ttft,
        avg_ttft=avg_ttft,
        p50_ttft=p50_ttft,
        p99_ttft=p99_ttft,
        tpot_values=tpot,
        avg_tpot=avg_tpot,
        p50_tpot=p50_tpot,
        p99_tpot=p99_tpot,
    )


def plot_throughput_comparison(results: dict[str, SimulationSummary]) -> None:
    fig, ax = plt.subplots()
    labels: list[str] = list(results.keys())
    values: list[float] = [results[label].throughput for label in labels]
    ax.bar(labels, values)
    ax.set_title("Throughput")
    ax.set_xlabel("Batching Strategy")
    ax.set_ylabel("Throughput (tokens/sec)")
    fig.savefig("results/throughput_comparison.png")
    plt.close(fig)


def plot_ttft_distribution(results: dict[str, SimulationSummary]) -> None:
    # Input: completed requests from both runs
    # Histogram or CDF of TTFT, two series overlaid
    # Save to results/ttft_distribution.png
    fig, ax = plt.subplots()

    continuous_ttft_ms = [1000 * ttft for ttft in results["continuous"].ttft_values]
    static_ttft_ms = [1000 * ttft for ttft in results["static"].ttft_values]

    ax.hist(
        continuous_ttft_ms,
        bins=50,
        label="Continuous Batching",
        alpha=0.5,
    )
    ax.hist(
        static_ttft_ms,
        bins=50,
        label="Static Batching",
        alpha=0.5,
    )
    ax.set_xlabel("Time to First Token (ms)")
    ax.legend()

    fig.suptitle("Time to First Token Distributions")

    fig.savefig("results/ttft_distribution.png")
    plt.close(fig)


def plot_memory_over_time(histories: dict[str, list[SchedulerMetrics]]) -> None:
    fig, ax = plt.subplots()
    labels: list[str] = list(histories.keys())
    for label in labels:
        timestamps = [1000 * h.timestamp for h in histories[label]]
        memory_utilisation = [h.memory_utilisation for h in histories[label]]
        ax.plot(timestamps, memory_utilisation, label=label)
    ax.set_title("Memory Utilisation")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Memory utilisation")
    ax.legend()
    fig.savefig("results/memory_utilisation.png")
    plt.close(fig)


def plot_batch_size_over_time(history: list[SchedulerMetrics]) -> None:
    fig, ax = plt.subplots()
    timestamps = [1000 * h.timestamp for h in history]
    batch_size = [h.batch_size for h in history]
    ax.plot(timestamps, batch_size)
    ax.set_title("Batch Size")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Batch Size")
    fig.savefig("results/batch_size.png")
    plt.close(fig)
