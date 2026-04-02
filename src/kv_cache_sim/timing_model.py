from kv_cache_sim.models import GPUConfig, ModelConfig


class TimingModel:
    """Analytical timing model for LLM inference.

    Estimates prefill and decode latency from hardware and model
    parameters. Prefill is modelled as compute-bound (FLOPS-limited),
    decode as memory-bandwidth-bound (weight-loading-limited).
    """

    def __init__(self, gpu: GPUConfig, model: ModelConfig) -> None:
        self.gpu: GPUConfig = gpu
        self.model: ModelConfig = model

    def prefill_time(self, prompt_length: int, batch_size: int = 1) -> float:
        """Compute-bound prefill latency in seconds."""
        flops_per_request: float = 2.0 * self.model.num_params * prompt_length
        total_flops: float = flops_per_request * batch_size

        return total_flops / self.gpu.compute_flops

    def decode_step_time(self) -> float:
        """Wall-clock time for one decode step (load weights once for entire batch)"""
        weights_bytes: int = self.model.num_params * self.model.dtype_bytes
        return weights_bytes / (self.gpu.hbm_bandwidth)

    def kv_cache_bytes_per_token(self) -> int:
        """KV cache memory footprint per token in bytes."""
        return (
            2
            * self.model.num_layers
            * self.model.num_kv_heads
            * self.model.head_dim
            * self.model.dtype_bytes
        )

    def max_context_for_memory(self, available_bytes: int) -> int:
        """Maximum context length that fits in available_bytes."""
        return available_bytes // self.kv_cache_bytes_per_token()
