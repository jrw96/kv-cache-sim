from kv_cache_sim.models import GPUConfig, ModelConfig


class TimingModel:
    def __init__(self, gpu: GPUConfig, model: ModelConfig) -> None:
        self.gpu = gpu
        self.model = model

    def prefill_time(self, prompt_length: int, batch_size: int = 1) -> float:
        flops_per_request: float = 2.0 * self.model.num_params * prompt_length
        total_flops: float = flops_per_request * batch_size

        return total_flops / self.gpu.compute_flops

    def decode_step_time(self) -> float:
        """Wall-clock time for one decode step (load weights once for entire batch)"""
        weights_bytes: int = self.model.num_params * self.model.dtype_bytes
        return weights_bytes / (self.gpu.hbm_bandwidth)

    def kv_cache_bytes_per_token(self) -> int:
        return (
            2
            * self.model.num_layers
            * self.model.num_kv_heads
            * self.model.head_dim
            * self.model.dtype_bytes
        )

    def max_context_for_memory(self, available_bytes: int) -> int:
        return available_bytes // self.kv_cache_bytes_per_token()
