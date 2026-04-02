from dataclasses import dataclass
from enum import Enum, auto


class RequestState(Enum):
    WAITING = auto()
    PREFILLING = auto()
    RUNNING = auto()
    COMPLETED = auto()


@dataclass
class Request:
    request_id: int
    arrival_time: float
    prompt_length: int
    max_output_length: int
    generated_tokens: int
    state: RequestState = RequestState.WAITING
    prefill_start_time: float | None = None
    first_token_time: float | None = None
    completion_time: float | None = None


@dataclass
class GPUConfig:
    name: str
    hbm_bandwidth: float  # bytes/sec
    compute_flops: float  # FLOPS (FP16)
    total_memory: int  # bytes


@dataclass
class ModelConfig:
    num_params: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int  # 2 for FP16, 4 for FP32


A100_80GB = GPUConfig(
    name="A100-80GB",
    hbm_bandwidth=2.0e12,
    compute_flops=312e12,
    total_memory=80 * 1024**3,
)

LLAMA_70B = ModelConfig(
    num_params=70_000_000_000,
    num_layers=80,
    num_kv_heads=8,
    head_dim=128,
    dtype_bytes=2,
)
