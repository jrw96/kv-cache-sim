class BlockAllocator:
    """
    Paged KV cache memory manager.

    Manages a fixed pool of memory blocks, each holding block_size tokens
    of KV cache. Blocks are allocated on demand and tracked via per-request
    block tables, modelling the PagedAttention memory management scheme.
    """

    def __init__(self, total_blocks: int, block_size: int) -> None:
        self.total_blocks: int = total_blocks
        self.block_size: int = block_size
        self.free_blocks: set[int] = set(range(total_blocks))
        self.block_tables: dict[int, list[int]] = {}
        self.token_counts: dict[int, int] = {}

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if enough free blocks exist to hold num_tokens."""
        return self._num_blocks_required(num_tokens) <= len(self.free_blocks)

    def allocate(self, request_id: int, num_tokens: int) -> bool:
        """Allocate blocks for a new request. Returns False if insufficient memory."""
        if request_id in self.block_tables:
            return False

        if not self.can_allocate(num_tokens):
            return False

        self.block_tables[request_id] = []
        num_blocks_required: int = self._num_blocks_required(num_tokens)
        for _ in range(num_blocks_required):
            self.block_tables[request_id].append(self.free_blocks.pop())
        self.token_counts[request_id] = num_tokens

        return True

    def append(self, request_id: int, num_new_tokens: int = 1) -> bool:
        """
        Extend a request's allocation by num_new_tokens.

        Allocates a new block only when the current last block is full.
        Returns False on OOM.
        """
        space: int = self._block_space_remaining(request_id)
        overflow: int = num_new_tokens - space

        if overflow > 0:
            new_blocks_needed: int = self._num_blocks_required(overflow)
            if new_blocks_needed > len(self.free_blocks):
                return False
            for _ in range(new_blocks_needed):
                self.block_tables[request_id].append(self.free_blocks.pop())

        self.token_counts[request_id] += num_new_tokens
        return True

    def free(self, request_id: int) -> None:
        """Release all blocks for a completed request."""
        self.free_blocks.update(self.block_tables[request_id])
        del self.block_tables[request_id]
        del self.token_counts[request_id]

    def get_utilisation(self) -> float:
        """Fraction of total blocks currently allocated (0.0 to 1.0)."""
        return 1.0 - len(self.free_blocks) / self.total_blocks

    def get_wasted_tokens(self) -> int:
        """Total unused token slots across all allocated blocks."""
        return sum(
            self._block_space_remaining(request_id) for request_id in self.block_tables
        )

    def _num_blocks_required(self, num_tokens: int) -> int:
        """Total number of blocks required to allocate space for num_tokens tokens."""
        # Ceiling division, avoids importing math library
        return -(-num_tokens // self.block_size)

    def _block_space_remaining(self, request_id: int) -> int:
        """Space remaining in the last block allocated to the given request."""
        blocks: list[int] = self.block_tables[request_id]
        return len(blocks) * self.block_size - self.token_counts[request_id]
