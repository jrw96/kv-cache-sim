import pytest

from kv_cache_sim.block_allocator import BlockAllocator


@pytest.fixture
def allocator() -> BlockAllocator:
    """10 blocks of 16 tokens each = 160 token capacity."""
    return BlockAllocator(total_blocks=10, block_size=16)


# --- Allocation ---


def test_allocate_success(allocator: BlockAllocator) -> None:
    """Allocate blocks for a single request and verify state."""
    result = allocator.allocate(0, 32)
    assert result
    assert allocator.get_utilisation() == pytest.approx(2 / 10)
    assert 0 in allocator.block_tables


def test_allocate_exact_block_boundary(allocator: BlockAllocator) -> None:
    """16 tokens should take exactly 1 block, not 2."""
    allocator.allocate(0, 16)
    assert allocator.get_utilisation() == pytest.approx(1 / 10)


def test_allocate_insufficient_memory(allocator: BlockAllocator) -> None:
    """Reject allocation when not enough blocks remain."""
    result = allocator.allocate(0, 161)
    assert not result
    assert allocator.get_utilisation() == pytest.approx(0.0)
    assert len(allocator.block_tables) == 0


def test_allocate_duplicate_request_id(allocator: BlockAllocator) -> None:
    """Reject allocation for an already-allocated request."""
    allocator.allocate(0, 16)
    result = allocator.allocate(0, 16)
    assert not result


def test_allocate_all_memory(allocator: BlockAllocator) -> None:
    """Fill all blocks, verify next allocation fails."""
    result = allocator.allocate(0, 160)
    assert result
    assert allocator.get_utilisation() == pytest.approx(1.0)

    result = allocator.allocate(1, 1)
    assert not result


# --- Free ---


def test_free_releases_blocks(allocator: BlockAllocator) -> None:
    """Freed blocks are available for reallocation."""
    allocator.allocate(0, 32)
    allocator.free(0)
    assert allocator.get_utilisation() == pytest.approx(0.0)

    result = allocator.allocate(1, 32)
    assert result


def test_free_cleans_up_state(allocator: BlockAllocator) -> None:
    """Free removes request from both block_tables and token_counts."""
    allocator.allocate(0, 32)
    allocator.free(0)
    assert 0 not in allocator.block_tables
    assert 0 not in allocator.token_counts


# --- Append ---


def test_append_within_existing_block(allocator: BlockAllocator) -> None:
    """Appending tokens that fit in the current block allocates no new blocks."""
    allocator.allocate(0, 1)
    allocator.append(0, 1)
    assert allocator.get_utilisation() == pytest.approx(1 / 10)


def test_append_triggers_new_block(allocator: BlockAllocator) -> None:
    """Appending past block boundary allocates a new block."""
    allocator.allocate(0, 16)
    allocator.append(0, 1)
    assert allocator.get_utilisation() == pytest.approx(2 / 10)


def test_append_oom(allocator: BlockAllocator) -> None:
    """Append fails gracefully when no blocks remain."""
    allocator.allocate(0, 160)
    result = allocator.append(0, 1)
    assert not result


# --- Utilisation ---


def test_utilisation_empty(allocator: BlockAllocator) -> None:
    """Empty allocator has 0.0 utilisation."""
    assert allocator.get_utilisation() == pytest.approx(0.0)


def test_utilisation_full(allocator: BlockAllocator) -> None:
    """Fully allocated has 1.0 utilisation."""
    allocator.allocate(0, 160)
    assert allocator.get_utilisation() == pytest.approx(1.0)


# --- Fragmentation ---


def test_wasted_tokens(allocator: BlockAllocator) -> None:
    """Internal fragmentation from partially filled blocks."""
    allocator.allocate(0, 17)
    assert allocator.get_wasted_tokens() == 15


def test_wasted_tokens_exact_fit(allocator: BlockAllocator) -> None:
    """No waste when tokens exactly fill blocks."""
    allocator.allocate(0, 16)
    assert allocator.get_wasted_tokens() == 0


# --- Can Allocate ---


def test_can_allocate_without_side_effects(allocator: BlockAllocator) -> None:
    """can_allocate checks without modifying state."""
    allocator.can_allocate(16)
    assert allocator.get_utilisation() == pytest.approx(0.0)
    assert len(allocator.block_tables) == 0
