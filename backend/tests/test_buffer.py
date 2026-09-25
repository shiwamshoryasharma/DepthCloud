import numpy as np
import pytest

from depthcloud.camera.buffer import FrameBuffer, FramePacket, recommend_capacity


def frame(seq, size=4):
    return FramePacket(seq, 1700000000.0 + seq, float(seq), np.full((size, size, 3), seq, dtype=np.uint8))


def test_fifo_discards_oldest_and_retains_latest_without_copying():
    buffer = FrameBuffer(capacity=3, max_bytes=1000)
    packets = [frame(i) for i in range(1, 6)]
    for packet in packets:
        buffer.push(packet)
    assert [p.sequence_id for p in buffer.snapshot()] == [3, 4, 5]
    assert buffer.latest() is packets[-1]
    assert buffer.stats()["evicted"] == 2


def test_byte_limit_caps_retention_even_when_frame_count_is_large():
    buffer = FrameBuffer(capacity=100, max_bytes=100)
    for i in range(10):
        buffer.push(frame(i))
    assert buffer.stats()["bytes"] <= 100
    assert len(buffer.snapshot()) == 2


def test_oversized_frame_is_rejected_instead_of_exceeding_budget():
    buffer = FrameBuffer(capacity=2, max_bytes=10)
    with pytest.raises(ValueError, match="memory"):
        buffer.push(frame(1))
    assert buffer.latest() is None


def test_resize_preserves_newest_and_clear_releases_all_frames():
    buffer = FrameBuffer(capacity=5, max_bytes=1000)
    for i in range(5):
        buffer.push(frame(i))
    buffer.resize(2)
    assert [p.sequence_id for p in buffer.snapshot()] == [3, 4]
    buffer.clear()
    assert buffer.latest() is None
    assert buffer.stats()["bytes"] == 0
    assert buffer.snapshot() == []


@pytest.mark.parametrize("capacity", [0, -1])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError):
        FrameBuffer(capacity, 1000)


def test_automatic_capacity_responds_to_latency_and_memory():
    fast = recommend_capacity(30, 10, 1_000_000, 100_000_000, 128)
    slow = recommend_capacity(30, 200, 1_000_000, 100_000_000, 128)
    limited = recommend_capacity(30, 200, 1_000_000, 2_000_000, 128)
    assert 1 <= fast < slow <= 128
    assert limited == 2


def test_automatic_capacity_handles_no_measurements():
    assert recommend_capacity(0, 0, 1_000_000, 10_000_000, 128) >= 1

