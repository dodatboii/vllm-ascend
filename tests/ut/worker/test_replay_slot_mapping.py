# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
##

"""SWA bounded replay: the worker-side write path (S6).

A replayed request keeps its prefix hit's blocks, which stay shared with every
other request that hit the same prefix. The cacheable groups must therefore not
write the replayed positions -- their KV already exists, and rewriting it would
corrupt the other requests sharing those blocks. The group that owns the replay
(``prefix_cacheable=False``) does write them, and that write is what rebuilds
its sliding window.

These cases run on CPU: the rule is exercised through the CP slot-mapping path,
which is plain torch. The triton kernels need an NPU and are covered by the e2e
kernel tests.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendSlidingWindowMLASpec,
    is_prefix_cacheable,
)

BLOCK_SIZE = 16
WINDOW = 32
# 100 prompt tokens -> a 96-token hit -> replay_start 64, recompute [64, 96).
REPLAY_START = 64
REPLAY_END = REPLAY_START + WINDOW
# A chunk wholly inside the replayed run, and one straddling its end.
INSIDE = np.arange(REPLAY_START, REPLAY_START + 4, dtype=np.int64)
ACROSS_END = np.arange(REPLAY_END - 2, REPLAY_END + 2, dtype=np.int64)
NO_REPLAY = 0
PAD = -1


def _swa_spec(*, bounded_replay: bool = True) -> AscendSlidingWindowMLASpec:
    return AscendSlidingWindowMLASpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=WINDOW,
        model_version="deepseek_v4",
        bounded_replay=bounded_replay,
    )


def _full_spec() -> FullAttentionSpec:
    return FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32)


def _dcp_group_patch(dcp_world_size: int, dcp_rank: int):
    mock_group = MagicMock(spec=GroupCoordinator)
    mock_group.world_size = dcp_world_size
    mock_group.rank_in_group = dcp_rank
    return patch("vllm_ascend.worker.block_table.get_dcp_group", return_value=mock_group)


def _block_table(*, kv_cache_group=None, dcp_world_size: int = 1, dcp_rank: int = 0):
    with _dcp_group_patch(dcp_world_size, dcp_rank):
        from vllm_ascend.worker.block_table import BlockTable

        table = BlockTable(
            block_size=BLOCK_SIZE,
            max_num_reqs=4,
            max_num_blocks_per_req=64,
            max_num_batched_tokens=512,
            pin_memory=False,
            device=torch.device("cpu"),
            kernel_sizes=[BLOCK_SIZE],
            cp_kv_cache_interleave_size=1,
            kv_cache_group=kv_cache_group,
        )
    # Enough distinct block ids that a wrong block index cannot look right.
    table.add_row(list(range(1, 9)), 0)
    return table


def _slot_mapping(table, positions, replay_start: int | None, replay_window: int = WINDOW) -> list[int]:
    """One CP slot mapping for request 0 over ``positions``.

    ``replay_start`` is what the kernels receive per request -- 0 is the value a
    request that is not replaying always carries, and None stands for the
    pre-replay call shape that does not pass the argument at all.
    """
    req_indices = torch.zeros(len(positions), dtype=torch.int32)
    per_req = None if replay_start is None else torch.tensor([replay_start], dtype=torch.int32)
    table._compute_dcp_slot_mapping(req_indices, torch.from_numpy(positions), per_req, replay_window)
    return table.slot_mapping.cpu[: len(positions)].tolist()


def _cacheable_table(**kwargs):
    return _block_table(kv_cache_group=KVCacheGroupSpec(["full"], _full_spec()), **kwargs)


def _replaying_table(**kwargs):
    return _block_table(kv_cache_group=KVCacheGroupSpec(["swa"], _swa_spec()), **kwargs)


# --- which group pads, which group writes ------------------------------------


def test_bounded_replay_group_is_not_prefix_cacheable():
    table = _replaying_table()

    assert table.is_prefix_cacheable is False
    assert is_prefix_cacheable(_swa_spec()) is False


def test_plain_groups_stay_cacheable():
    full = _cacheable_table()
    swa_without_replay = _block_table(kv_cache_group=KVCacheGroupSpec(["swa"], _swa_spec(bounded_replay=False)))

    assert full.is_prefix_cacheable is True
    assert swa_without_replay.is_prefix_cacheable is True


def test_table_without_a_group_is_cacheable():
    """No group means no replay, so the pre-existing behaviour is unchanged."""
    assert _block_table(kv_cache_group=None).is_prefix_cacheable is True


# --- the rule, through the CP path -------------------------------------------


def test_cacheable_group_pads_the_whole_replayed_chunk():
    """A chunk inside the replayed run writes none of its KV."""
    slots = _slot_mapping(_cacheable_table(), INSIDE, REPLAY_START)

    assert slots == [PAD] * len(INSIDE)


def test_cacheable_group_writes_past_the_replayed_run():
    """The replayed run ends at replay_start + window; past that the KV is new."""
    table = _cacheable_table()
    baseline = _slot_mapping(table, ACROSS_END, NO_REPLAY)

    slots = _slot_mapping(table, ACROSS_END, REPLAY_START)

    assert slots[:2] == [PAD, PAD]
    assert slots[2:] == baseline[2:]
    assert PAD not in baseline


def test_the_window_is_what_ends_the_padded_run():
    """Without a window nothing is below the write start, so nothing is padded."""
    table = _cacheable_table()

    slots = _slot_mapping(table, ACROSS_END, REPLAY_START, replay_window=0)

    assert PAD not in slots


def test_the_replaying_group_writes_everything():
    """The owning group's write is what rebuilds its sliding window."""
    table = _replaying_table()
    baseline = _slot_mapping(table, INSIDE, NO_REPLAY)

    slots = _slot_mapping(table, INSIDE, REPLAY_START)

    assert slots == baseline
    assert PAD not in slots


def test_nothing_changes_when_no_request_replays():
    """A replay start of 0 must be inert even with a window configured.

    The kernels guard on the replay start, not on the end of the replayed run,
    so a request that is not replaying is never mistaken for one replaying its
    first window of tokens.
    """
    table = _cacheable_table()

    for chunk in (INSIDE, ACROSS_END):
        absent = _slot_mapping(table, chunk, None)
        assert absent == _slot_mapping(table, chunk, NO_REPLAY)
        assert PAD not in absent


@pytest.mark.parametrize(("dcp_world_size", "dcp_rank"), [(2, 0), (2, 1), (4, 3)])
def test_padding_composes_with_context_parallelism(dcp_world_size, dcp_rank):
    """The replay rule must not un-pad a position another rank owns."""
    table = _cacheable_table(dcp_world_size=dcp_world_size, dcp_rank=dcp_rank)
    baseline = _slot_mapping(table, ACROSS_END, NO_REPLAY)

    slots = _slot_mapping(table, ACROSS_END, REPLAY_START)

    assert slots[:2] == [PAD, PAD]
    assert slots[2:] == baseline[2:]


# --- the fused path gets one flag per group, in group order ------------------


def test_fused_group_flags_follow_the_group_order():
    from vllm_ascend.worker.block_table import MultiGroupBlockTable

    with _dcp_group_patch(1, 0):
        tables = MultiGroupBlockTable(
            max_num_reqs=4,
            max_model_len=1024,
            max_num_batched_tokens=512,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[BLOCK_SIZE, BLOCK_SIZE],
            max_num_blocks=[64, 64],
            kernel_sizes=[[BLOCK_SIZE], [BLOCK_SIZE]],
            kv_cache_groups=[
                KVCacheGroupSpec(["full"], _full_spec()),
                KVCacheGroupSpec(["swa"], _swa_spec()),
            ],
        )

    assert tables._can_fuse_slot_mapping
    # Group order is the launch's group_idx order, so [cacheable, replaying].
    assert tables._fused_is_prefix_cacheable.tolist() == [1, 0]
