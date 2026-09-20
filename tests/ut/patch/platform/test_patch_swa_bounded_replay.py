# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the SWA bounded replay scheduler plumbing patch."""

import dataclasses
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.request import Request

import vllm_ascend.patch.platform.patch_swa_bounded_replay  # noqa: F401  (applies the patch)
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.utils import vllm_version_is

BLOCK_SIZE = 16
WINDOW = 32

pytestmark = pytest.mark.skipif(
    vllm_version_is("0.28.0"),
    reason="the release lane predates the bounded replay API, so the patch is a no-op",
)


def _spec(*, bounded_replay: bool) -> AscendSlidingWindowMLASpec:
    return AscendSlidingWindowMLASpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=WINDOW,
        model_version="deepseek_v4",
        bounded_replay=bounded_replay,
    )


def _request(replay_start: int = 64) -> SimpleNamespace:
    return SimpleNamespace(request_id="req-0", replay_start=replay_start)


# --------------------------------------------------------------- scheduler ---


def test_request_has_a_replay_start_default():
    assert Request.replay_start == 0


def test_new_request_data_carries_the_replay_start():
    # Declared, because the payload is msgpack-encoded field by field.
    assert "replay_start" in {field.name for field in dataclasses.fields(NewRequestData)}

    request = MagicMock()
    request.replay_start = WINDOW
    new_req_data = NewRequestData.from_request(request, ((1,),))
    assert new_req_data.replay_start == WINDOW


def test_cached_request_data_carries_the_replay_start_of_resumed_requests():
    """Ascend-only: the V1 runner resumes a preempted request through this
    payload, where the V2 runner uses the new-request list."""
    assert "replay_start" in {field.name for field in dataclasses.fields(CachedRequestData)}
    assert "replay_start" in inspect.signature(CachedRequestData.__init__).parameters

    def cached_req_data(req_id: str) -> CachedRequestData:
        return CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids={req_id},
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[0],
            num_output_tokens=[0],
        )

    first = cached_req_data("req-0")
    second = cached_req_data("req-1")
    # Absent means "no replay", and one payload never shares state with another.
    assert first.replay_start == {}
    first.replay_start["req-0"] = WINDOW
    assert first.replay_start == {"req-0": WINDOW}
    assert second.replay_start == {}


# -------------------------------------------------------------- kv manager ---


def test_allocate_slots_guard_allows_adopting_without_allocating():
    # The guard is a precondition check on the caller, so the relaxed form has
    # to be in the installed method itself.
    assert "and num_new_computed_tokens == 0" in inspect.getsource(KVCacheManager.allocate_slots)

    with pytest.raises(ValueError, match="computed tokens to adopt"):
        KVCacheManager.allocate_slots(
            MagicMock(),
            _request(),
            num_new_tokens=0,
            num_new_computed_tokens=0,
            num_external_computed_tokens=0,
        )


def test_cache_blocks_skips_non_cacheable_groups():
    manager = MagicMock()
    manager.kv_cache_spec = _spec(bounded_replay=True)

    SingleTypeKVCacheManager.cache_blocks(manager, _request(), 0)

    manager.num_cached_block.get.assert_not_called()


def test_cache_blocks_passes_cacheable_groups_through():
    manager = MagicMock()
    manager.kv_cache_spec = _spec(bounded_replay=False)
    manager.block_size = BLOCK_SIZE
    manager.num_cached_block.get.return_value = 0

    # Zero tokens means the upstream early return, which is all this test needs
    # to observe: the wrapper let the call through to the original.
    SingleTypeKVCacheManager.cache_blocks(manager, _request(), 0)

    manager.num_cached_block.get.assert_called_once_with("req-0", 0)
