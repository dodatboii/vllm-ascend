# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
    is_prefix_cacheable,
)


def _mla_spec():
    return AscendMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )


def test_get_storage_block_size_and_dcp_memory():
    spec = _mla_spec()
    # On main, storage_block_size is an optional dataclass field and may be
    # None. Ascend derives physical rows from block_size / compression ratio.
    expected = spec.block_size // get_kv_cache_compression_ratio(spec)
    assert get_storage_block_size(spec) == expected

    uniform = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"layer": spec})
    assert get_storage_block_size(uniform) == expected

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
    )
    assert spec.max_memory_usage_bytes(vllm_config) > 0


def test_sliding_window_mla_storage_and_page_size():
    spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    assert spec.storage_block_size == 16
    assert spec.real_page_size_bytes == 16 * 128 * 2


def _swa_spec(bounded_replay=False, sliding_window=64):
    return AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=sliding_window,
        bounded_replay=bounded_replay,
    )


class TestSWABoundedReplaySpec:
    """Bounded-replay spec semantics ported from upstream #56227."""

    def test_default_is_prefix_cacheable_without_replay(self):
        spec = _swa_spec()
        assert spec.prefix_cacheable
        assert spec.prefix_replay_tokens == 0
        assert is_prefix_cacheable(spec)

    def test_bounded_replay_opts_out_of_prefix_caching(self):
        spec = _swa_spec(bounded_replay=True, sliding_window=128)
        assert not spec.prefix_cacheable
        assert spec.prefix_replay_tokens == 128
        assert not is_prefix_cacheable(spec)

    def test_merge_propagates_bounded_replay(self):
        merged = AscendSlidingWindowMLASpec.merge([_swa_spec(bounded_replay=True), _swa_spec(bounded_replay=True)])
        assert merged.bounded_replay
        assert not merged.prefix_cacheable
        assert merged.prefix_replay_tokens == 64

    def test_merge_rejects_mixed_replay_policy(self):
        with pytest.raises(AssertionError, match="replay policy"):
            AscendSlidingWindowMLASpec.merge([_swa_spec(bounded_replay=True), _swa_spec(bounded_replay=False)])
