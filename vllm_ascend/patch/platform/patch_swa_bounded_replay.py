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

"""Apply the scheduler-side SWA bounded replay plumbing (vLLM #56227).

The replay-aware scheduler (`vllm_ascend.core.swa_replay_scheduler`) drives
these, and they are runner-independent:

1. ``Request.replay_start`` / ``NewRequestData.replay_start`` /
   ``CachedRequestData.replay_start``: the per-request replay start that the
   model runner turns into the attention window clamp and the padded slot
   range. ``Request`` and ``NewRequestData`` mirror upstream #56227;
   ``CachedRequestData`` is Ascend-only (see below).
2. ``KVCacheManager.allocate_slots``: allow a step that adopts computed tokens
   while allocating no slots of its own, so a chunk ending inside the replayed
   range can still make progress.
3. ``SingleTypeKVCacheManager.cache_blocks``: a non-cacheable group (the
   replayed sliding-window group, and the existing scratch groups) must never
   publish blocks to the prefix cache.
4. ``CachedRequestData.replay_start`` (Ascend-only, beyond #56227): the V1
   runner resumes a preempted request through ``CachedRequestData``, whereas
   the V2 runner folds resumed requests into the new-request list. Without
   this, a resumed request that replays again would replay with the worker
   unaware, which would rewrite shared prefix-cache blocks.

Only the ``allocate_slots`` guard change sits inside a method body, so that
method is a fork of the pinned source: every line but the marked
``[SWA-REPLAY S8]`` hunk is a verbatim copy. A vLLM pin bump must re-check it,
the same way `swa_replay_scheduler.py` must be re-checked.

The release lane (``vllm_version_is("0.28.0")``) predates the whole
replay/spec API surface, so this module is a no-op there: the switch is
downgraded to False in `vllm_ascend.platform` (S13).
"""

import dataclasses
import functools
import inspect
from typing import Any

from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
from vllm_ascend.utils import vllm_version_is


def _add_dataclass_field(cls: type, name: str, default: Any, annotation: Any) -> None:
    """Declare one more field on an already-processed dataclass.

    ``SchedulerOutput`` and its payloads cross to the worker as msgpack, which
    only sees *declared* fields, so an attribute written onto an instance never
    arrives. Re-running ``dataclasses.dataclass`` regenerates ``__init__``, but
    it only sets a method when it is absent from the class ``__dict__``, so the
    generated ``__init__`` is dropped first. ``__repr__`` and ``__eq__`` are
    left alone: both payloads define their own (with token ids redacted), and
    clobbering those would leak them into logs.
    """
    if name in cls.__dataclass_fields__:
        return
    cls.__annotations__[name] = annotation
    # A mutable default must be a factory, or every instance would share it.
    setattr(cls, name, dataclasses.field(default_factory=default) if default is dict else default)
    delattr(cls, "__init__")
    dataclasses.dataclass(cls)
    assert name in cls.__dataclass_fields__, f"{cls.__name__}.{name} was not declared"
    assert name in inspect.signature(cls.__init__).parameters, (
        f"{cls.__name__}.__init__ does not accept {name}; msgspec could not rebuild it"
    )


def _patch_request_replay_start() -> None:
    """``Request.replay_start``: where a replayed request's window begins.

    A class-level default is equivalent for every reader and costs nothing per 
    request; the scheduler writes the per-request value when it rewinds a prefix 
    hit.
    """
    if hasattr(Request, "replay_start"):
        return
    Request.replay_start = 0


def _patch_new_request_data_replay_start() -> None:
    """Carry the replay start of a freshly admitted request to the worker."""
    _add_dataclass_field(NewRequestData, "replay_start", 0, int)

    original = NewRequestData.__dict__["from_request"].__func__

    @functools.wraps(original)
    def from_request(cls, request, *args, **kwargs):
        new_req_data = original(cls, request, *args, **kwargs)
        # Writing the attribute is safe because the field is declared above.
        new_req_data.replay_start = request.replay_start
        return new_req_data

    NewRequestData.from_request = classmethod(from_request)


def _patch_cached_request_data_replay_start() -> None:
    """Carry the replay start of a resumed request to the V1 worker.

    Ascend-only: see the module docstring. A dict rather than a list parallel
    to ``req_ids`` because an absent entry is the common case and has to mean
    "no replay" without depending on the payload's length.
    """
    _add_dataclass_field(CachedRequestData, "replay_start", dict, dict[str, int])


def _patch_allocate_slots_guard() -> None:
    """Install the S8 fork of the pinned ``allocate_slots``."""
    expected = (
        "self",
        "request",
        "num_new_tokens",
        "num_new_computed_tokens",
        "new_computed_blocks",
        "num_lookahead_tokens",
        "num_external_computed_tokens",
        "delay_cache_blocks",
        "num_encoder_tokens",
        "full_sequence_must_fit",
        "reserved_blocks",
        "has_scheduled_reqs",
    )
    current = tuple(inspect.signature(KVCacheManager.allocate_slots).parameters)
    if current != expected:
        raise RuntimeError(
            "Cannot apply the SWA bounded replay allocate_slots patch: "
            f"unexpected KVCacheManager.allocate_slots signature {current}"
        )

def _allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.
            full_sequence_must_fit: Only allocate blocks if the KV cache has enough
                free blocks to hold the full sequence, accounting for prefix cache hits
                and sliding window. Used as an admission gate to prevent over-admitting
                requests when chunked prefill would otherwise only check the first chunk
            reserved_blocks: Number of free blocks that must be left available for
                other in-flight sequences to complete. The actual allocation is only
                made if it fits within (free blocks - reserved_blocks). Used to gate
                async KV-connector loads so their initial allocation cannot consume
                blocks an already in-flight (prefilling) sequence is relying on.
            has_scheduled_reqs: Whether any requests are already scheduled to run
                this step, controls whether watermark is applied.

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # [SWA-REPLAY S8] A step may need no slots of its own while still
        # adopting computed tokens: an async KV load, or a chunk that ends
        # inside the replayed range of a hit (SWA bounded replay).
        if (
            num_new_tokens == 0
            and num_external_computed_tokens == 0
            and num_new_computed_tokens == 0
        ):
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "computed tokens to adopt"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )

        watermark_blocks = 0
        # The watermark is applied to waiting/preempted requests only, and only
        # when there's at least one request already scheduled.
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks

        if full_sequence_must_fit:
            # First check and fail if the full request sequence won't fit.
            full_num_tokens = min(request.num_tokens, self.max_model_len)

            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_local_computed_tokens=num_local_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks
            if required_blocks > self.block_pool.get_num_free_blocks():
                return None

        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        # Free on the processed-token basis: in-flight steps' attention windows
        # still read blocks below the optimistic boundary, and rejected spec
        # tokens can roll it back.
        self.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, total_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        # Keep `reserved_blocks` free for other in-flight sequences, and an
        # additional watermark of headroom for waiting/preempted admissions.
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)


KVCacheManager.allocate_slots = _allocate_slots


def _patch_cache_blocks_skips_non_cacheable() -> None:
    """Never publish a non-cacheable group's blocks to the prefix cache.

    The replayed sliding-window group opts out of prefix caching entirely, so
    its blocks must not be hashed, retained or handed to a connector. Upstream
    #56227 returns early on ``prefix_cacheable``; a wrapper cannot drift from
    the method's signature, and on a lane without the property the group stays
    cacheable (the behavior from before this feature).
    """
    original = SingleTypeKVCacheManager.cache_blocks

    @functools.wraps(original)
    def cache_blocks(self, request, num_tokens, *args, **kwargs):
        if not is_prefix_cacheable(self.kv_cache_spec):
            return
        return original(self, request, num_tokens, *args, **kwargs)

    SingleTypeKVCacheManager.cache_blocks = cache_blocks


if not vllm_version_is("0.28.0"):
    _patch_request_replay_start()
    _patch_new_request_data_replay_start()
    _patch_cached_request_data_replay_start()
    _patch_allocate_slots_guard()
    _patch_cache_blocks_skips_non_cacheable()
    logger.debug_once(
        "SWA bounded replay: scheduler plumbing patched "
        "(Request/NewRequestData/CachedRequestData.replay_start, "
        "allocate_slots guard, non-cacheable cache_blocks)."
    )
else:
    logger.debug_once(
        "SWA bounded replay: release lane detected, scheduler plumbing not patched."
    )
