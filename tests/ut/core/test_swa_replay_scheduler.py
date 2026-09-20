# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SWA bounded replay: scheduler-side tests.

Ported from ``tests/v1/core/test_prefix_replay.py`` (vLLM #56227) with
two differences: the KV cache groups are Ascend ones, and the connector
scenarios drive a deliberately minimal connector double that reports a fixed
matched-token count, because the repo's shared KV-connector harness is
Mooncake-specific and cannot be told what to match.

After a prefix hit the scheduler rewinds by the replayed window: it keeps the
hit's blocks, recomputes the hit's last window, and hands the worker
``replay_start``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import create_model_runner_output, create_vllm_config
from vllm_ascend.core.kv_cache_interface import (
    AscendSlidingWindowMLASpec,
    get_prefix_replay_tokens,
    is_prefix_cacheable,
    resolve_replay_window,
)
from vllm_ascend.core.swa_replay_scheduler import (
    AsyncSwaReplayScheduler,
    SwaReplayScheduler,
)

BLOCK_SIZE = 16
WINDOW = 32
NUM_PROMPT_TOKENS = 100
# 100 tokens -> 6 full blocks cached -> a 96-token hit.
HIT_TOKENS = NUM_PROMPT_TOKENS // BLOCK_SIZE * BLOCK_SIZE
FULL, SWA = 0, 1
SAMPLED_TOKEN_ID = 1000

_hash_initialized = False


def _swa_spec(*, bounded_replay: bool = True, sliding_window: int = WINDOW) -> AscendSlidingWindowMLASpec:
    return AscendSlidingWindowMLASpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=sliding_window,
        model_version="deepseek_v4",
        bounded_replay=bounded_replay,
    )


def _full_spec() -> FullAttentionSpec:
    return FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32)


def _kv_cache_config(*, uniform_swa_group: bool = False) -> KVCacheConfig:
    """One prefix-cacheable full-attention group and one replayed SWA group."""
    swa_spec = _swa_spec()
    if uniform_swa_group:
        # DeepSeek-V4 groups its specs behind a uniform wrapper, so the
        # replayed group's window has to resolve through it.
        swa_spec = UniformTypeKVCacheSpecs(kv_cache_specs={"swa": swa_spec})
    return KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full"], _full_spec()),
            KVCacheGroupSpec(["swa"], swa_spec),
        ],
    )


class _MockKVConnector:
    """A connector double reporting a fixed matched-token count.

    Only the methods whose *return shape* the scheduler unpacks are
    implemented; everything else the scheduler reaches for resolves to a no-op
    ``MagicMock``.
    """

    def __init__(self, matched_tokens: int, is_async: bool) -> None:
        self.matched_tokens = matched_tokens
        self.is_async = is_async
        self.requires_kv_delivery = False

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        return self.matched_tokens, self.is_async

    def request_finished(self, *args, **kwargs) -> tuple[bool, None]:
        return False, None

    def request_finished_all_groups(self, *args, **kwargs) -> tuple[bool, None]:
        return False, None

    def __getattr__(self, name: str) -> MagicMock:
        return MagicMock()


def _replay_scheduler(
    *,
    long_prefill_token_threshold: int = 0,
    connector: _MockKVConnector | None = None,
    uniform_swa_group: bool = False,
) -> Scheduler:
    vllm_config = create_vllm_config(max_num_seqs=16, max_num_batched_tokens=8192, block_size=BLOCK_SIZE)
    # No connector unless a test installs one, and it is installed after
    # construction so its matching behavior stays under the test's control.
    vllm_config.kv_transfer_config = None
    vllm_config.scheduler_config.long_prefill_token_threshold = long_prefill_token_threshold
    kv_cache_config = _kv_cache_config(uniform_swa_group=uniform_swa_group)
    vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

    scheduler = SwaReplayScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    if connector is not None:
        scheduler.connector = connector
    assert scheduler.prefix_replay_tokens == WINDOW
    return scheduler


def _create_requests(
    num_requests: int,
    num_tokens: int = NUM_PROMPT_TOKENS,
    *,
    same_prompt: bool = False,
    req_ids: list[str] | None = None,
) -> list[Request]:
    global _hash_initialized
    if not _hash_initialized:
        init_none_hash(sha256)
        _hash_initialized = True

    block_hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    requests = []
    for index in range(num_requests):
        req_id = req_ids[index] if req_ids else f"{index}"
        # A shared prompt is one every replaying request sees. The sampled
        # token on completion is non-zero so an output never extends the
        # shared prompt by accident and creates an accidental deeper hit.
        prompt_token_ids = [1] * num_tokens if same_prompt else [index * num_tokens + i for i in range(num_tokens)]
        requests.append(
            Request(
                request_id=req_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None,
                block_hasher=block_hasher,
            )
        )
    return requests


def _step_output(out, requests: list[Request]) -> ModelRunnerOutput:
    """Model output for every scheduled request: a token once its prefill ends.

    ``schedule()`` has already advanced ``num_computed_tokens`` past this
    step's chunk.
    """
    scheduled = [request for request in requests if request.request_id in out.num_scheduled_tokens]
    return ModelRunnerOutput(
        req_ids=[request.request_id for request in scheduled],
        req_id_to_index={request.request_id: index for index, request in enumerate(scheduled)},
        sampled_token_ids=[
            [SAMPLED_TOKEN_ID] if request.num_computed_tokens >= request.num_prompt_tokens else []
            for request in scheduled
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
    )


def _new_req_data(out, request: Request):
    return next(new_req for new_req in out.scheduled_new_reqs if new_req.req_id == request.request_id)


def _prefill(scheduler: Scheduler, request: Request) -> None:
    scheduler.add_request(request)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[request.request_id] == request.num_prompt_tokens
    scheduler.update_from_output(out, _step_output(out, [request]))


# --------------------------------------------------------------- scheduler ---


def test_replay_scheduler_forks_upstream_methods():
    """The fork must be a real override, not an inherited method."""
    assert SwaReplayScheduler.schedule is not Scheduler.schedule
    assert SwaReplayScheduler._update_waiting_for_remote_kv is not Scheduler._update_waiting_for_remote_kv
    assert issubclass(AsyncSwaReplayScheduler, SwaReplayScheduler)


def test_hit_replays_window_without_reallocating():
    scheduler = _replay_scheduler()
    first, second = _create_requests(2, same_prompt=True)
    _prefill(scheduler, first)
    manager = scheduler.kv_cache_manager

    scheduler.add_request(second)
    out = scheduler.schedule()

    replay_start = HIT_TOKENS - WINDOW
    new_req = _new_req_data(out, second)
    assert new_req.num_computed_tokens == replay_start
    assert new_req.replay_start == replay_start
    assert out.num_scheduled_tokens[second.request_id] == NUM_PROMPT_TOKENS - replay_start

    first_blocks = manager.get_blocks(first.request_id).blocks
    second_blocks = manager.get_blocks(second.request_id).blocks
    # The full-attention hit is shared, replayed tokens included; only the tail
    # block is new.
    num_hit_blocks = HIT_TOKENS // BLOCK_SIZE
    assert second_blocks[FULL][:num_hit_blocks] == first_blocks[FULL][:num_hit_blocks]
    assert all(block.ref_cnt == 2 for block in second_blocks[FULL][:num_hit_blocks])
    assert len(second_blocks[FULL]) == num_hit_blocks + 1
    # The window group never hits: blocks below the window are null, the window
    # (replayed and new tokens) gets fresh blocks.
    num_skipped_blocks = (HIT_TOKENS - WINDOW + 1) // BLOCK_SIZE
    assert all(block.is_null for block in second_blocks[SWA][:num_skipped_blocks])
    num_window_blocks = len(second_blocks[SWA]) - num_skipped_blocks
    assert num_window_blocks == -(-NUM_PROMPT_TOKENS // BLOCK_SIZE) - num_skipped_blocks
    assert not any(block.is_null for block in second_blocks[SWA][num_skipped_blocks:])

    scheduler.update_from_output(out, _step_output(out, [first, second]))
    assert second.num_computed_tokens == NUM_PROMPT_TOKENS


def test_replay_only_chunks_make_progress():
    """A chunk cap smaller than the replay window schedules replay-only chunks;
    they advance the request instead of stalling it (S8)."""
    scheduler = _replay_scheduler(long_prefill_token_threshold=BLOCK_SIZE)
    first, second = _create_requests(2, same_prompt=True)
    scheduler.add_request(first)
    while first.num_computed_tokens < NUM_PROMPT_TOKENS:
        out = scheduler.schedule()
        scheduler.update_from_output(out, _step_output(out, [first]))

    scheduler.add_request(second)

    def full_blocks():
        return scheduler.kv_cache_manager.get_blocks(second.request_id).blocks[FULL]

    scheduled: list[int] = []
    while second.num_computed_tokens < NUM_PROMPT_TOKENS:
        out = scheduler.schedule()
        if not scheduled:
            new_req = _new_req_data(out, second)
            assert new_req.num_computed_tokens == HIT_TOKENS - WINDOW
            assert new_req.replay_start == HIT_TOKENS - WINDOW
            # A replay-only chunk adopts the hit and allocates nothing new.
            assert len(full_blocks()) == HIT_TOKENS // BLOCK_SIZE
        scheduled.append(out.num_scheduled_tokens[second.request_id])
        scheduler.update_from_output(out, _step_output(out, [first, second]))

    # 32 replayed + 4 new tokens in 16-token chunks: two replay-only chunks.
    assert scheduled == [BLOCK_SIZE, BLOCK_SIZE, NUM_PROMPT_TOKENS - HIT_TOKENS]
    assert len(full_blocks()) == -(-NUM_PROMPT_TOKENS // BLOCK_SIZE)
    assert second.status == RequestStatus.RUNNING


def test_async_remote_kv_hit_replays_after_load():
    """A KV-connector hit loaded asynchronously carries no window state either;
    the replay is applied when the load completes (S5)."""
    matched = 64
    scheduler = _replay_scheduler(connector=_MockKVConnector(matched_tokens=matched, is_async=True))
    request = _create_requests(1)[0]
    scheduler.add_request(request)
    out = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.request_id not in out.num_scheduled_tokens

    scheduler.update_from_output(out, create_model_runner_output([], finished_recving={request.request_id}))

    out = scheduler.schedule()
    new_req = _new_req_data(out, request)
    assert new_req.num_computed_tokens == matched - WINDOW
    assert new_req.replay_start == matched - WINDOW
    assert out.num_scheduled_tokens[request.request_id] == NUM_PROMPT_TOKENS - matched + WINDOW


def test_remote_kv_hit_is_taken_in_whole_blocks():
    """A hit ending one token short of a block boundary would put the replay
    window's first token in a block the sliding-window group retires, so a
    connector hit is cut back to whole blocks (S4)."""
    matched = 4 * BLOCK_SIZE - 1
    scheduler = _replay_scheduler(connector=_MockKVConnector(matched_tokens=matched, is_async=False))
    request = _create_requests(1)[0]
    scheduler.add_request(request)
    out = scheduler.schedule()

    new_req = _new_req_data(out, request)
    hit = 3 * BLOCK_SIZE
    assert new_req.replay_start == hit - WINDOW
    assert new_req.num_computed_tokens == hit - WINDOW
    assert out.num_scheduled_tokens[request.request_id] == NUM_PROMPT_TOKENS - hit + WINDOW
    swa_manager = scheduler.kv_cache_manager.coordinator.single_type_managers[SWA]
    swa_blocks = scheduler.kv_cache_manager.get_blocks(request.request_id).blocks[SWA]
    assert swa_blocks[new_req.replay_start // BLOCK_SIZE] is not swa_manager._null_block


@pytest.mark.parametrize("uniform_swa_group", [False, True])
def test_hit_no_longer_than_window_is_ignored(uniform_swa_group):
    """Such a hit would be recomputed in full anyway. It is not adopted, so a
    request has computed tokens iff it replays (S3)."""
    scheduler = _replay_scheduler(uniform_swa_group=uniform_swa_group)
    _prefill(scheduler, _create_requests(1, WINDOW, same_prompt=True)[0])
    request = _create_requests(1, same_prompt=True, req_ids=["long"])[0]

    scheduler.add_request(request)
    out = scheduler.schedule()
    new_req = _new_req_data(out, request)
    assert request.status == RequestStatus.RUNNING
    assert new_req.num_computed_tokens == 0
    assert new_req.replay_start == 0
    assert out.num_scheduled_tokens[request.request_id] == NUM_PROMPT_TOKENS


def test_connector_hit_no_longer_than_window_is_ignored():
    """The same drop on the connector path: the async load is not even started."""
    scheduler = _replay_scheduler(connector=_MockKVConnector(matched_tokens=WINDOW, is_async=True))
    request = _create_requests(1)[0]

    scheduler.add_request(request)
    out = scheduler.schedule()
    assert request.status != RequestStatus.WAITING_FOR_REMOTE_KVS
    assert out.num_scheduled_tokens[request.request_id] == NUM_PROMPT_TOKENS


# ------------------------------------------------------------ replay helper --


def _bare_scheduler(prefix_replay_tokens: int) -> SwaReplayScheduler:
    scheduler = SwaReplayScheduler.__new__(SwaReplayScheduler)
    scheduler.prefix_replay_tokens = prefix_replay_tokens
    scheduler.connector = MagicMock()
    scheduler.kv_cache_manager = MagicMock()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    return scheduler


def test_mark_prefix_replay_rewinds_by_the_window():
    scheduler = _bare_scheduler(WINDOW)
    request = SimpleNamespace(replay_start=-1)

    assert scheduler._mark_prefix_replay(request, HIT_TOKENS) == WINDOW
    assert request.replay_start == HIT_TOKENS - WINDOW

    # A hit shorter than the window replays in full.
    assert scheduler._mark_prefix_replay(request, BLOCK_SIZE) == BLOCK_SIZE
    assert request.replay_start == 0

    # No hit at all: nothing replays, and a stale value is cleared.
    assert scheduler._mark_prefix_replay(request, 0) == 0
    assert request.replay_start == 0


def test_mark_prefix_replay_is_inert_without_a_window():
    scheduler = _bare_scheduler(0)
    request = SimpleNamespace()

    assert scheduler._mark_prefix_replay(request, HIT_TOKENS) == 0
    assert not hasattr(request, "replay_start")


def test_update_waiting_for_remote_kv_falls_back_to_recomputing_the_last_token():
    """Without a replay window the full-prompt-hit behavior is unchanged."""
    scheduler = _bare_scheduler(0)
    request = SimpleNamespace(
        request_id="req",
        num_computed_tokens=NUM_PROMPT_TOKENS,
        num_tokens=NUM_PROMPT_TOKENS,
        replay_start=0,
    )

    scheduler._update_waiting_for_remote_kv(request)

    assert request.num_computed_tokens == NUM_PROMPT_TOKENS - 1
    assert "req" in scheduler.finished_recving_kv_req_ids


def test_update_waiting_for_remote_kv_replays_the_hit_tail():
    """With a window the replay covers the last token, so the full-prompt-hit
    fallback is skipped and the computed count rewinds instead."""
    scheduler = _bare_scheduler(WINDOW)
    request = SimpleNamespace(
        request_id="req",
        num_computed_tokens=NUM_PROMPT_TOKENS,
        num_tokens=NUM_PROMPT_TOKENS,
        replay_start=0,
    )

    scheduler._update_waiting_for_remote_kv(request)

    assert request.replay_start == NUM_PROMPT_TOKENS - WINDOW
    assert request.num_computed_tokens == NUM_PROMPT_TOKENS - WINDOW


# ------------------------------------------------------- replay window spec --


def test_get_prefix_replay_tokens_resolves_the_uniform_wrapper():
    assert get_prefix_replay_tokens(_swa_spec()) == WINDOW
    assert get_prefix_replay_tokens(_swa_spec(bounded_replay=False)) == 0
    # A spec without the API at all (the release lane) reports no window.
    assert get_prefix_replay_tokens(MagicMock(spec=[])) == 0
    wrapped = UniformTypeKVCacheSpecs(kv_cache_specs={"swa": _swa_spec()})
    assert get_prefix_replay_tokens(wrapped) == WINDOW
    assert get_prefix_replay_tokens(wrapped) == max(
        get_prefix_replay_tokens(spec) for spec in wrapped.kv_cache_specs.values()
    )


def test_resolve_replay_window_agrees_across_groups():
    assert resolve_replay_window(_kv_cache_config()) == WINDOW
    assert resolve_replay_window(_kv_cache_config(uniform_swa_group=True)) == WINDOW
    # No group replays: not a replay run at all.
    assert (
        resolve_replay_window(
            KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=[KVCacheGroupSpec(["full"], _full_spec())],
            )
        )
        == 0
    )


def test_resolve_replay_window_rejects_disagreeing_windows():
    """One rewind has to match every group's allocation, so the windows must
    agree."""
    with pytest.raises(AssertionError, match="should agree"):
        resolve_replay_window(
            KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=[
                    KVCacheGroupSpec(["swa"], _swa_spec()),
                    KVCacheGroupSpec(["swa2"], _swa_spec(sliding_window=WINDOW * 2)),
                ],
            )
        )


def test_bounded_replay_group_is_not_prefix_cacheable():
    assert not is_prefix_cacheable(_swa_spec())
    assert is_prefix_cacheable(_swa_spec(bounded_replay=False))
    assert not is_prefix_cacheable(UniformTypeKVCacheSpecs(kv_cache_specs={"swa": _swa_spec()}))
