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

"""Tests for selecting (or downgrading) SWA bounded replay from platform.py.

The switch defaults to on, so every unsupported environment has to degrade to
"the sliding-window group stays prefix-cacheable" with a warning rather than
fail the run.
"""

from types import SimpleNamespace

import pytest

from vllm_ascend.platform import _apply_swa_bounded_replay, _get_swa_replay_scheduler_cls
from vllm_ascend.utils import vllm_version_is

SYNC_CLS = "vllm_ascend.core.swa_replay_scheduler.SwaReplayScheduler"
ASYNC_CLS = "vllm_ascend.core.swa_replay_scheduler.AsyncSwaReplayScheduler"

pytestmark = pytest.mark.skipif(
    vllm_version_is("0.28.0"),
    reason="the release lane has no scheduler hooks for replay, so the switch always degrades",
)

_SCHEDULER_MODE_FLAGS = {
    "recompute_scheduler_enable": False,
    "dyntra_lb_config": SimpleNamespace(enabled=False),
    "profiling_chunk_config": SimpleNamespace(enabled=False),
    "batch_job_sched_config": SimpleNamespace(enabled=False),
    "short_request_first_config": SimpleNamespace(enabled=False),
    "enable_balance_scheduling": False,
}


def _config(
    *,
    model_type: str = "deepseek_v4",
    enabled: bool = True,
    async_scheduling: bool = True,
    decode_context_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    kv_role: str | None = None,
    scheduler_cls: str = "vllm.v1.core.sched.scheduler.Scheduler",
    **mode_flags,
):
    flags = {**_SCHEDULER_MODE_FLAGS, **mode_flags}
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(swa_bounded_replay=enabled),
        additional_config={},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=decode_context_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
        ),
        kv_transfer_config=None if kv_role is None else SimpleNamespace(kv_role=kv_role),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling, scheduler_cls=scheduler_cls),
    )
    ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(**flags))
    return vllm_config, ascend_config


def _apply(**kwargs):
    vllm_config, ascend_config = _config(**kwargs)
    _apply_swa_bounded_replay(vllm_config, ascend_config)
    return vllm_config


def test_scheduler_cls_follows_async_scheduling():
    assert _get_swa_replay_scheduler_cls(async_scheduling=True) == ASYNC_CLS
    assert _get_swa_replay_scheduler_cls(async_scheduling=False) == SYNC_CLS


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_replay_scheduler_is_selected_for_deepseek_v4(async_scheduling):
    vllm_config = _apply(async_scheduling=async_scheduling)

    assert vllm_config.scheduler_config.scheduler_cls == (ASYNC_CLS if async_scheduling else SYNC_CLS)
    assert vllm_config.cache_config.swa_bounded_replay
    assert "swa_bounded_replay" not in vllm_config.additional_config


def test_switch_off_leaves_the_scheduler_alone():
    vllm_config = _apply(enabled=False)

    assert vllm_config.scheduler_config.scheduler_cls == "vllm.v1.core.sched.scheduler.Scheduler"


def test_other_models_keep_the_switch_and_the_default_scheduler():
    """The switch only reaches the DeepSeek-V4 SWA group, so this is a no-op
    rather than a misconfiguration."""
    vllm_config = _apply(model_type="opt")

    assert vllm_config.scheduler_config.scheduler_cls == "vllm.v1.core.sched.scheduler.Scheduler"
    assert vllm_config.cache_config.swa_bounded_replay
    assert "swa_bounded_replay" not in vllm_config.additional_config


@pytest.mark.parametrize("dcp,pcp", [(2, 1), (1, 2)])
def test_context_parallel_degrades_replay(dcp, pcp):
    vllm_config = _apply(decode_context_parallel_size=dcp, prefill_context_parallel_size=pcp)

    assert not vllm_config.cache_config.swa_bounded_replay
    assert vllm_config.additional_config["swa_bounded_replay"] is False
    assert vllm_config.scheduler_config.scheduler_cls == "vllm.v1.core.sched.scheduler.Scheduler"


@pytest.mark.parametrize(
    "mode_kwargs",
    [
        {"recompute_scheduler_enable": True, "kv_role": "kv_consumer"},
        {"dyntra_lb_config": SimpleNamespace(enabled=True)},
        {"profiling_chunk_config": SimpleNamespace(enabled=True)},
        {"batch_job_sched_config": SimpleNamespace(enabled=True)},
        {"short_request_first_config": SimpleNamespace(enabled=True)},
        {"enable_balance_scheduling": True},
    ],
    ids=[
        "recompute_scheduler_enable",
        "dyntra_lb_config",
        "profiling_chunk_config",
        "batch_job_sched_config",
        "short_request_first_config",
        "enable_balance_scheduling",
    ],
)
def test_conflicting_scheduler_modes_degrade_replay(mode_kwargs):
    """Each of these owns the scheduler itself, so replay cannot share it."""
    vllm_config = _apply(**mode_kwargs)

    assert not vllm_config.cache_config.swa_bounded_replay
    assert vllm_config.additional_config["swa_bounded_replay"] is False
    assert vllm_config.scheduler_config.scheduler_cls == "vllm.v1.core.sched.scheduler.Scheduler"


def test_recompute_outside_a_decode_node_is_not_a_conflict():
    """recompute only claims the scheduler on a PD-disaggregated decode node."""
    vllm_config = _apply(recompute_scheduler_enable=True, kv_role="kv_both")

    assert vllm_config.cache_config.swa_bounded_replay
    assert vllm_config.scheduler_config.scheduler_cls == ASYNC_CLS
