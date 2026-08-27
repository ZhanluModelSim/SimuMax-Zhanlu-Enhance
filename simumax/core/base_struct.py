"""Basic data structures for simumax"""

from copy import deepcopy
from dataclasses import dataclass, field
from abc import ABC
from typing import ClassVar, List, Tuple, Dict
from collections import defaultdict, deque
import heapq
import itertools
import math
import re
import time
import types
import multiprocessing
try:
    from mpi4py import MPI
    enable_mpi = True
except ImportError:
    enable_mpi = False
import json
import os
from simumax.core.tensor import TensorSize
from simumax.core.config import StrategyConfig, SystemConfig, get_capture_graph_only, SIMU_DEBUG, TMP_PATH
from simumax.core.model_struct import (
    ActivationInfo,
    InputOutputInfo,
    ModuleComputeInfo,
    ModuleCostInfo,
    ModuleMemoryInfo,
    PathDebugContext,
    RecomputeStatus,
)
from simumax.core.simu_events import EventSink
from simumax.core.fusion import FusionPolicy
from simumax.core.simu_memory import OpMemoryProfile
from simumax.core.utils import get_point_name, to_json_string
from simumax.core.utils import get_rank_group
from simumax.core.utils import group_node_stats, estimate_straggler_increase_ratio
from simumax.core.utils import group_level_span
from simumax.core.graph import SimuONNXGraphBuilder


_COLLECTIVE_LIFECYCLE_COMPONENTS = (
    "post",
    "task_dispatch",
    "data_transfer",
    "completion",
    "wait",
    "consumer_release",
)


def _collective_lifecycle_facts(
    op_id,
    group_size=None,
    payload_bytes=0,
    comm_role=None,
    comm_stage=None,
    ctx=None,
):
    """Build model-side collective lifecycle facts.

    This is deliberately a structural description.  It does not read a
    profiler and it never turns a measured duration into a cost.  The
    optional runtime profile contributes only declared launch/task policies;
    completion and wait values remain ``None`` when the target runtime does
    not expose a portable fact for them.
    """

    token = "_".join(
        str(value or "").lower()
        for value in (op_id, comm_role, comm_stage)
    ).replace("-", "_")
    # ``all2all`` is the fixed-count semantic exchange used by the router
    # metadata call.  ``alltoallv`` is the variable-count exchange used by
    # dispatch/combine and must remain a separate family in the lifecycle
    # ledger.  The old substring rule classified both as alltoallv, which
    # corrupted family counts even though the cost formula itself was still
    # selected through the compatibility fallback.
    if "alltoallv" in token or "all2allv" in token or "a2av" in token:
        collective = "alltoallv"
        algorithm = "pairwise_exchange"
        stages = max(0, int(group_size or 1) - 1)
    elif "alltoall" in token or "all2all" in token:
        collective = "alltoall"
        algorithm = "pairwise_exchange"
        stages = max(0, int(group_size or 1) - 1)
    elif "allreduce" in token or "all_reduce" in token:
        collective = "all_reduce"
        algorithm = "ring"
        stages = 2 * max(0, int(group_size or 1) - 1)
    elif (
        "reducescatter" in token
        or "reduce_scatter" in token
        or re.search(r"(?:^|_)rs(?:_|$)", token)
    ):
        collective = "reduce_scatter"
        algorithm = "ring"
        stages = max(0, int(group_size or 1) - 1)
    elif (
        "allgather" in token
        or "all_gather" in token
        or re.search(r"(?:^|_)ag(?:_|$)", token)
    ):
        collective = "all_gather"
        algorithm = "ring"
        stages = max(0, int(group_size or 1) - 1)
    else:
        collective = "communication"
        algorithm = "unknown"
        stages = None

    payload = max(0, int(payload_bytes or 0))
    system = getattr(ctx, "system", None) if ctx is not None else None
    runtime_cfg = {}
    if system is not None:
        getter = getattr(system, "_hccl_network_config", None)
        if callable(getter):
            network_cfg = dict(getter() or {})
            runtime_cfg = dict(network_cfg.get("call_runtime", {}) or {})

    descriptor_bytes = runtime_cfg.get("descriptor_chunk_bytes")
    if descriptor_bytes:
        descriptor_bytes = max(1, int(descriptor_bytes))
        chunk_count = max(1, math.ceil(payload / descriptor_bytes)) if payload else 1
        chunk_source = "system_config.hccl_runtime.call_runtime.descriptor_chunk_bytes"
    else:
        # No physical HCCL bucket count is invented when the target profile
        # does not declare one.  One logical call is still a valid internal
        # representation, while the provenance tells consumers that physical
        # decomposition is unresolved.
        chunk_count = 1
        chunk_source = "portable_logical_call_default"

    payload_per_chunk = math.ceil(payload / chunk_count) if payload else 0
    completion = dict(runtime_cfg.get("completion", {}) or {})
    launch = runtime_cfg.get("call_launch_latency_us")
    task = runtime_cfg.get("task_launch_latency_us", launch)
    tasks_per_stage = runtime_cfg.get("tasks_per_stage")
    descriptor_tasks = max(0, chunk_count - 1) * int(
        runtime_cfg.get("tasks_per_additional_chunk", 0) or 0
    )
    task_count = None
    if stages is not None and tasks_per_stage is not None:
        task_count = (
            stages * max(1, len(getattr(ctx, "levels", []) or [1]))
            * int(tasks_per_stage)
            + descriptor_tasks
        )
    return {
        "schema": "collective_call_v1",
        "collective": collective,
        "algorithm": algorithm,
        "algorithm_stages": stages,
        "group_size": int(group_size) if group_size is not None else None,
        "payload_bytes": payload,
        "chunk_count": chunk_count,
        "payload_per_chunk_bytes": payload_per_chunk,
        "chunk_count_source": chunk_source,
        "lifecycle_components": list(_COLLECTIVE_LIFECYCLE_COMPONENTS),
        "runtime": {
            "call_launch_latency_us": launch,
            "task_dispatch_latency_us": task,
            "tasks_per_stage": tasks_per_stage,
            "runtime_task_count": task_count,
            "completion_latency_us": completion.get("completion_latency_us"),
            "wait_latency_us": completion.get("wait_latency_us"),
            "barrier_latency_us": completion.get("barrier_latency_us"),
            "unknown_fields": [
                name for name in (
                    "completion_latency_us",
                    "wait_latency_us",
                    "barrier_latency_us",
                )
                if not isinstance(completion.get(name), (int, float))
            ],
        },
        "provenance": {
            "source": (
                "model_strategy_system_config"
                if system is not None
                else "model_structure_portable_default"
            ),
            "measured_duration_used": False,
            "physical_kernel_identity_used": False,
        },
    }


def _lifecycle_with_times(
    metadata,
    *,
    stage=None,
    phase=None,
    post_time=None,
    completion_time=None,
    consumer_release_time=None,
    entry=None,
):
    """Copy lifecycle metadata and attach simulator-clock timestamps."""

    result = dict(metadata or {})
    lifecycle = dict(result.get("lifecycle") or {})
    if phase is not None:
        lifecycle["phase"] = phase
    if stage is not None:
        lifecycle["event_stage"] = stage
    if entry is not None:
        post_time = post_time if post_time is not None else entry.issue_t
        completion_time = (
            completion_time if completion_time is not None else entry.completion_t
        )
        consumer_release_time = (
            consumer_release_time
            if consumer_release_time is not None
            else entry.consumer_release_t
        )
    if post_time is not None:
        lifecycle["post_time_ms"] = post_time
    if completion_time is not None:
        lifecycle["completion_time_ms"] = completion_time
    if consumer_release_time is not None:
        lifecycle["consumer_release_time_ms"] = consumer_release_time
    result["lifecycle"] = lifecycle
    return result

class FwdQue:
    def __init__(
        self,
        call_stk='',
        que=None,
        mem_profile: OpMemoryProfile = None,
        phase: str = "fwd",
        batch_blocking_comm: bool = False,
    ):
        self.que = que if que else []
        self.call_stk = call_stk
        self.st = None
        self.mem_profile = mem_profile
        self.phase = phase
        self._mem_started = False
        self._mem_finished = False
        self.batch_blocking_comm = batch_blocking_comm
        # Phase-line classification latched from a single contained op on the
        # first step; None/None keeps the exporter's structural fallback.
        self._span_kind = None
        self._span_lane = None

    def step(self, t, ctx):
        # t is the per-rank lane-clock dict (lanes from the resource registry,
        # see SimuThread); FwdQue advances the "comp" lane.
        if self.st is None:
            self.st = t["comp"]
            # A queue wrapping exactly one op inherits that op's span
            # classification (e.g. a lone comm op); multi-op queues keep
            # None/None so the exporter applies its structural fallback.
            # batch_blocking_comm queues (batch_pp_*) always keep None/None:
            # the exporter's batch_pp name rule classifies them as scope.
            if not self.batch_blocking_comm and len(self.que) == 1:
                self._span_kind = getattr(self.que[0], "simu_kind", None)
                self._span_lane = getattr(self.que[0], "simu_lane", None)
        if (
            self.mem_profile is not None
            and not self._mem_started
            and getattr(ctx, "memory_tracker", None) is not None
        ):
            ctx.memory_tracker.phase_start(
                rank=ctx.current_rank,
                ts=self.st,
                profile=self.mem_profile,
                phase=self.phase,
            )
            self._mem_started = True

        ok, blk = self._step(t, ctx)
        if ok:
            if (
                self.mem_profile is not None
                and not self._mem_finished
                and getattr(ctx, "memory_tracker", None) is not None
            ):
                ctx.memory_tracker.phase_end(
                    rank=ctx.current_rank,
                    ts=t["comp"],
                    profile=self.mem_profile,
                    phase=self.phase,
                )
                self._mem_finished = True
            ctx.event_sink.emit_span(self.call_stk, self.phase, self.st, t['comp'],
                                     kind=self._span_kind, lane=self._span_lane)
            return True, None
        return False, blk

    def _step(self, t, ctx):
        if self.batch_blocking_comm:
            batch_submit_t = max(t["comp"], t["comm"])
            blocked_key = None
            remaining = []
            for op in self.que:
                if hasattr(op, "_prime_batch_submit"):
                    op._prime_batch_submit(self.phase, batch_submit_t)
                ok, blk = op.step(t, ctx)
                if ok:
                    continue
                if isinstance(blk, tuple) and blk:
                    if blk[0] == "yield_done":
                        continue
                    if blk[0] in ("yield_done", "yield_keep"):
                        self.que = [op] + list(self.que[len(remaining) + 1 :])
                        return False, blk
                remaining.append(op)
                if blocked_key is None:
                    blocked_key = blk
            self.que = remaining
            if self.que:
                return False, blocked_key
            t["comp"] += 2e-3  # tracing
            return True, None

        while self.que:
            ok, blk = self.que[0].step(t, ctx)   # LeafModel.step now returns (ok, blk)
            if not ok:
                if isinstance(blk, tuple) and blk:
                    if blk[0] == "yield_done":
                        self.que.pop(0)
                    if blk[0] in ("yield_done", "yield_keep"):
                        return False, blk
                return False, blk
            self.que.pop(0)

        t["comp"] += 2e-3  # tracing
        return True, None

    def append(self, x):
        self.que.append(x)

    def __bool__(self):
        return bool(self.que)


class BwdStk:
    def __init__(self, call_stk='', stk=None, mem_profile: OpMemoryProfile = None):
        self.stk = stk if stk else []
        self.call_stk = call_stk
        self.st_bwd = None
        self.mem_profile = mem_profile
        self._mem_started = False
        self._mem_finished = False
        # Phase-line classification latched from a single contained op on the
        # first bwd; None/None keeps the exporter's structural fallback.
        self._span_kind = None
        self._span_lane = None

    def bwd(self, t, ctx):
        if self.st_bwd is None:
            self.st_bwd = t["comp"]
            # A stack wrapping exactly one op inherits that op's span
            # classification (e.g. a lone comm op); multi-op stacks keep
            # None/None so the exporter applies its structural fallback.
            if len(self.stk) == 1:
                self._span_kind = getattr(self.stk[0], "simu_kind", None)
                self._span_lane = getattr(self.stk[0], "simu_lane", None)
        if (
            self.mem_profile is not None
            and not self._mem_started
            and getattr(ctx, "memory_tracker", None) is not None
        ):
            ctx.memory_tracker.phase_start(
                rank=ctx.current_rank,
                ts=self.st_bwd,
                profile=self.mem_profile,
                phase="bwd",
            )
            self._mem_started = True

        ok, blk = self._bwd(t, ctx)
        if ok:
            if (
                self.mem_profile is not None
                and not self._mem_finished
                and getattr(ctx, "memory_tracker", None) is not None
            ):
                ctx.memory_tracker.phase_end(
                    rank=ctx.current_rank,
                    ts=t["comp"],
                    profile=self.mem_profile,
                    phase="bwd",
                )
                self._mem_finished = True
            ctx.event_sink.emit_span(self.call_stk, "bwd", self.st_bwd, t['comp'],
                                     kind=self._span_kind, lane=self._span_lane)
            return True, None
        return False, blk

    def _bwd(self, t, ctx):
        while self.stk:
            ok, blk = self.stk[-1].bwd(t, ctx)
            if not ok:
                if isinstance(blk, tuple) and blk:
                    if blk[0] == "yield_done":
                        self.stk.pop(-1)
                    if blk[0] in ("yield_done", "yield_keep"):
                        return False, blk
                return False, blk
            self.stk.pop(-1)

        t["comp"] += 2e-3  # tracing
        return True, None

    def append(self, x):
        self.stk.append(x)

    def __bool__(self):
        return bool(self.stk)


class RecomputeBlockJob:
    """Replay a checkpointed forward block before running its backward."""

    def __init__(self, call_stk='', fwd_jobs=None, bwd_jobs=None):
        self.call_stk = call_stk
        self._has_recompute = bool(fwd_jobs)
        self.recompute_fwd = FwdQue(
            call_stk=f"{call_stk}-recompute_block",
            que=fwd_jobs if fwd_jobs else [],
            phase="recompute_fwd",
        )
        self.bwd_stk = BwdStk(
            call_stk=f"{call_stk}-checkpoint_bwd",
            stk=bwd_jobs if bwd_jobs else [],
        )
        self._recompute_done = False

    def bwd(self, t, ctx):
        if self._has_recompute and not self._recompute_done:
            ok, blk = self.recompute_fwd.step(t, ctx)
            if not ok:
                return False, blk
            self._recompute_done = True
        elif not self._has_recompute:
            self._recompute_done = True
        return self.bwd_stk.bwd(t, ctx)

    

class BaseModel: #templete for non-leaf model
    def __init__(self, specific_name=''):
        # self.call_stk = call_stk+f'-{self.__class__.__name__}'
        self.call_stk = f'-{self.__class__.__name__}'
        self.specific_name = specific_name
        if specific_name:
            self.call_stk =f'-{specific_name}'
        self.layers = [] #layer require prefill_fwd/bwd, could be (non-)leaf model

    def prefill(self, args, call_stk='', com_buff=None):
        #
        pass

    def prefill_fwd(self):
        # return a fwd job:  FwdQue or LeafModel
        fwd = FwdQue(call_stk=self.call_stk)
        for layer in self.layers:
            fwd.append(layer.prefill_fwd())
        return fwd
    
    def prefill_bwd(self):
        # return a bwd job:  BwdStk or LeafModel
        bwd = BwdStk(call_stk=self.call_stk)
        for layer in self.layers:
            bwd.append(layer.prefill_bwd())
        return bwd

class PostInitMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        if hasattr(obj, '__post_init__'):
            obj.__post_init__()
        return obj

class MetaModule(BaseModel, metaclass = PostInitMeta):
    """
    Assume that there are two types of modules:
    1. The most basic module that does not have children modules
    2. A module composed of children modules, except for children modules,
       there are no other calculations
    """

    dtype_to_element_size = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}
    id_counter = 0
    def __init__(self, strategy:StrategyConfig, system:SystemConfig, specific_name='', parent_module = None) -> None:
        super().__init__(specific_name)
        self.strategy = strategy
        self.system = system
        self.offload_inputs = False

        self.children_ordered_module:List[MetaModule] = []
        self.children_modules:List[MetaModule] = []  # Unordered list of all child modules.
        self.children_modules_names:Dict[MetaModule, str] = {}
        self.default_dtype = strategy.dtype 
        self._init_strategy = False
        self.input_info = None
        self.output_info_ = None
        # self.cache_info = []
        self.enable_recompute = False
        self.recompute_granularity = "full"
        self.enable_block_recompute_schedule = False
        self.parent_module:MetaModule = parent_module
        self._reset_infos()
        self.is_leaf_module = False
        self.cache_inputs = False
        self.cache_outputs = False
        self.recompute_status:str = RecomputeStatus.NO_RECOMPUTE # "first", "middle", "last", default = middle
        self.is_breakpoints = False
        self.ordered_module_hooks:List[callable] = None
        self.forward_pre_hooks:List[callable] = None
        self.forward_post_hooks:List[callable] = None
        self.init_ready = False
        self.is_recompute_forward_finished = False
        self.full_name = "self"
        self.name = ''
        self.call_idx = -1
        # Instance-level override hook for cost-model efficiency lookup:
        # a module class or user may set this to a string to replace the
        # default class-name-based key (see get_cost_keys).
        self.cost_op_key = None

        # for Selective recompute strategy
        self.all_recompute_nodes:List[MetaModule] = []
        self.all_leaf_nodes:List[MetaModule] = []
        self.status_ready = False
        self.is_variance_node = False
        self.use_variance_tail_model = bool(strategy.recompute_variance)
        self.id = MetaModule.id_counter
        MetaModule.id_counter += 1

    def __post_init__(self):
        self.is_leaf_module = self.set_children_modules()
        self.cache_inputs = not self.enable_recompute 
        self.init_ready = True

    def set_children_modules(self):
        is_leaf = True
        for name, member in vars(self).items():
            if isinstance(member, MetaModule):
                is_leaf = False
                if member.parent_module is None:
                    member.parent_module = self
                    self.children_modules.append(member) 
                    self.children_modules_names[member] = name
        return is_leaf
    
    def set_variance_node(self, is_variance_node:bool):
        if self.use_variance_tail_model:
            self.is_variance_node = is_variance_node

    @property
    def _fsdp_net_resolved(self):
        """Return fsdp_net if explicitly set, else the resolved dp_net
        (design doc design_simu_system_net_ext.md Part A, section 3.2).
        After analysis_net() runs, fsdp_net is already a concrete value;
        this property is a safety fallback for direct DES access."""
        fsdp_net = getattr(self.strategy, 'fsdp_net', 'auto')
        if fsdp_net and fsdp_net != 'auto':
            return fsdp_net
        return self.strategy.dp_net

    @property
    def _fsdp_moe_net_resolved(self):
        """Return fsdp_moe_net if explicitly set, else the resolved edp_net."""
        fsdp_moe_net = getattr(self.strategy, 'fsdp_moe_net', 'auto')
        if fsdp_moe_net and fsdp_moe_net != 'auto':
            return fsdp_moe_net
        return self.strategy.edp_net

    @property
    def output_info(self):
        if self.output_info_ is None:
            self.output_info_ = self.create_output_info()
        return self.output_info_
    
    def set_leaf_full_name(self, parent_name:str):
        for child, name in self.children_modules_names.items():
            child.full_name = parent_name + '.' + name
            child.name = name
            child.set_leaf_full_name(child.full_name)

    def get_cost_keys(self):
        """(class_key, path_key) for cost-model efficiency lookup."""
        class_key = self.cost_op_key or type(self).__name__
        full = getattr(self, "full_name", "") or ""
        path_key = full[5:] if full.startswith("self.") else (full or None)
        return class_key, path_key
    
    def _reset_infos(self):
        self._act_info = ActivationInfo()
        self._act_info_with_recomp = ActivationInfo()
        self._model_info = ModuleMemoryInfo()
        self._compute_info = ModuleComputeInfo()
        self._cost_info = ModuleCostInfo()
        self.path_debug_context = None
        self.parent = None
        self.current = None
        self._info_ready = False
        self.is_recompute_forward_finished = False
        self.children_ordered_module:List[MetaModule] = [] 
        self.children_modules:List[MetaModule] = [] 
        self.all_recompute_nodes:List[MetaModule] = []
        self.all_leaf_nodes:List[MetaModule] = []

    def get_root_module(self):
        module = self
        while getattr(module, "parent_module", None) is not None:
            module = module.parent_module
        return module

    def is_last_leaf_in_root(self):
        root = self.get_root_module()
        leaf_nodes = getattr(root, "all_leaf_nodes", None)
        return bool(leaf_nodes) and leaf_nodes[-1] is self

    def build_simu_mem_profile(self, phase: str = "fwd"):
        if not self.is_leaf_module or not self._info_ready:
            return None

        act_info = self.get_act_info()
        cache_size_bytes = 0
        cache_alloc_phase = None
        if self.strategy.enable_recompute and self.enable_recompute:
            recompute_peak_mem_no_cache = act_info.fwd_peak_mem_no_cache
            if self.recompute_status == RecomputeStatus.FIRST:
                if not self.offload_inputs:
                    cache_size_bytes = self.all_input_element_num()
                    cache_alloc_phase = "fwd"
            else:
                cache_size_bytes = act_info.total_activation_mem_cache
                cache_alloc_phase = "recompute_fwd"
        else:
            cache_size_bytes = act_info.total_activation_mem_cache
            cache_alloc_phase = "fwd"
            recompute_peak_mem_no_cache = 0

        if self.use_variance_tail_model and self.is_variance_node:
            if cache_alloc_phase == "recompute_fwd":
                cache_size_bytes = 0
                cache_alloc_phase = None

        bwd_peak_mem_no_cache = act_info.bwd_peak_mem_no_cache

        return OpMemoryProfile(
            op_name=self.full_name or self.call_stk,
            fwd_peak_mem_no_cache=int(act_info.fwd_peak_mem_no_cache),
            bwd_peak_mem_no_cache=int(bwd_peak_mem_no_cache),
            recompute_peak_mem_no_cache=int(recompute_peak_mem_no_cache),
            cache_size_bytes=int(cache_size_bytes),
            cache_alloc_phase=cache_alloc_phase,
            cache_token_scope=self.call_stk,
        )

    def prefill_fwd(self):
        fwd = FwdQue(
            call_stk=self.call_stk,
            mem_profile=self.build_simu_mem_profile(phase="fwd") if self.is_leaf_module else None,
        )
        for layer in self.layers:
            if isinstance(layer, LeafModel):
                layer.set_event_metadata(self._event_metadata_for_leaf(layer))
            fwd.append(layer.prefill_fwd())
        return fwd

    def prefill_recompute_fwd(self, recompute_cost_override=None):
        fwd = FwdQue(
            call_stk=self.call_stk,
            mem_profile=self.build_simu_mem_profile(phase="recompute_fwd") if self.is_leaf_module else None,
            phase="recompute_fwd",
        )
        recompute_cost = self._cost_info.recompute_compute_time if self.is_leaf_module else recompute_cost_override
        rc_ratio = getattr(self.strategy, 'recompute_cost_ratio', 1.0)
        if recompute_cost is not None:
            recompute_cost = recompute_cost * rc_ratio
        for layer in self.layers:
            fwd.append(layer.prefill_recompute_fwd(recompute_cost))
        return fwd

    def _use_block_recompute_schedule(self):
        if self.is_leaf_module or not self.enable_block_recompute_schedule:
            return False
        nodes = self.get_all_leaf_modules() if self.status_ready else self.layers
        return any(getattr(node, "enable_recompute", False) for node in nodes)

    def _append_checkpoint_segment(self, bwd, segment):
        if not segment:
            return
        recompute_jobs = [
            layer.prefill_recompute_fwd()
            for layer in segment
            if not (
                getattr(layer, "use_variance_tail_model", False)
                and getattr(layer, "is_variance_node", False)
            )
        ]
        bwd_jobs = [layer.prefill_bwd() for layer in segment]
        bwd.append(
            RecomputeBlockJob(
                call_stk=self.call_stk,
                fwd_jobs=recompute_jobs,
                bwd_jobs=bwd_jobs,
            )
        )

    def prefill_bwd(self):
        if self._use_block_recompute_schedule():
            bwd = BwdStk(call_stk=self.call_stk)
            nodes = self.get_all_leaf_modules() if self.status_ready else self.layers
            checkpoint_segment = []
            for node in nodes:
                if getattr(node, "enable_recompute", False):
                    if (
                        checkpoint_segment
                        and getattr(node, "recompute_status", RecomputeStatus.MIDDLE) == RecomputeStatus.FIRST
                    ):
                        self._append_checkpoint_segment(bwd, checkpoint_segment)
                        checkpoint_segment = []
                    checkpoint_segment.append(node)
                    if getattr(node, "recompute_status", RecomputeStatus.MIDDLE) == RecomputeStatus.LAST:
                        self._append_checkpoint_segment(bwd, checkpoint_segment)
                        checkpoint_segment = []
                    continue

                self._append_checkpoint_segment(bwd, checkpoint_segment)
                checkpoint_segment = []
                bwd.append(node.prefill_bwd())

            self._append_checkpoint_segment(bwd, checkpoint_segment)
            return bwd

        bwd = BwdStk(
            call_stk=self.call_stk,
            mem_profile=self.build_simu_mem_profile(phase="bwd") if self.is_leaf_module else None,
        )
        for layer in self.layers:
            if isinstance(layer, LeafModel):
                layer.set_event_metadata(self._event_metadata_for_leaf(layer))
            bwd.append(layer.prefill_bwd())
        return bwd

    @staticmethod
    def _tensor_shape_metadata(value):
        """Return JSON-safe shapes/dtypes from a TensorSize or IO container."""
        if value is None:
            return [], []
        tensors = getattr(value, "tensors", None)
        if tensors is None:
            tensors = [value]
        shapes = [list(getattr(tensor, "shape", [])) for tensor in tensors]
        dtypes = [getattr(tensor, "dtype", None) for tensor in tensors]
        return shapes, dtypes

    def _event_metadata_for_leaf(self, leaf):
        """Build structural event metadata from this module's model tensors.

        This is deliberately attached during DES prefill, after the model
        graph has propagated ``input_info``/``output_info``.  It therefore
        remains portable when batch, sequence length, expert count, or
        parallelism changes, and it does not consult a profiler trace.
        """
        input_shapes, input_dtypes = self._tensor_shape_metadata(self.input_info)
        output_shapes, output_dtypes = self._tensor_shape_metadata(self.output_info)
        shape_desc_by_stage = {}
        for stage in ("fwd", "bwd_grad_act", "bwd_grad_w"):
            try:
                desc = self.get_input_shapes_desc(stage)
            except (AttributeError, AssertionError, IndexError, TypeError, ValueError):
                desc = ""
            if desc:
                shape_desc_by_stage[stage] = desc
        if not shape_desc_by_stage:
            shape_desc_by_stage["fwd"] = (
                f"inputs={input_shapes}, outputs={output_shapes}"
            )
        # The output dtype is useful for fused casts (e.g. VWN out); retain
        # both sides so a consumer never has to guess from the operator name.
        dtype = next((d for d in output_dtypes if d), None)
        if dtype is None:
            dtype = next((d for d in input_dtypes if d), None)
        metadata = {
            "shape_desc_by_stage": shape_desc_by_stage,
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "input_dtypes": input_dtypes,
            "output_dtypes": output_dtypes,
            "dtype": dtype,
        }
        # Layout-bearing modules may expose a portable physical-work contract
        # in addition to the ordinary logical tensor shapes. Keep unknown
        # physical facts explicit instead of guessing a CANN kernel format.
        # The contract is metadata only; it never changes DES cost.
        # A sequence module supplies the common contract while each AtomModel
        # may refine the physical stage (bytes, owner, fusion boundary).  Merge
        # the leaf refinement after the parent so stage-local facts are not
        # overwritten when the parent attaches metadata during prefill.
        layout_contract = dict(getattr(self, "layout_contract", {}) or {})
        child_metadata = dict(getattr(leaf, "event_metadata", {}) or {})
        layout_contract.update(
            dict(child_metadata.get("layout_contract") or {}))
        if layout_contract:
            physical_work_id = layout_contract.get("physical_work_id")
            if (physical_work_id in (None, "auto")
                    or str(physical_work_id).startswith("auto/")):
                owner = getattr(self, "full_name", None) or self.call_stk
                stage_name = (
                    layout_contract.get("physical_stage_role")
                    or getattr(leaf, "specific_name", None)
                    or getattr(self, "specific_name", None)
                    or "layout")
                if str(physical_work_id).startswith("auto/"):
                    stage_name = str(physical_work_id).split("/", 1)[1]
                layout_contract["physical_work_id"] = (
                    f"{owner}/{stage_name}")
            layout_contract.setdefault("logical_input_shape", input_shapes)
            layout_contract.setdefault("logical_output_shape", output_shapes)
            layout_contract.setdefault("dtype", dtype)
            layout_contract.setdefault("source_format", None)
            layout_contract.setdefault("target_format", None)
            layout_contract.setdefault("transpose_dims", None)
            layout_contract.setdefault("padding", None)
            layout_contract.setdefault("contiguous", None)
            # Zero is a physical claim.  Keep temporary traffic unknown unless
            # the model explicitly declares that no temporary tensor exists.
            layout_contract.setdefault("temporary_bytes", None)
            layout_contract.setdefault("physical_shape", None)
            layout_contract.setdefault("physical_shape_status", "unknown")
            metadata["layout_contract"] = layout_contract
            for field_name in (
                    "fusion_scope", "physical_work_id",
                    "memory_transaction_owner", "physical_stage_role"):
                value = child_metadata.get(field_name)
                if value is None:
                    value = layout_contract.get(field_name)
                if value is not None:
                    metadata[field_name] = value
        return metadata
        
    def get_all_leaf_modules(self):
        assert self.status_ready, f"{self.__class__.__name__} is not ready yet, please run set_first_last_recompute_status() first"
        return self.all_leaf_nodes

    def set_first_last_recompute_status(self):
        self.pre_enable_recompute = False
        self.p_recom_m: MetaModule = None
        self.all_recompute_nodes = []
        self.all_leaf_nodes = []

        def dfs(module: MetaModule):
            ordered = module.children_ordered_module or module.children_modules
            if module.is_leaf_module or len(ordered) == 0:
                module.call_idx = len(self.all_leaf_nodes)
                self.all_leaf_nodes.append(module)

                if module.enable_recompute:
                    module.recompute_status = RecomputeStatus.MIDDLE
                    self.all_recompute_nodes.append(module)

                if not self.pre_enable_recompute and module.enable_recompute:
                    module.recompute_status = RecomputeStatus.FIRST
                if self.pre_enable_recompute and not module.enable_recompute and self.p_recom_m is not None:
                    self.p_recom_m.recompute_status = RecomputeStatus.LAST
                if module.enable_recompute:
                    self.p_recom_m = module
                self.pre_enable_recompute = module.enable_recompute
                return

            for child in ordered:
                dfs(child)

        dfs(self)
        if self.pre_enable_recompute and self.p_recom_m is not None:
            self.p_recom_m.recompute_status = RecomputeStatus.LAST
    
    def get_weight(self) -> TensorSize:
        return None
    
    def register_add_ordered_module_hooks(self, hook):
        assert self.init_ready, f"Module {self.__class__.__name__} must be initialized before registering hooks"
        self.add_ordered_module_hooks(hook)
        for module in self.children_modules:
            module.register_add_ordered_module_hooks(hook)

    def register_add_forward_pre_hook(self, hook):
        assert self.init_ready, f"Module {self.__class__.__name__} must be initialized before registering hooks"
        self.add_forward_pre_hooks(hook)
        for module in self.children_modules:
            module.register_add_forward_pre_hook(hook)

    def register_forward_post_hook(self, hook):
        assert self.init_ready, f"Module {self.__class__.__name__} must be initialized before registering hooks"
        self.add_forward_post_hooks(hook)
        for module in self.children_modules:
            module.register_forward_post_hook(hook)

    def add_ordered_module_hooks(self, hook):
        if self.ordered_module_hooks is None:
            self.ordered_module_hooks = []
        self.ordered_module_hooks.append(hook)
    def add_forward_pre_hooks(self, hook):
        if self.forward_pre_hooks is None:
            self.forward_pre_hooks = []
        self.forward_pre_hooks.append(hook)
    def add_forward_post_hooks(self, hook):
        if self.forward_post_hooks is None:
            self.forward_post_hooks = []
        self.forward_post_hooks.append(hook)
    def call_add_ordered_module_hooks(self, *args):
        if self.ordered_module_hooks is not None:
            for hook in self.ordered_module_hooks:
                hook(self, *args)
    def call_forward_pre_hook(self, *args):
        if self.forward_pre_hooks is not None:
            for hook in self.forward_pre_hooks:
                hook(self, *args)
    def call_forward_post_hook(self, *args):
        if self.forward_post_hooks is not None:
            for hook in self.forward_post_hooks:
                hook(self, *args)

    def register_module(self, sub_module):
        self.children_ordered_module.append(sub_module)
        # TODO(sherry): support register hooks
        self.call_add_ordered_module_hooks(sub_module)
    
    def set_dtype(self, dtype: str):
        assert dtype in ["fp32", "fp16", "bf16"]
        self.dtype = dtype
    
    def parse_recompute_node(self):
        all_ordered_leaf_module:List[MetaModule] = []
        def dfs_traverse_leaf_module(module:MetaModule):
            if module.is_leaf_module:
                all_ordered_leaf_module.append(module)
            else:
                for module in self.children_ordered_module:
                    dfs_traverse_leaf_module(module)
        dfs_traverse_leaf_module(self)
      
        self.all_ordered_leaf_module = all_ordered_leaf_module

        fisrt_recomps:List[MetaModule] = []
        last_recomps:List[MetaModule] = []
        pre_enabled_recompute = False
        for leaf in all_ordered_leaf_module:
            if not pre_enabled_recompute and leaf.enable_recompute:
                leaf.recompute_status = "first"
                fisrt_recomps.append(leaf)
            if pre_enabled_recompute and not leaf.enable_recompute:
                leaf.recompute_status = "last"
                last_recomps.append(leaf)
            pre_enabled_recompute = leaf.enable_recompute

        # for i, p in enumerate(fisrt_recomps):
        #     print(f"{i}first recomputable module, path={p.}")

    @property
    def element_size(self):
        dtype = self.default_dtype
        if getattr(self, "dtype", False):
            dtype = self.dtype
        return self.dtype_to_element_size[dtype]

    @property
    def main_grad_element_size(self):
        """Main gradient precision used by memory/communication modeling."""
        if self.strategy.grad_reduce_in_bf16 or (not self.strategy.use_fp32_accum_grad):
            return self.dtype_to_element_size["bf16"]
        return self.dtype_to_element_size["fp32"]

    @property
    def first_compute_module(self):
        return self.children_ordered_module[0] if len(self.children_ordered_module) > 0 else self
    # =========================
    # Basic Compute Related
    # =========================
    def compute_end2end_time(self, compute_time, mem_time):
        return self.system.compute_end2end_time(compute_time, mem_time)

    def all_input_element_num(self):
        res = 0

        if isinstance(self.input_info, InputOutputInfo):
            input_info = [self.input_info]
        else:
            input_info = self.input_info
        for ii in input_info:
            if isinstance(ii, InputOutputInfo):
                for x in ii.tensors:
                    res += x.get_memory_size()
            elif isinstance(ii, TensorSize):
                res += ii.get_memory_size()
        return res

    def all_output_element_num(self):
        res = 0
        # element_size = self.element_size
        if isinstance(self.output_info, InputOutputInfo):
            output_info = [self.output_info]
        else:
            output_info = self.output_info
        for oi in output_info:
            if isinstance(oi, InputOutputInfo):
                for x in oi.tensors:
                    res += x.get_memory_size()
            elif isinstance(oi, TensorSize):
                res += oi.get_memory_size()
        return res
    
    def set_input_state_info(self, input_info: InputOutputInfo):
        # self.input_info = deepcopy(input_info)
        self.input_info = input_info # reference assignments are allowed here

    def set_path_debug_context(self, path_debug_context: PathDebugContext):
        self.path_debug_context = deepcopy(path_debug_context)

    def create_output_info(self):
        return InputOutputInfo([])

    # =========================
    # Pre/Post Porcess Related
    # =========================
    def _pre_op(self):
        pass

    def _post_op(self):
        pass

    # =========================
    # Memory Related
    # =========================
    def _comp_submod_cache_info_impl(self):
        ...
    def _comp_leaf_act_info_impl(self):
        self._act_info.activation_mem_cache = 0
        self._act_info.fwd_peak_mem_no_cache = 0
        self._act_info.bwd_peak_mem_no_cache = 0

    def _comp_act_info(self):
        if len(self.children_ordered_module) == 0:
            self._comp_leaf_act_info_impl()
            # leaf module act info is the same with recompute,
            # because _act_info_with_recomp is used to distinguish
            # the case of recompute in the combined module
            if self.is_variance_node:
                # print("Warning: variance node change peak")
                # self._act_info.activation_mem_cache = 0
                # self._act_info.bwd_peak_mem_no_cache += self._act_info.activation_mem_cache
                pass
            self._act_info_with_recomp = deepcopy(self._act_info)
        else:
            for module in self.children_ordered_module:
                self._act_info.activation_mem_cache = self._act_info.activation_mem_cache + module._act_info.activation_mem_cache

    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    def _comp_model_info(self):
        if len(self.children_ordered_module) > 0:
            for module in self.children_ordered_module:
                self._model_info = self._model_info + module.get_model_info()
        else:
            self._comp_leaf_model_info_impl()

    # =========================
    # Compute or Communicate Related
    # =========================

    def _comp_leaf_flops_info(self):
        self._compute_info.fwd_flops = 0
        self._compute_info.recompute_flops = 0
        self._compute_info.bwd_grad_act_flops = 0
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_mem_accessed_info(self):
        self._compute_info.fwd_accessed_mem = 0
        self._compute_info.bwd_grad_act_accessed_mem = 0
        self._compute_info.bwd_grad_w_accessed_mem = 0
        self._compute_info.recompute_accessed_mem = 0

    def _comp_leaf_intra_net_info(self):
        pass

    def _comp_compute_info(self):
        if len(self.children_ordered_module) > 0:
            for module in self.children_ordered_module:
                self._compute_info = self._compute_info + module.get_compute_info()
        else:
            self._comp_leaf_flops_info()
            self._comp_leaf_mem_accessed_info()
            self._comp_leaf_intra_net_info()
            if self.use_variance_tail_model and self.is_variance_node:
                self._compute_info.recompute_accessed_mem = 0
                self._compute_info.recompute_flops = 0
                self._cost_info.recompute_net_time = 0
                self._cost_info.recompute_net_exposed_time = 0
                if SIMU_DEBUG:
                    print(f"- {self.full_name} is variance node, recompute_accessed_mem and recompute_flops are set to 0")

    def _comp_cost_info(self):
        if len(self.children_ordered_module) > 0:
            for module in self.children_ordered_module:
                self._cost_info = self._cost_info + module.get_cost_info()
        else:
            # raise NotImplementedError
            self._comp_cost_info_impl(
                fwd_op="default",
                bwd_grad_act_op="default",
                bwd_grad_w_op="default",
                enable_recompute=self.enable_recompute,
            )  
                 
        if (
            self.path_debug_context
            and self.path_debug_context.target_point is not None
        ):
            # get the parent path of the current module
            path = get_point_name(
                parent=self.parent, current=self.current, sep=" -> "
            )
            if path in self.path_debug_context.target_point:
                file_path = f'{TMP_PATH}/cost_log.json'
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        try:
                            existing_data = json.load(file) 
                        except json.JSONDecodeError:
                            existing_data = {}
                else:
                    existing_data = {}
                existing_data.update(
                    {path:{"cost_F": self._cost_info.fwd_compute_time,
                            "cost_B": self._cost_info.bwd_grad_act_time,
                            "cost_W": self._cost_info.bwd_grad_w_time,
                            "recompute_F": self._cost_info.recompute_compute_time,
                            "net_F": self._cost_info.fwd_net_time,
                            "net_B": self._cost_info.bwd_net_time,
                            }
                            }
                )
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(existing_data, file, indent=4, ensure_ascii=False)
    def set_details(
        self, stage, compute_details, io_details
    ):
        if not hasattr(self, 'details'):
            self.details = {}
        self.details[stage] = {
            "compute_details" : deepcopy(compute_details),
            "io_details" : deepcopy(io_details),
        }

    def get_input_shapes_desc(self, stage):
        if isinstance(self, LinearBase):
            bmnk_info = self.get_gemm_bmnk(stage)
            b, m, n, k = bmnk_info['B'], bmnk_info['M'], bmnk_info['N'], bmnk_info['K']
            layout = bmnk_info['layout']
            accumulate = bmnk_info['accumulate'] # TODO(sherry): in bwd_grad_w, accumulate is True
            out_dtype = bmnk_info['out_dtype']
            return f'b={b}, m={m}, k={k}, n={n}, layout={layout}, accumulate={accumulate}, out_dtype={out_dtype}'
        else:
            return ""
    def _comp_cost_info_impl(
        self,
        fwd_op="default",
        bwd_grad_act_op="default",
        bwd_grad_w_op="default",
        enable_recompute=False,
    ):
        def compute_details(op_name, stage, flops, accessed_mem):
            #compute_details include compute time, tflops of accelerator, flops of current op, etc.
            class_key, path_key = self.get_cost_keys()
            shape_desc = self.get_input_shapes_desc(stage)
            compute_details = self.system.compute_op_accuracy_time(
                op_name, flops, shape_desc=shape_desc,
                reture_detail=True, class_key=class_key, path_key=path_key,
                accessed_mem=accessed_mem, stage=stage)

            # io_details include io time, gbps of accelerator, io size of current op, etc.
            io_details = self.system.compute_mem_access_time(op_name,
                accessed_mem, reture_detail=True, shape_desc=shape_desc,
                stage=stage,
            )

            # Get final time, we can set "roofline" or "compute_only" in accelerator config, default is roofline
            # if rooline, final time = max(compute_time, mem_time)
            # if compute_only, final time = compute_time
            end2end_time = self.compute_end2end_time(
                compute_time=compute_details['compute_only_time'], mem_time=io_details['io_time'],
            )

            # save details for each stage, for analysis
            self.set_details(stage, compute_details, io_details)
            # end2end_time is in ms (from compute_op_accuracy_time /
            # compute_mem_access_time). Keep _cost_info.*_time fields in ms:
            # net times (fwd_net_time, etc.) are ms and consumers (AtomModel
            # fwd_cost, perf_llm fwd_compute, DES clock) mix them directly.
            return end2end_time
        # 1. forward   
        self._cost_info.fwd_compute_time = compute_details(fwd_op, 'fwd', self._compute_info.fwd_flops, self._compute_info.fwd_accessed_mem)
        self._cost_info.bwd_grad_act_time = compute_details(bwd_grad_act_op, 'bwd_grad_act', self._compute_info.bwd_grad_act_flops, self._compute_info.bwd_grad_act_accessed_mem)
        self._cost_info.bwd_grad_w_time = compute_details(bwd_grad_w_op, 'bwd_grad_w', self._compute_info.bwd_grad_w_flops, self._compute_info.bwd_grad_w_accessed_mem)

        self._cost_info.recompute_compute_time = self._cost_info.fwd_time if self.enable_recompute else 0

        if self.enable_recompute and self.is_variance_node:
            self._cost_info.recompute_compute_time = 0
            if SIMU_DEBUG:
            # if 1:
                print(f'%% {self.name} is variance node, recompute_compute_time is 0')

        # if (
        #     self.path_debug_context
        #     and self.path_debug_context.target_point is not None
        # ):
        #     # get the parent path of the current module
        #     path = get_point_name(
        #         parent=self.parent, current=self.current, sep=" -> "
        #     )
        #     if path in self.path_debug_context.target_point:
        #         file_path = f'{TMP_PATH}/cost_log.json'
        #         os.makedirs(TMP_PATH, exist_ok=True)
        #         if os.path.exists(file_path):
        #             with open(file_path, 'r', encoding='utf-8') as file:
        #                 try:
        #                     existing_data = json.load(file) 
        #                 except json.JSONDecodeError:
        #                     existing_data = {}
        #         else:
        #             existing_data = {}
        #         existing_data.update(
        #             {path:{"cost_F": self._cost_info.fwd_compute_time,
        #                     "cost_B": self._cost_info.bwd_grad_act_time,
        #                     "cost_W": self._cost_info.bwd_grad_w_time,
        #                     "recompute_F": self._cost_info.recompute_compute_time,
        #                     "net_F": self._cost_info.fwd_net_time,
        #                     "net_B": self._cost_info.bwd_net_time,
        #                     }
        #                     }
        #         )
        #         with open(file_path, 'w', encoding='utf-8') as file:
        #             json.dump(existing_data, file, indent=4, ensure_ascii=False)

    # =========================
    # Agg Related
    # =========================
                    
    def get_compute_info(self) -> ModuleComputeInfo:
        assert (
            self._info_ready
        ), "flops/mem info not ready, please call the module to compute info"
        return self._compute_info

    def get_act_info(self) -> ActivationInfo:
        assert (
            self._info_ready
        ), "act info not ready, please call the module to compute info"
        return self._act_info

    def get_act_info_with_recomp(self) -> ActivationInfo:
        assert (
            self._info_ready
        ), "act info with recompute not ready, please call the module to compute info"
        return self._act_info_with_recomp

    def get_model_info(self) -> ModuleMemoryInfo:
        assert (
            self._info_ready
        ), f"model {self.__class__.__name__} info not ready, please call the module to compute info"
        return self._model_info

    def get_cost_info(self) -> ModuleCostInfo:
        assert (
            self._info_ready
        ), "cost info not ready, please call the module to compute info"
        return self._cost_info
    
    def forward(self, input_info: InputOutputInfo, path_debug_context: PathDebugContext) -> InputOutputInfo:
        raise NotImplementedError   
    
    def __call__(
        self, input_info: InputOutputInfo, path_debug_context: PathDebugContext
    ) -> InputOutputInfo:
        is_capture_only = get_capture_graph_only()
        if isinstance(input_info, TensorSize):
            input_info = InputOutputInfo([input_info])

        self.call_forward_pre_hook(input_info)

        # reset last result info
        self._reset_infos()
        
        self.set_input_state_info(input_info) # record the input
        self.set_path_debug_context(path_debug_context) # copy path debug context
        
        if self.parent_module and self not in self.parent_module.children_ordered_module:
            self.parent_module.register_module(self) # Non-leaf nodes also register themselves in the children module on the previous layer.
        # Debug, record the parent module and
        if self.path_debug_context:
            idx = len(self.parent_module.children_ordered_module)-1 if self.parent_module else 0
            current_repr = "(" + str(idx) + ")" + self.__class__.__name__

            self.path_debug_context.path_list.append(current_repr)
            
            self.parent = get_point_name(
                parent=path_debug_context.parent, 
                current=path_debug_context.current, sep=" -> "
            ) 
            self.current = current_repr
            self.current_full_module_path = get_point_name(parent=self.parent, current=self.current, sep=" -> ") #FIXME(sherry): path_debug_context is deepcopy to module. How to modify the parent of the temporary variable and pass it to the next module?

        # call once, return all fwd, bwd info
        self._pre_op()
        output_info = None        

        if not self.is_leaf_module:
            output_info = self.forward(input_info, self.path_debug_context)
        else:
            output_info = output_info if output_info else self.output_info # output_info = None, return leaf output
            if is_capture_only:
                graph_builder = SimuONNXGraphBuilder()
                graph_builder.add_node(op = self,
                                    op_type = self.__class__.__name__, 
                                    inputs = input_info.tensors if isinstance(input_info, InputOutputInfo) else [input_info],
                                    outputs = output_info.tensors  if isinstance(output_info, InputOutputInfo) else [output_info]
                                    )
        
        if not is_capture_only:
            # aggregate the info or compute the leaf info
            self._comp_model_info()  #static model memory usage
            self._comp_act_info()  #activation
            self._comp_compute_info()
            self._post_op()
            self._comp_cost_info()
        
        self._info_ready = True
        
        if isinstance(output_info, InputOutputInfo) and len(output_info.tensors) == 1:
            output_info = output_info.tensors[0]

        # path = get_point_name(parent=self.parent, current=self.current, sep=" -> ")
        self.call_forward_post_hook(input_info, output_info)
        return output_info

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        """
        Set the extra representation of the module
        """
        return ""

    # modified from
    # https://github.com/pytorch/pytorch/blob/08b5e07/torch/ao/nn/quantized/modules/utils.py#L114  # pylint: disable=line-too-long
    def __repr__(self) -> str:
        # pylint: disable=invalid-name
        def get_variable_name(var, namespace):
            for name, value in namespace.items():
                if value is var:
                    return name
            return None

        def _addindent(s_, numSpaces):
            s = s_.split("\n")
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        prev_mod_str = None
        prev_start_idx = 0
        show_full_name = False
        for idx, module in enumerate(self.children_ordered_module):
            if show_full_name:
                mod_str = module.full_name + " " + repr(module)
            else:
                mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)

            if prev_mod_str == mod_str:
                # merge
                if child_lines:
                    child_lines.pop()
                child_lines.append(
                    "(" + str(prev_start_idx) + "->" + str(idx) + "): " + mod_str
                )
            else:
                child_lines.append("(" + str(idx) + "): " + mod_str)
                prev_start_idx = idx
            prev_mod_str = mod_str

        lines = extra_lines + child_lines
        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        

        module = self
        
        # TODO(sherry): delete this, for debug

        main_str += ")"

        show_details = True
        if show_details:
            cost_info = module._cost_info
            main_str += f"\n\t1. cost: (total_time={cost_info.all_time:.2f} ms, fwd_details=(sum={cost_info.fwd_time+cost_info.fwd_net_time:.2f} ms, compute={cost_info.fwd_compute_time*1000:.2f} us, net={cost_info.fwd_net_time*1000:.2f} us), bwd_details=(sum={cost_info.bwd_time+cost_info.bwd_net_time:.2f} ms, compute={cost_info.bwd_compute_time*1000:.2f} us, net={cost_info.bwd_net_time*1000:.2f} us), variance_node={self.is_variance_node} flops={sum(module._compute_info.get_all_flops())/1e12:.2f} T) "

            module_info = module._model_info
            main_str += f"\n\t2. memory: (d_w={module_info.dense_weight_bytes}, d_g={module_info.dense_grad_bytes}, d_s={module_info.dense_state_bytes}, m_w={module_info.moe_weight_bytes}, m_g={module_info.moe_grad_bytes}, m_s={module_info.moe_state_bytes})"

        return main_str

class RecomputeBreakModule(MetaModule):
    def __init__(self, strategy, system, specific_name='', parent_module=None):
        super().__init__(strategy, system, specific_name, parent_module=parent_module)
        self.enable_recompute = False
    
    # TODO(sherry): no memory and not cost. Need to be implemented
    def create_output_info(self):
        output_info = InputOutputInfo(tensors=[t.new() for t in self.input_info.tensors])
        return output_info
class CostShape:
    """Unified cost-shape override for a linear layer: decouples the cost
    model's gemm dimensions from the forward topology, so a single object
    carries all cost-shape corrections instead of scattered per-op hooks:
      - output_size: cost N width (16p F 族 QKV 4608/5120 vs structural 5120)
      - seq_mult:    cost M scaling (16p lm_head full-seq 131072 vs per-rank)
      - skip_dw:     skip bwd_grad_w (lm_head weight tied to embedding, no dW)
    None / defaults = use structural values (backward-compatible)."""
    __slots__ = ('output_size', 'seq_mult', 'skip_dw')

    def __init__(self, output_size=None, seq_mult=1, skip_dw=False):
        self.output_size = output_size
        self.seq_mult = seq_mult
        self.skip_dw = skip_dw


class LinearBase(MetaModule):
    def __init__(self, input_size, output_size, strategy, system, specific_name='', parent_module=None):
        super().__init__(strategy, system, specific_name, parent_module)
        self.input_size = input_size
        self.output_size = output_size

    @property
    def micro_input_tensor(self) -> TensorSize:
        return TensorSize(shape=[])
    
    def get_weight(self):
        return TensorSize(shape=(self.output_size, self.input_size), dtype='fp8' if self.strategy.fp8 else 'bf16')

    def _record_te_dummy_wgrad_shape(self, output_size=None, input_size=None, grouped_linear=False):
        version_enabled = (
            self.strategy.te_grouped_linear_dummy_wgrad_memory_enabled
            if grouped_linear
            else self.strategy.te_dummy_wgrad_memory_enabled
        )
        if not (
            self.strategy.use_fused_grad_accumulation
            and version_enabled
        ):
            return
        output_size = self.output_size if output_size is None else output_size
        input_size = self.input_size if input_size is None else input_size
        # TE caches dummy tensors by (rows, cols, dtype). The dtype is the parameter dtype,
        # not the main_grad accumulation dtype.
        elem_size = self.dtype_to_element_size.get(self.strategy.dtype, self.dtype_to_element_size["bf16"])
        self._model_info.te_dummy_wgrad_shapes.add((int(output_size), int(input_size), int(elem_size)))
    
    def get_gemm_mnk(self, stage, format=False):
        """Get the m, n, k of the gemm operation, include forward and backward(bwd_act, bwd_w) pass"""
        inp_tensor = self.micro_input_tensor
        if inp_tensor.ndim == 2:
            bs = inp_tensor.shape[0]
        else:
            bs = inp_tensor.shape[0] * inp_tensor.shape[1]
        print(self.input_info.tensors[0])
        inp = self.input_size
        out = self.output_size
        if stage == 'fwd':
            return [[bs, inp], [inp, out], [bs, out]] if format else bs, inp, out
        elif stage == 'bwd_act':
            return [[bs, out], [out, inp], [bs, inp]] if format else bs, out, inp
        elif stage == 'bwd_w':
            return [[out, bs], [bs, inp], [out, inp]] if format else out, bs, inp
        elif stage == 'all':
            # get ms, ns, ks for all stages, fwd, bwd_act, bwd_w
            return [bs, bs, out], [inp, out, bs], [out, inp, inp]
        
    def get_gemm_bmnk(self, stage, format=False):
        """Get the b, m, k, n of the gemm operation, include forward and backward(bwd_grad_act, bwd_grad_w) pass sequently"""
        inp_tensor = self.micro_input_tensor
        if inp_tensor.ndim == 2:
            bs, seq_len = 1, inp_tensor.shape[0]
        else:
            bs, seq_len = inp_tensor.shape[:2] 
        inp = self.input_size
        # Unified cost-shape override (16p): the structural output_size / seq_len
        # are kept for the forward topology, but the cost model's gemm shape
        # (N / M here) reflects the real per-layer width / full-seq token count
        # when a CostShape is attached. No CostShape = no-op for every other module.
        cs = getattr(self, 'cost_shape', None)
        out = (cs.output_size if cs is not None and cs.output_size is not None
               else self.output_size)
        seq_len = int(seq_len * (cs.seq_mult if cs is not None else 1))
        bs, seq_len, inp, out = int(bs), int(seq_len), int(inp), int(out)
        if stage == 'fwd':
            return [[bs, seq_len, inp], [inp, out], [bs, out]] if format else dict(B=bs, M=seq_len, K=inp, N=out, layout='TN', accumulate=False, out_dtype='bf16')
        elif stage == 'bwd_grad_act':
            return [[bs, seq_len, out], [out, inp], [bs, inp]] if format else dict(B=bs, M=seq_len, K=out, N=inp, layout='NN', accumulate=False, out_dtype='bf16')
        elif stage == 'bwd_grad_w':
            return [[1, out, bs*seq_len], [bs*seq_len, inp], [out, inp]] if format else dict(B=1, M=out, K=bs*seq_len, N=inp, layout='NT', accumulate=True, out_dtype='bf16' if self.strategy.grad_reduce_in_bf16 else 'fp32')
        elif stage == 'all':
            # get bs, ms,  ks, ns for all stages, fwd, bwd_grad_act, bwd_grad_w, sequently
            return dict(B=[bs, bs, 1], M=[seq_len, seq_len, out], K=[inp, out, bs*seq_len], N=[out, inp, inp], layout=['TN', 'NN', 'NT'], accumulate=[False, False, True], out_dtype=['bf16', 'bf16', 'bf16' if self.strategy.grad_reduce_in_bf16 else 'fp32'])


    def parse_fwd_bwd_gemm_shape(self):
        x = self.input_info
        if x.tensors[0].ndim == 3:
            batch_size = int(x.tensors[0].shape[0] * x.tensors[0].shape[1])
        elif x.tensors[0].ndim == 2:
            batch_size = int(x.tensors[0].shape[0])
        else:
            raise NotImplementedError("Only support 2D and 3D tensors")
     
        fwd_lhs_shape, fwd_rhs_shape, fwd_out_shape = self.get_gemm_mnk('fwd', format=True)
        bwd_a_lhs_shape, bwd_a_rhs_shape, bwd_a_out_shape = self.get_gemm_mnk('bwd', format=True)
        bwd_w_lhs_shape, bwd_w_rhs_shape, bwd_w_out_shape = self.get_gemm_mnk('bwd_w', format=True)
        
        return {
            "fwd_lhs_shape": fwd_lhs_shape,
            "fwd_rhs_shape": fwd_rhs_shape,
            "fwd_out_shape": fwd_out_shape,
            "bwd_a_lhs_shape": bwd_a_lhs_shape,
            "bwd_a_rhs_shape": bwd_a_rhs_shape,
            "bwd_a_out_shape": bwd_a_out_shape,
            "bwd_w_lhs_shape": bwd_w_lhs_shape,
            "bwd_w_rhs_shape": bwd_w_rhs_shape,
            "bwd_w_out_shape": bwd_w_out_shape
        }  

class GroupLinearBase(LinearBase):
    """Base class for GroupGemm"""
    def __init__(self, local_expert_num, input_size: int, output_size: int,  strategy, system, specific_name='', parent_module=None) -> None:
        super().__init__(input_size, output_size, strategy, system, specific_name, parent_module)
        self.local_expert_num = local_expert_num    

    def get_input_shapes_desc(self, stage):
        assert self.input_info.tensors[0].size(0) % self.local_expert_num == 0, f'input size {self.input_info.tensors[0].size(0)} is not divisible by local_expert_num {self.local_expert_num} {self.strategy.parallelism}'
        num_tokens = self.input_info.tensors[0].size(0) // self.local_expert_num
        shape_str = f'ng={self.local_expert_num}, M={num_tokens}, N={self.output_size}, K={self.input_size}'

        dtype_str = f", dtype={'fp8' if self.strategy.fp8 else 'bf16'}, out_dtype=bf16, main_grad_dtype={'bf16' if self.strategy.grad_reduce_in_bf16 else 'fp32'}"
        # if self.strategy.fp8:
        shape_str += dtype_str
        if stage == 'fwd':
            shape_str += ', stage=fwd, grad=False, accumulate=False, use_split_accumulator=False, single_output=True'
        elif stage == 'bwd_grad_act':
            shape_str += ', stage=bwd_grad_act, grad=True, accumulate=False, use_split_accumulator=True, single_output=False'
        elif stage == 'bwd_grad_w':
            shape_str += ', stage=bwd_grad_w, grad=True, accumulate=True, use_split_accumulator=True, single_output=False'
        else:
            raise ValueError(f'Invalid stage: {stage}') 
        return shape_str
class Result:
    """A simple class to wrap the result dict"""

    def __init__(self, result: dict) -> None:
        self.data = result

    def get(self, key: str):
        return self.data.get(key, None)

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return to_json_string(self.data)

    def __str__(self):
        return self.to_json_string()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"


class BarrierBackend:
    def __init__(self):
        # gid -> state
        self.st = {}
        self.done = {}  # gid -> (end_t, set(waiters))

    def arrive(self, gid, rank, ready_t, expected, cost):
        # If this gid already completed and the same rank participated in that
        # completion, return the cached done state directly.
        d = self.done.get(gid)
        if d is not None:
            end_t, waiters = d
            if rank in waiters:
                return True, list(waiters), end_t

        s = self.st.get(gid)
        if s is None:
            s = {"expected": expected, "arrived": 0, "max_ready": 0.0, "waiters": [], "cost": cost}
            self.st[gid] = s
        elif rank in s["waiters"]:
            # Blocking comm jobs may be retried locally while still waiting for
            # their peer. Keep the original arrival instead of double-counting
            # the same rank and spuriously completing the barrier.
            return False, None, None

        s["arrived"] += 1
        s["max_ready"] = max(s["max_ready"], ready_t)
        s["waiters"].append(rank)

        if s["arrived"] == s["expected"]:
            end_t = s["max_ready"] + s["cost"]
            waiters = set(s["waiters"])
            del self.st[gid]
            self.done[gid] = (end_t, waiters)  # Cache completion for local retries.
            return True, list(waiters), end_t

        return False, None, None


class P2PBackend:
    """Dedicated backend for point-to-point send_recv-* rendezvous."""

    def __init__(self):
        self.st = {}
        self.done = {}

    def arrive(self, gid, rank, ready_t, cost):
        d = self.done.get(gid)
        if d is not None:
            end_t, waiters = d
            if rank in waiters:
                return True, list(waiters), end_t

        s = self.st.get(gid)
        if s is None:
            s = {"arrived": 0, "waiters": [], "arrivals": []}
            self.st[gid] = s
        elif rank in s["waiters"]:
            return False, None, None

        s["arrived"] += 1
        s["waiters"].append(rank)
        s["arrivals"].append((rank, ready_t, cost))

        if s["arrived"] == 2:
            end_t = max(arrival_ready + arrival_cost for _, arrival_ready, arrival_cost in s["arrivals"])
            waiters = set(s["waiters"])
            del self.st[gid]
            self.done[gid] = (end_t, waiters)
            return True, list(waiters), end_t

        return False, None, None


class NetworkFabric:
    """Per-GPU NIC servers (+ reserved per-node ToR servers, + optional
    per-level link servers).

    One NIC server per GPU, keyed by global rank; one reserved ToR server per
    node, keyed by ``rank // num_per_node``. Server state is a single tail
    clock per server (``nic_tail[rank]`` / ``tor_tail[node]``), mirroring
    ``rank_comm_tail``. Fabric servers never modify an op's ``cost``; they
    only shift its ``launch_t``/``ready_t`` (network-fabric design doc 5.3).
    ToR servers default to non-constraining (pass-through) until
    ``tor_enabled`` is switched on (design doc 5.1/5.5, decision 3).

    Hierarchical level servers (hierarchical-network design doc section 8,
    ``fabric_model="nic+levels"``) add one logical link server per
    (level, unit), keyed ``level_tail[(level_idx, unit)]`` for
    ``level_idx >= 1``; level 0 is the node level and its server stays the
    existing per-node ToR (``tor_tail``) — the two are unified, never
    duplicated. Until ``set_level_topology()`` is called the fabric behaves
    exactly as the NIC(+ToR)-only model.
    """

    def __init__(self, num_per_node, tor_enabled=False, tor_node_share=1,
                 tor_capacity_gbps=None):
        self.num_per_node = num_per_node
        self.tor_enabled = tor_enabled
        self.tor_node_share = tor_node_share
        # ToR service rate; when None, ToR occupancy falls back to the
        # cost-based formula below (which keeps ToR from binding harder than
        # the per-GPU NIC for isomorphic node traffic).
        self.tor_capacity_gbps = tor_capacity_gbps
        self.nic_tail = defaultdict(float)  # rank -> NIC busy until
        self.tor_tail = defaultdict(float)  # node -> ToR busy until
        # Hierarchical level servers; inert until set_level_topology().
        self.levels = None              # raw topology levels, innermost first
        self.level_spans = []           # cumulative GPUs per unit of level i
        self.level_capacities = []      # gbps, index-aligned with levels
        self.level_share = []           # occupancy amplification per level
        self.level_tail = defaultdict(float)  # (level_idx>=1, unit) -> busy until

    def set_level_topology(self, levels, level_capacities, merge_lanes):
        """Activate per-level link servers (hierarchical-network doc sec. 8).

        ``levels``: topology levels (innermost first), each
        ``{"name", "size", "net"}``; ``level_capacities[i]`` is the service
        rate (gbps) of level i's links, resolved by the caller from
        ``networks[levels[i]["net"]]``. ``level_share[i]`` is the merge_lanes
        amplification — active ranks per unit / simulated ranks per unit,
        i.e. the unit's span under merge_lanes, else 1.
        """
        self.levels = levels
        self.level_capacities = list(level_capacities)
        cumulative = 1
        self.level_spans = []
        for entry in levels:
            cumulative *= int(entry["size"])
            self.level_spans.append(cumulative)
        self.level_share = [span if merge_lanes else 1 for span in self.level_spans]
        self.level_tail = defaultdict(float)

    def level_topology_active(self):
        return self.levels is not None

    def level_crossings(self, rank_a, rank_b):
        """Levels (>= 1) whose unit boundary separates the two ranks.

        Symmetric in the endpoints, so either op of a p2p pair can compute it.
        Level 0 (node) is never reported: inter-node pairs are already ToR
        charged, and level_tail holds no level-0 keys.
        """
        if self.levels is None:
            return []
        return [i for i in range(1, len(self.level_spans))
                if rank_a // self.level_spans[i] != rank_b // self.level_spans[i]]

    def node_of(self, rank):
        return rank // self.num_per_node

    def acquire(self, rank, t, crossed_levels=None):
        """Earliest start on rank's NIC (and ToR / crossed level links).

        ``crossed_levels=None`` keeps the legacy NIC(+ToR) behavior of the
        existing call sites; an explicit list additionally serializes on the
        crossed (level, unit) link servers.
        """
        t = max(t, self.nic_tail[rank])
        if self.tor_enabled:
            t = max(t, self.tor_tail[self.node_of(rank)])
        if crossed_levels:
            for i in crossed_levels:
                if i < 1 or i >= len(self.level_spans):
                    continue  # Level 0 is the ToR, handled above.
                t = max(t, self.level_tail[(i, rank // self.level_spans[i])])
        return t

    def tor_occupancy(self, cost, size_bytes):
        """ToR service time of one entry (ms), amplified by node_share.

        With a capacity: size / tor_capacity * share. With the defaults
        (tor_capacity = node uplink, share = num_per_node) this equals the
        per-NIC service time, so ToR never binds harder than the NIC for
        isomorphic node traffic; it only binds when the user models
        oversubscription (capacity < num_per_node * per-NIC bandwidth).
        Fallback without size/capacity: cost * share / num_per_node, which
        has the same neutral default.
        """
        if size_bytes and self.tor_capacity_gbps:
            return (size_bytes / (self.tor_capacity_gbps * 1024**3) * 1e3
                    * self.tor_node_share)
        return cost * self.tor_node_share / self.num_per_node

    def level_occupancy(self, level_idx, cost, size_bytes):
        """Service time of one entry on a level link (ms), amplified by
        level_share. Mirrors ``tor_occupancy``: size-based when capacity and
        size are known; fallback ``cost * share / span`` (neutral under
        merge_lanes, where share == span, like the ToR default).
        """
        capacity = (self.level_capacities[level_idx]
                    if level_idx < len(self.level_capacities) else None)
        if size_bytes and capacity:
            return (size_bytes / (capacity * 1024**3) * 1e3
                    * self.level_share[level_idx])
        return cost * self.level_share[level_idx] / self.level_spans[level_idx]

    def charge(self, rank, end_t, cost, size_bytes=0):
        self.nic_tail[rank] = max(self.nic_tail[rank], end_t)
        if self.tor_enabled:
            node = self.node_of(rank)
            launch_t = end_t - cost
            self.tor_tail[node] = max(
                self.tor_tail[node], launch_t + self.tor_occupancy(cost, size_bytes)
            )

    def charge_levels(self, rank, end_t, cost, size_bytes=0, crossed_levels=None):
        """``charge()`` plus one charge per crossed level link server.

        The unit index derives from THIS rank (``rank // span_i``), so the
        two endpoints of a p2p pair — charged separately, each with its own
        rank — cover both sides' unit servers (design doc 8 route:
        NIC(src), link(level, src_unit), ..., NIC(dst)). Level 0 is skipped:
        the node level's server is the ToR, already charged by the base path
        when enabled. With no level topology or no crossed levels this
        degenerates to exactly ``charge()``.
        """
        self.charge(rank, end_t, cost, size_bytes=size_bytes)
        if not crossed_levels or self.levels is None:
            return
        launch_t = end_t - cost
        for i in crossed_levels:
            if i < 1 or i >= len(self.level_spans):
                continue  # Level 0 is the ToR, handled by charge().
            unit = rank // self.level_spans[i]
            self.level_tail[(i, unit)] = max(
                self.level_tail[(i, unit)],
                launch_t + self.level_occupancy(i, cost, size_bytes),
            )


@dataclass
class CommEntry:
    eid: int
    rank: int
    gid: tuple
    cost: float
    issue_t: float
    stream: str
    mode: str
    backend_kind: str
    expected: int | None = None
    status: str = "queued"
    ready_t: float | None = None
    # CollectiveCall lifecycle timestamps.  These are simulator-clock facts,
    # not values copied from a profiler.  ``dispatch_t`` is the modelled
    # submit point; a target runtime may later provide a more detailed split
    # without changing the semantic call identity.
    post_t: float | None = None
    dispatch_t: float | None = None
    transfer_start_t: float | None = None
    completion_t: float | None = None
    consumer_release_t: float | None = None
    launch_t: float | None = None
    end_t: float | None = None
    log_call_stk: str | None = None
    log_id: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class AsyncP2PState:
    gid: tuple
    cost: float = 0.0
    ready_t: float | None = None
    pair_logged: bool = False
    finalize_enqueued: bool = False
    post_unblock_enqueued: bool = False
    send_rank: int | None = None
    recv_rank: int | None = None
    send_eid: int | None = None
    recv_eid: int | None = None
    send_post_t: float | None = None
    recv_post_t: float | None = None
    send_post_order: int | None = None
    recv_post_order: int | None = None
    send_meta: dict = field(default_factory=dict)
    recv_meta: dict = field(default_factory=dict)


class State_Thread:
    def __init__(self):
        self.comm_order = 0
        # Accumulates all layer-wise FSDP RS ops across microbatches for this
        # thread (rank). LLMBlock._build_fsdp_rs_ops appends here; the
        # OptimizerSimulator tail creates an async_wait_collective over this
        # list to ensure all RS complete before optimizer step (FSDP2 gap
        # analysis doc section 3.6, decision #3).
        self.fsdp_rs_ops = []


# Built-in resource lanes used when no system-level registry is supplied
# (design doc 4.2). Must match SystemConfig.simu_resource_lanes()'s built-ins.
# ``offload`` is the asynchronous activation D2H/H2D lane.  It is kept
# separate from ``comp`` and ``comm`` so host-transfer work can overlap with
# device compute without inventing a measured overlap coefficient.
DEFAULT_RESOURCE_LANES = ("comp", "comm", "pp_fwd", "pp_bwd", "offload")


class SimuThread:
    def __init__(self, rank=None, lanes=None):
        self.rank = rank  # Exposed so SimuSystem can manage per-rank scheduling.
        self.job = []
        if lanes is None:
            lanes = DEFAULT_RESOURCE_LANES
        self.t = {lane: 0.0 for lane in lanes}
        self.thread_state = State_Thread()

    def _sync_time(self):
        # Optional lane merge for legacy behavior.
        m = max(self.t.values()) if self.t else 0.0
        for lane in list(self.t.keys()):
            self.t[lane] = m

    def step(self, ctx):
        ctx.current_rank = self.rank
        progressed = False
        while self.job:
            if isinstance(self.job[0], FwdQue):
                ok, blk = self.job[0].step(self.t, ctx)  # Returns (ok, block_key).
                if not ok:
                    if ctx.sync_lanes:
                        self._sync_time()
                    return "BLOCKED", blk
            else:
                ok, blk = self.job[0].bwd(self.t, ctx)
                if not ok:
                    if ctx.sync_lanes:
                        self._sync_time()
                    return "BLOCKED", blk

            progressed = True

            if not self.job[0]:
                self.job.pop(0)

            if ctx.sync_lanes:
                self._sync_time()

        return ("DONE", None) if not progressed else ("PROGRESSED", None)



class SimuSystem:
    def __init__(self, resource_lanes=None):
        self.threads = []  # thread must have .rank and .step()
        # Lane names from the system resource registry (design doc 4.2);
        # informational here — each SimuThread already carries its lane dict.
        self.resource_lanes = list(resource_lanes) if resource_lanes is not None else None

    def simu(self, ctx):
        # ctx.backend: BarrierBackend
        # ctx.threads_by_rank: dict[int, thread]
        threads_by_rank = {th.rank: th for th in self.threads}
        ctx.threads_by_rank = threads_by_rank

        ver = {r: 0 for r in threads_by_rank}
        blocked = set()
        heap = []
        blocked_on = {} 
        def cur_time(r):
            th = threads_by_rank[r]
            if ctx.sync_lanes:
                return max(th.t.values()) if th.t else 0.0
            # Overlap mode: schedule rank as soon as one lane can make progress.
            # Replicate the legacy defaultdict semantics: comp/comm always
            # count; other lanes count once they have been touched (> 0),
            # so untouched pp lanes do not distort heap priorities.
            active = [v for lane, v in th.t.items()
                      if v > 0 or lane in ("comp", "comm")]
            return min(active) if active else 0.0

        def push(r):
            ver[r] += 1
            heapq.heappush(heap, (cur_time(r), r, ver[r]))

        for r in threads_by_rank:
            push(r)

        done = set()
        # gids of blocking p2p completions already charged to the fabric, so
        # retried/cached completions of the same gid never double-charge.
        fabric_charged_gids = set()
        while len(done) < len(threads_by_rank):
            if not heap:
                print("DEADLOCK: heap empty")
                print("done", len(done), "blocked", len(blocked), "total", len(threads_by_rank))
                # Print ranks that have not finished yet.
                alive = [r for r in threads_by_rank if r not in done]
                print("alive ranks:", alive[:50], "..." if len(alive) > 50 else "")
                print("blocked_on sample:", list(blocked_on.items())[:20])
                blocked_gids = [key[1] for key in blocked_on.values() if isinstance(key, tuple) and len(key) > 1 and key[0] == "async_wait"]
                if blocked_gids:
                    async_meta = {}
                    for gid in blocked_gids[:12]:
                        state = ctx.get_async_state(gid)
                        async_meta[gid] = {
                            "send": state.send_meta,
                            "recv": state.recv_meta,
                            "send_rank": state.send_rank,
                            "recv_rank": state.recv_rank,
                            "send_post": state.send_post_t,
                            "recv_post": state.recv_post_t,
                        }
                    print("blocked async meta:", async_meta)
                if hasattr(ctx, "rank_comm_queue"):
                    queue_state = {}
                    for rr, q in ctx.rank_comm_queue.items():
                        if q:
                            queue_state[rr] = list(q)[:6]
                    print("rank_comm_queue sample:", dict(list(queue_state.items())[:20]))
                if hasattr(ctx, "comm_entries"):
                    sample_entries = {}
                    for rr, q in getattr(ctx, "rank_comm_queue", {}).items():
                        if q:
                            eid = q[0]
                            sample_entries[rr] = ctx.comm_entries.get(eid)
                    print("head comm entries:", sample_entries)
                if hasattr(ctx, "pending_async_posts"):
                    print("pending_async_posts:", ctx.pending_async_posts[:20])
                if hasattr(ctx, "async_states"):
                    async_state_sample = {}
                    for i, (gid, state) in enumerate(ctx.async_states.items()):
                        async_state_sample[gid] = {
                            "ready_t": state.ready_t,
                            "send_rank": state.send_rank,
                            "recv_rank": state.recv_rank,
                            "send_post_t": state.send_post_t,
                            "recv_post_t": state.recv_post_t,
                            "send_eid": state.send_eid,
                            "recv_eid": state.recv_eid,
                            "pair_logged": state.pair_logged,
                        }
                        if i >= 19:
                            break
                    print("async_states sample:", async_state_sample)
                # Print unfinished gids still tracked by the barrier backend.
                print("pending barriers:", len(ctx.backend.st))
                # Print a few concrete gids with expected/arrived counts.
                for i, (gid, s) in enumerate(ctx.backend.st.items()):
                    # if i >= 10: break
                    print(gid, "arrived", s["arrived"], "expected", s["expected"], "waiters_sample", s["waiters"][:8])
                raise RuntimeError("deadlock")

            t, r, v = heapq.heappop(heap)
            if v != ver[r]:
                continue
            if r in blocked or r in done:
                continue

            status, key = threads_by_rank[r].step(ctx)  # run-until-block
            ctx.pump_comm_queue()
            if status == "BLOCKED":
                blocked_on[r] = key

            # Handle completions triggered by this step via pending_completions.
            while ctx.pending_completions:
                gid, waiters, end_t, stream, fabric_charge = ctx.pending_completions.pop()
                for w in waiters:
                    th = threads_by_rank[w]
                    # Blocking collectives are synchronous at rank level:
                    # once completed, both compute and comm lanes should observe end_t.
                    th.t["comm"] = max(th.t["comm"], end_t)
                    th.t["comp"] = max(th.t["comp"], end_t)
                    if stream not in ("comm", "comp"):
                        th.t[stream] = max(th.t[stream], end_t)

                    # Only unblock ranks that are actually waiting on this gid.
                    if blocked_on.get(w) == ("barrier", gid):
                        del blocked_on[w]
                        push(w)
                # Blocking p2p charges BOTH ends (decision 1): every waiter's
                # NIC is set to the common barrier end_t, applied in the same
                # drain that raises their lane clocks. Keyed on the gid so
                # retried/cached completions never double-charge. Under a
                # level topology the payload also carries the pair's crossed
                # levels; each waiter's charge_levels derives its own unit
                # (rank // span_i), so both endpoints' link servers charge.
                if fabric_charge is not None and gid not in fabric_charged_gids:
                    fabric_charged_gids.add(gid)
                    charge_end_t, charge_cost, charge_size, charge_crossed = fabric_charge
                    for w in waiters:
                        ctx.fabric.charge_levels(w, charge_end_t, charge_cost,
                                                 size_bytes=charge_size,
                                                 crossed_levels=charge_crossed)
            while ctx.pending_comm_entry_completions:
                eid = ctx.pending_comm_entry_completions.pop()
                to_unblock = [w for w, wait_key in list(blocked_on.items()) if wait_key == ("comm_entry", eid)]
                for w in to_unblock:
                    del blocked_on[w]
                    push(w)
            ctx.flush_async_pair_logs()
            while ctx.pending_async_posts:
                gid = ctx.pop_async_post_unblock()
                to_unblock = [w for w, key in list(blocked_on.items()) if key in (("async_recv", gid), ("async_wait", gid))]
                for w in to_unblock:
                    del blocked_on[w]
                    push(w)
                    
            if status == "DONE":
                done.add(r)
                continue
            if status == "BLOCKED":
                if isinstance(key, tuple) and key and key[0] in ("yield", "yield_done", "yield_keep"):
                    blocked_on.pop(r, None)
                    push(r)
                    continue
                continue

            # PROGRESSED
            push(r)



        # Iteration end is the latest completed lane across all ranks.
        end_t = 0.0
        for th in threads_by_rank.values():
            if th.t:
                end_t = max(end_t, max(th.t.values()))
        print(f'end in {end_t}')
        return end_t

# Phase C "virtual waiters" (network-fabric design doc section 8): collective
# group kinds recognized in comm ids, as (marker, group_kind) pairs. Ordering
# matters — "dp_cp_group:" must be tried before "cp_group:" because the latter
# is a substring of the former. Ids look like
# "7-Embedding-tp_group:pp:0-cp:0-dp:0", so match with `in`.
_SKEW_GROUP_KINDS = (("dp_cp_group:", "dp_cp"), ("edp_group:", "edp"),
                     ("ep_group:", "ep"), ("cp_group:", "cp"),
                     ("tp_group:", "tp"), ("pp_group:", "pp"))


def _parse_group_kind(id_str):
    """Group kind encoded in a comm id; None when not a collective group."""
    for marker, kind in _SKEW_GROUP_KINDS:
        if marker in id_str:
            return kind
    return None


class SimuContext:
    def __init__(self, backend, merge_lanes=True, log_path='./tmp/log.log', sync_lanes=False,
                 resource_lanes=None, fabric=None):
        self.backend = backend
        self.p2p_backend = P2PBackend()
        self.pending_completions = []  # list[(gid, waiters, end_t, stream, fabric_charge)]
        # fabric_charge: None, or (end_t, cost, size_bytes, crossed_levels)
        self.pending_comm_entry_completions = []  # list[eid]
        self.pending_async_finalizations = []  # list[gid], LIFO for compatibility
        self.pending_async_posts = []  # list[gid], LIFO for compatibility
        self.pending_async_slot_releases = []  # legacy, unused in single-stream async p2p
        self.async_states = {}  # gid -> AsyncP2PState
        self.host_issue_seq = 0
        self.comm_entry_seq = 0
        self.comm_entries = {}  # eid -> entry dict
        self.rank_comm_queue = {}  # rank -> deque[eid]
        self.rank_comm_tail = {}  # rank -> end_t of last completed comm entry
        self.threads_by_rank = None
        self.merge_lanes = merge_lanes
        self.sync_lanes = sync_lanes
        self.log_path = log_path
        self.event_sink = EventSink()
        self.current_rank = None
        self.memory_tracker = None
        # Set by the simulation runner from the model's SystemConfig.  It is
        # used only to expose declared runtime/lifecycle policy in events;
        # measured traces never enter this context.
        self.system = None
        # Resource-lane names shared by all threads (design doc 4.2); kept on
        # the context so comm code can resolve lane membership if needed.
        self.resource_lanes = resource_lanes
        # NetworkFabric servers (network-fabric design doc 5); None = off,
        # which reproduces the pre-fabric behavior bit for bit.
        self.fabric = fabric


    @staticmethod
    def comm_lane_key(rank, stream):
        return (rank, stream)

    def get_async_state(self, gid):
        state = self.async_states.get(gid)
        if state is None:
            state = AsyncP2PState(gid=gid)
            self.async_states[gid] = state
        return state

    @staticmethod
    def p2p_channel(op_id):
        if "-backward-" in op_id:
            return "backward"
        return "forward"

    def register_async_send(
        self,
        *,
        gid,
        rank,
        post_t,
        cost,
        order,
        call_stk,
        log_id,
        meta=None,
    ):
        state = self.get_async_state(gid)
        state.cost = cost
        state.send_rank = rank
        state.send_post_t = post_t
        state.send_post_order = order
        state.send_meta = {"call_stk": call_stk, "id": log_id, **(meta or {})}

    def register_async_recv(
        self,
        *,
        gid,
        rank,
        post_t,
        cost,
        order,
        call_stk,
        log_id,
        meta=None,
    ):
        state = self.get_async_state(gid)
        state.cost = cost
        state.recv_rank = rank
        state.recv_post_t = post_t
        state.recv_post_order = order
        state.recv_meta = {"call_stk": call_stk, "id": log_id, **(meta or {})}

    def post_async_send_entry(
        self,
        *,
        gid,
        rank,
        post_t,
        cost,
        stream,
        mode,
        call_stk,
        log_id,
        net=None,
        size_bytes=0,
        meta=None,
    ):
        order = self.next_issue_seq()
        self.register_async_send(
            gid=gid,
            rank=rank,
            post_t=post_t,
            cost=cost,
            order=order,
            call_stk=call_stk,
            log_id=log_id,
            meta=meta,
        )
        eid = self.issue_comm_entry(
            rank=rank,
            gid=gid,
            cost=cost,
            issue_t=post_t,
            stream=stream,
            mode=mode,
            backend_kind="p2p",
            expected=2,
            log_call_stk=call_stk,
            log_id=log_id,
            meta={"post_order": order, "post_ts": post_t, "net": net,
                  "size_bytes": size_bytes, **(meta or {})},
        )
        self.attach_async_send_eid(gid, eid)
        self._stamp_async_crossed_levels(gid)
        self.pump_comm_queue()
        return eid

    def post_async_recv_entry(
        self,
        *,
        gid,
        rank,
        post_t,
        cost,
        stream,
        mode,
        call_stk,
        log_id,
        net=None,
        size_bytes=0,
        meta=None,
    ):
        order = self.next_issue_seq()
        self.register_async_recv(
            gid=gid,
            rank=rank,
            post_t=post_t,
            cost=cost,
            order=order,
            call_stk=call_stk,
            log_id=log_id,
            meta=meta,
        )
        eid = self.issue_comm_entry(
            rank=rank,
            gid=gid,
            cost=cost,
            issue_t=post_t,
            stream=stream,
            mode=mode,
            backend_kind="p2p",
            expected=2,
            log_call_stk=call_stk,
            log_id=log_id,
            meta={"post_order": order, "post_ts": post_t, "net": net,
                  "size_bytes": size_bytes, **(meta or {})},
        )
        self.attach_async_recv_eid(gid, eid)
        self._stamp_async_crossed_levels(gid)
        self.pump_comm_queue()
        return eid

    def _stamp_pair_levels_meta(self, meta, send_rank, recv_rank):
        """Compute the fabric level meta of one p2p pair (both ranks known).

        ``crossed_levels``: levels (>= 1) whose unit boundary separates the
        endpoints (symmetric in the ranks). For net="levels" entries also
        ``fabric_egress``: whether the pair sits on different nodes (level-0
        units) — same-node pairs stay on the intra-node fabric and engage
        no fabric server at all. net="inter_node" entries need no egress
        flag: their explicit marking already engages NIC(+ToR) as before.
        """
        fabric = self.fabric
        meta["crossed_levels"] = fabric.level_crossings(send_rank, recv_rank)
        if meta.get("net") == "levels":
            span0 = fabric.level_spans[0]
            meta["fabric_egress"] = send_rank // span0 != recv_rank // span0

    def _stamp_async_crossed_levels(self, gid):
        """Backfill ``crossed_levels`` into an async p2p pair's entry metas.

        No-op unless the hierarchical fabric is active (ctx.levels set and
        fabric level topology on) — legacy paths keep their exact meta shape.
        Computed once both endpoint ranks are known (the second post sees
        both); entries not yet done can still be stamped, and a p2p entry
        cannot complete before both posts, so stamping here always precedes
        the fabric charge. ``_complete_comm_entry`` re-checks at completion.
        """
        fabric = self.fabric
        if (fabric is None or not fabric.level_topology_active()
                or not getattr(self, "levels", None)):
            return
        state = self.get_async_state(gid)
        if state.send_rank is None or state.recv_rank is None:
            return
        for eid in (state.send_eid, state.recv_eid):
            if eid is None:
                continue
            entry = self.comm_entries.get(eid)
            if entry is None or entry.status == "done":
                continue
            if entry.meta.get("net") not in ("inter_node", "levels"):
                continue
            if "crossed_levels" not in entry.meta:
                self._stamp_pair_levels_meta(
                    entry.meta, state.send_rank, state.recv_rank)

    def _fabric_entry_engagement(self, entry):
        """(engages, crossed_levels) of the fabric for one comm entry.

        net=="inter_node" (legacy explicit marking): always engages when a
        fabric exists; crossed comes from the meta (None without a level
        topology, so charge/acquire degenerate to NIC+ToR exactly as before).
        net=="levels" (net-field semantics C, design doc section 7): engages
        only under an active level topology, and only for traffic leaving
        the node — collectives with a phase at level >= 1 (composition
        c_i > 1), p2p pairs on different nodes (meta["fabric_egress"];
        conservatively True before the peer post stamps it, matching the
        legacy unconditional NIC acquire at arrival).
        """
        if self.fabric is None:
            return False, None
        net = entry.meta.get("net")
        if net == "inter_node":
            return True, entry.meta.get("crossed_levels")
        if (net == "levels" and self.fabric.level_topology_active()
                and getattr(self, "levels", None)):
            crossed = entry.meta.get("crossed_levels")
            if entry.backend_kind == "p2p":
                return entry.meta.get("fabric_egress", True), crossed
            return bool(crossed) and any(i >= 1 for i in crossed), crossed
        return False, None

    def attach_async_send_eid(self, gid, eid):
        state = self.get_async_state(gid)
        state.send_eid = eid
        if state.send_meta is not None:
            state.send_meta["eid"] = eid

    def attach_async_recv_eid(self, gid, eid):
        state = self.get_async_state(gid)
        state.recv_eid = eid
        if state.recv_meta is not None:
            state.recv_meta["eid"] = eid

    def get_async_send_eid(self, gid):
        state = self.get_async_state(gid)
        if state.send_eid is not None:
            return state.send_eid
        return state.send_meta.get("eid")

    def get_async_recv_eid(self, gid):
        state = self.get_async_state(gid)
        if state.recv_eid is not None:
            return state.recv_eid
        return state.recv_meta.get("eid")

    def has_async_posted_send(self, gid):
        return self.get_async_state(gid).send_post_t is not None

    def has_async_posted_recv(self, gid):
        return self.get_async_state(gid).recv_post_t is not None

    def set_async_ready_t(self, gid, ready_t):
        state = self.get_async_state(gid)
        state.ready_t = ready_t

    def get_async_ready_t(self, gid):
        return self.get_async_state(gid).ready_t

    def queue_async_post_unblock(self, gid):
        state = self.get_async_state(gid)
        if state.post_unblock_enqueued:
            return
        self.pending_async_posts.append(gid)
        state.post_unblock_enqueued = True

    def pop_async_post_unblock(self):
        gid = self.pending_async_posts.pop()
        self.get_async_state(gid).post_unblock_enqueued = False
        return gid

    def queue_async_finalize(self, gid):
        state = self.get_async_state(gid)
        if state.finalize_enqueued:
            return
        self.pending_async_finalizations.append(gid)
        state.finalize_enqueued = True

    def pop_async_finalize(self):
        gid = self.pending_async_finalizations.pop()
        self.get_async_state(gid).finalize_enqueued = False
        return gid

    def flush_async_pair_logs(self):
        while self.pending_async_finalizations:
            gid = self.pop_async_finalize()
            self.emit_async_pair_logs(gid)

    def next_issue_seq(self):
        seq = self.host_issue_seq
        self.host_issue_seq += 1
        return seq

    def next_comm_entry_seq(self):
        seq = self.comm_entry_seq
        self.comm_entry_seq += 1
        return seq

    def issue_comm_entry(
        self,
        *,
        rank,
        gid,
        cost,
        issue_t,
        stream,
        mode,
        backend_kind,
        expected=None,
        log_call_stk=None,
        log_id=None,
        meta=None,
    ):
        eid = self.next_comm_entry_seq()
        entry = {
            "eid": eid,
            "rank": rank,
            "gid": gid,
            "cost": cost,
            "issue_t": issue_t,
            "stream": stream,
            "mode": mode,
            "backend_kind": backend_kind,
            "expected": expected,
            "post_t": issue_t,
            "dispatch_t": issue_t,
            "log_call_stk": log_call_stk,
            "log_id": log_id,
            "meta": meta or {},
        }
        self.comm_entries[eid] = CommEntry(**entry)
        lane_key = self.comm_lane_key(rank, stream)
        self.rank_comm_queue.setdefault(lane_key, deque()).append(eid)
        return eid

    def get_entry(self, eid):
        return self.comm_entries.get(eid)

    def entry_done(self, eid):
        entry = self.comm_entries.get(eid)
        return bool(entry) and entry.status == "done"

    def get_entry_end(self, eid):
        entry = self.comm_entries.get(eid)
        return None if entry is None else entry.end_t

    def get_rank_comm_tail(self, rank, stream):
        return self.rank_comm_tail.get(self.comm_lane_key(rank, stream), 0.0)

    def _complete_comm_entry(self, eid, launch_t, end_t):
        entry = self.comm_entries[eid]
        rank = entry.rank
        lane_key = self.comm_lane_key(rank, entry.stream)
        queue = self.rank_comm_queue.setdefault(lane_key, deque())
        if not queue or queue[0] != eid:
            raise RuntimeError(
                f"comm queue out of order on lane {lane_key}: expected head {eid}, got {queue[0] if queue else None}"
            )
        if launch_t + 1e-9 < self.get_rank_comm_tail(rank, entry.stream):
            raise RuntimeError(
                f"comm launch regressed on lane {lane_key}: launch_t={launch_t}, "
                f"tail={self.get_rank_comm_tail(rank, entry.stream)}, gid={entry.gid}"
            )
        entry.status = "done"
        entry.launch_t = launch_t
        entry.end_t = end_t
        entry.transfer_start_t = launch_t
        entry.completion_t = end_t
        lifecycle = dict(entry.meta.get("lifecycle") or {})
        lifecycle.update({
            "post_time_ms": entry.post_t,
            "task_dispatch_time_ms": entry.dispatch_t,
            "transfer_start_time_ms": entry.transfer_start_t,
            "completion_time_ms": entry.completion_t,
            "time_provenance": "simulator_clock",
        })
        entry.meta["lifecycle"] = lifecycle
        queue.popleft()
        self.rank_comm_tail[lane_key] = end_t
        # Uniform fabric charge (network-fabric design doc 5.3): covers local
        # entries, rendezvous waiters, and async p2p send/recv entries (both
        # ends get their own entries, so both ends are charged — decision 1).
        # Under a level topology the entry's crossed_levels (stamped at issue
        # for collectives, at post for async p2p) also charge the per-level
        # link servers (hierarchical-network design doc section 8).
        if self.fabric is not None and entry.backend_kind == "p2p":
            # Completion-time fallback for async p2p: both endpoint ranks
            # are known once the pair completed, even if the posts never got
            # to stamp the meta (covers "levels" egress too).
            if ("crossed_levels" not in entry.meta
                    and entry.meta.get("net") in ("inter_node", "levels")
                    and self.fabric.level_topology_active()
                    and getattr(self, "levels", None)):
                state = self.async_states.get(entry.gid)
                if (state is not None and state.send_rank is not None
                        and state.recv_rank is not None):
                    self._stamp_pair_levels_meta(
                        entry.meta, state.send_rank, state.recv_rank)
        engages, crossed = self._fabric_entry_engagement(entry)
        if engages:
            self.fabric.charge_levels(rank, end_t, entry.cost,
                                      size_bytes=entry.meta.get("size_bytes", 0),
                                      crossed_levels=crossed)
        if self.threads_by_rank is not None and rank in self.threads_by_rank:
            self.threads_by_rank[rank].t[entry.stream] = max(self.threads_by_rank[rank].t.get(entry.stream, 0.0), end_t)
        self.pending_comm_entry_completions.append(eid)
        self._maybe_finalize_async_ready(entry.gid)
        self._queue_async_finalize(entry.gid)
    def _maybe_finalize_async_ready(self, gid):
        state = self.get_async_state(gid)
        if state.ready_t is not None:
            return state.ready_t
        send_eid = self.get_async_send_eid(gid)
        recv_eid = self.get_async_recv_eid(gid)
        if send_eid is None or recv_eid is None:
            return None
        if not self.entry_done(send_eid) or not self.entry_done(recv_eid):
            return None
        send_entry = self.get_entry(send_eid)
        recv_entry = self.get_entry(recv_eid)
        if not send_entry or not recv_entry:
            return None
        if send_entry.end_t is None or recv_entry.end_t is None:
            return None
        ready_t = max(send_entry.end_t, recv_entry.end_t)
        self.set_async_ready_t(gid, ready_t)
        self.queue_async_post_unblock(gid)
        return ready_t

    def _queue_async_finalize(self, gid):
        state = self.get_async_state(gid)
        if state.pair_logged:
            return
        if self._maybe_finalize_async_ready(gid) is None:
            return
        self.queue_async_finalize(gid)

    def _pump_local_entry(self, eid):
        entry = self.comm_entries[eid]
        rank = entry.rank
        launch_t = max(entry.issue_t, self.get_rank_comm_tail(rank, entry.stream))
        # Engaged ops additionally serialize on the rank's NIC (and ToR when
        # enabled, and any crossed level link servers); the fabric only
        # shifts launch_t, never cost.
        engages, crossed = self._fabric_entry_engagement(entry)
        if engages:
            launch_t = self.fabric.acquire(rank, launch_t, crossed)
        end_t = launch_t + entry.cost
        # Phase C "virtual waiters" (network-fabric design doc section 8): a
        # local collective really completes when the slowest member of its
        # group arrives, so inflate its duration by the analytical straggler
        # ratio of the group's node count. p2p and rendezvous entries sync
        # with real peers and never pass through here, so they are never
        # skewed. Off by default; the skewed end_t flows downstream exactly
        # like the legacy one (lane tail, fabric charge, event end).
        if getattr(self, "collective_skew", None) == "virtual_waiters":
            strategy = getattr(self, "strategy", None)
            kind = _parse_group_kind(entry.gid[1])
            if strategy is not None and kind is not None:
                nodes = group_node_stats(kind, strategy, self.num_per_node)[1]
                end_t = launch_t + entry.cost * estimate_straggler_increase_ratio(nodes)
        self._complete_comm_entry(eid, launch_t, end_t)

    def _pump_rendezvous_entry(self, eid):
        entry = self.comm_entries[eid]
        rank = entry.rank
        if entry.status == "done":
            return
        if entry.status == "waiting":
            # This rank has already arrived at the rendezvous. Re-arriving the
            # same queued head would double-count the participant and can make
            # a p2p/collective appear to complete locally before its peer(s)
            # actually arrive.
            return
        ready_t = max(entry.issue_t, self.get_rank_comm_tail(rank, entry.stream))
        # Each engaged waiter acquires its NIC (and ToR when enabled, and any
        # crossed level link servers) before arriving at the backend
        # (network-fabric design doc 5.3).
        engages, crossed = self._fabric_entry_engagement(entry)
        if engages:
            ready_t = self.fabric.acquire(rank, ready_t, crossed)
        entry.ready_t = ready_t
        if entry.backend_kind == "p2p":
            done, waiters, end_t = self.p2p_backend.arrive(entry.gid, rank, ready_t, entry.cost)
        else:
            done, waiters, end_t = self.backend.arrive(
                entry.gid, rank, ready_t, entry.expected, entry.cost
            )
        entry.status = "waiting"
        if not done:
            return
        for waiter_rank in waiters:
            # Find the matching head on any lane for this waiting gid.
            queue = None
            waiter_eid = None
            waiter_entry = None
            for lane_key, candidate_queue in self.rank_comm_queue.items():
                if lane_key[0] != waiter_rank or not candidate_queue:
                    continue
                candidate_eid = candidate_queue[0]
                candidate_entry = self.comm_entries[candidate_eid]
                if candidate_entry.gid == entry.gid:
                    queue = candidate_queue
                    waiter_eid = candidate_eid
                    waiter_entry = candidate_entry
                    break
            if queue is None:
                raise RuntimeError(f"comm completion without queued head on rank {waiter_rank} for {entry.gid}")
            if waiter_entry.gid != entry.gid:
                raise RuntimeError(
                    f"comm completion gid mismatch on rank {waiter_rank}: head={waiter_entry.gid} done={entry.gid}"
                )
            waiter_ready_t = waiter_entry.ready_t
            if waiter_ready_t is None:
                waiter_ready_t = max(
                    waiter_entry.issue_t, self.get_rank_comm_tail(waiter_rank, waiter_entry.stream)
                )
                w_engages, w_crossed = self._fabric_entry_engagement(waiter_entry)
                if w_engages:
                    waiter_ready_t = self.fabric.acquire(
                        waiter_rank, waiter_ready_t, w_crossed)
                waiter_entry.ready_t = waiter_ready_t
            launch_t = max(waiter_ready_t, end_t - waiter_entry.cost)
            self._complete_comm_entry(waiter_eid, launch_t, end_t)
        return

    def pump_comm_queue(self):
        progressed = True
        while progressed:
            progressed = False
            for lane_key in sorted(self.rank_comm_queue):
                queue = self.rank_comm_queue.get(lane_key)
                if not queue:
                    continue
                eid = queue[0]
                entry = self.comm_entries[eid]
                before_status = entry.status
                if entry.backend_kind == "local":
                    self._pump_local_entry(eid)
                else:
                    self._pump_rendezvous_entry(eid)
                if self.entry_done(eid) or self.comm_entries[eid].status != before_status:
                    progressed = True

    def ensure_async_ready(self, gid):
        state = self.get_async_state(gid)
        ready_t = self._maybe_finalize_async_ready(gid)
        if ready_t is None:
            self.pump_comm_queue()
            ready_t = self._maybe_finalize_async_ready(gid)
        return ready_t

    def emit_async_pair_logs(self, gid):
        state = self.get_async_state(gid)
        if state.pair_logged:
            return state.ready_t
        ready_t = state.ready_t
        if ready_t is None:
            return None
        send_eid = self.get_async_send_eid(gid)
        recv_eid = self.get_async_recv_eid(gid)
        send_entry = self.get_entry(send_eid)
        recv_entry = self.get_entry(recv_eid)
        if not send_entry or not recv_entry:
            return None
        if send_entry.end_t is None or recv_entry.end_t is None:
            return None
        send_meta = state.send_meta
        recv_meta = state.recv_meta
        if send_meta is not None and recv_meta is not None:
            send_post = state.send_post_t if state.send_post_t is not None else send_entry.launch_t
            recv_post = state.recv_post_t if state.recv_post_t is not None else recv_entry.launch_t
            send_order = state.send_post_order if state.send_post_order is not None else -1
            recv_order = state.recv_post_order if state.recv_post_order is not None else -1
            self.event_sink.emit_span(
                send_meta['call_stk'], gid[0], send_entry.launch_t, send_entry.end_t,
                gid=send_meta['id'], post=send_post, order=send_order,
                stream=send_entry.stream, kind="comm", lane=None,
                metadata=send_meta,
            )
            self.event_sink.emit_span(
                recv_meta['call_stk'], gid[0], recv_entry.launch_t, recv_entry.end_t,
                gid=recv_meta['id'], post=recv_post, order=recv_order,
                stream=recv_entry.stream, kind="comm", lane=None,
                metadata=recv_meta,
            )
            if ready_t > recv_entry.end_t + 1e-9:
                wait_call_stk = recv_meta["call_stk"].replace("-async_recv", "-async_wait_recv")
                self.event_sink.emit_span(
                    wait_call_stk, gid[0], recv_entry.end_t, ready_t,
                    gid=recv_meta['id'], post=recv_post, order=recv_order,
                    stream="comp", kind="wait", lane=None,
                    metadata=recv_meta,
                )
        state.pair_logged = True
        return ready_t

    def finalize_async_p2p(self, gid, stream="comm"):
        ready_t = self.ensure_async_ready(gid)
        if ready_t is None:
            return None
        return self.emit_async_pair_logs(gid)


class LeafModel():
    # Phase-1 span classification for the trace exporter (see
    # docs/design_simu_kind_resource_model.md section 4.1). Pure annotation:
    # scheduling, lanes, and durations are unaffected.
    simu_kind: ClassVar[str | None] = "compute"
    simu_lane: ClassVar[str | None] = None

    def _event_metadata(
        self,
        ctx=None,
        phase=None,
        lifecycle_stage=None,
        post_time=None,
        completion_time=None,
        consumer_release_time=None,
        entry=None,
    ):
        """Return portable metadata shared by all leaf event kinds."""
        structural = dict(getattr(self, "event_metadata", {}) or {})
        op_id = str(getattr(self, "id", ""))
        lowered = op_id.lower()
        group_kind = getattr(self, "group_kind", None)
        if group_kind is None:
            for candidate in ("dp_cp", "edp", "etp", "cp", "ep", "tp", "pp"):
                if f"{candidate}_group" in lowered or f"{candidate}group" in lowered:
                    group_kind = candidate
                    break
        if group_kind is None and ("default_group" in lowered or "send_recv-" in lowered):
            group_kind = "pp"
        comm_stage = getattr(self, "comm_stage", None)
        marker = "-stage:"
        if comm_stage is None and marker in lowered:
            comm_stage = op_id[lowered.index(marker) + len(marker):].split("-")[0]
        owner = getattr(self, "comm_owner", None) or {
            "cp": "attention_cp", "ep": "moe_ep", "dp_cp": "fsdp_dense",
            "edp": "fsdp_moe", "tp": "tensor_parallel",
            "etp": "expert_tensor_parallel", "pp": "pipeline",
        }.get(group_kind)
        stage_lower = str(comm_stage or "").lower()
        if "dispatch" in stage_lower:
            owner = "moe_dispatch"
        elif "combine" in stage_lower:
            owner = "moe_combine"
        elif "router" in stage_lower:
            owner = "moe_router"
        defaults = {
            "group_kind": group_kind,
            "group_size": getattr(self, "group_size", None),
            "payload_bytes": getattr(self, "size_bytes", 0),
            "size_bytes": getattr(self, "size_bytes", 0),
            "net": getattr(self, "net", None),
            "comm_stage": comm_stage,
            "comm_owner": owner,
            "comm_role": (getattr(self, "comm_role", None)
                           or comm_stage or group_kind),
        }
        # Explicit model metadata wins over communication defaults.  This is
        # what allows compute atoms and communication atoms to share one event
        # schema without changing their scheduling behavior.
        defaults.update(structural)
        if getattr(self, "simu_kind", None) in ("comm", "wait") or group_kind is not None:
            lifecycle = _collective_lifecycle_facts(
                op_id=op_id,
                group_size=getattr(self, "group_size", None),
                payload_bytes=getattr(self, "size_bytes", 0),
                comm_role=getattr(self, "comm_role", None),
                comm_stage=comm_stage,
                ctx=ctx,
            )
            supplied_lifecycle = dict(defaults.get("lifecycle") or {})
            lifecycle.update(supplied_lifecycle)
            defaults["lifecycle"] = lifecycle
            defaults = _lifecycle_with_times(
                defaults,
                stage=lifecycle_stage,
                phase=phase,
                post_time=post_time,
                completion_time=completion_time,
                consumer_release_time=consumer_release_time,
                entry=entry,
            )
        return defaults

    def __init__(self, specific_name='', event_metadata=None):
        self.st = None
        self.st_bwd = None
        self.call_stk =f'-{self.__class__.__name__}'
        self.forward_op = "fwd"
        if specific_name:
            self.call_stk =f'-{specific_name}'
        self.event_metadata = dict(event_metadata or {})

    def set_event_metadata(self, metadata):
        """Attach model/config-derived metadata without affecting DES cost."""
        if metadata:
            self.event_metadata.update(metadata)

    # def step(self, t, ctx):
    #     # Default behavior is to call _step; subclasses can override it.
    #     out = self._step(t, ctx)
    #     return out if isinstance(out, tuple) else (bool(out), None)
    
    def step(self, t, ctx):
        # t is the per-rank lane-clock dict (lanes from the resource registry,
        # see SimuThread); leaf ops advance the "comp" lane.
        if self.st is None:
            self.st = t["comp"]

        out = self._step(t, ctx)
        ok, blk = out if isinstance(out, tuple) else (bool(out), None)
        if ok:
            if t['comp'] == self.st:
                return True, None
            ctx.event_sink.emit_span(self.call_stk, self.forward_op, self.st, t['comp'],
                                     kind=self.simu_kind, lane=self.simu_lane,
                                     metadata=self._event_metadata())
            return True, None
        return False, blk
    
    # def bwd(self, t, ctx):
    #     out = self._bwd(t, ctx)
    #     return out if isinstance(out, tuple) else (bool(out), None)

    def bwd(self, t, ctx):
        if self.st_bwd is None:
            self.st_bwd = t["comp"]
        out = self._bwd(t, ctx)
        ok, blk = out if isinstance(out, tuple) else (bool(out), None)
        if ok:
            if t['comp'] == self.st_bwd:
                return True, None
            ctx.event_sink.emit_span(self.call_stk, "bwd", self.st_bwd, t['comp'],
                                     kind=self.simu_kind, lane=self.simu_lane,
                                     metadata=self._event_metadata())
            return True, None
        return False, blk
    
    def _step(self, t, ctx):
        return True  # Default leaf behavior: no blocking.

    def _bwd(self, t, ctx):
        return True
    
    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
    
    def prefill_fwd(self):
        return self

    def prefill_recompute_fwd(self, recompute_cost_override=None):
        return self.prefill_fwd()

    def prefill_bwd(self):
        return self


    
class AtomModel(LeafModel):
    #simplify LeafModel with cost information
    def __init__(self, fwd_cost, bwd_cost, specific_name='', recompute_cost=None,
                 skip_recompute=False, metadata=None):
        super().__init__(specific_name, event_metadata=metadata)
        self.fwd_cost = fwd_cost
        self.bwd_cost = bwd_cost
        self.skip_recompute = skip_recompute
        self.recompute_cost = fwd_cost if recompute_cost is None else recompute_cost
        # self.fwd_cost = fwd_cost*(1+random.random()*0.6)
        # self.bwd_cost = bwd_cost*(1+random.random()*0.6)
    def _step(self, t, ctx):
        t["comp"] += self.fwd_cost
        return True

    def _bwd(self, t, ctx):
        t["comp"] += self.bwd_cost
        return True

    def prefill_recompute_fwd(self, recompute_cost_override=None):
        # A bwd-only atom (fwd_cost=0, skip_recompute=True) models a
        # backward-only kernel that has no forward/recompute phase; its
        # recompute clone must stay zero-cost even when a parent leaf-module
        # recompute override would otherwise be applied.
        if self.skip_recompute:
            recompute_cost = 0.0
        else:
            recompute_cost = self.recompute_cost if recompute_cost_override is None else recompute_cost_override
        clone = AtomModel(
            fwd_cost=recompute_cost,
            bwd_cost=self.bwd_cost,
            recompute_cost=recompute_cost,
            metadata=self.event_metadata,
        )
        clone.call_stk = self.call_stk
        clone.forward_op = "recompute_fwd"
        return clone
    

# Module-level id source for FusedOp correlation gids (design doc 4.3): all
# lane slices of one fused op phase share the gid so the trace exporter can
# render them as correlated slices.
_FUSED_OP_ID_SEQ = itertools.count(1)


def _next_fused_op_id():
    return f"fused-{next(_FUSED_OP_ID_SEQ)}"


class FusedOp(LeafModel):
    """A fused multi-resource operator (design doc 4.3, Phase 3).

    One logical op that occupies several resource lanes at once — e.g. a TP
    all-gather fused with the following GEMM. The per-lane busy costs are
    given up front; a FusionPolicy (simumax/core/fusion.py) composes them
    into the op's total span and the per-lane busy durations.

    Anchor rule: the op starts only when every declared lane AND the rank
    anchor clock t["comp"] are free; on completion t["comp"] is set to
    start + span, so the rank's next queued op sequences after the whole
    fused unit (t["comp"] stays the queue-sequencing anchor clock), while
    each declared lane advances independently to its own busy end.

    step/bwd are overridden (like Com) because the base LeafModel wrapper
    would emit a single comp-anchored span; a fused op instead emits one
    slice per occupied lane, all sharing one gid (kind "fused",
    stream/lane = the occupied lane) so the trace exporter can render them
    as correlated slices on multiple lanes.

    Builders are responsible for placing FusedOp instances into job queues;
    no builder uses it yet in Phase 3 — this class is only the scheduling
    and emission mechanism.
    """

    simu_kind = "fused"

    def __init__(self, costs: Dict[str, float], policy: FusionPolicy,
                 specific_name='', bwd_costs: Dict[str, float] = None, op_id=None,
                 metadata=None):
        # costs: dict lane_name -> busy ms, e.g. {"comp": 10.0, "comm": 8.0}.
        # bwd_costs defaults to costs.
        super().__init__(specific_name, event_metadata=metadata)
        assert costs, "FusedOp requires a non-empty costs dict"
        assert all(c >= 0 for c in costs.values()), f"negative lane cost in {costs}"
        self.costs = dict(costs)
        self.bwd_costs = dict(costs) if bwd_costs is None else dict(bwd_costs)
        assert self.bwd_costs and all(c >= 0 for c in self.bwd_costs.values()), \
            f"invalid bwd_costs {bwd_costs}"
        self.policy = policy
        # One gid per phase; every lane slice of a phase shares it. bwd uses a
        # derived id so fwd/bwd slices never correlate with each other.
        self.op_id = op_id if op_id is not None else _next_fused_op_id()
        self._bwd_op_id = f"{self.op_id}-bwd"

    @property
    def simu_resources(self):
        """Resource lanes this op occupies (from the forward costs)."""
        return tuple(self.costs.keys())

    def step(self, t, ctx):
        # Overridden (like Com) so the base LeafModel wrapper does not emit
        # its single comp-anchored span; _step emits the per-lane slices.
        out = self._step(t, ctx)
        ok, blk = out if isinstance(out, tuple) else (bool(out), None)
        return (True, None) if ok else (False, blk)

    def bwd(self, t, ctx):
        out = self._bwd(t, ctx)
        ok, blk = out if isinstance(out, tuple) else (bool(out), None)
        return (True, None) if ok else (False, blk)

    def _step(self, t, ctx):
        return self._run_fused(t, ctx, self.costs, self.op_id, self.forward_op)

    def _bwd(self, t, ctx):
        return self._run_fused(t, ctx, self.bwd_costs, self._bwd_op_id, "bwd")

    def _run_fused(self, t, ctx, costs, gid, phase):
        # Never blocks: all sequencing inputs are local lane clocks.
        lanes = set(costs) | {"comp"}
        start = max(t[l] for l in lanes)
        span = self.policy.span(costs)
        lane_durs = self.policy.lane_durations(costs)
        for lane, dur in lane_durs.items():
            t[lane] = start + dur
        # Anchor last: the rank's next queued op sequences after the fused
        # unit even when "comp" itself is a declared (shorter) lane.
        t["comp"] = start + span
        for lane, dur in lane_durs.items():
            if dur <= 0:
                # Zero-cost lanes still advance their clock above but emit no
                # slice — a zero-duration fused slice carries no information.
                continue
            ctx.event_sink.emit_span(
                self.call_stk, phase, start, start + dur,
                gid=gid, kind="fused", stream=lane, lane=lane,
                metadata=self._event_metadata(),
            )
        return True

class Com(LeafModel):
    simu_kind = "comm"

    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0,
                 call_stk='', global_rank=None, stream="comm", net=None, size_bytes=0,
                 group_kind=None, comm_stage=None, comm_owner=None, comm_role=None,
                 **kwargs):
        super().__init__(event_metadata=kwargs.pop("metadata", None))
        self.call_stk = call_stk + f'{self.call_stk}'
        self.id = id
        self.rank = rank
        self.group_size = group_size
        self.fwd_cost = fwd_cost
        self.bwd_cost = bwd_cost
        self.global_rank = global_rank
        self.stream = stream
        # Resolved strategy net name ("inter_node"/"intra_node"/...) and
        # payload size; net is None for unmigrated call sites, which keeps the
        # op out of fabric charging (network-fabric design doc 5.2).
        self.net = net
        self.size_bytes = size_bytes
        # Communication semantics are model/config facts.  Keep them on the
        # op so every completion/post span can export the same ledger fields.
        # ``kwargs`` intentionally absorbs legacy constructor extras (for
        # example ``strategy``) without changing scheduling behaviour.
        self.group_kind = group_kind
        self.comm_stage = comm_stage
        self.comm_owner = comm_owner
        self.comm_role = comm_role
        self._completed = set()  # store completed gid for this rank/op
        self._fwd_launch_st = None
        self._bwd_launch_st = None
        self._fwd_issue_order = None
        self._bwd_issue_order = None
        self._fwd_entry_eid = None
        self._bwd_entry_eid = None
        self._blocking_start_by_gid = {}
        self._fwd_done_t = None
        self._bwd_done_t = None
        self._batch_submit_by_gid = {}

    def _event_metadata(
        self,
        ctx=None,
        phase=None,
        lifecycle_stage=None,
        post_time=None,
        completion_time=None,
        consumer_release_time=None,
        entry=None,
    ):
        """Return portable communication ownership metadata for trace spans."""
        group_kind = self.group_kind
        comm_stage = self.comm_stage
        # Existing model ids already encode these structural labels.  Parsing
        # them avoids changing every historical call site while preserving
        # portability across rank/world-size changes.
        lowered = str(self.id).lower()
        if group_kind is None:
            for candidate in ("dp_cp", "edp", "etp", "cp", "ep", "tp", "pp"):
                if f"{candidate}_group" in lowered or f"{candidate}group" in lowered:
                    group_kind = candidate
                    break
        if group_kind is None and ("default_group" in lowered or "send_recv-" in lowered):
            group_kind = "pp"
        if comm_stage is None:
            marker = "-stage:"
            if marker in lowered:
                comm_stage = str(self.id)[lowered.index(marker) + len(marker):].split("-")[0]
        owner = self.comm_owner
        if owner is None:
            owner = {
                "cp": "attention_cp", "ep": "moe_ep", "dp_cp": "fsdp_dense",
                "edp": "fsdp_moe", "tp": "tensor_parallel",
                "etp": "expert_tensor_parallel", "pp": "pipeline",
            }.get(group_kind)
        stage_lower = str(comm_stage or "").lower()
        if "dispatch" in stage_lower:
            owner = "moe_dispatch"
        elif "combine" in stage_lower:
            owner = "moe_combine"
        elif "router" in stage_lower:
            owner = "moe_router"
        metadata = {
            "comm_id": str(self.id),
            "group_kind": group_kind,
            "group_size": self.group_size,
            "payload_bytes": self.size_bytes,
            "size_bytes": self.size_bytes,
            "net": self.net,
            "comm_stage": comm_stage,
            "comm_owner": owner,
            "comm_role": self.comm_role or comm_stage or group_kind,
        }
        metadata.update(getattr(self, "event_metadata", {}) or {})
        lifecycle = _collective_lifecycle_facts(
            op_id=self.id,
            group_size=self.group_size,
            payload_bytes=self.size_bytes,
            comm_role=metadata.get("comm_role"),
            comm_stage=comm_stage,
            ctx=ctx,
        )
        lifecycle.update(dict(metadata.get("lifecycle") or {}))
        metadata["lifecycle"] = lifecycle
        metadata = _lifecycle_with_times(
            metadata,
            stage=lifecycle_stage,
            phase=phase,
            post_time=post_time,
            completion_time=completion_time,
            consumer_release_time=consumer_release_time,
            entry=entry,
        )
        return metadata

    def _dp_comm_push(self, ctx, t, end_t, launch_st):
        """Advance compute for a blocking communication completion.

        FSDP overlap is represented only by async post/wait graph edges.
        """
        return end_t

    def _prime_batch_submit(self, phase, submit_t):
        gid = (phase, self.id)
        self._batch_submit_by_gid.setdefault(gid, submit_t)

    def _event_start_t(self, entry):
        # For rendezvous/barrier-style communications, the profile-visible
        # event should include local waiting before common completion.
        if entry.backend_kind == "barrier" or self.id.startswith("send_recv-"):
            return entry.issue_t
        return entry.launch_t

    def _emit_post_marker(self, ctx, phase, issue_t):
        # Faithful post/wait trace shape for blocking comm (design doc 9.3):
        # a zero-duration marker at the moment the op posts (CommEntry issued
        # / barrier reached). The completion span is still emitted by the
        # step/bwd wrappers; this marker carries no timing effect. The display
        # name is the completion span's name (last call_stk segment) + "-post".
        ctx.event_sink.emit_span(
            self.call_stk, phase, issue_t, issue_t,
            gid=self.id, stream=self.stream,
            kind="comm", lane=self.simu_lane,
            name=self.call_stk.split("-")[-1] + "-post",
            metadata=self._event_metadata(
                ctx=ctx, phase=phase, lifecycle_stage="post",
                post_time=issue_t,
            ),
        )

    def _forward_phase(self):
        """Return the semantic phase for the forward-side lifecycle.

        Normal forward calls use ``fwd``. Recompute calls are cloned by
        :meth:`prefill_recompute_fwd` and set ``forward_op`` to
        ``recompute_fwd``; they execute while the backward queue replays
        saved activations, so retaining that label is important for phase and
        overlap analysis. The value is supplied by the model scheduler, not
        inferred from measured trace names or timings.
        """
        return getattr(self, "forward_op", "fwd") or "fwd"

    def step(self, t, ctx):
        out = self._step(t, ctx)
        ok, blk = out if isinstance(out, tuple) else (bool(out), None)
        if ok:
            done_t = self._fwd_done_t if self._fwd_done_t is not None else t[self.stream]
            if self._fwd_launch_st is None or done_t == self._fwd_launch_st:
                return True, None
            entry = ctx.get_entry(self._fwd_entry_eid)
            if entry is not None:
                entry.consumer_release_t = done_t
            ctx.event_sink.emit_span(
                self.call_stk, self._forward_phase(), self._fwd_launch_st, done_t,
                gid=self.id, stream=self.stream,
                kind=self.simu_kind, lane=self.simu_lane,
                metadata=self._event_metadata(
                    ctx=ctx, phase=self._forward_phase(), lifecycle_stage="completion",
                    completion_time=done_t,
                    entry=entry,
                ),
            )
            self._fwd_launch_st = None
            self._fwd_done_t = None
            return True, None
        return False, blk

    def bwd(self, t, ctx):
        out = self._bwd(t, ctx)
        ok, blk = out if isinstance(out, tuple) else (bool(out), None)
        if ok:
            done_t = self._bwd_done_t if self._bwd_done_t is not None else t[self.stream]
            if self._bwd_launch_st is None or done_t == self._bwd_launch_st:
                return True, None
            entry = ctx.get_entry(self._bwd_entry_eid)
            if entry is not None:
                entry.consumer_release_t = done_t
            ctx.event_sink.emit_span(
                self.call_stk, "bwd", self._bwd_launch_st, done_t,
                gid=self.id, stream=self.stream,
                kind=self.simu_kind, lane=self.simu_lane,
                metadata=self._event_metadata(
                    ctx=ctx, phase="bwd", lifecycle_stage="completion",
                    completion_time=done_t,
                    entry=entry,
                ),
            )
            self._bwd_launch_st = None
            self._bwd_done_t = None
            return True, None
        return False, blk

    def _issue_meta(self, ctx, phase=None):
        """Meta dict for the comm entry issued by _step/_bwd.

        Under an active level topology (ctx.levels set by the runner AND
        fabric.set_level_topology called), collectives additionally carry
        ``crossed_levels``: the topology levels whose links this collective's
        traffic crosses, from the T2 composition — level i is crossed iff
        composition[i] > 1 (a phase exists at that level). This applies to
        net=="inter_node" entries and to net=="levels" entries (net-field
        semantics C): for the latter the fabric engages only when a level
        >= 1 is crossed (traffic leaves the node; a [0]-only crossed list
        is an intra-node phase and engages nothing). Level 0 has no
        level_tail entry — its server is the ToR.
        """
        meta = self._event_metadata(
            ctx=ctx, phase=phase or self._forward_phase())
        levels = getattr(ctx, "levels", None)
        fabric = ctx.fabric
        if (self.net in ("inter_node", "levels") and levels
                and fabric is not None and fabric.level_topology_active()):
            kind = _parse_group_kind(self.id)
            strategy = getattr(ctx, "strategy", None)
            if kind is not None and strategy is not None:
                composition, _ = group_level_span(kind, strategy, levels)
                meta["crossed_levels"] = [
                    i for i, c in enumerate(composition) if c > 1]
        return meta

    def _step(self, t, ctx):
        if self.global_rank is None:
            raise RuntimeError(f"Com {self.id}: global_rank is None")

        if self.fwd_cost == 0 or self.group_size <= 1:
            return True, None

        gid = ("fwd", self.id)
        if gid in self._completed:
            return True, None

        if self._fwd_issue_order is None:
            self._fwd_issue_order = ctx.next_issue_seq()
        if self._fwd_entry_eid is None:
            expected = 2 if self.id.startswith("send_recv-") else self.group_size
            backend_kind = "barrier"
            if self.id.startswith("send_recv-"):
                backend_kind = "p2p"
            elif ctx.merge_lanes and 'default_group' not in self.id:
                backend_kind = "local"
            elif 'default_group' in self.id:
                expected = int(self.id.split('pp_size:')[1])
            self._fwd_entry_eid = ctx.issue_comm_entry(
                rank=self.global_rank,
                gid=gid,
                cost=self.fwd_cost,
                # dp_comm (FSDP) is issued no earlier than the current compute
                # progress: a fwd AG follows its layer's fwd pipeline stage and a
                # bwd RS/AG follows the layer's bwd, instead of firing on the
                # stale dp_comm lane clock (which was left at the fwd-phase tail
                # and pulled all bwd FSDP comm into the fwd phase).
                issue_t=(max(t["dp_comm"], t["comp"]) if "dp_comm" in t else t["comp"]) if self.stream == "dp_comm" else t["comp"],
                stream=self.stream,
                mode="sync",
                backend_kind=backend_kind,
                expected=expected,
                log_call_stk=self.call_stk,
                log_id=self.id,
                meta=self._issue_meta(ctx, phase=self._forward_phase()),
            )
            # Faithful post marker (design doc 9.3): the post happens at issue;
            # the completion span follows from the step wrapper as before.
            self._emit_post_marker(ctx, self._forward_phase(), t["comp"])
            ctx.pump_comm_queue()
        if not ctx.entry_done(self._fwd_entry_eid):
            return False, ("comm_entry", self._fwd_entry_eid)
        end_t = ctx.get_entry_end(self._fwd_entry_eid)
        entry = ctx.get_entry(self._fwd_entry_eid)
        self._fwd_launch_st = self._event_start_t(entry)
        self._fwd_done_t = end_t
        t[self.stream] = max(t[self.stream], end_t)
        # FSDP（dp_comm lane）与计算**部分**重叠：按 ctx.fsdp_exposure_ratio
        # 暴露率推进 comp lane（掩盖率 = 1 − 暴露率，结构推导）。未设 ratio
        # 时保持旧语义（全掩盖，不推 comp）。
        t["comp"] = max(t["comp"], self._dp_comm_push(ctx, t, end_t, self._fwd_launch_st))
        self._completed.add(gid)
        return True, None

    def _bwd(self, t, ctx):
        if self.global_rank is None:
            raise RuntimeError(f"Com {self.id}: global_rank is None")
        if self.bwd_cost == 0 or self.group_size <= 1:
            return True, None
        gid = ("bwd", self.id)
        if gid in self._completed:
            return True, None
        if self._bwd_issue_order is None:
            self._bwd_issue_order = ctx.next_issue_seq()
        if self._bwd_entry_eid is None:
            expected = 2 if self.id.startswith("send_recv-") else self.group_size
            backend_kind = "barrier"
            if self.id.startswith("send_recv-"):
                backend_kind = "p2p"
            elif ctx.merge_lanes and 'default_group' not in self.id:
                backend_kind = "local"
            elif 'default_group' in self.id:
                expected = int(self.id.split('pp_size:')[1])
            self._bwd_entry_eid = ctx.issue_comm_entry(
                rank=self.global_rank,
                gid=gid,
                cost=self.bwd_cost,
                # Same as fwd: bwd FSDP comm must wait for the layer's bwd
                # compute (comp lane), not fire on the stale dp_comm clock.
                issue_t=(max(t["dp_comm"], t["comp"]) if "dp_comm" in t else t["comp"]) if self.stream == "dp_comm" else t["comp"],
                stream=self.stream,
                mode="sync",
                backend_kind=backend_kind,
                expected=expected,
                log_call_stk=self.call_stk,
                log_id=self.id,
                meta=self._issue_meta(ctx, phase="bwd"),
            )
            # Faithful post marker (design doc 9.3): the post happens at issue;
            # the completion span follows from the bwd wrapper as before.
            self._emit_post_marker(ctx, "bwd", t["comp"])
            ctx.pump_comm_queue()
        if not ctx.entry_done(self._bwd_entry_eid):
            return False, ("comm_entry", self._bwd_entry_eid)
        end_t = ctx.get_entry_end(self._bwd_entry_eid)
        entry = ctx.get_entry(self._bwd_entry_eid)
        self._bwd_launch_st = self._event_start_t(entry)
        self._bwd_done_t = end_t
        t[self.stream] = max(t[self.stream], end_t)
        # 同 fwd：FSDP（dp_comm）按暴露率部分推 comp
        t["comp"] = max(t["comp"], self._dp_comm_push(ctx, t, end_t, self._bwd_launch_st))
        self._completed.add(gid)
        return True, None

    def prefill_recompute_fwd(self, recompute_cost_override=None):
        """Clone for recompute forward so the _completed idempotency guard
        doesn't skip re-issue. LeafModel's default returns self, so a
        recomputed comm op re-enters _step with gid already in _completed and
        is dropped — the DES trace then has no recompute-forward comm events
        (CP/MoE a2a recompute segments missing). The clone carries
        recompute_cost as fwd_cost; bwd_cost stays 0 (recompute only runs
        _step, not _bwd). Mirrors AtomModel.prefill_recompute_fwd.

        Recompute cost is the op's own fwd_cost (replaying a communication
        moves the same payload again). An incoming recompute_cost_override
        comes from compute-stage rc modelling (e.g. Permutation's
        recompute_compute_time = permute-only mem) and must NOT shrink a comm
        op's payload — otherwise MoE a2a recompute segments collapse to the
        permute mem time (~2ms) instead of the full alltoall volume (~21ms).
        """
        recompute_cost = self.fwd_cost
        clone = type(self)(
            self.id, self.rank, self.group_size,
            com_buff=None,
            fwd_cost=recompute_cost, bwd_cost=0,
            call_stk=self.call_stk,
            global_rank=self.global_rank,
            stream=self.stream, net=self.net,
            size_bytes=self.size_bytes, group_kind=self.group_kind,
            comm_stage=self.comm_stage, comm_owner=self.comm_owner,
            comm_role=self.comm_role, metadata=self.event_metadata)
        clone.forward_op = "recompute_fwd"
        return clone

    def _blocking_step_impl(self, t, ctx, *, phase):
        if self.global_rank is None:
            raise RuntimeError(f"Com {self.id}: global_rank is None")
        cost = self.fwd_cost if phase == "fwd" else self.bwd_cost
        if cost == 0 or self.group_size <= 1:
            return True, None
        gid = (phase, self.id)
        if gid in self._completed:
            return True, None
        m = max(t["comp"], t["comm"])
        t["comp"] = t["comm"] = m
        ready_t = self._batch_submit_by_gid.get(gid, t[self.stream])
        # Blocking p2p acquires this rank's NIC (and ToR when enabled) at
        # arrival; only first arrivals use ready_t, so retries re-acquire
        # harmlessly (network-fabric design doc 5.3). net=="levels" pairs
        # acquire conservatively here too (the egress node check needs the
        # peer rank, known only at completion; the charge below is exact).
        topology_on = (ctx.fabric is not None
                       and ctx.fabric.level_topology_active()
                       and getattr(ctx, "levels", None))
        if ctx.fabric is not None and (
                self.net == "inter_node" or (self.net == "levels" and topology_on)):
            ready_t = ctx.fabric.acquire(self.global_rank, ready_t)
        first_arrival = gid not in self._blocking_start_by_gid
        done, waiters, end_t = ctx.backend.arrive(gid, self.global_rank, ready_t, 2, cost)
        if first_arrival:
            # Faithful post marker (design doc 9.3): the post happens at the
            # first barrier arrival, whether the op blocks or completes now.
            self._emit_post_marker(ctx, phase, ready_t)
        if not done:
            self._blocking_start_by_gid.setdefault(gid, ready_t)
            return False, ("barrier", gid)
        # Blocking communication should cover the local call interval from the
        # moment this rank enters the communication until the common completion
        # time. Any rendezvous wait is part of the visible blocking comm span.
        event_start_t = self._blocking_start_by_gid.pop(gid, ready_t)
        done_t = end_t
        if phase == "fwd":
            self._fwd_launch_st = event_start_t
            self._fwd_done_t = done_t
        else:
            self._bwd_launch_st = event_start_t
            self._bwd_done_t = done_t
        # Retried blocking p2p ops may observe a cached completion whose end_t
        # is earlier than the rank's current visible time (for example, when a
        # longer sibling op in the same batch finished later). Never move local
        # time backwards on replay.
        fabric_charge = None
        if ctx.fabric is not None and self.net == "inter_node":
            # Both waiters' NICs are charged to the COMMON barrier end_t (not
            # the replay-adjusted local one) by the pending_completions drain,
            # keyed on the gid so retries never double-charge (decision 1).
            crossed = None
            if topology_on and len(waiters) == 2:
                # The pair's crossed levels, computed once here by whichever
                # op observes the completion (the relation is symmetric in
                # the two ranks). The drain calls each waiter's charge_levels
                # with its own rank, so the per-unit index (rank // span_i)
                # covers both endpoints' link servers. Arrival-time acquire
                # above stays NIC+ToR only: the peer rank (and thus the
                # crossed set) is unknown until both waiters arrive.
                crossed = ctx.fabric.level_crossings(waiters[0], waiters[1])
            fabric_charge = (end_t, cost, self.size_bytes or 0, crossed)
        elif ctx.fabric is not None and self.net == "levels" and topology_on:
            # Semantics-C pair: engage the fabric only when the endpoints
            # sit on different nodes (level-0 units); same-node pairs stay
            # on the intra-node fabric and charge nothing. Crossed level
            # servers (>= 1) follow the same endpoint rule as inter_node.
            if len(waiters) == 2:
                rank_a, rank_b = waiters
                span0 = ctx.fabric.level_spans[0]
                if rank_a // span0 != rank_b // span0:
                    fabric_charge = (end_t, cost, self.size_bytes or 0,
                                     ctx.fabric.level_crossings(rank_a, rank_b))
        end_t = max(end_t, t["comp"], t["comm"])
        t["comp"] = t["comm"] = end_t
        self._batch_submit_by_gid.pop(gid, None)
        self._completed.add(gid)
        ctx.pending_completions.append((gid, waiters, end_t, self.stream, fabric_charge))
        return True, None


    
class all_gather(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all_gather'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk, **kwargs)
        # self.call_stk = self.call_stk + '-all_gather'
class all_gatherv(Com):
    """Variable-count all-gather.

    Scheduling semantics are identical to :class:`all_gather`; the distinct
    type keeps the model graph/trace faithful while SystemConfig resolves the
    operation onto the same physical network levels.
    """
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0,
                 bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all_gatherv' + id, rank, group_size, com_buff,
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost,
                         call_stk=call_stk, **kwargs)
class all_gather_fwd(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all_gather'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk, **kwargs)
        # self.call_stk = self.call_stk + '-all_gather'

    def _bwd(self, t, ctx):
        return True

class all_gather_bwd(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all_gather'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)
        # self.call_stk = self.call_stk + '-all_gather'

    def _step(self, t, ctx):
        return True

class reduce_scatter(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('reduce_scatter'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)
        # self.call_stk = self.call_stk + '-reduce_scatter'
class reduce_scatterv(Com):
    """Variable-count reduce-scatter; physically routed through levels."""
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0,
                 bwd_cost=0, call_stk='', **kwargs):
        super().__init__('reduce_scatterv' + id, rank, group_size, com_buff,
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost,
                         call_stk=call_stk, **kwargs)
class all_reduce(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all_reduce'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)
        # self.call_stk = self.call_stk + '-all_reduce'
class all2all(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all2all'+id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)


class alltoallv(Com):
    """Variable-count all-to-all used by token and CP redistributions."""
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0,
                 bwd_cost=0, call_stk='', **kwargs):
        super().__init__('alltoallv' + id, rank, group_size, com_buff,
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost,
                         call_stk=call_stk, **kwargs)


class all2all_fwd(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all2all'+id, rank, group_size, com_buff,
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk, **kwargs)

    def _bwd(self, t, ctx):
        return True


class all2all_bwd(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        super().__init__('all2all'+id, rank, group_size, com_buff,
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk, **kwargs)

    def _step(self, t, ctx):
        return True


class alltoallv_fwd(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0,
                 bwd_cost=0, call_stk='', **kwargs):
        super().__init__('alltoallv' + id, rank, group_size, com_buff,
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost,
                         call_stk=call_stk, **kwargs)

    def _bwd(self, t, ctx):
        return True


class alltoallv_bwd(Com):
    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0,
                 bwd_cost=0, call_stk='', **kwargs):
        super().__init__('alltoallv' + id, rank, group_size, com_buff,
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost,
                         call_stk=call_stk, **kwargs)

    def _step(self, t, ctx):
        return True


class batch_send_recv(Com):
    """One batched point-to-point exchange between two global ranks.

    The batch may contain several tensors, but they share one rendezvous and
    one levels-derived P2P cost. Callers pass the structurally summed payload.
    """
    def __init__(self, id, rank, peer_rank, com_buff=None, fwd_cost=0,
                 bwd_cost=0, call_stk='', global_rank=None, **kwargs):
        pair = sorted((int(global_rank), int(peer_rank)))
        pair_id = f"send_recv-{pair[0]}-{pair[1]}-batch-{id}"
        local_rank = 0 if int(global_rank) == pair[0] else 1
        super().__init__(pair_id, local_rank, 2, com_buff,
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost,
                         call_stk=call_stk, global_rank=global_rank, **kwargs)

    def _step(self, t, ctx):
        return self._blocking_step_impl(t, ctx, phase="fwd")

    def _bwd(self, t, ctx):
        return self._blocking_step_impl(t, ctx, phase="bwd")

class send(Com):
    simu_lane = "pp_detail"

    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        assert (rank==0 and group_size==2)
        super().__init__(id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)

    def _step(self, t, ctx):
        return self._blocking_step_impl(t, ctx, phase="fwd")

    def _bwd(self, t, ctx):
        return self._blocking_step_impl(t, ctx, phase="bwd")

class recv(Com):
    simu_lane = "pp_detail"

    def __init__(self, id, rank, group_size, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', **kwargs):
        assert (rank==1 and group_size==2)
        super().__init__(id, rank, group_size, com_buff, 
                         fwd_cost=fwd_cost, bwd_cost=bwd_cost, call_stk=call_stk,**kwargs)

    def _step(self, t, ctx):
        return self._blocking_step_impl(t, ctx, phase="fwd")

    def _bwd(self, t, ctx):
        return self._blocking_step_impl(t, ctx, phase="bwd")

class recv_prev(recv):
    def __init__(self, id, rank, group_size=2, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', pp_size=1, **kwargs):
        prev_rank = (rank-1)%pp_size
        id = f"send_recv-{prev_rank}-{rank}-{id}"
        local_rank = 1
        super().__init__(id, local_rank, group_size, com_buff, fwd_cost, bwd_cost, call_stk, **kwargs)
        if pp_size<=1:
            self.step = lambda *args:True

class send_next(send):
    def __init__(self, id, rank, group_size=2, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', pp_size=1, **kwargs):
        next_rank = (rank+1)%pp_size
        id = f"send_recv-{rank}-{next_rank}-{id}"
        local_rank = 0
        super().__init__(id, local_rank, group_size, com_buff, fwd_cost, bwd_cost, call_stk, **kwargs)
        if pp_size<=1:
            self.step = lambda *args:True

class recv_next(recv):
    def __init__(self, id, rank, group_size=2, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', pp_size=1, **kwargs):
        next_rank = (rank+1)%pp_size
        id = f"send_recv-{next_rank}-{rank}-{id}"
        local_rank = 1
        super().__init__(id, local_rank, group_size, com_buff, fwd_cost, bwd_cost, call_stk, **kwargs)
        if pp_size<=1:
            self.step = lambda *args:True
class send_prev(send):
    def __init__(self, id, rank, group_size=2, com_buff=None, fwd_cost=0, bwd_cost=0, call_stk='', pp_size=1, **kwargs):
        prev_rank = (rank-1)%pp_size
        id = f"send_recv-{rank}-{prev_rank}-{id}"
        local_rank = 0
        super().__init__(id, local_rank, group_size, com_buff, fwd_cost, bwd_cost, call_stk, **kwargs)
        if pp_size<=1:
            self.step = lambda *args:True


class async_all_gather(LeafModel):
    """Posts an all_gather CommEntry and yields WITHOUT blocking the comp lane.

    Layer-wise FSDP unshard is split into a non-blocking post (this op) and a
    blocking wait (``async_wait_collective``). The post issues the CommEntry,
    pumps the comm queue so the AG runs on the comm lane, then returns
    ``("yield_keep", gid)`` — the op stays queued and the rank is re-pushed,
    so on retry it reports done (already posted) and the queue advances to the
    next op (typically the block's compute), which starts on the comp lane
    while the AG is in flight on the comm lane. ``yield_keep`` (not
    ``yield_done``) is required because these posts sit inside a multi-op
    FwdQue nested in the model's top FwdQue; ``yield_done`` would pop the head
    and, when bubbled up to the outer FwdQue, drop the whole inner sub-queue.
    The matching wait, placed later in the queue, re-syncs the comp lane to the
    AG completion.

    Supports both forward and backward posts (FSDP2 gap analysis doc section
    3.2/3.3). When ``reshard_after_forward=True`` (FULL_SHARD), the same op
    posts in both fwd (unshard for forward) and bwd (re-unshard for backward).
    Per-phase ``_posted`` flags ensure each phase posts exactly once.

    Issue logic mirrors ``Com._step`` (skip when cost==0 / group_size<=1;
    backend_kind ``local`` under merge_lanes) but never blocks and never raises
    ``t["comp"]``. See docs/design_simu_fsdp2_gap_analysis.md section 3.3.
    """
    simu_kind = "comm"

    def __init__(self, id, rank, group_size, fwd_cost=0, bwd_cost=0,
                 global_rank=None, stream="comm", net=None, size_bytes=0,
                 call_stk='', group_kind=None, comm_stage=None,
                 comm_owner=None, comm_role=None, **kwargs):
        super().__init__(event_metadata=kwargs.pop("metadata", None))
        self.call_stk = call_stk + f'-{self.__class__.__name__}'
        self.id = id
        self.rank = rank
        self.group_size = group_size
        self.fwd_cost = fwd_cost
        self.bwd_cost = bwd_cost
        self.global_rank = global_rank
        self.stream = stream
        # Resolved strategy net name + payload size, forwarded into the posted
        # entry's meta; None keeps the entry out of fabric charging.
        self.net = net
        self.size_bytes = size_bytes
        self.group_kind = group_kind
        self.comm_stage = comm_stage
        self.comm_owner = comm_owner
        self.comm_role = comm_role
        self._eid = None
        self._posted_fwd = False
        self._posted_bwd = False
        # A collective may be observed by an inflight wait and by the final
        # optimizer barrier.  Keep completion logging at the operation level
        # so a second wait cannot manufacture a duplicate semantic event.
        self._completion_logged = set()

    def _issue_meta(self, ctx=None, phase=None):
        # Keep the communication entry metadata identical to its eventual
        # completion span.  This matters for level routing and for async FSDP
        # posts: a payload/domain override must not disappear at the queue
        # boundary.
        return self._event_metadata(ctx=ctx, phase=phase)

    def _backend_kind(self, ctx):
        expected = 2 if self.id.startswith("send_recv-") else self.group_size
        if self.id.startswith("send_recv-"):
            backend_kind = "p2p"
        elif ctx.merge_lanes and 'default_group' not in self.id:
            backend_kind = "local"
        elif 'default_group' in self.id:
            backend_kind = "barrier"
            expected = int(self.id.split('pp_size:')[1])
        else:
            backend_kind = "barrier"
        return backend_kind, expected

    def _post(self, t, ctx, phase):
        if self.global_rank is None:
            raise RuntimeError(f"async_all_gather {self.id}: global_rank is None")
        cost = self.fwd_cost if phase == "fwd" else self.bwd_cost
        if cost == 0 or self.group_size <= 1:
            return True, None
        gid = (phase, self.id)
        posted_attr = f"_posted_{phase}"
        if getattr(self, posted_attr):
            return True, None
        backend_kind, expected = self._backend_kind(ctx)
        self._eid = ctx.issue_comm_entry(
            rank=self.global_rank,
            gid=gid,
            cost=cost,
            issue_t=t["comp"],
            stream=self.stream,
            mode="sync",
            backend_kind=backend_kind,
            expected=expected,
            log_call_stk=self.call_stk,
            log_id=self.id,
            meta=self._issue_meta(ctx=ctx, phase=phase),
        )
        # Faithful post marker (design doc 9.3): a zero-duration marker at the
        # moment the op posts; the completion span is emitted by the wait.
        ctx.event_sink.emit_span(
            self.call_stk, phase, t["comp"], t["comp"],
            gid=self.id, stream=self.stream,
            kind="comm", lane=self.simu_lane,
            name=self.call_stk.split("-")[-1] + "-post",
            metadata=self._event_metadata(
                ctx=ctx, phase=phase, lifecycle_stage="post",
                post_time=t["comp"],
            ),
        )
        ctx.pump_comm_queue()
        setattr(self, posted_attr, True)
        return False, ("yield_keep", gid)

    def _step(self, t, ctx):
        return self._post(t, ctx, "fwd")

    def _bwd(self, t, ctx):
        return self._post(t, ctx, "bwd")

    def step(self, t, ctx):
        return self._step(t, ctx)

    def bwd(self, t, ctx):
        return self._bwd(t, ctx)

    def prefill_fwd(self):
        return self


class async_reduce_scatter(LeafModel):
    """Posts a reduce_scatter CommEntry and yields WITHOUT blocking the comp lane.

    The layer-wise FSDP reshard mirror of ``async_all_gather``: the post runs
    in backward (``bwd_cost``) so the RS overlaps with the previous block's
    backward compute on the comp lane. Forward is a no-op (AG is fwd-only).

    Supports per-phase ``_posted`` flags (FSDP2 gap analysis doc section 3.3)
    for robustness, though RS currently only posts in backward.
    """
    simu_kind = "comm"

    def __init__(self, id, rank, group_size, bwd_cost=0, fwd_cost=0,
                 global_rank=None, stream="comm", net=None, size_bytes=0,
                 call_stk='', group_kind=None, comm_stage=None,
                 comm_owner=None, comm_role=None, **kwargs):
        super().__init__(event_metadata=kwargs.pop("metadata", None))
        self.call_stk = call_stk + f'-{self.__class__.__name__}'
        self.id = id
        self.rank = rank
        self.group_size = group_size
        self.fwd_cost = fwd_cost
        self.bwd_cost = bwd_cost
        self.global_rank = global_rank
        self.stream = stream
        self.net = net
        self.size_bytes = size_bytes
        self.group_kind = group_kind
        self.comm_stage = comm_stage
        self.comm_owner = comm_owner
        self.comm_role = comm_role
        self._eid = None
        self._posted_fwd = False
        self._posted_bwd = False
        # See async_all_gather._completion_logged.  Reduce-scatter is commonly
        # waited first by the inflight limiter and again before optimizer.
        self._completion_logged = set()

    def _issue_meta(self, ctx=None, phase=None):
        # Keep domain/owner/group metadata on the posted entry.  Losing it at
        # this queue boundary made async reduce-scatter completion records look
        # like unowned communication to structural trace consumers.
        return self._event_metadata(ctx=ctx, phase=phase)

    def _backend_kind(self, ctx):
        return async_all_gather._backend_kind(self, ctx)

    def _post(self, t, ctx, phase):
        if self.global_rank is None:
            raise RuntimeError(f"async_reduce_scatter {self.id}: global_rank is None")
        cost = self.bwd_cost if phase == "bwd" else self.fwd_cost
        if cost == 0 or self.group_size <= 1:
            return True, None
        gid = (phase, self.id)
        posted_attr = f"_posted_{phase}"
        if getattr(self, posted_attr):
            return True, None
        backend_kind, expected = self._backend_kind(ctx)
        self._eid = ctx.issue_comm_entry(
            rank=self.global_rank,
            gid=gid,
            cost=cost,
            issue_t=t["comp"],
            stream=self.stream,
            mode="sync",
            backend_kind=backend_kind,
            expected=expected,
            log_call_stk=self.call_stk,
            log_id=self.id,
            meta=self._issue_meta(ctx=ctx, phase=phase),
        )
        ctx.event_sink.emit_span(
            self.call_stk, phase, t["comp"], t["comp"],
            gid=self.id, stream=self.stream,
            kind="comm", lane=self.simu_lane,
            name=self.call_stk.split("-")[-1] + "-post",
            metadata=self._event_metadata(
                ctx=ctx, phase=phase, lifecycle_stage="post",
                post_time=t["comp"],
            ),
        )
        ctx.pump_comm_queue()
        setattr(self, posted_attr, True)
        return False, ("yield_keep", gid)

    def _step(self, t, ctx):
        return self._post(t, ctx, "fwd")

    def _bwd(self, t, ctx):
        return self._post(t, ctx, "bwd")

    def step(self, t, ctx):
        return self._step(t, ctx)

    def bwd(self, t, ctx):
        return self._bwd(t, ctx)

    def prefill_bwd(self):
        return self


class async_wait_collective(LeafModel):
    """Blocks the comp lane until a posted async collective completes.

    Pairs with ``async_all_gather`` (forward wait) or ``async_reduce_scatter``
    (backward wait). On entry it checks whether every referenced post op's
    CommEntry is done: if so it raises ``t["comp"]`` to the latest completion
    time (the comp stall that realizes any non-overlapped comm) and returns
    done; otherwise it blocks on ``("comm_entry", eid)`` and the runner re-runs
    it once that entry completes. The op itself emits no comp-anchored span
    (LeafModel.step/bwd are bypassed); the comm-lane completion spans and a
    comp-lane wait span for the stall are emitted here, mirroring the async
    p2p post/wait trace shape (design doc 4.1/9.3).
    """
    simu_kind = "wait"

    def __init__(self, ag_ops, call_stk='', wait_phase=None,
                 consumer_phase=None, metadata=None):
        super().__init__(event_metadata=metadata)
        self.call_stk = call_stk or '-async_wait_collective'
        # ag_ops: a single async_all_gather/async_reduce_scatter instance, or a
        # list (dense + MoE sub-ops of one block are waited together).
        self.ag_ops = list(ag_ops) if isinstance(ag_ops, (list, tuple)) else [ag_ops]
        self._completed = set()
        # ``wait_phase`` is the semantic phase of the collective being
        # completed, which can differ from the DES container phase.  For
        # example, the final layer-wise FSDP wait is stored in an optimizer
        # FwdQue but completes backward reduce-scatter operations.
        self.wait_phase = wait_phase
        # The lifecycle phase being completed and the phase that owns the
        # consumer are deliberately separate.  A backward RS can therefore
        # remain a backward communication event while its final barrier is
        # explicitly attributed to the optimizer consumer.
        self.consumer_phase = consumer_phase

    def _wait(self, t, ctx, phase):
        effective_phase = self.wait_phase or phase
        if effective_phase in self._completed:
            return True, None
        wait_start = t["comp"]
        # First pass: every referenced post must be done; find the latest end.
        end_t = wait_start
        pending_eid = None
        for op in self.ag_ops:
            eid = getattr(op, '_eid', None)
            if eid is None:
                # Post was skipped (cost==0 / group_size<=1): nothing to wait.
                continue
            if not ctx.entry_done(eid):
                pending_eid = eid
                break
            done_t = ctx.get_entry_end(eid)
            if done_t is not None and done_t > end_t:
                end_t = done_t
        if pending_eid is not None:
            return False, ("comm_entry", pending_eid)
        dependency_refs = [
            f"comm/{getattr(op, 'id', 'unknown')}/{effective_phase}"
            for op in self.ag_ops
            if getattr(op, '_eid', None) is not None
        ]
        # A wait is the model-side consumer edge.  Keep the edge on both the
        # consumer marker and the completion span so a trace consumer can
        # reconstruct ``post -> completion -> consumer`` without relying on
        # event name matching or duration ordering.  The optional fields are
        # supplied by the schedule from layer/phase structure; when legacy
        # callers do not provide them, the lifecycle remains explicit but
        # owner-unknown.
        wait_semantic = self._event_metadata()
        consumer_fields = {
            key: wait_semantic.get(key)
            for key in ("consumer_id", "consumer_event", "consumer_phase")
            if wait_semantic.get(key) is not None
        }
        # Emit the comm-lane completion span for each posted op (the AG/RS
        # activity that ran in parallel with compute), plus a comp-lane wait
        # span for the stall (zero-duration when fully overlapped).
        for op in self.ag_ops:
            eid = getattr(op, '_eid', None)
            if eid is None:
                continue
            entry = ctx.get_entry(eid)
            if entry is None or entry.launch_t is None or entry.end_t is None:
                continue
            logged = getattr(op, "_completion_logged", None)
            if logged is None:
                logged = set()
                op._completion_logged = logged
            # Multiple structural waits may observe the same communication
            # lifecycle.  The first wait owns the semantic completion span;
            # later waits are dependency checks only.
            if effective_phase in logged:
                continue
            completion_metadata = op._event_metadata(
                ctx=ctx, phase=effective_phase, lifecycle_stage="completion")
            # The completion remains a communication event in the phase in
            # which it was issued, while the explicit consumer link records
            # which later queue/compute owns the dependency.  ``depends_on``
            # is retained on the completion for portable audit consumers.
            if consumer_fields:
                completion_metadata.update(consumer_fields)
            if dependency_refs:
                completion_metadata["depends_on"] = dependency_refs
            entry.consumer_release_t = end_t
            lifecycle = dict(entry.meta.get("lifecycle") or {})
            lifecycle["consumer_release_time_ms"] = end_t
            lifecycle["time_provenance"] = "simulator_clock"
            entry.meta["lifecycle"] = lifecycle
            completion_metadata = _lifecycle_with_times(
                completion_metadata,
                stage="completion",
                phase=effective_phase,
                entry=entry,
                consumer_release_time=end_t,
            )
            ctx.event_sink.emit_span(
                op.call_stk, effective_phase, entry.launch_t, entry.end_t,
                gid=op.id, stream=op.stream, kind="comm", lane=op.simu_lane,
                metadata=completion_metadata)
            logged.add(effective_phase)
        # A consumer-owned barrier is useful even when the dependency is
        # completely overlapped and therefore has zero stall time.  Emit a
        # zero-duration structural marker in that case; it does not advance
        # any lane or alter the performance result, but makes the lifecycle
        # visible to trace consumers.
        if end_t > wait_start + 1e-12 or self.consumer_phase is not None:
            wait_metadata = self._event_metadata(
                ctx=ctx, phase=effective_phase,
                lifecycle_stage="consumer_release",
                consumer_release_time=end_t,
            )
            defaults = {
                # This is a model/DES dependency declaration: it identifies
                # the lifecycle entries that the consumer barrier observes.
                # It is not inferred from profiler timing or kernel names.
                "dependency_kind": "consumer_barrier",
                "dependency_status": "explicit" if dependency_refs else "implicit_lifecycle",
                "ready_rule": "all_dependencies_complete",
                "overlap_policy": "wait_for_dependencies",
                "overlap_lanes": ["comp"],
            }
            for key, value in defaults.items():
                wait_metadata.setdefault(key, value)
            wait_metadata["depends_on"] = dependency_refs
            # Preserve an explicit stable semantic consumer id supplied by
            # the model; fall back to the display call stack for legacy waits.
            wait_metadata.setdefault("consumer_id", self.call_stk)
            if self.consumer_phase is not None:
                wait_metadata["consumer_phase"] = self.consumer_phase
            if wait_semantic.get("consumer_event") is not None:
                wait_metadata["consumer_event"] = wait_semantic["consumer_event"]
            wait_lifecycle = dict(wait_metadata.get("lifecycle") or {})
            wait_lifecycle["consumer_release_time_ms"] = end_t
            wait_lifecycle["time_provenance"] = "simulator_clock"
            wait_metadata["lifecycle"] = wait_lifecycle
            ctx.event_sink.emit_span(
                self.call_stk, effective_phase, wait_start, end_t,
                kind="wait", lane=None,
                name=("optimizer_gradient_sync_barrier"
                      if self.consumer_phase == "optimizer" else None),
                metadata=wait_metadata)
        t["comp"] = max(t["comp"], end_t)
        self._completed.add(effective_phase)
        return True, None

    def _step(self, t, ctx):
        return self._wait(t, ctx, "fwd")

    def _bwd(self, t, ctx):
        return self._wait(t, ctx, "bwd")

    def step(self, t, ctx):
        return self._step(t, ctx)

    def bwd(self, t, ctx):
        return self._bwd(t, ctx)

    def prefill_fwd(self):
        return self

    def prefill_bwd(self):
        return self


class async_send(LeafModel):
    simu_kind = "comm"

    def __init__(self, id, fwd_cost=0, call_stk='', global_rank=None, stream="comm",
                 net=None, size_bytes=0):
        super().__init__()
        self.call_stk = call_stk + f'{self.call_stk}'
        self.id = id
        self.fwd_cost = fwd_cost
        self.global_rank = global_rank
        self.stream = stream
        # Resolved strategy net name + payload size, forwarded into the posted
        # entry's meta; None keeps the entry out of fabric charging.
        self.net = net
        self.size_bytes = size_bytes
        self._completed = set()
        self._entry_by_gid = {}

    def _step(self, t, ctx, phase="fwd"):
        if self.global_rank is None:
            raise RuntimeError(f"async_send {self.id}: global_rank is None")
        gid = (phase, self.id)
        if gid in self._completed:
            return True, None
        start_t = t["comp"]
        eid = ctx.post_async_send_entry(
            gid=gid,
            rank=self.global_rank,
            post_t=start_t,
            cost=self.fwd_cost,
            stream=self.stream,
            mode="async_send",
            call_stk=self.call_stk,
            log_id=f"{phase}:{self.id}",
            net=self.net,
            size_bytes=self.size_bytes,
            meta=self._event_metadata(),
        )
        self._entry_by_gid[gid] = eid
        self._completed.add(gid)
        return False, ("yield_done", gid)

    def _bwd(self, t, ctx):
        return self._step(t, ctx, phase="bwd")

    def step(self, t, ctx):
        return self._step(t, ctx, phase="fwd")

    def bwd(self, t, ctx):
        return self._bwd(t, ctx)


class async_recv(LeafModel):
    simu_kind = "comm"

    def __init__(self, id, call_stk='', global_rank=None, stream="comm", fwd_cost=0,
                 net=None, size_bytes=0):
        super().__init__()
        self.call_stk = call_stk + f'{self.call_stk}'
        self.id = id
        self.global_rank = global_rank
        self.stream = stream
        self.fwd_cost = fwd_cost
        # Resolved strategy net name + payload size, forwarded into the posted
        # entry's meta; None keeps the entry out of fabric charging.
        self.net = net
        self.size_bytes = size_bytes
        self._launched = set()
        self._entry_by_gid = {}

    def _step(self, t, ctx, phase="fwd"):
        if self.global_rank is None:
            raise RuntimeError(f"async_recv {self.id}: global_rank is None")
        gid = (phase, self.id)
        if gid in self._launched:
            return True, None
        eid = ctx.post_async_recv_entry(
            gid=gid,
            rank=self.global_rank,
            post_t=t["comp"],
            cost=self.fwd_cost,
            stream=self.stream,
            mode="async_recv",
            call_stk=self.call_stk,
            log_id=f"{phase}:{self.id}",
            net=self.net,
            size_bytes=self.size_bytes,
            meta=self._event_metadata(),
        )
        self._entry_by_gid[gid] = eid
        self._launched.add(gid)
        return False, ("yield_done", gid)

    def _bwd(self, t, ctx):
        return self._step(t, ctx, phase="bwd")

    def step(self, t, ctx):
        return self._step(t, ctx, phase="fwd")

    def bwd(self, t, ctx):
        return self._bwd(t, ctx)


class async_recv_prev(async_recv):
    def __init__(self, id, rank, call_stk='', pp_size=1, **kwargs):
        prev_rank = (rank - 1) % pp_size
        id = f"send_recv-{prev_rank}-{rank}-{id}"
        kwargs.setdefault("stream", "pp_fwd")
        super().__init__(id, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True


class async_send_next(async_send):
    def __init__(self, id, rank, fwd_cost=0, call_stk='', pp_size=1, **kwargs):
        next_rank = (rank + 1) % pp_size
        id = f"send_recv-{rank}-{next_rank}-{id}"
        kwargs.setdefault("stream", "pp_fwd")
        super().__init__(id, fwd_cost=fwd_cost, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True


class async_recv_next(async_recv):
    def __init__(self, id, rank, call_stk='', pp_size=1, **kwargs):
        next_rank = (rank + 1) % pp_size
        id = f"send_recv-{next_rank}-{rank}-{id}"
        kwargs.setdefault("stream", "pp_bwd")
        super().__init__(id, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True


class async_send_prev(async_send):
    def __init__(self, id, rank, fwd_cost=0, call_stk='', pp_size=1, **kwargs):
        prev_rank = (rank - 1) % pp_size
        id = f"send_recv-{rank}-{prev_rank}-{id}"
        kwargs.setdefault("stream", "pp_bwd")
        super().__init__(id, fwd_cost=fwd_cost, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True

class async_wait_recv(LeafModel):
    # This op never emits its own span: the visible wait line is produced by
    # SimuContext.emit_async_pair_logs with kind="wait". The ClassVar only
    # declares the op's semantics (design doc 4.1: async_wait_recv -> wait).
    simu_kind = "wait"

    def __init__(self, id, call_stk='', global_rank=None, stream="comm", fwd_cost=0,
                 net=None, size_bytes=0):
        super().__init__()
        self.call_stk = call_stk + f'{self.call_stk}'
        self.id = id
        self.global_rank = global_rank
        self.stream = stream
        self.fwd_cost = fwd_cost
        # Resolved strategy net name + payload size, forwarded into the posted
        # entry's meta; None keeps the entry out of fabric charging.
        self.net = net
        self.size_bytes = size_bytes
        self._completed = set()

    def _step(self, t, ctx, phase="fwd"):
        if self.global_rank is None:
            raise RuntimeError(f"async_wait_recv {self.id}: global_rank is None")
        gid = (phase, self.id)
        if gid in self._completed:
            return True, None
        ready_t = ctx.get_async_ready_t(gid)
        if ready_t is None:
            if not ctx.has_async_posted_send(gid) or not ctx.has_async_posted_recv(gid):
                return False, ("async_wait", gid)
            ready_t = ctx.ensure_async_ready(gid)
            if ready_t is None:
                return False, ("async_wait", gid)
        t["comp"] = max(t["comp"], ready_t)
        self._completed.add(gid)
        return True, None

    def _bwd(self, t, ctx):
        return self._step(t, ctx, phase="bwd")

    def _event_call_stk(self):
        return self.call_stk.replace("async_wait_recv", "async_recv")

    def _emit_async_pair_logs(self, ctx, gid, ready_t, op):
        return

    def step(self, t, ctx):
        gid = ("fwd", self.id)
        if not ctx.has_async_posted_recv(gid):
            eid = ctx.post_async_recv_entry(
                gid=gid,
                rank=self.global_rank,
                post_t=t["comp"],
                cost=self.fwd_cost,
                stream=self.stream,
                mode="async_recv",
                call_stk=self._event_call_stk(),
                log_id=f"fwd:{self.id}",
                net=self.net,
                size_bytes=self.size_bytes,
                meta=self._event_metadata(),
            )
            return False, ("yield_keep", gid)
        ok, blk = self._step(t, ctx, phase="fwd")
        if ok:
            ready = ctx.get_async_ready_t(gid) or t[self.stream]
            self._emit_async_pair_logs(ctx, gid, ready, "fwd")
            return True, None
        return False, blk

    def bwd(self, t, ctx):
        gid = ("bwd", self.id)
        if not ctx.has_async_posted_recv(gid):
            eid = ctx.post_async_recv_entry(
                gid=gid,
                rank=self.global_rank,
                post_t=t["comp"],
                cost=self.fwd_cost,
                stream=self.stream,
                mode="async_recv",
                call_stk=self._event_call_stk(),
                log_id=f"bwd:{self.id}",
                net=self.net,
                size_bytes=self.size_bytes,
                meta=self._event_metadata(),
            )
            return False, ("yield_keep", gid)
        ok, blk = self._bwd(t, ctx)
        if ok:
            ready = ctx.get_async_ready_t(gid) or t[self.stream]
            self._emit_async_pair_logs(ctx, gid, ready, "bwd")
            return True, None
        return False, blk


class async_wait_recv_prev(async_wait_recv):
    def __init__(self, id, rank, call_stk='', pp_size=1, **kwargs):
        prev_rank = (rank - 1) % pp_size
        id = f"send_recv-{prev_rank}-{rank}-{id}"
        kwargs.setdefault("stream", "pp_fwd")
        super().__init__(id, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True


class async_wait_recv_next(async_wait_recv):
    def __init__(self, id, rank, call_stk='', pp_size=1, **kwargs):
        next_rank = (rank + 1) % pp_size
        id = f"send_recv-{next_rank}-{rank}-{id}"
        kwargs.setdefault("stream", "pp_bwd")
        super().__init__(id, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True


class sync_send(async_send):
    simu_kind = "comm"

    def _step(self, t, ctx, phase="fwd"):
        if self.global_rank is None:
            raise RuntimeError(f"sync_send {self.id}: global_rank is None")
        gid = (phase, self.id)
        if not ctx.has_async_posted_send(gid):
            eid = ctx.post_async_send_entry(
                gid=gid,
                rank=self.global_rank,
                post_t=t["comp"],
                cost=self.fwd_cost,
                stream=self.stream,
                mode="sync_send",
                call_stk=self.call_stk,
                log_id=f"{phase}:{self.id}",
                net=self.net,
                size_bytes=self.size_bytes,
                meta=self._event_metadata(),
            )
            self._entry_by_gid[gid] = eid
        ready_t = ctx.ensure_async_ready(gid)
        if ready_t is None:
            return False, ("comm_entry", self._entry_by_gid[gid])
        t["comp"] = max(t["comp"], ready_t)
        self._completed.add(gid)
        return True, None

    def _bwd(self, t, ctx):
        return self._step(t, ctx, phase="bwd")


class sync_send_next(async_send_next):
    def __init__(self, *args, **kwargs):
        kwargs["stream"] = "comm"
        super().__init__(*args, **kwargs)


class sync_send_prev(async_send_prev):
    def __init__(self, *args, **kwargs):
        kwargs["stream"] = "comm"
        super().__init__(*args, **kwargs)


class sync_wait_recv(async_wait_recv):
    # Overrides async_wait_recv's "wait": sync_wait_recv runs on stream "comm"
    # and its visible span is the recv line classified as comm (legacy mapping).
    simu_kind = "comm"

    def _step(self, t, ctx, phase="fwd"):
        if self.global_rank is None:
            raise RuntimeError(f"sync_wait_recv {self.id}: global_rank is None")
        gid = (phase, self.id)
        if gid in self._completed:
            return True, None
        if not ctx.has_async_posted_recv(gid):
            eid = ctx.post_async_recv_entry(
                gid=gid,
                rank=self.global_rank,
                post_t=t["comp"],
                cost=self.fwd_cost,
                stream=self.stream,
                mode="sync_recv",
                call_stk=self._event_call_stk(),
                log_id=f"{phase}:{self.id}",
                net=self.net,
                size_bytes=self.size_bytes,
                meta=self._event_metadata(),
            )
        ready_t = ctx.ensure_async_ready(gid)
        if ready_t is None:
            return False, ("comm_entry", ctx.get_async_recv_eid(gid))
        t[self.stream] = max(t[self.stream], ready_t)
        t["comp"] = max(t["comp"], ready_t)
        self._completed.add(gid)
        return True, None

    def step(self, t, ctx):
        return self._step(t, ctx, phase="fwd")

    def bwd(self, t, ctx):
        return self._step(t, ctx, phase="bwd")

    def _event_call_stk(self):
        return self.call_stk.replace("sync_wait_recv", "sync_recv")


class sync_wait_recv_prev(sync_wait_recv):
    def __init__(self, id, rank, call_stk='', pp_size=1, **kwargs):
        kwargs["stream"] = "comm"
        prev_rank = (rank - 1) % pp_size
        id = f"send_recv-{prev_rank}-{rank}-{id}"
        super().__init__(id, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True


class sync_wait_recv_next(sync_wait_recv):
    def __init__(self, id, rank, call_stk='', pp_size=1, **kwargs):
        kwargs["stream"] = "comm"
        next_rank = (rank + 1) % pp_size
        id = f"send_recv-{next_rank}-{rank}-{id}"
        super().__init__(id, call_stk=call_stk, **kwargs)
        if pp_size <= 1:
            self.step = lambda *args: True
COM_BUFF={}
COM_BUFF=None
# COM_BUFF = Manager.dict()

def get_comm_group(strategy):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    rank_info = get_rank_group(rank, strategy)
    group_name  = [
        "tp_group_id",
        "cp_group_id",
        "pp_group_id",
        "dp_group_id",
        "dp_cp_group_id",
        "ep_group_id",
        "edp_group_id",
    ]
    local_group_id = {k:rank_info[k] for k in group_name}
    local_group = {k:[] for k in group_name}
    for i in range(size):
        rank_info_i = get_rank_group(i, strategy)
        for name in group_name:
            if rank_info_i[name] == local_group_id[name]:
                local_group[name].append(i)
    group = comm.Get_group()
    comm_group = {k.split('_id')[0]:comm.Create(group.Incl(ranks)) for k, ranks in local_group.items()}
    # for k,v in comm_group.items():
    #     print(f"{k} group={v}, size={v.Get_size()}")
    # comm_group = {k.split('_id')[0]:sub_comm.Create_group(ranks) for k, ranks in local_group.items()}
    return comm_group
