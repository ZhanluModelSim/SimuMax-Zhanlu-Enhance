"""Simulator replay orchestration helpers."""

from __future__ import annotations

import math
import os
import time
import pickle
import json
from types import SimpleNamespace

from simumax.core.base_struct import (
    BarrierBackend,
    InputOutputInfo,
    NetworkFabric,
    PathDebugContext,
    SimuContext,
    SimuSystem,
    SimuThread,
    TensorSize,
)
from simumax.core.generate_tracing import write_trace_file
from simumax.core.simu_events import write_debug_log
from simumax.core.simu_artifacts import (
    append_memory_events_to_trace,
    export_simu_memory_artifacts,
    should_enable_simu_memory_timeline,
)
from simumax.core.simu_memory import SimuMemoryTracker
from simumax.core.fsdp_summary import summarize_fsdp_trace
from simumax.core.transformer.pipeline_schedule import OptimizerSimulator, PpSchedule
from simumax.core.utils import get_pp_stage_representative_rank, get_rank_group


def _solve_fsdp_exposure(C, S, F, alpha_cover=1.0):
    """FSDP 暴露率（1 − 掩盖率）从结构量解方程（治本模型，无 trace）：

        T = C + S + F × (1 − α_cover × C/T)

    C=计算时间、S=串行通信（a2a+sync，100% 暴露）、F=fsdp 通信 dur
    （传输 + 启动，与计算部分重叠）。掩盖率 = α_cover × C/T，
    α_cover 是通信-计算重叠效率（fsdp 集中在计算密集段时 >1，16p 实测 ~1.16）。
    解析解正根：

        T = [(C+S+F) + sqrt((C+S+F)² − 4·α_cover·F·C)] / 2，暴露率 = 1 − α_cover·C/T

    全部从结构量推导（DES lane 时间 + 通信分类），对任意配置自动适用。
    """
    if F <= 0:
        return 0.0
    a = C + S + F
    disc = a * a - 4.0 * alpha_cover * F * C
    if disc <= 0:
        return 0.0
    T = (a + math.sqrt(disc)) / 2.0
    if T <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - alpha_cover * C / T))


def _serial_comm_dur(events):
    """DES 事件里非 dp_comm（串行暴露）通信的墙钟 dur 总和（秒）。

    dp_comm（FSDP）与计算重叠、部分掩盖；a2a/sync/辅助走 comp lane、
    100% 暴露，构成方程里的 S。
    """
    total = 0.0
    for e in events:
        if e.kind != "comm":
            continue
        if e.stream == "dp_comm":
            continue
        total += e.cost / 1e3  # SimuEvent.cost 是 ms
    return total


def _fsdp_dur(events):
    """FSDP 通信 dur 累计（秒）——方程里的 F。

    只统计 fsdp_* 事件（fsdp_ag / fsdp_bwd_ag / fsdp_rs）；dp_comm lane 上
    的模型级辅助 all_gather/reduce_scatter（model_embed_ag 等）不属于 fsdp
    权重/梯度通信，不计入 F。
    """
    total = 0.0
    for e in events:
        if e.kind == "comm" and e.stream == "dp_comm":
            name = e.name or ""
            if name.startswith("fsdp_"):
                total += e.cost / 1e3
    return total


def run_simulation(perf_model, save_path, merge_lanes=True):
    """Run simulator replay for a configured PerfLLM-like object.

    Two-pass: pass 1 runs with FSDP fully hidden (legacy masking) to read the
    lane clocks and derive C/S/F; the structural exposure equation solves the
    FSDP exposure ratio; pass 2 re-runs with that ratio (FSDP partially pushes
    comp) and writes the final trace.
    """

    model_base = perf_model.model_chunk_dict["first_stage_chunk"]
    # Resource lanes are computed once from the system config (design doc 4.2)
    # and shared by the SimuSystem, every SimuThread lane dict, and the ctx.
    resource_lanes = perf_model.system.simu_resource_lanes()
    t0 = time.time()
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, "log.log")
    output_json_path = os.path.join(save_path, "tracing_logs.json")

    # Network fabric servers (network-fabric design doc sections 5-6);
    # None = off, which reproduces the current behavior. Computed once,
    # shared by both passes.
    fabric = None
    levels = None
    model = getattr(perf_model.system, "fabric_model", None)
    if model in ("nic", "nic+tor"):
        topo = perf_model.system.topology or {}
        share = topo.get("tor_node_share", "auto")
        if share == "auto":
            share = perf_model.system.num_per_node if merge_lanes else 1
        # ToR capacity defaults to the node uplink (inter_node bandwidth);
        # set topology.tor_capacity_gbps below that to model oversubscription.
        tor_capacity = topo.get("tor_capacity_gbps")
        if tor_capacity is None:
            inter = perf_model.system.networks.get("inter_node")
            tor_capacity = inter.bandwidth.gbps if inter is not None else None
        fabric = NetworkFabric(
            perf_model.system.num_per_node,
            tor_enabled=(model == "nic+tor"),
            tor_node_share=share,
            tor_capacity_gbps=tor_capacity,
        )
    elif model == "nic+levels":
        # Hierarchical fabric (hierarchical-network design doc section 8):
        # per-GPU NIC servers plus one logical link server per (level, unit).
        # topology["levels"] is required by the SystemConfig sanity check.
        levels = perf_model.system.topology["levels"]
        # Per-level link capacity in gbps, resolved from each level's net
        # profile (level["net"] -> networks[net].bandwidth.gbps), with
        # topology-kind-aware convergence (design doc Part C, section 5.6):
        # CLOS levels divide by convergence_ratio; FullMesh levels keep
        # full bandwidth (the level server is pass-through).
        level_capacities = []
        for level in levels:
            net_bw = perf_model.system.networks[level["net"]].bandwidth.gbps
            kind = level.get("kind", "clos")
            conv = level.get("convergence_ratio", 1.0)
            if kind == "clos" and conv > 1.0:
                net_bw /= conv
            level_capacities.append(net_bw)
        fabric = NetworkFabric(perf_model.system.num_per_node)
        fabric.set_level_topology(levels, level_capacities, merge_lanes)

    if merge_lanes:
        simu_ranks = perf_model.strategy.pp_size
    else:
        simu_ranks = perf_model.strategy.world_size

    def _run_des(exposure):
        """Build ctx + per-rank threads and run one DES pass.

        exposure: FSDP exposure ratio (None = legacy full masking: dp_comm
        never pushes comp). The ctx, SimuSystem and threads are rebuilt each
        pass so no event-sink / lane state leaks between passes.
        """
        nonlocal model_base
        simu = SimuSystem(resource_lanes=resource_lanes)
        ctx = SimuContext(BarrierBackend(), merge_lanes=merge_lanes, log_path=log_path,
                          resource_lanes=resource_lanes)
        # Phase C virtual waiters (network-fabric design doc section 8)
        ctx.collective_skew = getattr(perf_model.strategy, "collective_skew", None)
        ctx.strategy = perf_model.strategy
        ctx.num_per_node = perf_model.system.num_per_node
        if exposure is not None:
            ctx.fsdp_exposure_ratio = exposure
        if should_enable_simu_memory_timeline(perf_model.strategy, perf_model._vp_size()):
            ctx.memory_tracker = SimuMemoryTracker()
        ctx.fabric = fabric
        # Level routing context of the DES; set only under "nic+levels". A
        # topology["levels"] list may still exist in the config for the
        # analytical levels cost path (net field "auto") — fabric charging
        # stays off unless fabric_model is set.
        ctx.levels = levels

        for rank_i in range(simu_ranks):
            rank = (
                get_pp_stage_representative_rank(rank_i, perf_model.strategy)
                if merge_lanes
                else rank_i
            )
            thread = SimuThread(rank=rank, lanes=resource_lanes)

            args = SimpleNamespace(thread_state=thread.thread_state, rank=rank, microbatch=0)
            rank_info = get_rank_group(rank, model_base.strategy)
            if rank_info["pp_rank"] == 0:
                model_base = perf_model.model_chunk_dict["first_stage_chunk"]
                model_name = "first_stage_chunk"
                stage_key = "first_stage_chunk"
            elif rank_info["pp_rank"] < model_base.strategy.pp_size - 1:
                model_base = perf_model.model_chunk_dict["middle_stage_chunk"]
                model_name = "middle_stage_chunk"
                stage_key = "middle_stage_chunk"
            else:
                model_base = perf_model.model_chunk_dict["last_stage_chunk"]
                model_name = "last_stage_chunk"
                stage_key = "last_stage_chunk"

            # Root fix for DES comm events missing input_info: the DES prefill path
            # (model.prefill()) never sets input_info, so cost leaves that read
            # self.input_info (e.g. HybridAttentionBlock._get_shared_cp_a2a_stage_specs,
            # CP a2a) see None and silently produce no comm ops. analysis() has them
            # because _run() calls forward first (perf_llm.py:3409). Replicate that
            # here: set input_info from the same per-rank-token derivation so the DES
            # trace matches analysis for every input_info-dependent op (not just CP a2a).
            if getattr(model_base, 'input_info', None) is None:
                sc = perf_model.strategy
                mc = perf_model.model_config
                td = getattr(mc, 'trunk_cp_divisor', 1)
                per_rank_cp = max(1, sc.cp_size // td)
                per_rank_tokens = max(1, sc.seq_len // per_rank_cp)
                fake_input = InputOutputInfo(
                    tensors=[TensorSize(shape=(sc.micro_batch_size, per_rank_tokens))]
                )
                # Must be a real PathDebugContext (has .parent) — a plain
                # SimpleNamespace makes forward() raise AttributeError that DES
                # swallows, so input_info never gets set and CP a2a ops are skipped.
                fake_pdc = PathDebugContext()
                fake_pdc.path_list = []
                model_base(fake_input, fake_pdc)

            vp_size = perf_model._vp_size()
            if vp_size > 1 and perf_model.vpp_stage_chunk_names.get(stage_key):
                stage_models = [
                    perf_model.vpp_chunk_dict[name]
                    for name in perf_model.vpp_stage_chunk_names[stage_key]
                ]
            else:
                stage_models = [model_base]

            pp_simu = PpSchedule(perf_model.strategy, perf_model.system, stage_models)
            if ctx.memory_tracker is not None:
                stage_static_bytes = sum(model.get_model_info().all for model in stage_models)
                # FSDP AG buffer transient memory (design_simu_fsdp_mem_mfu_fix.md
                # Part A.3.2): set as a static offset on top of sharded params.
                # FULL_SHARD layer-wise: (1+prefetch) × per_block_full_params
                # SHARD_GRAD_OP / model-wise: full_chunk_full_params
                ag_buffer_bytes = 0
                if perf_model.strategy.zero_state >= 3:
                    fsdp_mode = getattr(perf_model.strategy, 'fsdp_mode', 'model-wise')
                    reshard = getattr(perf_model.strategy, 'reshard_after_forward', True)
                    prefetch = getattr(perf_model.strategy, 'fsdp_prefetch_layers', 1)
                    dp_gs = perf_model.strategy.fsdp_dense_group_size
                    edp_gs = perf_model.strategy.fsdp_moe_group_size
                    mi = stage_models[0].get_model_info()
                    if fsdp_mode == 'layer-wise' and reshard:
                        layer_num = getattr(stage_models[0], 'layer_num', 1)
                        block_mi = (stage_models[0].layer_0.get_model_info()
                                    if hasattr(stage_models[0], 'layer_0') else mi)
                        per_block = (block_mi.dense_weight_bytes * dp_gs
                                     + block_mi.moe_weight_bytes * edp_gs)
                        ag_buffer_bytes = (1 + prefetch) * per_block
                    else:
                        ag_buffer_bytes = (mi.dense_weight_bytes * dp_gs
                                           + mi.moe_weight_bytes * edp_gs)
                ctx.memory_tracker.init_rank(rank, stage_static_bytes)
                if ag_buffer_bytes > 0:
                    ctx.memory_tracker._transient_bytes[rank] = int(ag_buffer_bytes)
                    # Record the initial transient allocation
                    total = stage_static_bytes + int(ag_buffer_bytes)
                    ctx.memory_tracker._append_counter(
                        rank, 0.0, total, "fwd", "fsdp_ag_buffer",
                        "transient_init", "fsdp")

            thread.job = pp_simu.prefill_batch(args, com_buff=None)

            op_block = OptimizerSimulator(perf_model, model_name)
            op_block.prefill(args, com_buff=None)
            # FSDP tail wiring (docs/design_simu_zero3_fsdp.md sections 4.2/5.2):
            # - layer-wise: per-block AG/RS live inside LLMBlock prefill_fwd/bwd, so
            #   the OptimizerSimulator tail is just the optimizer step (appended).
            # - model-wise: unshard (all-gather params) is prepended before the PP
            #   forward; reshard (reduce-scatter grads) + optim_step is appended
            #   after the PP backward.
            # - otherwise (zero_state <= 1): the legacy ZeRO-1 tail (RS -> barrier ->
            #   optim -> AG) is appended as a single block.
            if getattr(perf_model.strategy, 'fsdp_mode', 'model-wise') == 'layer-wise' \
                    and perf_model.strategy.zero_state >= 3:
                thread.job.append(op_block.prefill_step_only_fwd())  # optim_step only, no AG/RS
            elif getattr(perf_model.strategy, 'fsdp_mode', 'model-wise') == 'model-wise' \
                    and perf_model.strategy.zero_state >= 3:
                thread.job.insert(0, op_block.prefill_unshard_fwd())  # Phase 1
                thread.job.append(op_block.prefill_reshard_step_fwd())
            else:
                thread.job.append(op_block.prefill_fwd())  # legacy

            simu.threads.append(thread)

        simu.simu(ctx)
        if os.environ.get('DES_DEBUG'):
            for th in simu.threads:
                print(f"[DES] { {k:round(v,3) for k,v in th.t.items()} }", flush=True)
        return simu, ctx

    # One structural DES pass. Explicit async post/wait edges, prefetch
    # distance, and max-inflight limits determine communication exposure.
    simu, ctx = _run_des(None)

    print("wall time", time.time() - t0)

    write_debug_log(ctx.event_sink.events, log_path)
    write_trace_file(ctx.event_sink.events, output_json_path)
    if ctx.memory_tracker is not None:
        append_memory_events_to_trace(output_json_path, ctx.memory_tracker)
        export_simu_memory_artifacts(save_path, ctx.memory_tracker, pickle_module=pickle)

    # FSDP communication summary (FSDP2 gap analysis doc section 3.7):
    # aggregate per-block AG/RS durations, exposed time, and overlap
    # statistics from the DES trace. Only written when FSDP events exist.
    fsdp_summary = summarize_fsdp_trace(output_json_path, save_path=save_path)
    if fsdp_summary is not None:
        t = fsdp_summary["total"]
        print(f"FSDP summary: comm={t['total_comm_time']:.1f}us "
              f"exposed={t['exposed_time']:.1f}us "
              f"overlap={t['overlap_percentage']:.1f}%")

    # DES wall-clock duration = max clock over every thread and resource lane.
    # Communication exposure comes from graph dependencies in the single pass.
    des_wall_ms = max(max(th.t.values()) for th in simu.threads)
    des_summary = {"duration_time_per_iter_ms": des_wall_ms}
    des_summary_path = os.path.join(save_path, "des_summary.json")
    with open(des_summary_path, "w", encoding="utf-8") as f:
        json.dump(des_summary, f, indent=2)
    print(f"[DES] aligned wall-clock = {des_wall_ms / 1e3:.4f} s "
          f"-> {des_summary_path}")
    return des_summary
