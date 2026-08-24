"""Portable, forward-derived communication plans.

This module is deliberately independent from profiler data.  A plan is built
from model-generated DES event metadata and/or the forward derivation records
emitted by :class:`SystemConfig`.  It describes a semantic collective call and
the generic physical facts that SimuMax used to cost it; it does not replay
CANN/HCCL kernel names or use measured duration as an input.

The schema is intentionally JSON-first.  Keeping the builder here (instead of
spreading serialization logic through the DES and trace writers) gives trace
consumers one stable contract while preserving the existing timing formulas.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


COMM_PLAN_SCHEMA_VERSION = "comm_plan_v1"


def _get(value: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dataclass event or a mapping."""

    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _event_cost(value: Any) -> float:
    raw = _get(value, "cost")
    if raw is None:
        raw = _get(value, "cost_ms", 0.0)
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _event_rank(value: Any) -> Optional[int]:
    raw = _get(value, "rank")
    if isinstance(raw, str):
        match = re.search(r"(\d+)$", raw)
        raw = match.group(1) if match else None
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _event_interval(value: Any) -> Optional[Tuple[float, float]]:
    start = _get(value, "st")
    end = _get(value, "ed")
    if start is None:
        start = _get(value, "st_ms")
    if end is None:
        end = _get(value, "ed_ms")
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _interval_intersection_ms(
        left: Sequence[Tuple[float, float]],
        right: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    for start, end in left:
        for other_start, other_end in right:
            total += max(0.0, min(end, other_end) - max(start, other_start))
    return total


def _simulated_overlap(rows: Sequence[Any]) -> Dict[str, Any]:
    """Compute separate event-accounting and occupancy masking metrics.

    ``event_duration_sum_ms`` is the sum of positive communication/wait
    spans, while ``rank_union_duration_ms`` is the occupied interval union on
    the selected rank.  The latter is the only metric used for the simulated
    exposed occupancy because overlapping DES spans must not be counted twice.
    Keeping both values prevents a later validator from comparing a union on
    the simulated side with a raw kernel-duration sum on the measured side.
    """

    comm_by_rank: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    compute_by_rank: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        rank = _event_rank(row)
        interval = _event_interval(row)
        if rank is None or interval is None:
            continue
        kind = _get(row, "kind")
        if kind in ("comm", "wait") and not str(_get(row, "name") or "").endswith("-post"):
            comm_by_rank[rank].append(interval)
        elif kind in ("compute", "fused"):
            compute_by_rank[rank].append(interval)
    per_rank = []
    for rank in sorted(comm_by_rank):
        comm_union = _merge_intervals(comm_by_rank[rank])
        compute_union = _merge_intervals(compute_by_rank.get(rank, []))
        event_sum = sum(end - start for start, end in comm_by_rank[rank])
        union_duration = sum(end - start for start, end in comm_union)
        overlap = _interval_intersection_ms(comm_union, compute_union)
        event_overlap = sum(
            _interval_intersection_ms([interval], compute_union)
            for interval in comm_by_rank[rank]
        )
        per_rank.append({
            "rank": rank,
            # Keep the legacy field as the occupancy-union value, but expose
            # its definition explicitly alongside the event accounting sum.
            "raw_duration_ms": union_duration,
            "raw_duration_definition": "rank_union_of_comm_and_wait_spans",
            "event_duration_sum_ms": event_sum,
            "rank_union_duration_ms": union_duration,
            "overlap_with_compute_ms": overlap,
            "exposed_duration_ms": max(0.0, union_duration - overlap),
            "event_exposed_duration_sum_ms": max(0.0, event_sum - event_overlap),
        })
    if not per_rank:
        return {
            "raw_duration_ms": None,
            "raw_duration_definition": "unknown",
            "event_duration_sum_ms": None,
            "rank_union_duration_ms": None,
            "overlap_with_compute_ms": None,
            "exposed_duration_ms": None,
            "event_exposed_duration_sum_ms": None,
            "overlap_ratio": None,
            "metric": "max_rank_union",
            "status": "unknown",
        }
    selected = max(per_rank, key=lambda row: row["raw_duration_ms"])
    raw = selected["rank_union_duration_ms"]
    overlap = selected["overlap_with_compute_ms"]
    return {
        **selected,
        "overlap_ratio": overlap / raw if raw > 0 else 0.0,
        "metric": "max_rank_union",
        "status": "known",
        "per_rank": per_rank,
    }


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or ""))
    return token.strip("_") or "unknown"


def _normalised_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def collective_algorithm(op_name: Any, group_size: Optional[int]) -> Tuple[str, Optional[int]]:
    """Return the generic algorithm family and structural stage count.

    Stage counts are algorithm facts, not observations.  They are intentionally
    independent of the HCCL kernel implementation chosen at runtime.
    """

    token = _normalised_name(op_name)
    n = int(group_size) if group_size is not None else None
    if token in {"p2p", "send", "recv", "sendprev", "sendnext", "recvprev", "recvnext"}:
        return "point_to_point", 1
    if "allreduce" in token or token in {"synchallreduce", "reduce"}:
        return "ring", 2 * max(0, n - 1) if n is not None else None
    if "alltoall" in token or "all2all" in token or "a2a" in token:
        return "pairwise_exchange", max(0, n - 1) if n is not None else None
    if "reducescatter" in token or token in {"rs", "modelmoers", "fsdprs"}:
        # Ring reduce-scatter has one reduce/send phase per peer.  The second
        # (all-gather) half belongs to all-reduce, not reduce-scatter.
        return "ring", max(0, n - 1) if n is not None else None
    if "allgather" in token or token in {"ag", "modelmoeag", "fsdpag"}:
        return "ring", max(0, n - 1) if n is not None else None
    return "unknown", None


def _owner_for_group(group_kind: Any, stage: Any = None) -> Optional[str]:
    token = str(stage or "").lower()
    if "dispatch" in token:
        return "moe_dispatch"
    if "combine" in token:
        return "moe_combine"
    if "router" in token:
        return "moe_router"
    return {
        "cp": "attention_cp",
        "ep": "moe_ep",
        "dp_cp": "fsdp_dense",
        "edp": "fsdp_moe",
        "dp": "fsdp_dense",
        "tp": "tensor_parallel",
        "etp": "expert_tensor_parallel",
        "pp": "pipeline",
    }.get(str(group_kind).lower() if group_kind is not None else "")


def _collective_family(value: Any) -> str:
    token = _normalised_name(value)
    if "alltoall" in token or "all2all" in token or "a2a" in token:
        return "alltoallv"
    if "allreduce" in token or "synchallreduce" in token:
        return "all_reduce"
    if "reducescatter" in token or token in {"rs", "modelmoers", "fsdprs"}:
        return "reduce_scatter"
    if "allgather" in token or token in {"ag", "modelmoeag", "fsdpag"}:
        return "all_gather"
    if token in {"p2p", "send", "recv"} or "sendrecv" in token:
        return "p2p"
    return str(value or "communication")


def _event_collective_family(event: Any) -> str:
    """Infer a generic collective family from model metadata only.

    Communication stage labels such as ``Attention_FWD_CP2_Q_redist`` are
    semantic roles, not operation names.  The generated collective id/name is
    therefore checked first, while the stage remains a fallback for older
    call-sites that did not expose an explicit id.
    """

    candidates = [
        _get(event, "comm_id"),
        _get(event, "name"),
        _get(event, "comm_role"),
        _get(event, "comm_stage"),
    ]
    known = {"alltoallv", "all_reduce", "reduce_scatter", "all_gather", "p2p"}
    for candidate in candidates:
        family = _collective_family(candidate)
        if family in known:
            return family
    return _collective_family(next((candidate for candidate in candidates if candidate), None))


def _stage_direction(stage: Any) -> Optional[str]:
    token = str(stage or "").lower()
    if "bwd" in token or "backward" in token or "grad" in token:
        return "bwd"
    if "fwd" in token or "forward" in token:
        return "fwd"
    return None


def _source_map(system: Any) -> Dict[str, Any]:
    sources = dict(getattr(system, "profile_sources", {}) or {}) if system else {}
    return {
        "model_strategy": "model_config+strategy_config",
        "hardware": sources.get("hardware", "system_config.hardware_spec"),
        "topology": sources.get("topology", "system_config.topology"),
        "cann_runtime": sources.get("cann", "system_config.cann_runtime"),
        "hccl_runtime": sources.get("hccl_runtime", "system_config.hccl_runtime"),
        "algorithm": "generic_collective_definition",
    }


def _provenance(system: Any) -> Dict[str, Any]:
    return {
        "sources": _source_map(system),
        "performance_observations_used_as_parameters": False,
        "measured_results_role": "validation_only",
    }


def _status(unknown_fields: Sequence[str]) -> str:
    if not unknown_fields:
        return "known"
    return "partial" if len(unknown_fields) < 4 else "unknown"


def _runtime_facts(system: Any, op_name: Any, group_size: Optional[int],
                   payload_bytes: Optional[int], active_levels: int = 1) -> Dict[str, Any]:
    """Derive runtime facts from the HCCL profile without side effects."""

    if system is None:
        return {"status": "unknown", "unknown_reason": "system_profile_missing"}
    cfg = {}
    getter = getattr(system, "_hccl_network_config", None)
    if callable(getter):
        cfg = dict(getter() or {})
    runtime_cfg = dict(cfg.get("call_runtime", {}) or {})
    compute_cfg = {}
    compute_getter = getattr(system, "_cann_compute_config", None)
    if callable(compute_getter):
        compute_cfg = dict(compute_getter() or {})
    if group_size is None or payload_bytes is None:
        return {
            "status": "partial",
            "call_runtime_overhead_us": None,
            "algorithm_stages": None,
            "runtime_task_count": None,
            "unknown_reason": "group_or_payload_missing",
        }
    algorithm, stages = collective_algorithm(op_name, group_size)
    if stages is None:
        return {
            "status": "partial",
            "algorithm": algorithm,
            "algorithm_stages": None,
            "runtime_task_count": None,
            "call_runtime_overhead_us": None,
            "unknown_reason": "algorithm_stage_count_missing",
        }
    # Match the canonical SystemConfig runtime formula.  Missing optional
    # runtime keys may use declared analytical defaults, but the output keeps
    # the assumption list so a consumer does not mistake it for a measured
    # runtime observation.
    assumptions = []
    default_launch = compute_cfg.get("kernel_launch_latency_us")
    launch = runtime_cfg.get("call_launch_latency_us", default_launch)
    if "call_launch_latency_us" not in runtime_cfg:
        assumptions.append("call_launch_latency_us:compute_default")
    task_launch = runtime_cfg.get("task_launch_latency_us", launch)
    if "task_launch_latency_us" not in runtime_cfg:
        assumptions.append("task_launch_latency_us:call_default")
    tasks_per_stage = runtime_cfg.get("tasks_per_stage", 1)
    if "tasks_per_stage" not in runtime_cfg:
        assumptions.append("tasks_per_stage:portable_default_1")
    chunk_bytes = runtime_cfg.get("descriptor_chunk_bytes", 0)
    tasks_per_chunk = runtime_cfg.get("tasks_per_additional_chunk", 0)
    missing = []
    if launch is None:
        missing.append("call_launch_latency_us")
    if task_launch is None:
        missing.append("task_launch_latency_us")
    if tasks_per_stage is None:
        missing.append("tasks_per_stage")
    if missing:
        return {
            "status": "partial",
            "execution_engine": runtime_cfg.get("execution_engine"),
            "algorithm": algorithm,
            "algorithm_stages": stages,
            "runtime_task_count": None,
            "call_runtime_overhead_us": None,
            "unknown_reason": ";".join(missing),
            "assumptions": assumptions,
        }
    chunks = 1
    if chunk_bytes and payload_bytes:
        chunks = max(1, (int(payload_bytes) + int(chunk_bytes) - 1) // int(chunk_bytes))
    stage_tasks = int(stages) * max(1, int(active_levels)) * int(tasks_per_stage)
    descriptor_tasks = max(0, chunks - 1) * int(tasks_per_chunk)
    task_count = stage_tasks + descriptor_tasks
    overhead = float(launch) + task_count * float(task_launch)
    return {
        "status": "known",
        "execution_engine": runtime_cfg.get("execution_engine", "host_cpu_ts"),
        "algorithm": algorithm,
        "algorithm_stages": int(stages),
        "active_network_levels": int(max(1, active_levels)),
        "payload_chunks": chunks,
        "stage_runtime_tasks": stage_tasks,
        "descriptor_runtime_tasks": descriptor_tasks,
        "runtime_task_count": task_count,
        "call_launch_latency_us": float(launch),
        "task_launch_latency_us": float(task_launch),
        "call_runtime_overhead_us": overhead,
        "assumptions": assumptions,
        "formula": "T_runtime=L_call+(stages*active_levels*tasks_per_stage+descriptor_tasks)*L_task",
    }


def _level_rows_for_event(event: Any, derivation_records: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not derivation_records:
        return []
    level_rows = list((derivation_records.get("network_layers") or {}).values())
    family = _event_collective_family(event)
    group_kind = _get(event, "group_kind")
    group_size = _get(event, "group_size")
    direction = _get(event, "operation")
    matching_call = []
    for row in level_rows:
        row_family = _collective_family(row.get("op_name"))
        if row_family != family:
            continue
        if group_kind and row.get("group_kind") and str(row.get("group_kind")) != str(group_kind):
            continue
        if group_size and row.get("comm_num") and int(row.get("comm_num")) != int(group_size):
            continue
        row_direction = _stage_direction(row.get("stage"))
        if row_direction and direction and row_direction != direction:
            continue
        matching_call.append(row)
    # Keep deterministic, compact output.  Repeated calls share the same
    # derivation fact; the event plan carries the distinct semantic plan_id.
    unique = {}
    for row in matching_call:
        key = (row.get("network_level") or row.get("stage"),
               row.get("message_bytes"), row.get("comm_num"), row.get("net"))
        unique[key] = dict(row)
    return _consolidate_level_rows(
        [unique[key] for key in sorted(unique, key=lambda value: str(value))])


def _call_row_for_event(event: Any,
                        derivation_records: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the call-level timing fact matching an event's structure."""

    if not derivation_records:
        return None
    family = _event_collective_family(event)
    group_kind = _get(event, "group_kind")
    group_size = _get(event, "group_size")
    direction = _get(event, "operation")
    candidates = []
    for row in (derivation_records.get("communications") or {}).values():
        stage = str(row.get("stage") or "")
        if stage.startswith("levels:") and not stage.endswith(":call"):
            continue
        row_family = _collective_family(row.get("algorithm_family") or row.get("op_name"))
        if row_family != family:
            continue
        if group_kind and row.get("group_kind") and str(row.get("group_kind")) != str(group_kind):
            continue
        if group_size and row.get("comm_num") and int(row.get("comm_num")) != int(group_size):
            continue
        row_direction = _stage_direction(stage)
        if row_direction and direction and row_direction != direction:
            continue
        candidates.append(row)
    # The derivation table can contain multiple calls with the same family,
    # group and direction but different payloads.  Collapse exact duplicate
    # facts first, then refuse to select by dictionary order when structure is
    # insufficient.  A silent first-candidate choice creates false plan
    # identities and can make the report look more precise than it is.
    unique_candidates = {}
    for row in candidates:
        key = (
            row.get("stage"), row.get("message_bytes"), row.get("group_kind"),
            row.get("comm_num"), row.get("derived_time_ms"),
        )
        unique_candidates[key] = row
    candidates = list(unique_candidates.values())
    if not candidates:
        return None
    payload = _get(event, "payload_bytes")
    candidate_count_before_payload = len(candidates)
    candidate_payloads = sorted({
        row.get("message_bytes") for row in candidates
        if row.get("message_bytes") is not None
    })
    if payload is not None:
        exact = [row for row in candidates
                 if row.get("message_bytes") is not None
                 and int(row.get("message_bytes")) == int(payload)]
        if not exact:
            return {
                "mapping_status": "payload_mismatch",
                "event_payload_bytes": payload,
                "candidate_count": candidate_count_before_payload,
                "message_bytes_candidates": candidate_payloads,
                "stage_candidates": sorted({
                    str(row.get("stage") or "") for row in candidates
                }),
            }
        candidates = exact
    if len(candidates) == 1:
        resolved = dict(candidates[0])
        resolved["mapping_status"] = "unique"
        return resolved
    return {
        "mapping_status": "ambiguous",
        "candidate_count": len(candidates),
        "message_bytes_candidates": sorted({
            row.get("message_bytes") for row in candidates
            if row.get("message_bytes") is not None
        }),
        "stage_candidates": sorted({
            str(row.get("stage") or "") for row in candidates
        }),
    }


def _ideal_transfer_time_ms(levels: Sequence[Mapping[str, Any]],
                            composition_policy: Optional[str] = None) -> Optional[float]:
    phase_times = []
    for level in levels:
        beta = level.get("beta_gb_per_s")
        payload = level.get("payload_bytes")
        if beta and payload:
            phase_times.append(float(payload) / (float(beta) * 1e9) * 1e3)
    if not phase_times:
        return None
    if composition_policy == "max":
        return max(phase_times)
    return sum(phase_times)


def _consolidate_level_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Keep physical facts while refusing ambiguous payload assignments."""

    grouped: Dict[Any, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("network_level") or row.get("stage")].append(row)
    consolidated = []
    for level, candidates in grouped.items():
        payloads = sorted({row.get("message_bytes") for row in candidates
                           if row.get("message_bytes") is not None})
        first = dict(candidates[0])
        if len(payloads) > 1:
            # The same physical level is reused by different semantic calls.
            # Do not select a payload by duration or by dictionary order.
            first["message_bytes"] = None
            first["payload_candidates"] = payloads
            first["payload_mapping_status"] = "ambiguous"
        else:
            first["payload_mapping_status"] = "unique"
        consolidated.append(first)
    return consolidated


@dataclass
class CommunicationPlan:
    """JSON-serializable semantic communication plan."""

    plan_id: str
    comm_id: Optional[str] = None
    owner_path: Optional[str] = None
    semantic_id: Optional[str] = None
    phase_id: Optional[str] = None
    operation: Optional[str] = None
    comm_owner: Optional[str] = None
    comm_role: Optional[str] = None
    group_kind: Optional[str] = None
    group_size: Optional[int] = None
    payload_bytes: Optional[int] = None
    dtype: Optional[str] = None
    algorithm: Optional[str] = None
    algorithm_stages: Optional[int] = None
    composition_policy: Optional[str] = None
    derived_time_ms: Optional[float] = None
    call_mapping_status: Optional[str] = None
    structural_execution_efficiency: Optional[float] = None
    attainable_efficiency: Optional[float] = None
    min_reachable_beta_gb_per_s: Optional[float] = None
    max_reachable_beta_gb_per_s: Optional[float] = None
    topology_levels: List[Dict[str, Any]] = field(default_factory=list)
    runtime: Dict[str, Any] = field(default_factory=dict)
    lifecycle: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    status: str = "unknown"
    unknown_fields: List[str] = field(default_factory=list)
    source: str = "des_event"
    ranks: List[int] = field(default_factory=list)
    identity_scope: str = "semantic_call"

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "plan_version": COMM_PLAN_SCHEMA_VERSION,
            **self.__dict__,
        }
        return data


def _plan_key(event: Any, ordinal: int) -> Tuple[Any, ...]:
    semantic = _get(event, "semantic_id") or _get(event, "owner_path") or "event"
    operation = _get(event, "operation") or "unknown"
    comm_id = (
        _get(event, "comm_id") or _get(event, "gid")
        or _get(event, "plan_id") or f"event-{ordinal}"
    )
    sequence = _get(event, "comm_sequence")
    segment = _get(event, "comm_segment_index")
    return (str(semantic), str(operation), str(comm_id), sequence, segment)


def build_plans_from_events(events: Iterable[Any], system: Any = None,
                            strategy: Any = None,
                            derivation_records: Optional[Mapping[str, Any]] = None,
                            source: str = "des_event") -> List[Dict[str, Any]]:
    """Build one plan per semantic communication call from DES events."""

    grouped: Dict[Tuple[Any, ...], List[Any]] = defaultdict(list)
    for ordinal, event in enumerate(events or []):
        if _get(event, "kind") not in ("comm", "wait"):
            continue
        if (_get(event, "comm_id") is None and _get(event, "gid") is None
                and _get(event, "plan_id") is None):
            continue
        grouped[_plan_key(event, ordinal)].append(event)

    plans: List[Dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        rows = grouped[key]
        primary = next((row for row in rows
                        if _get(row, "kind") == "comm"
                        and not str(_get(row, "name") or "").endswith("-post")
                        and _event_cost(row) > 0), rows[0])
        semantic_id, operation, comm_id, sequence, segment = key
        group_kind = _get(primary, "group_kind")
        group_size = _get(primary, "group_size")
        payload = _get(primary, "payload_bytes")
        op_name = (_get(primary, "comm_stage") or _get(primary, "comm_role")
                   or _get(primary, "name") or "communication")
        family = _event_collective_family(primary)
        algorithm, stages = collective_algorithm(family, group_size)
        plan_id = _get(primary, "plan_id") or (
            f"derived/{_safe_token(semantic_id)}/{_safe_token(operation)}"
            f"/s{sequence if sequence is not None else 'u'}"
            f"/{_safe_token(comm_id)}")
        call_row = _call_row_for_event(primary, derivation_records)
        call_mapping_status = call_row.get("mapping_status") if call_row else "not_found"
        composition_policy = call_row.get("composition_policy") if call_row else None
        levels = []
        if call_row and call_row.get("physical_levels"):
            # The levels embedded in the call record are produced by the same
            # forward formula invocation, so their payloads are unambiguous.
            for row in call_row["physical_levels"]:
                levels.append({
                    "level": row.get("level"),
                    "net": row.get("net"),
                    "units_touched": row.get("units_touched"),
                    "payload_bytes": row.get("payload_bytes"),
                    "beta_gb_per_s": row.get("beta_gb_per_s"),
                    "bandwidth_unit": row.get("bandwidth_unit", "GB/s"),
                    "physical_latency_us": row.get("physical_latency_us"),
                    "hop_count": row.get("hop_count"),
                    "port_utilization": row.get("port_utilization"),
                    "packet_efficiency": row.get("packet_efficiency"),
                    "attainable_bandwidth_efficiency": row.get(
                        "attainable_bandwidth_efficiency"),
                    "reachable_beta_gb_per_s": row.get(
                        "reachable_beta_gb_per_s"),
                    "latency_formula": row.get("latency_formula"),
                    "source": row.get("source", "forward_formula"),
                })
        else:
            level_rows = _level_rows_for_event(primary, derivation_records)
            for row in level_rows:
                levels.append({
                    "level": row.get("network_level") or row.get("stage"),
                    "net": row.get("net"),
                    "units_touched": row.get("units_touched"),
                    "payload_bytes": row.get("message_bytes"),
                    "payload_candidates": row.get("payload_candidates"),
                    "payload_mapping_status": row.get("payload_mapping_status"),
                    "beta_gb_per_s": row.get("effective_beta_gb_per_s")
                    or row.get("effective_beta_gib_per_s")
                    or row.get("derived_beta_gb_per_s")
                    or row.get("derived_beta_gib_per_s"),
                    "reachable_beta_gb_per_s": row.get(
                        "reachable_beta_gb_per_s")
                    or row.get("effective_beta_gb_per_s")
                    or row.get("effective_beta_gib_per_s")
                    or row.get("derived_beta_gb_per_s")
                    or row.get("derived_beta_gib_per_s"),
                    "bandwidth_unit": row.get("bandwidth_unit", "GB/s"),
                    "physical_latency_us": row.get("physical_propagation_latency_us")
                    or row.get("network_layer_latency_us")
                    or row.get("base_latency_us"),
                    "attainable_bandwidth_efficiency": row.get(
                        "attainable_bandwidth_efficiency"),
                    "hop_count": row.get("hop_count") or row.get("physical_hop_count"),
                    "source": "forward_derivation_records",
                })
        derived_time_ms = call_row.get("derived_time_ms") if call_row else None
        ideal_transfer_ms = _ideal_transfer_time_ms(levels, composition_policy)
        structural_efficiency = (
            ideal_transfer_ms / derived_time_ms
            if ideal_transfer_ms is not None and derived_time_ms and derived_time_ms > 0
            else None)
        attainable_efficiency = (
            call_row.get("communication_attainable_efficiency")
            if call_row else structural_efficiency)
        reachable_betas = [
            level.get("reachable_beta_gb_per_s") or level.get("beta_gb_per_s")
            for level in levels
            if level.get("reachable_beta_gb_per_s") or level.get("beta_gb_per_s")
        ]
        runtime = _runtime_facts(
            system, family, group_size, payload, active_levels=max(1, len(levels)))
        unknown = []
        for field_name, value in (("group_size", group_size), ("payload_bytes", payload),
                                  ("comm_owner", _get(primary, "comm_owner"))):
            if value is None:
                unknown.append(field_name)
        if not levels:
            unknown.append("topology_levels")
        if any(level.get("payload_bytes") is None for level in levels):
            unknown.append("level_payload_mapping")
        if runtime.get("status") != "known":
            unknown.append("runtime")
        if call_mapping_status in {"ambiguous", "payload_mismatch"}:
            unknown.append("communication_call_mapping")
        post_count = sum(1 for row in rows
                         if str(_get(row, "name") or "").endswith("-post"))
        wait_count = sum(1 for row in rows if _get(row, "kind") == "wait")
        completion_count = sum(
            1 for row in rows
            if _get(row, "kind") == "comm"
            and not str(_get(row, "name") or "").endswith("-post")
            and _event_cost(row) > 0)
        lifecycle = {
            "event_count": len(rows),
            "rank_count": len({_get(row, "rank") for row in rows}),
            "post_count": post_count,
            "completion_count": completion_count,
            "wait_count": wait_count,
        }
        lifecycle.update(_simulated_overlap(rows))
        edges = []
        if post_count and completion_count:
            edges.append({"from": "post", "to": "completion", "type": "lifecycle"})
        if completion_count and wait_count:
            edges.append({"from": "completion", "to": "wait_release", "type": "consumer_barrier"})
        explicit_consumer_edges = []
        for row in rows:
            consumer_id = _get(row, "consumer_id")
            depends_on = (_get(row, "depends_on")
                          or _get(row, "dependency_ids")
                          or _get(row, "consumer_ids"))
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            for dependency in depends_on or []:
                explicit_consumer_edges.append({
                    "from": str(plan_id),
                    "to": str(dependency),
                    "type": "explicit_consumer",
                    "consumer_id": str(consumer_id) if consumer_id is not None else None,
                })
        dependencies = {
            "edges": edges + explicit_consumer_edges,
            "status": (
                "explicit_consumer_dependency"
                if explicit_consumer_edges else
                "structural_lifecycle_only" if edges else "not_explicit"
            ),
            "consumer_dependency_known": bool(explicit_consumer_edges),
        }
        plans.append(CommunicationPlan(
            plan_id=str(plan_id),
            comm_id=str(comm_id) if comm_id is not None else None,
            owner_path=_get(primary, "owner_path"),
            semantic_id=semantic_id,
            phase_id=_get(primary, "phase_id"),
            operation=operation,
            comm_owner=_get(primary, "comm_owner") or _owner_for_group(group_kind, op_name),
            comm_role=_get(primary, "comm_role") or _get(primary, "comm_stage"),
            group_kind=group_kind,
            group_size=int(group_size) if group_size is not None else None,
            payload_bytes=int(payload) if payload is not None else None,
            dtype=_get(primary, "dtype"),
            algorithm=algorithm,
            algorithm_stages=stages,
            composition_policy=composition_policy,
            derived_time_ms=derived_time_ms,
            call_mapping_status=call_mapping_status,
            structural_execution_efficiency=structural_efficiency,
            attainable_efficiency=attainable_efficiency,
            min_reachable_beta_gb_per_s=(
                min(reachable_betas) if reachable_betas else (
                    call_row.get("min_reachable_beta_gb_per_s")
                    if call_row else None)),
            max_reachable_beta_gb_per_s=(
                max(reachable_betas) if reachable_betas else (
                    call_row.get("max_reachable_beta_gb_per_s")
                    if call_row else None)),
            topology_levels=levels,
            runtime=runtime,
            lifecycle=lifecycle,
            dependencies=dependencies,
            provenance=_provenance(system),
            status=_status(unknown),
            unknown_fields=sorted(set(unknown)),
            source=source,
            ranks=sorted({rank for row in rows
                          if (rank := _event_rank(row)) is not None}),
            identity_scope="semantic_call",
        ).to_dict())
    return plans


def build_plans_from_derivations(derivation_records: Optional[Mapping[str, Any]],
                                 system: Any = None) -> List[Dict[str, Any]]:
    """Build configuration-only plans before a DES trace exists."""

    if not derivation_records:
        return []
    rows = list((derivation_records.get("communications") or {}).values())
    level_rows = list((derivation_records.get("network_layers") or {}).values())
    plans = []
    for row in rows:
        stage = row.get("stage") or "unknown"
        # A level row is a physical fact; the semantic plan is emitted once for
        # a call.  Non-level communication records are already call-level.
        if str(stage).startswith("levels:") and not str(stage).endswith(":call"):
            continue
        op_name = row.get("op_name") or row.get("algorithm_family") or "communication"
        group_kind = row.get("group_kind")
        group_size = row.get("comm_num")
        payload = row.get("message_bytes")
        algorithm = row.get("algorithm")
        stages = row.get("algorithm_stages")
        if algorithm is None:
            algorithm, stages = collective_algorithm(op_name, group_size)
        matching_levels = []
        if row.get("physical_levels"):
            for level in row["physical_levels"]:
                matching_levels.append({
                    "level": level.get("level"),
                    "net": level.get("net"),
                    "units_touched": level.get("units_touched"),
                    "payload_bytes": level.get("payload_bytes"),
                    "beta_gb_per_s": level.get("beta_gb_per_s"),
                    "bandwidth_unit": level.get("bandwidth_unit", "GB/s"),
                    "physical_latency_us": level.get("physical_latency_us"),
                    "hop_count": level.get("hop_count"),
                    "port_utilization": level.get("port_utilization"),
                    "packet_efficiency": level.get("packet_efficiency"),
                    "attainable_bandwidth_efficiency": level.get(
                        "attainable_bandwidth_efficiency"),
                    "reachable_beta_gb_per_s": level.get(
                        "reachable_beta_gb_per_s") or level.get("beta_gb_per_s"),
                    "source": level.get("source", "forward_formula"),
                })
        else:
            family = _collective_family(row.get("algorithm_family") or op_name)
            for level in level_rows:
                if _collective_family(level.get("op_name")) != family:
                    continue
                if group_size and level.get("comm_num") and int(level["comm_num"]) != int(group_size):
                    continue
                matching_levels.append({
                    "level": level.get("network_level"),
                    "net": level.get("net"),
                    "units_touched": level.get("units_touched"),
                    "payload_bytes": level.get("message_bytes"),
                    "beta_gb_per_s": level.get("effective_beta_gb_per_s")
                    or level.get("effective_beta_gib_per_s")
                    or level.get("derived_beta_gb_per_s")
                    or level.get("derived_beta_gib_per_s"),
                    "reachable_beta_gb_per_s": level.get(
                        "reachable_beta_gb_per_s")
                    or level.get("effective_beta_gb_per_s")
                    or level.get("effective_beta_gib_per_s")
                    or level.get("derived_beta_gb_per_s")
                    or level.get("derived_beta_gib_per_s"),
                    "bandwidth_unit": level.get("bandwidth_unit", "GB/s"),
                    "physical_latency_us": level.get("physical_propagation_latency_us")
                    or level.get("network_layer_latency_us")
                    or level.get("base_latency_us"),
                    "hop_count": level.get("hop_count") or level.get("physical_hop_count"),
                    "attainable_bandwidth_efficiency": level.get(
                        "attainable_bandwidth_efficiency"),
                    "source": "forward_derivation_records",
                })
        composition_policy = row.get("composition_policy")
        ideal_transfer_ms = _ideal_transfer_time_ms(matching_levels, composition_policy)
        structural_efficiency = (
            ideal_transfer_ms / row.get("derived_time_ms")
            if ideal_transfer_ms is not None and row.get("derived_time_ms")
            and row.get("derived_time_ms") > 0 else None)
        attainable_efficiency = row.get(
            "communication_attainable_efficiency", structural_efficiency)
        reachable_betas = [
            level.get("reachable_beta_gb_per_s") or level.get("beta_gb_per_s")
            for level in matching_levels
            if level.get("reachable_beta_gb_per_s") or level.get("beta_gb_per_s")
        ]
        runtime = dict(row.get("call_runtime") or {})
        if not runtime:
            runtime = _runtime_facts(system, op_name, group_size, payload,
                                     active_levels=max(1, len(matching_levels)))
        unknown = []
        for field_name, value in (("semantic_id", None), ("owner_path", None),
                                  ("group_size", group_size), ("payload_bytes", payload)):
            if value is None:
                unknown.append(field_name)
        if not matching_levels:
            unknown.append("topology_levels")
        if any(level.get("payload_bytes") is None for level in matching_levels):
            unknown.append("level_payload_mapping")
        if runtime.get("status") == "partial" or runtime.get("call_runtime_overhead_us") is None:
            unknown.append("runtime")
        safe_stage = _safe_token(stage)
        plan_id = (
            f"template/{_safe_token(op_name)}/{safe_stage}/"
            f"n{group_size if group_size is not None else 'u'}/"
            f"bytes{int(payload) if payload is not None else 'u'}")
        plans.append(CommunicationPlan(
            plan_id=plan_id,
            comm_id=plan_id,
            owner_path=None,
            semantic_id=None,
            phase_id=stage,
            operation=_stage_direction(stage),
            comm_owner=_owner_for_group(group_kind, stage),
            comm_role=stage,
            group_kind=group_kind,
            group_size=int(group_size) if group_size is not None else None,
            payload_bytes=int(payload) if payload is not None else None,
            dtype=row.get("dtype"),
            algorithm=algorithm,
            algorithm_stages=stages,
            composition_policy=composition_policy,
            derived_time_ms=row.get("derived_time_ms"),
            call_mapping_status="unique",
            structural_execution_efficiency=structural_efficiency,
            attainable_efficiency=attainable_efficiency,
            min_reachable_beta_gb_per_s=(
                min(reachable_betas) if reachable_betas else None),
            max_reachable_beta_gb_per_s=(
                max(reachable_betas) if reachable_betas else None),
            topology_levels=matching_levels,
            runtime=runtime,
            lifecycle={"event_count": 0, "source": "derivation_only"},
            dependencies={"edges": [], "status": "not_explicit"},
            provenance=_provenance(system),
            status=_status(unknown),
            unknown_fields=sorted(set(unknown)),
            source="forward_derivation",
            ranks=[],
            identity_scope="configuration_template",
        ).to_dict())
    return plans


def build_communication_plan_document(
    events: Optional[Iterable[Any]] = None,
    system: Any = None,
    strategy: Any = None,
    derivation_records: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the canonical JSON document for a run."""

    event_plans = build_plans_from_events(
        events or [], system=system, strategy=strategy,
        derivation_records=derivation_records)
    derivation_plans = build_plans_from_derivations(derivation_records, system=system)
    plans = event_plans if event_plans else derivation_plans
    # Runtime event plans are the active per-call view.  Keep the
    # configuration-only templates separately instead of silently dropping
    # them or concatenating them into the per-call count.
    templates = derivation_plans if event_plans else []
    unknown = sum(1 for plan in plans if plan.get("status") == "unknown")
    partial = sum(1 for plan in plans if plan.get("status") == "partial")
    return {
        "plan_version": COMM_PLAN_SCHEMA_VERSION,
        "mode": "forward_derived",
        "plans": plans,
        "plan_templates": templates,
        "summary": {
            "plan_count": len(plans),
            "des_event_plan_count": len(event_plans),
            "derivation_plan_count": len(derivation_plans),
            "plan_template_count": len(templates),
            "known_count": len(plans) - unknown - partial,
            "partial_count": partial,
            "unknown_count": unknown,
            "performance_observations_used_as_parameters": False,
            "measured_results_role": "validation_only",
        },
        "provenance": _provenance(system),
    }
