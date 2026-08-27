"""Structured simulation event stream.

Contract module for the simulate() DES rework (see
docs/design_simu_kind_resource_model.md). Phase 0 replaces the private
text log with an in-memory stream of SimuEvent objects; Phase 1 fills in
the classification fields (``kind``/``lane``) so the trace exporter can
stop guessing from name prefixes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

_RANK_PREFIX = re.compile(r"^rank(\d+)-")
_LAYER_RE = re.compile(r"(?:mxxmodellayer|layer)[_-]?(\d+)", re.IGNORECASE)
_MICROBATCH_RANK = re.compile(r"^(microbatch\d+)rank\d+$", re.IGNORECASE)
_MICROBATCH = re.compile(r"^microbatch(\d+)$", re.IGNORECASE)


def _normalise_token(value: object) -> str:
    """Return a comparison token without making any hardware assumptions."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _group_gemm_projection(name: str) -> Optional[str]:
    """Return Gate/Up/Down when a GroupGemm semantic name exposes it."""
    token = _normalise_token(name)
    if "groupgemm" not in token:
        return None
    projection_token = token.split("groupgemm", 1)[-1]
    if "down" in projection_token:
        return "down"
    if "gate" in projection_token:
        return "gate"
    if "up" in projection_token:
        return "up"
    return None


def _kernel_role(name: str, semantic_stage: Optional[str] = None) -> Optional[str]:
    """Derive a portable kernel role from model-generated names.

    The role is intentionally semantic (for example ``gate`` or
    ``bwd_grad_w``), not a CANN kernel name.  It lets a consumer distinguish
    the three GroupGemm projections and backward phases even when the real
    profiler exposes only a generic kernel family.
    """
    token = _normalise_token(name)
    if "groupgemm" in token or "groupgemm" in _normalise_token(semantic_stage):
        if "bwdact" in token or "gradact" in token:
            return "bwd_grad_act"
        if "bwdw" in token or "gradw" in token:
            return "bwd_grad_w"
        # Strip the fixed GroupGemm prefix before checking projections;
        # otherwise the ``up`` substring in ``group`` mislabels Down/unknown
        # projections as Up.
        projection = _group_gemm_projection(name)
        if projection:
            return projection
        return "group_gemm"
    if "router" in token or "routing" in token:
        return "router"
    if "vwnwidth" in token:
        return "vwn_width"
    if "vwn_depth" in name.lower() or "vwndepth" in token:
        return "vwn_depth_prenorm" if "prenorm" in token else "vwn_depth"
    if "vwnout" in token:
        return "vwn_out"
    if "vwnin" in token:
        return "vwn_in"
    if "transpose" in token or "layout" in token:
        return "layout"
    if "rmsnorm" in token or token.endswith("norm") or "norm" in token:
        return "normalization"
    if "rope" in token:
        return "rope"
    if "swa" in token:
        return "swa"
    if "matmul" in token or "linear" in token or token.endswith("bmm"):
        return "matmul"
    return None


def _layer_index(segments: List[str], text: str) -> Optional[int]:
    """Extract a model layer index from generated model names when present."""
    match = _LAYER_RE.search(text)
    if match:
        return int(match.group(1))
    # A few generic modules use ``layer-<n>`` as a stack segment.  Searching
    # individual segments keeps this parser independent of a specific model
    # implementation while still providing a stable layer field.
    for segment in segments:
        match = _LAYER_RE.search(segment)
        if match:
            return int(match.group(1))
    return None


def _normalise_call_segments(segments: List[str]) -> List[str]:
    """Remove rank/microbatch join artifacts from generated call stacks.

    MXX builds nested stacks by concatenating ``rank<N>-microbatch<M>``
    prefixes.  When a child already carries the rank prefix this historically
    produced a first segment such as ``microbatch0rank0`` (and sometimes a
    duplicate leading ``microbatch0``).  The rank is already a first-class
    event field, so retaining the suffix changes only semantic ownership and
    makes portable stage matching fail.  This cleanup is purely structural and
    does not alter scheduling or durations.
    """
    normalised = []
    for segment in segments:
        match = _MICROBATCH_RANK.match(str(segment))
        normalised.append(match.group(1) if match else segment)
    while (len(normalised) > 1
           and normalised[0].lower() == normalised[1].lower()
           and normalised[0].lower().startswith("microbatch")):
        normalised.pop(0)
    return normalised


def _canonical_stage(text: str, name: str, kind: Optional[str],
                     comm_stage: Optional[str] = None) -> Optional[str]:
    """Map generated semantic names to portable model-level stages.

    This is deliberately a structural classifier.  It uses names emitted by
    the model graph and communication constructor metadata only; no measured
    durations, alpha/beta values, or trace-derived constants are consulted.
    """
    token = _normalise_token(f"{text}/{name}")
    comm_token = _normalise_token(comm_stage)
    if kind in ("comm", "wait") or comm_stage:
        if "cp1" in comm_token or "cp1" in token:
            return "attention.cp1_comm"
        if "cp2" in comm_token or "cp2" in token:
            return "attention.cp2_comm"
        if "cp3" in comm_token or "cp3" in token:
            return "attention.cp3_comm"
        if "dispatch" in comm_token or "dispatch" in token:
            return "moe.dispatch_comm"
        if "combine" in comm_token or "combine" in token:
            return "moe.combine_comm"
        if "router" in comm_token or "router" in token:
            return "moe.router_comm"
        if "edp" in comm_token or "edpgroup" in token:
            return "fsdp.moe_comm"
        if "dpcp" in comm_token or "dpcpgroup" in token:
            return "fsdp.dense_comm"
        if comm_token in ("cp", "cpgroup"):
            return "attention.cp_comm"
        if comm_token in ("ep", "epgroup"):
            return "moe.ep_comm"
        if comm_token in ("tp", "tpgroup"):
            return "tensor_parallel_comm"
        if comm_token in ("etp", "etpgroup"):
            return "expert_tensor_parallel_comm"
        if comm_token in ("pp", "ppgroup"):
            return "pipeline_comm"
        if "tp" in comm_token or "tpgroup" in token:
            return "tensor_parallel_comm"
        if "etp" in comm_token or "etpgroup" in token:
            return "expert_tensor_parallel_comm"
        if "pp" in comm_token or "defaultgroup" in token or "sendrecv" in token:
            return "pipeline_comm"
        return "communication"

    # Most specific compute boundaries first; this avoids classifying a
    # fused ``...NormRoPE...`` event as a generic RMSNorm.
    if "groupgemm" in token:
        return "moe.group_gemm"
    if "swiglu" in token or "swishglu" in token:
        return "moe.swiglu"
    if "routeraux" in token or "routeownermap" in token or "experthistogram" in token:
        return "moe.router"
    if "vwnpre" in token and "norm" in token:
        return "normalization"
    if "attnvwnin" in token or "mlpvwnin" in token:
        return "vwn.width_in"
    if "attnvwnout" in token or "mlpvwnout" in token:
        return "vwn.depth_out"
    if "vwnwidth" in token:
        return "vwn.width_kernel"
    if "vwndepth" in token:
        return "vwn.depth_kernel"
    if "vwn" in token:
        vwn_stages = {
            "widthout": "vwn.width_out", "depthout": "vwn.depth_out",
            "widthin": "vwn.width_in", "depthin": "vwn.depth_in",
        }
        for suffix, stage_name in vwn_stages.items():
            if suffix in token:
                return stage_name
        return "vwn"
    if "normrope" in token:
        return "attention.norm_rope"
    if "rope" in token:
        return "attention.rope"
    # Branch-specific RMSNorms are normalization boundaries, not attention
    # kernels.  Check this before the generic ``swa`` classifier so names such
    # as SWAQueryRMSNorm do not move ahead of the configured RoPE/CP stages in
    # the semantic dependency graph.
    if "swa" in token and ("rmsnorm" in token or token.endswith("norm")):
        return "normalization"
    if "swa" in token:
        return "attention.swa_prepare" if "prepare" in token else "attention.swa"
    if "concatd" in token or "concat" in token:
        return "attention.concat"
    if "dim01transpose" in token or "transpose" in token:
        return "layout.transpose"
    if "latentbmm" in token:
        return "attention.latent_bmm"
    if "kvc" in token:
        return "attention.kvc_pack"
    if "qkv" in token:
        return "attention.qkv_projection"
    if "rmsnorm" in token or token.endswith("norm") or "norm" in token:
        return "normalization"
    if "matmul" in token or "linear" in token or "bmm" in token:
        return "matmul"
    return "compute"


def _derive_semantic_metadata(segments: List[str], name: str,
                              operation: str, kind: Optional[str],
                              metadata: Optional[Dict[str, object]],
                              stream: Optional[str] = None,
                              lane: Optional[str] = None) -> Dict[str, object]:
    """Derive stable semantic fields from model-generated event context."""
    supplied = dict(metadata or {})
    text = "/".join(segments)
    layer_idx = supplied.get("layer_idx")
    if layer_idx is None:
        layer_idx = _layer_index(segments, f"{text}/{name}")
    comm_stage = supplied.get("comm_stage")
    group_kind = supplied.get("group_kind")
    stage = supplied.get("semantic_stage") or _canonical_stage(
        text, name, kind, comm_stage or group_kind)
    comm_owner = supplied.get("comm_owner")
    comm_role = supplied.get("comm_role") or comm_stage
    supplied.setdefault("kernel_role", _kernel_role(name, stage))
    supplied.setdefault("projection", _group_gemm_projection(name))
    # Shape descriptors are kept as a per-stage map on the model object.  Pick
    # the most specific entry available for this emitted event without making
    # the DES depend on profiler output.
    shape_by_stage = supplied.get("shape_desc_by_stage") or {}
    if supplied.get("shape_desc") is None and shape_by_stage:
        role = supplied.get("kernel_role")
        shape_key = (
            "bwd_grad_act" if role == "bwd_grad_act" else
            "bwd_grad_w" if role == "bwd_grad_w" else
            "fwd" if operation == "fwd" else
            "fwd" if operation == "recompute_fwd" else
            "bwd_grad_act"
        )
        supplied["shape_desc"] = shape_by_stage.get(shape_key)
    if kind in ("comm", "wait") or comm_stage:
        if comm_owner is None:
            owner_by_group = {
                "cp": "attention_cp", "ep": "moe_ep", "dp_cp": "fsdp_dense",
                "edp": "fsdp_moe", "tp": "tensor_parallel",
                "etp": "expert_tensor_parallel", "pp": "pipeline",
            }
            comm_owner = owner_by_group.get(str(group_kind).lower())
        if comm_owner is None:
            comm_owner = {
                "attention.cp1_comm": "attention_cp",
                "attention.cp2_comm": "attention_cp",
                "attention.cp3_comm": "attention_cp",
                "moe.dispatch_comm": "moe_dispatch",
                "moe.combine_comm": "moe_combine",
                "moe.router_comm": "moe_router",
                "fsdp.dense_comm": "fsdp_dense",
                "fsdp.moe_comm": "fsdp_moe",
                "pipeline_comm": "pipeline",
            }.get(stage, "communication")
        supplied.setdefault("comm_owner", comm_owner)
        supplied.setdefault("comm_role", comm_role or stage)
        supplied.setdefault("group_kind", group_kind)
        # Keep one portable owner vocabulary for trace consumers.  The
        # resource-level ``comm_owner`` remains available, while
        # ``semantic_owner`` identifies the logical collective role.  Explicit
        # model declarations (for example model-level gradient sync) win.
        supplied.setdefault(
            "semantic_owner",
            comm_role or comm_stage or comm_owner or stage,
        )
        supplied.setdefault("semantic_phase", operation)
        supplied.setdefault(
            "layer_id",
            f"layer{layer_idx}" if layer_idx is not None else "model",
        )
        supplied.setdefault(
            "module_path",
            supplied.get("owner_path") or "/".join(segments[:-1]),
        )

    # These fields describe the model's structural dependency and overlap
    # contract.  They deliberately use categories and lane names rather than
    # profiler timings or CANN/HCCL kernel names.  Explicit model metadata is
    # retained when a caller has a stronger dependency declaration.
    resource_lane = str(lane or stream or "comp")
    explicit_dependency = bool(
        supplied.get("depends_on") or supplied.get("dependency_ids")
        or supplied.get("consumer_id")
    )
    if kind == "wait":
        dependency_defaults = {
            "dependency_kind": "consumer_barrier",
            "dependency_status": "explicit" if explicit_dependency else "implicit_lifecycle",
            "ready_rule": "all_dependencies_complete",
            "overlap_policy": "wait_for_dependencies",
            "overlap_lanes": ["comp"],
        }
    elif kind == "comm":
        blocking = resource_lane in ("comp", "compute")
        dependency_defaults = {
            "dependency_kind": "communication_lifecycle",
            "dependency_status": "explicit" if explicit_dependency else "implicit_lifecycle",
            "ready_rule": "post_then_completion",
            "overlap_policy": "blocking_collective" if blocking else "async_until_wait",
            "overlap_lanes": ["comp"] if blocking else [resource_lane, "comp"],
        }
    elif kind == "runtime":
        direction = str(supplied.get("offload_direction") or "").lower()
        dependency_defaults = {
            "dependency_kind": "runtime_transfer",
            "dependency_status": "explicit" if explicit_dependency else "implicit_lane_order",
            "ready_rule": (
                "producer_complete_before_d2h"
                if direction == "d2h" else "h2d_complete_before_consumer"
            ),
            "overlap_policy": supplied.get("overlap_policy") or "async",
            "overlap_lanes": (
                [resource_lane] if supplied.get("overlap_policy") == "serial"
                else [resource_lane, "comp"]
            ),
        }
    elif kind == "fused":
        dependency_defaults = {
            "dependency_kind": "resource_lane_order",
            "dependency_status": "explicit" if explicit_dependency else "implicit_lane_order",
            "ready_rule": "lane_start_after_inputs",
            "overlap_policy": "multi_lane_fused",
            "overlap_lanes": [resource_lane],
        }
    else:
        dependency_defaults = {
            "dependency_kind": "compute_lane_order",
            "dependency_status": "explicit" if explicit_dependency else "implicit_lane_order",
            "ready_rule": "previous_ready_on_lane",
            "overlap_policy": "compute_lane_serial",
            "overlap_lanes": [resource_lane],
        }
    for key, value in dependency_defaults.items():
        supplied.setdefault(key, value)
    supplied.setdefault("semantic_stage", stage)
    supplied.update({
        "layer_idx": layer_idx,
        "stage_role": (
            "communication" if kind == "comm" else
            "wait" if kind == "wait" else
            "scope" if kind == "scope" else
            "compute"
        ),
        "direction": operation,
    })
    return supplied


@dataclass
class SimuEvent:
    """One completed op/phase span on a simulated rank.

    Time fields are milliseconds, matching the legacy log format.
    ``kind``/``lane`` are Phase-1 extensions; leave them None in Phase 0.
    """

    rank: int
    name: str                  # last call-stack segment (display name)
    call_stack: List[str]      # call-stack segments, rank prefix removed
    operation: str             # 'fwd' | 'bwd' | 'recompute_fwd'
    cost: float                # ms, == ed - st
    st: float                  # ms
    ed: float                  # ms
    gid: Optional[str] = None  # comm group/op id, None for pure compute
    post: Optional[float] = None   # ms, async p2p post timestamp
    order: Optional[int] = None    # async p2p post order
    stream: str = "comp"       # lane clock the span ran on
    kind: Optional[str] = None     # Phase 1: compute|comm|wait|scope|fused
    lane: Optional[str] = None     # Phase 1: explicit display lane
    owner_path: Optional[str] = None  # parent semantic owner, '/'-joined
    semantic_id: Optional[str] = None  # stable full call-stack identity
    phase_id: Optional[str] = None     # semantic_id plus operation phase
    scope: Optional[str] = None        # model|optimizer|pipeline|utility
    semantic_stage: Optional[str] = None
    layer_idx: Optional[int] = None
    stage_role: Optional[str] = None
    comm_owner: Optional[str] = None
    comm_role: Optional[str] = None
    group_kind: Optional[str] = None
    group_size: Optional[int] = None
    payload_bytes: Optional[int] = None
    net: Optional[str] = None
    comm_stage: Optional[str] = None
    # Structural fields used by portable trace consumers.  They are derived
    # from model tensors/configuration and communication construction order;
    # no measured timings or fitted alpha/beta values are stored here.
    shape_desc: Optional[str] = None
    shape_desc_by_stage: Optional[Dict[str, str]] = None
    input_shapes: Optional[List[List[int]]] = None
    output_shapes: Optional[List[List[int]]] = None
    input_dtypes: Optional[List[str]] = None
    output_dtypes: Optional[List[str]] = None
    dtype: Optional[str] = None
    kernel_role: Optional[str] = None
    projection: Optional[str] = None
    comm_sequence: Optional[int] = None
    comm_segment_index: Optional[int] = None
    comm_id: Optional[str] = None
    # Stable semantic communication-plan identity.  It is generated from the
    # model call stack/phase and structural sequence, never from profiler gid
    # values or measured timings.
    plan_id: Optional[str] = None
    # Optional model-declared consumer dependency.  Empty means the current
    # event stream proves only lifecycle ordering, not the downstream tensor
    # consumer that determines communication masking.
    consumer_id: Optional[str] = None
    # Semantic phase of the consumer that owns a wait/barrier.  This is
    # separate from ``operation`` because a backward reduce-scatter may be
    # issued in the backward queue but consumed by the optimizer tail.
    consumer_phase: Optional[str] = None
    depends_on: Optional[List[str]] = None
    # Structural dependency and overlap categories.  These are not measured
    # runtime factors and do not encode CANN/HCCL kernel identities.
    dependency_kind: Optional[str] = None
    dependency_status: Optional[str] = None
    ready_rule: Optional[str] = None
    overlap_policy: Optional[str] = None
    overlap_lanes: Optional[List[str]] = None
    # Stable structural occurrence information.  These counters are assigned
    # while the model emits the event stream and never depend on profiler
    # event ids or measured durations.
    event_index: Optional[int] = None
    semantic_occurrence: Optional[int] = None
    microbatch_index: Optional[int] = None
    aggregation_policy: Optional[str] = None
    logical_substep_count: Optional[int] = None
    # CollectiveCall lifecycle metadata.  The nested object carries the
    # portable structural contract and simulator-clock timestamps; measured
    # profiler duration is never stored as a model parameter here.
    collective: Optional[str] = None
    lifecycle_stage: Optional[str] = None
    algorithm: Optional[str] = None
    algorithm_stages: Optional[int] = None
    chunk_count: Optional[int] = None
    payload_per_chunk_bytes: Optional[int] = None
    post_time_ms: Optional[float] = None
    completion_time_ms: Optional[float] = None
    consumer_release_time_ms: Optional[float] = None
    lifecycle: Optional[Dict[str, object]] = None
    semantic_owner: Optional[str] = None
    layer_id: Optional[str] = None
    module_path: Optional[str] = None
    semantic_phase: Optional[str] = None
    consumer_event: Optional[str] = None
    iteration_boundary: Optional[str] = None
    physical_decomposition: Optional[str] = None
    # Explicitly false for ordinary model-generated events.  A calibrated
    # caller must opt in with ``True`` so provenance cannot be confused with
    # an omitted/null field in exported ledgers.
    measured_duration_used: bool = False
    # Portable layout/physical-work ownership.  These fields describe how a
    # semantic model event may be materialised by a backend; they are not
    # profiler kernel ids and never contain measured durations.
    fusion_scope: Optional[str] = None
    physical_work_id: Optional[str] = None
    memory_transaction_owner: Optional[str] = None
    physical_stage_role: Optional[str] = None
    layout_contract: Optional[Dict[str, object]] = None


class EventSink:
    """Collects SimuEvents during a simulation run."""

    def __init__(self) -> None:
        self.events: List[SimuEvent] = []
        # Per-rank structural communication sequence.  A sequence is assigned
        # to a model-generated communication id on first emission and reused
        # by its post/completion/wait spans.  This replaces duration sorting as
        # the primary simulation-side ordering key.
        self._comm_sequence_next: Dict[tuple, int] = {}
        self._comm_sequence_by_id: Dict[tuple, int] = {}
        self._event_index = 0
        self._semantic_occurrence_next: Dict[tuple, int] = {}
        # Spans whose call_stk lacks a 'rank<N>-' prefix are dropped here,
        # counted but otherwise ignored. This mirrors the legacy behavior
        # where such log lines were written to log.log but silently dropped
        # by the trace parser (e.g. utility modules in function.py whose
        # queues never received a rank-prefixed call_stk).
        self.dropped = 0

    def emit_span(self, call_stk, operation, st, ed, gid=None, post=None,
                  order=None, stream="comp", kind=None, lane=None, name=None,
                  owner_path=None, semantic_id=None, phase_id=None, scope=None,
                  metadata=None):
        """Append one span. ``call_stk`` keeps the legacy 'rankN-...' form.

        ``name`` overrides the display name (default: last call-stack
        segment) without changing the call-stack segments.
        """
        match = _RANK_PREFIX.match(call_stk)
        if not match:
            self.dropped += 1
            return
        segments = _normalise_call_segments(call_stk[match.end():].split("-"))
        semantic_id = semantic_id or "/".join(segments)
        owner_path = owner_path or "/".join(segments[:-1])
        phase_id = phase_id or f"{semantic_id}:{operation}"
        if scope is None:
            lowered = "/".join(segments).lower()
            if "optimizer" in lowered or "optimizer" in (name or "").lower():
                scope = "optimizer"
            elif "batch_pp" in lowered or "pipeline" in lowered:
                scope = "pipeline"
            else:
                scope = "model"
        semantic_meta = _derive_semantic_metadata(
            segments, name or segments[-1], operation, kind, metadata,
            stream=stream, lane=lane)
        if kind in ("comm", "wait"):
            lifecycle = dict(semantic_meta.get("lifecycle") or {})
            lifecycle.setdefault("phase", operation)
            if kind == "wait":
                lifecycle.setdefault("event_stage", "consumer_release")
                lifecycle.setdefault("consumer_release_time_ms", ed)
            elif str(name or "").endswith("-post"):
                lifecycle.setdefault("event_stage", "post")
                lifecycle.setdefault("post_time_ms", st)
            else:
                lifecycle.setdefault("event_stage", "completion")
                lifecycle.setdefault("completion_time_ms", ed)
            lifecycle.setdefault("time_provenance", "simulator_clock")
            semantic_meta["lifecycle"] = lifecycle
        microbatch_index = None
        for segment in segments:
            microbatch_match = _MICROBATCH.match(str(segment))
            if microbatch_match:
                microbatch_index = int(microbatch_match.group(1))
                break
        # Count occurrences within a semantic stage, rather than within the
        # full call-stack identity.  The latter is unique for every layer and
        # consequently made every ``semantic_occurrence`` equal to zero.  A
        # stage-local ordinal is useful to portable consumers (for example,
        # the Nth VWN/Norm/MatMul occurrence in a microbatch) while remaining
        # independent of profiler ids, durations, and hardware-specific
        # kernel names.
        occurrence_key = (
            int(match.group(1)), operation, semantic_meta.get("semantic_stage"),
            kind or "",
        )
        semantic_occurrence = self._semantic_occurrence_next.get(occurrence_key, 0)
        self._semantic_occurrence_next[occurrence_key] = semantic_occurrence + 1
        if kind in ("comm", "wait"):
            comm_id = semantic_meta.get("comm_id") or gid
            comm_key = (
                int(match.group(1)), operation,
                semantic_meta.get("semantic_stage"),
                semantic_meta.get("group_kind"),
                semantic_meta.get("comm_stage"),
                semantic_meta.get("comm_owner"),
                # Sequence is a structural occurrence ordinal for one
                # microbatch/stage/owner family.  Do not include the full
                # semantic_id here: it contains layer identity and would
                # reset every sequence to zero, making cross-layer order
                # unusable.  The call-level comm_id/phase_id below still
                # guarantees that post/completion/wait spans reuse one entry.
                microbatch_index,
            )
            id_key = (comm_key, str(comm_id), phase_id)
            if id_key not in self._comm_sequence_by_id:
                sequence = self._comm_sequence_next.get(comm_key, 0)
                self._comm_sequence_next[comm_key] = sequence + 1
                self._comm_sequence_by_id[id_key] = sequence
            semantic_meta.setdefault(
                "comm_sequence", self._comm_sequence_by_id[id_key])
            semantic_meta.setdefault(
                "comm_segment_index", self._comm_sequence_by_id[id_key])
            # A call-level id is deliberately separate from gid because gid can
            # be a tuple in the DES backend and is not always JSON-friendly.
            semantic_meta.setdefault("comm_id", str(comm_id) if comm_id is not None else None)
            # Do not include rank in plan_id.  The same semantic call emitted
            # on several ranks should be consumable as one abstract plan.
            plan_id = semantic_meta.get("plan_id")
            if plan_id is None:
                plan_id = (
                    f"derived/{semantic_id}/{operation}/"
                    f"s{semantic_meta.get('comm_sequence', 'u')}/"
                    f"{comm_id if comm_id is not None else 'unknown'}"
                )
                semantic_meta["plan_id"] = plan_id
        event_index = self._event_index
        self._event_index += 1
        self.events.append(SimuEvent(
            rank=int(match.group(1)),
            name=name or segments[-1],
            call_stack=segments,
            operation=operation,
            cost=ed - st,
            st=st,
            ed=ed,
            gid=gid,
            post=post,
            order=order,
            stream=stream,
            kind=kind,
            lane=lane,
            owner_path=owner_path,
            semantic_id=semantic_id,
            phase_id=phase_id,
            scope=scope,
            semantic_stage=semantic_meta.get("semantic_stage"),
            layer_idx=semantic_meta.get("layer_idx"),
            stage_role=semantic_meta.get("stage_role"),
            comm_owner=semantic_meta.get("comm_owner"),
            comm_role=semantic_meta.get("comm_role"),
            group_kind=semantic_meta.get("group_kind"),
            group_size=semantic_meta.get("group_size"),
            payload_bytes=semantic_meta.get("payload_bytes", semantic_meta.get("size_bytes")),
            net=semantic_meta.get("net"),
            comm_stage=semantic_meta.get("comm_stage"),
            shape_desc=semantic_meta.get("shape_desc"),
            shape_desc_by_stage=semantic_meta.get("shape_desc_by_stage"),
            input_shapes=semantic_meta.get("input_shapes"),
            output_shapes=semantic_meta.get("output_shapes"),
            input_dtypes=semantic_meta.get("input_dtypes"),
            output_dtypes=semantic_meta.get("output_dtypes"),
            dtype=semantic_meta.get("dtype"),
            kernel_role=semantic_meta.get("kernel_role"),
            projection=semantic_meta.get("projection"),
            comm_sequence=semantic_meta.get("comm_sequence"),
            comm_segment_index=semantic_meta.get("comm_segment_index"),
            comm_id=semantic_meta.get("comm_id"),
            plan_id=semantic_meta.get("plan_id"),
            consumer_id=semantic_meta.get("consumer_id"),
            consumer_phase=semantic_meta.get("consumer_phase"),
            depends_on=semantic_meta.get("depends_on")
            or semantic_meta.get("dependency_ids"),
            dependency_kind=semantic_meta.get("dependency_kind"),
            dependency_status=semantic_meta.get("dependency_status"),
            ready_rule=semantic_meta.get("ready_rule"),
            overlap_policy=semantic_meta.get("overlap_policy"),
            overlap_lanes=semantic_meta.get("overlap_lanes"),
            event_index=event_index,
            semantic_occurrence=semantic_occurrence,
            microbatch_index=microbatch_index,
            aggregation_policy=semantic_meta.get("aggregation_policy"),
            logical_substep_count=semantic_meta.get("logical_substep_count"),
            collective=semantic_meta.get("collective")
            or (semantic_meta.get("lifecycle") or {}).get("collective"),
            lifecycle_stage=semantic_meta.get("lifecycle_stage")
            or (semantic_meta.get("lifecycle") or {}).get("event_stage"),
            algorithm=semantic_meta.get("algorithm")
            or (semantic_meta.get("lifecycle") or {}).get("algorithm"),
            algorithm_stages=semantic_meta.get("algorithm_stages")
            or (semantic_meta.get("lifecycle") or {}).get("algorithm_stages"),
            chunk_count=semantic_meta.get("chunk_count")
            or (semantic_meta.get("lifecycle") or {}).get("chunk_count"),
            payload_per_chunk_bytes=semantic_meta.get("payload_per_chunk_bytes")
            or (semantic_meta.get("lifecycle") or {}).get("payload_per_chunk_bytes"),
            post_time_ms=(semantic_meta.get("lifecycle") or {}).get("post_time_ms"),
            completion_time_ms=(semantic_meta.get("lifecycle") or {}).get(
                "completion_time_ms"),
            consumer_release_time_ms=(semantic_meta.get("lifecycle") or {}).get(
                "consumer_release_time_ms"),
            lifecycle=semantic_meta.get("lifecycle"),
            semantic_owner=semantic_meta.get("semantic_owner"),
            layer_id=semantic_meta.get("layer_id"),
            module_path=semantic_meta.get("module_path"),
            semantic_phase=(
                semantic_meta.get("semantic_phase")
                or semantic_meta.get("phase")
                or operation
            ),
            consumer_event=semantic_meta.get("consumer_event"),
            iteration_boundary=semantic_meta.get("iteration_boundary"),
            physical_decomposition=semantic_meta.get("physical_decomposition"),
            measured_duration_used=semantic_meta.get("measured_duration_used", False),
            fusion_scope=semantic_meta.get("fusion_scope"),
            physical_work_id=semantic_meta.get("physical_work_id"),
            memory_transaction_owner=semantic_meta.get("memory_transaction_owner"),
            physical_stage_role=semantic_meta.get("physical_stage_role"),
            layout_contract=semantic_meta.get("layout_contract"),
        ))


def event_to_record(event: SimuEvent) -> dict:
    """Adapt a SimuEvent to the dict shape the trace converter consumes.

    Values are rounded to 6 decimals to reproduce the legacy text
    round-trip exactly.
    """
    return {
        "rank": f"rank{event.rank}",
        "name": event.name,
        "call_stack": list(event.call_stack),
        "gid": event.gid,
        "operation": event.operation,
        "cost": round(event.cost, 6),
        "st": round(event.st, 6),
        "ed": round(event.ed, 6),
        "post": round(event.post, 6) if event.post is not None else None,
        "order": event.order,
        "stream": event.stream,
        "kind": event.kind,
        "lane": event.lane,
        "owner_path": event.owner_path,
        "semantic_id": event.semantic_id,
        "phase_id": event.phase_id,
        "scope": event.scope,
        "semantic_stage": event.semantic_stage,
        "layer_idx": event.layer_idx,
        "stage_role": event.stage_role,
        "comm_owner": event.comm_owner,
        "comm_role": event.comm_role,
        "group_kind": event.group_kind,
        "group_size": event.group_size,
        "payload_bytes": event.payload_bytes,
        "net": event.net,
        "comm_stage": event.comm_stage,
        "shape_desc": event.shape_desc,
        "shape_desc_by_stage": event.shape_desc_by_stage,
        "input_shapes": event.input_shapes,
        "output_shapes": event.output_shapes,
        "input_dtypes": event.input_dtypes,
        "output_dtypes": event.output_dtypes,
        "dtype": event.dtype,
        "kernel_role": event.kernel_role,
        "projection": event.projection,
        "comm_sequence": event.comm_sequence,
        "comm_segment_index": event.comm_segment_index,
        "comm_id": event.comm_id,
        "plan_id": event.plan_id,
        "consumer_id": event.consumer_id,
        "consumer_phase": event.consumer_phase,
        "depends_on": event.depends_on,
        "dependency_kind": event.dependency_kind,
        "dependency_status": event.dependency_status,
        "ready_rule": event.ready_rule,
        "overlap_policy": event.overlap_policy,
        "overlap_lanes": event.overlap_lanes,
        "event_index": event.event_index,
        "semantic_occurrence": event.semantic_occurrence,
        "microbatch_index": event.microbatch_index,
        "aggregation_policy": event.aggregation_policy,
        "logical_substep_count": event.logical_substep_count,
        "collective": event.collective,
        "lifecycle_stage": event.lifecycle_stage,
        "algorithm": event.algorithm,
        "algorithm_stages": event.algorithm_stages,
        "chunk_count": event.chunk_count,
        "payload_per_chunk_bytes": event.payload_per_chunk_bytes,
        "post_time_ms": event.post_time_ms,
        "completion_time_ms": event.completion_time_ms,
        "consumer_release_time_ms": event.consumer_release_time_ms,
        "lifecycle": event.lifecycle,
        "semantic_owner": event.semantic_owner,
        "layer_id": event.layer_id,
        "module_path": event.module_path,
        "semantic_phase": event.semantic_phase,
        "consumer_event": event.consumer_event,
        "iteration_boundary": event.iteration_boundary,
        "physical_decomposition": event.physical_decomposition,
        "measured_duration_used": event.measured_duration_used,
        "fusion_scope": event.fusion_scope,
        "physical_work_id": event.physical_work_id,
        "memory_transaction_owner": event.memory_transaction_owner,
        "physical_stage_role": event.physical_stage_role,
        "layout_contract": event.layout_contract,
    }


def format_event_line(event: SimuEvent) -> str:
    """Render one event in the legacy log line format (debug artifact)."""
    call_stk = f"rank{event.rank}-" + "-".join(event.call_stack)
    gid_part = f" gid {event.gid}" if event.gid is not None else ""
    tail = ""
    if event.post is not None:
        tail += f" post {event.post:.6f}"
    if event.order is not None:
        tail += f" order {event.order}"
    return (f"{call_stk}{gid_part} {event.operation} "
            f"cost {event.cost:.6f} st {event.st:.6f} ed {event.ed:.6f}{tail}")


def write_debug_log(events, log_path) -> None:
    """Write the legacy text log from the event stream (one-way, debug only)."""
    with open(log_path, "w") as f:
        for event in events:
            f.write(format_event_line(event) + "\n")
