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


def _normalise_token(value: object) -> str:
    """Return a comparison token without making any hardware assumptions."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


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
                              metadata: Optional[Dict[str, object]]) -> Dict[str, object]:
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
    supplied.update({
        "semantic_stage": stage,
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


class EventSink:
    """Collects SimuEvents during a simulation run."""

    def __init__(self) -> None:
        self.events: List[SimuEvent] = []
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
            segments, name or segments[-1], operation, kind, metadata)
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
