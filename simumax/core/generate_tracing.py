import json
import re
import heapq
import os

from simumax.core.simu_events import event_to_record


def parse_log_line(line):
    """Parse one simulator log line."""
    log_pattern = re.compile(
        r"(?P<rank>rank\d+)-(?P<call_stack>[\w-]+)\s+"
        r"(?:gid\s+(?P<gid>\S+)\s+)?"
        r"(?P<operation>\w+)\s+"
        r"cost\s+(?P<cost>\d+\.\d+)\s+"
        r"st\s+(?P<st>\d+\.\d+)\s+"
        r"ed\s+(?P<ed>\d+\.\d+)"
        r"(?:\s+post\s+(?P<post>\d+\.\d+))?"
        r"(?:\s+order\s+(?P<order>-?\d+))?"
    )
    match = log_pattern.match(line)
    if not match:
        return None

    return {
        "rank": match.group("rank"),
        "call_stack": match.group("call_stack").split("-"),
        "gid": match.group("gid"),
        "operation": match.group("operation"),
        "cost": float(match.group("cost")),
        "st": float(match.group("st")),
        "ed": float(match.group("ed")),
        "post": float(match.group("post")) if match.group("post") is not None else None,
        "order": int(match.group("order")) if match.group("order") is not None else None,
    }


def _rank_sort_key(pid):
    match = re.match(r"rank(\d+)", str(pid))
    return int(match.group(1)) if match else 0


def _is_post_marker_name(base_name):
    """Detect zero-duration post markers by their display-name suffix.

    Phase 2 (design doc 9.3): every blocking comm op emits an extra
    zero-duration marker at its issue time, displayed as the completion
    span's name plus "-post" and sharing the completion's comm gid. The
    pairing logic below assumes exactly 1 send + 1 recv per gid, so markers
    must be excluded from pairability detection and flow generation. They
    still render as normal zero-duration comm slices on their lane.
    """
    return str(base_name).endswith("-post")


def _ordered_tid(tid):
    if str(tid).startswith("pp_detail_"):
        return f"07_{tid}"
    order = {
        "fwd_scope": "00_fwd_scope",
        "bwd_scope": "01_bwd_scope",
        "fwd_compute": "02_fwd_compute",
        "bwd_compute": "03_bwd_compute",
        "pp_fwd": "04_pp_fwd",
        "pp_bwd": "05_pp_bwd",
        "pp_batch_scope": "06_pp_batch_scope",
        "pp_detail": "07_pp_detail",
        "comm": "08_comm",
        "wait": "09_wait",
    }
    return order.get(tid, f"99_{tid}")


def _display_tid(ordered_tid):
    parts = str(ordered_tid).split("_", 1)
    return parts[1] if len(parts) == 2 else str(ordered_tid)


def _thread_sort_index(tid):
    tid = str(tid)
    if tid.startswith("07_pp_detail_"):
        suffix = tid[len("07_pp_detail_") :]
        try:
            return 60 + int(suffix)
        except ValueError:
            return 60
    order = {
        "00_fwd_scope": 0,
        "01_bwd_scope": 1,
        "02_fwd_compute": 2,
        "03_bwd_compute": 3,
        "04_pp_fwd": 4,
        "05_pp_bwd": 5,
        "06_pp_batch_scope": 6,
        "08_comm": 80,
        "09_wait": 90,
    }
    return order.get(tid, 100)


def _flow_anchor_ts(event, prefer_end=False):
    """Anchor flow markers inside the slice to avoid boundary collisions."""
    ts = float(event["ts"])
    dur = float(event.get("dur") or 0.0)
    if dur <= 0.0:
        return ts

    # Keep the original "start vs end" intent, but move the anchor just inside
    # the slice so adjacent comm events that share a boundary do not visually
    # attach to the wrong slice in the trace viewer.
    pad = min(1e-3, dur * 0.25)
    if prefer_end:
        return ts + max(0.0, dur - pad)
    return ts + pad


def _pp_display_pad_ts(event):
    """Slightly shrink PP comm slices to avoid zero-gap viewer swallowing."""
    args = event.get("args", {}) or {}
    gid = args.get("gid")
    if not gid or "send_recv-" not in str(gid):
        return 0.0

    dur = float(event.get("dur") or 0.0)
    if dur <= 0.0:
        return 0.0

    # Keep the slice shrink smaller than the flow-anchor epsilon so flows still
    # land safely inside the visible body after the display-only trim.
    return min(5e-4, dur * 0.1)


def _apply_scope_recategorization(event):
    """Move one inclusive parent/container event onto its scope lane."""
    args = event.get("args", {})
    direction = args.get("direction", "fwd")
    event["cat"] = "scope"
    if direction == "recompute_fwd":
        event["tid"] = "bwd_scope"
        if args.get("base_name", "") == "recompute_block":
            event["name"] = "recompute_fwd"
    else:
        event["tid"] = f"{direction}_scope"


def _recategorize_scope_events(tracing_events):
    """Split inclusive parent/module envelopes from leaf compute events.

    This keeps real kernels on compute lanes while preserving hierarchy on a
    dedicated scope lane. Stack-based sweep per pid: compute events are sorted
    by (ts asc, dur desc) and scanned once while a stack of still-open
    candidate containers is maintained. An event becomes a scope when a later
    event's call_stack strictly extends its own while being temporally
    enclosed, or when it is one of the named recompute containers.
    """
    compute_by_pid = {}
    for event in tracing_events:
        if event.get("cat") != "compute" or event.get("ph") != "X":
            continue
        compute_by_pid.setdefault(event["pid"], []).append(event)

    for _, events in compute_by_pid.items():
        events.sort(key=lambda event: (event["ts"], -(event["dur"] or 0.0)))
        # Open containers that may still enclose later events. Each entry is
        # [end_ts, call_stack, event, already_scope].
        open_stack = []
        for event in events:
            args = event.get("args", {})
            call_stack = args.get("call_stack", [])
            if not call_stack:
                continue
            start_ts = event["ts"]
            end = start_ts + (event.get("dur") or 0.0)
            # Containers closed before this event starts (beyond the inclusive
            # epsilon) cannot enclose it or any later event.
            while open_stack and start_ts > open_stack[-1][0] + 1e-9:
                open_stack.pop()
            # Every still-open container whose call_stack is a strict prefix of
            # this event's is an inclusive parent envelope, i.e. a scope.
            for entry in open_stack:
                if entry[3]:
                    continue
                if len(call_stack) <= len(entry[1]):
                    continue
                if call_stack[: len(entry[1])] != entry[1]:
                    continue
                if end <= entry[0] + 1e-9:
                    _apply_scope_recategorization(entry[2])
                    entry[3] = True
            is_scope = args.get("base_name", "") in ("recompute_block", "checkpoint_bwd")
            if is_scope:
                _apply_scope_recategorization(event)
            open_stack.append([end, call_stack, event, is_scope])


def convert_to_tracing_format(parsed_logs):
    """
    Convert parsed logs to Chrome Tracing events.

    Records are dicts from parse_log_line (legacy text path) or
    event_to_record (SimuEvent stream). Classification comes from the
    explicit record kind/lane/stream annotations; records without them fall
    back to the structural batch_pp/compute rules.
    """
    tracing_events = []
    event_id_counter = 0

    # Only assign correlation/group id to gids that have both send and recv events.
    pairable_gid = {}
    for log in parsed_logs:
        gid = log.get("gid")
        if not gid:
            continue
        if log.get("kind") == "fused":
            # Fused slices share a gid across resource lanes but are not
            # send/recv pairs; they must not affect comm pairability.
            continue
        call_stack = log.get("call_stack", [])
        if not call_stack:
            continue
        base = log.get("name") or call_stack[-1]
        if _is_post_marker_name(base):
            # Post markers share the gid with their completion span; they are
            # not real send/recv events and must not affect pairability.
            continue
        st = pairable_gid.setdefault(gid, {"send": False, "recv": False})
        if base.startswith(("send_", "async_send", "sync_send")):
            st["send"] = True
        if base.startswith(("recv_", "async_recv", "sync_recv")):
            st["recv"] = True
    pairable_gid = {k for k, v in pairable_gid.items() if v["send"] and v["recv"]}

    gid_to_correlation_id = {}
    corr_counter = 0
    # Slices of one fused op (same gid) share an independent f<id> suffix.
    gid_to_fused_corr_id = {}
    fused_corr_counter = 0

    for log in parsed_logs:
        rank = log["rank"]
        call_stack = log["call_stack"]
        operation = log["operation"]
        # Simulator timestamps are in milliseconds; Chrome tracing uses microseconds.
        st = log["st"] * 1e3
        ed = log["ed"] * 1e3
        base_name = log.get("name") or call_stack[-1]
        gid = log.get("gid")

        # Phase 1: display classification comes from the explicit SimuEvent
        # annotations (kind/lane/stream). Records without a kind (legacy text
        # log lines, pipeline-schedule exports) keep the structural fallback.
        kind = log.get("kind")
        if kind:
            stream_type = kind
        elif base_name.startswith("batch_pp"):
            stream_type = "scope"
        else:
            stream_type = "compute"
        if stream_type == "comm":
            stream = log.get("stream") or "comp"
            lane = log.get("lane") or (stream if stream != "comp" else "comm")
        elif stream_type == "fused":
            # Each fused slice renders on the resource lane it occupies.
            lane = log.get("stream") or "comp"
        elif stream_type == "runtime":
            # Runtime transfers (currently activation offload) carry an
            # explicit resource lane.  Keep it visible in Chrome trace rather
            # than collapsing every runtime event into a generic compute lane.
            lane = log.get("lane") or log.get("stream") or "runtime"
        else:
            lane = stream_type
        if stream_type == "wait":
            tid = "wait"
        elif stream_type in ("comm", "fused", "runtime"):
            tid = lane
        elif stream_type == "scope" and base_name.startswith("batch_pp"):
            tid = "pp_batch_scope"
        else:
            tid = "bwd_compute" if operation == "recompute_fwd" else f"{operation}_{stream_type}"
        cat = stream_type
        corr_id = None
        if stream_type == "comm" and gid in pairable_gid:
            if gid not in gid_to_correlation_id:
                gid_to_correlation_id[gid] = corr_counter
                corr_counter += 1
            corr_id = gid_to_correlation_id[gid]
        fused_corr_id = None
        if stream_type == "fused" and gid:
            if gid not in gid_to_fused_corr_id:
                gid_to_fused_corr_id[gid] = fused_corr_counter
                fused_corr_counter += 1
            fused_corr_id = gid_to_fused_corr_id[gid]
        name = base_name
        if corr_id is not None:
            # Visual grouping without flow lines: same comm pair gets same g<id>.
            name = f"{base_name}[g{corr_id}]"
        if fused_corr_id is not None:
            # Slices of the same fused op share one f<id> across lane slices.
            name = f"{base_name}[f{fused_corr_id}]"

        event_id = event_id_counter
        event_id_counter += 1
        tracing_events.append(
            {
                "name": name,
                "cat": cat,
                "ph": "X",
                "ts": st,
                "dur": max(0.0, ed - st),
                "pid": rank,
                "tid": tid,
                "id": event_id,
                "args": {
                    "call_stack": call_stack,
                    "stream_type": stream_type,
                    "lane": lane,
                    "lane_base": lane,
                    "direction": operation,
                    "gid": gid,
                    "correlation_id": corr_id,
                    "base_name": base_name,
                    "owner_path": log.get("owner_path"),
                    "semantic_id": log.get("semantic_id"),
                    "phase_id": log.get("phase_id"),
                    "scope": log.get("scope"),
                    # Portable semantic ledger fields.  These are derived
                    # from the model call stack and communication object
                    # metadata; they do not contain measured timings or
                    # fitted alpha/beta values.
                    "semantic_stage": log.get("semantic_stage"),
                    "layer_idx": log.get("layer_idx"),
                    "stage_role": log.get("stage_role"),
                    "comm_owner": log.get("comm_owner"),
                    "comm_role": log.get("comm_role"),
                    "group_kind": log.get("group_kind"),
                    "group_size": log.get("group_size"),
                    "payload_bytes": log.get("payload_bytes", log.get("size_bytes")),
                    "net": log.get("net"),
                    "comm_stage": log.get("comm_stage"),
                    "shape_desc": log.get("shape_desc"),
                    "shape_desc_by_stage": log.get("shape_desc_by_stage"),
                    "input_shapes": log.get("input_shapes"),
                    "output_shapes": log.get("output_shapes"),
                    "input_dtypes": log.get("input_dtypes"),
                    "output_dtypes": log.get("output_dtypes"),
                    "dtype": log.get("dtype"),
                    "kernel_role": log.get("kernel_role"),
                    "projection": log.get("projection"),
                    "comm_sequence": log.get("comm_sequence"),
                    "comm_segment_index": log.get("comm_segment_index"),
                    "comm_id": log.get("comm_id"),
                    "plan_id": log.get("plan_id"),
                    "consumer_id": log.get("consumer_id"),
                    "consumer_phase": log.get("consumer_phase"),
                    "depends_on": log.get("depends_on"),
                    "dependency_kind": log.get("dependency_kind"),
                    "dependency_status": log.get("dependency_status"),
                    "ready_rule": log.get("ready_rule"),
                    "overlap_policy": log.get("overlap_policy"),
                    "overlap_lanes": log.get("overlap_lanes"),
                    # Stable model-side event identity.  These counters are
                    # assigned by EventSink from the generated call stack and
                    # operation phase; they are exported for portable trace
                    # consumers and never depend on profiler event ids or
                    # measured durations.
                    "event_index": log.get("event_index"),
                    "semantic_occurrence": log.get("semantic_occurrence"),
                    "microbatch_index": log.get("microbatch_index"),
                    "aggregation_policy": log.get("aggregation_policy"),
                    "logical_substep_count": log.get("logical_substep_count"),
                    "collective": log.get("collective"),
                    "lifecycle_stage": log.get("lifecycle_stage"),
                    "algorithm": log.get("algorithm"),
                    "algorithm_stages": log.get("algorithm_stages"),
                    "chunk_count": log.get("chunk_count"),
                    "payload_per_chunk_bytes": log.get("payload_per_chunk_bytes"),
                    "post_time_ms": log.get("post_time_ms"),
                    "completion_time_ms": log.get("completion_time_ms"),
                    "consumer_release_time_ms": log.get("consumer_release_time_ms"),
                    "lifecycle": log.get("lifecycle"),
                    "semantic_owner": log.get("semantic_owner"),
                    "layer_id": log.get("layer_id"),
                    "module_path": log.get("module_path"),
                    "semantic_phase": log.get("semantic_phase"),
                    "consumer_event": log.get("consumer_event"),
                    "iteration_boundary": log.get("iteration_boundary"),
                    "physical_decomposition": log.get("physical_decomposition"),
                    "measured_duration_used": log.get("measured_duration_used"),
                    "fusion_scope": log.get("fusion_scope"),
                    "physical_work_id": log.get("physical_work_id"),
                    "memory_transaction_owner": log.get("memory_transaction_owner"),
                    "physical_stage_role": log.get("physical_stage_role"),
                    "layout_contract": log.get("layout_contract"),
                    "post_ts": (log.get("post") * 1e3) if log.get("post") is not None else None,
                    "post_order": log.get("order"),
                },
            }
        )

    # Overlapping sync detail send/recv events need separate visual sub-lanes,
    # otherwise flow markers can attach ambiguously when multiple events share
    # the same pid/tid and overlap in time.
    detail_by_pid = {}
    for event in tracing_events:
        if event.get("ph") != "X" or event.get("cat") != "comm":
            continue
        if (event.get("args", {}) or {}).get("lane") != "pp_detail":
            continue
        detail_by_pid.setdefault(event["pid"], []).append(event)

    for _, events in detail_by_pid.items():
        events.sort(key=lambda event: (event["ts"], event.get("dur") or 0.0, event.get("id", 0)))
        active = []  # heap[(end_ts, lane_idx)]
        free_lanes = []
        next_lane = 0
        for event in events:
            start = float(event["ts"])
            while active and active[0][0] <= start + 1e-9:
                _, lane_idx = heapq.heappop(active)
                heapq.heappush(free_lanes, lane_idx)
            if free_lanes:
                lane_idx = heapq.heappop(free_lanes)
            else:
                lane_idx = next_lane
                next_lane += 1
            event["tid"] = f"pp_detail_{lane_idx}"
            event["args"]["lane"] = event["tid"]
            event["args"]["detail_lane_idx"] = lane_idx
            heapq.heappush(active, (start + float(event.get("dur") or 0.0), lane_idx))

    # When async PP comm has a known local post time and the rank's comm stream
    # is idle, pull the displayed event start left to that post time.
    # This removes false bubbles without changing end-to-end pairing or
    # introducing overlap on the single comm lane.
    comm_by_pid_lane = {}
    for event in tracing_events:
        if event.get("cat") == "comm" and event.get("ph") == "X":
            comm_by_pid_lane.setdefault((event["pid"], event.get("tid")), []).append(event)

    for _, events_on_rank in comm_by_pid_lane.items():
        events_on_rank.sort(key=lambda event: (event["ts"], event.get("id", 0)))
        prev_end = None
        for event in events_on_rank:
            args = event.get("args", {}) or {}
            post_ts = args.get("post_ts")
            if post_ts is None:
                prev_end = event["ts"] + (event.get("dur") or 0.0)
                continue
            original_ts = float(event["ts"])
            original_end = original_ts + (event.get("dur") or 0.0)
            post_ts = float(post_ts)
            candidate_ts = post_ts if prev_end is None else max(post_ts, prev_end)
            if candidate_ts < original_ts:
                event["ts"] = candidate_ts
                event["dur"] = max(0.0, original_end - candidate_ts)
            prev_end = event["ts"] + (event.get("dur") or 0.0)

    # Display-only polish for PP p2p events: shrink the visible slice slightly
    # so tightly adjacent zero-gap comm events remain individually visible in
    # trace viewers. Keep this smaller than the flow anchor epsilon.
    for event in tracing_events:
        if event.get("cat") != "comm" or event.get("ph") != "X":
            continue
        pad = _pp_display_pad_ts(event)
        if pad <= 0.0:
            continue
        ts = float(event["ts"])
        dur = float(event.get("dur") or 0.0)
        if dur <= 2.0 * pad:
            continue
        event["ts"] = ts + pad
        event["dur"] = max(0.0, dur - 2.0 * pad)

    # Split inclusive parent/module envelopes from leaf compute events.
    # This keeps real kernels on compute lanes while preserving hierarchy on a
    # dedicated scope lane.
    _recategorize_scope_events(tracing_events)

    # For PP point-to-point comm, emit one direct flow per pair. In async mode
    # the anchor is nudged inside the slice when post_ts exists; in sync mode
    # the anchor stays on the boundary so the line only represents pair
    # identity, not extra launch semantics.
    by_gid = {}
    for event in tracing_events:
        if event.get("cat") != "comm":
            continue
        gid = event.get("args", {}).get("gid")
        if not gid or "send_recv-" not in gid:
            continue
        by_gid.setdefault(gid, []).append(event)

    flow_id = 0
    for gid, events in by_gid.items():
        # Post markers ("-post" suffix) share the gid with their completion
        # span; exclude them so the exactly-1-send + 1-recv check below only
        # counts real comm events.
        sends = [
            event
            for event in events
            if event.get("args", {}).get("base_name", "").startswith(
                ("send_", "async_send", "sync_send")
            )
            and not _is_post_marker_name(event.get("args", {}).get("base_name", ""))
        ]
        recvs = [
            event
            for event in events
            if event.get("args", {}).get("base_name", "").startswith(
                ("recv_", "async_recv", "sync_recv")
            )
            and not _is_post_marker_name(event.get("args", {}).get("base_name", ""))
        ]
        if len(sends) != 1 or len(recvs) != 1:
            continue

        send = sends[0]
        recv = recvs[0]
        send_post = (send.get("args", {}) or {}).get("post_ts")
        recv_post = (recv.get("args", {}) or {}).get("post_ts")
        tracing_events.append(
            {
                "name": f"pair:{gid}",
                "cat": "comm_pair",
                "ph": "s",
                "ts": _flow_anchor_ts(send, prefer_end=False),
                "pid": send["pid"],
                "tid": send["tid"],
                "id": flow_id,
                "args": {"gid": gid},
            }
        )
        tracing_events.append(
            {
                "name": f"pair:{gid}",
                "cat": "comm_pair",
                "ph": "f",
                "ts": _flow_anchor_ts(recv, prefer_end=True),
                "pid": recv["pid"],
                "tid": recv["tid"],
                "bp": "e",
                "id": flow_id,
                "args": {"gid": gid},
            }
        )
        flow_id += 1

    for event in tracing_events:
        if "tid" in event:
            event["tid"] = _ordered_tid(event["tid"])

    process_ids = sorted({event["pid"] for event in tracing_events if "pid" in event}, key=_rank_sort_key)
    metadata_events = []
    for proc_idx, pid in enumerate(process_ids):
        metadata_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "args": {"name": pid},
            }
        )
        metadata_events.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "args": {"sort_index": proc_idx},
            }
        )
        metadata_events.append(
            {
                "name": "sort_index",
                "ph": "M",
                "pid": pid,
                "args": {"sort_index": proc_idx},
            }
        )
        tids = sorted({event["tid"] for event in tracing_events if event.get("pid") == pid and "tid" in event})
        for tid in tids:
            metadata_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": _display_tid(tid)},
                }
            )
            metadata_events.append(
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": _thread_sort_index(tid)},
                }
            )
            metadata_events.append(
                {
                    "name": "sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": _thread_sort_index(tid)},
                }
            )

    tracing_events = metadata_events + tracing_events

    return tracing_events


def process_log_file(log_path, output_json_path):
    """Read a simulator log file and write Chrome Tracing JSON."""
    parsed_logs = []

    with open(log_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if "cost" in line and "st" in line and "ed" in line:
                parsed_log = parse_log_line(line)
                if parsed_log:
                    parsed_logs.append(parsed_log)

    tracing_events = convert_to_tracing_format(parsed_logs)

    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(tracing_events, json_file, indent=4)

    print(f"Processed {len(parsed_logs)} logs. Saved to {output_json_path}.")


def write_trace_file(events, output_json_path, provenance=None):
    """Convert SimuEvents to Chrome trace JSON and a provenance ledger.

    ``provenance`` is supplied by the configured SystemConfig.  The fallback
    keeps this exporter compatible with standalone callers and only inspects
    explicit event flags; it never infers measured parameters from durations.
    """
    records = [event_to_record(event) for event in events]
    tracing_events = convert_to_tracing_format(records)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(tracing_events, f, indent=4)
    # Keep a machine-readable semantic ledger next to the Chrome trace.  The
    # ledger is intentionally derived from the model event stream only; it is
    # not a calibration table and contains no measured trace values.
    ledger_path = os.path.join(
        os.path.dirname(os.fspath(output_json_path)),
        "semantic_event_ledger.json",
    )
    ledger_events = []
    ledger_provenance = dict(provenance or {})
    if not ledger_provenance:
        performance_used = any(
            record.get("measured_duration_used") is True
            for record in records)
        ledger_provenance = {
            "mode": "unknown_export_context",
            "structural_observations_used": False,
            "shape_observations_used": False,
            "kernel_role_observations_used": False,
            "performance_duration_observations_used": performance_used,
            "performance_observations_used_as_parameters": performance_used,
            "model_structure_and_system_config_used": True,
            "calibration_parameter_count": 0,
            "calibration_parameters": [],
            "measured_source": None,
            "inference_basis": "explicit_event_flags_only",
        }
    for index, record in enumerate(records):
        ledger_events.append({
            "event_index": index,
            "rank": record.get("rank"),
            "name": record.get("name"),
            "call_stack": record.get("call_stack"),
            "operation": record.get("operation"),
            "kind": record.get("kind"),
            "lane": record.get("lane"),
            "st_ms": record.get("st"),
            "ed_ms": record.get("ed"),
            "cost_ms": record.get("cost"),
            "gid": record.get("gid"),
            "owner_path": record.get("owner_path"),
            "semantic_id": record.get("semantic_id"),
            "phase_id": record.get("phase_id"),
            "semantic_stage": record.get("semantic_stage"),
            "layer_idx": record.get("layer_idx"),
            "stage_role": record.get("stage_role"),
            "scope": record.get("scope"),
            "comm_owner": record.get("comm_owner"),
            "comm_role": record.get("comm_role"),
            "group_kind": record.get("group_kind"),
            "group_size": record.get("group_size"),
            "payload_bytes": record.get("payload_bytes"),
            "net": record.get("net"),
            "comm_stage": record.get("comm_stage"),
            "shape_desc": record.get("shape_desc"),
            "shape_desc_by_stage": record.get("shape_desc_by_stage"),
            "input_shapes": record.get("input_shapes"),
            "output_shapes": record.get("output_shapes"),
            "input_dtypes": record.get("input_dtypes"),
            "output_dtypes": record.get("output_dtypes"),
            "dtype": record.get("dtype"),
            "kernel_role": record.get("kernel_role"),
            "projection": record.get("projection"),
            "comm_sequence": record.get("comm_sequence"),
            "comm_segment_index": record.get("comm_segment_index"),
            "comm_id": record.get("comm_id"),
            "plan_id": record.get("plan_id"),
            "consumer_id": record.get("consumer_id"),
            "consumer_phase": record.get("consumer_phase"),
            "depends_on": record.get("depends_on"),
            "dependency_kind": record.get("dependency_kind"),
            "dependency_status": record.get("dependency_status"),
            "ready_rule": record.get("ready_rule"),
            "overlap_policy": record.get("overlap_policy"),
            "overlap_lanes": record.get("overlap_lanes"),
            "event_index": record.get("event_index"),
            "semantic_occurrence": record.get("semantic_occurrence"),
            "microbatch_index": record.get("microbatch_index"),
            "aggregation_policy": record.get("aggregation_policy"),
            "logical_substep_count": record.get("logical_substep_count"),
            "collective": record.get("collective"),
            "lifecycle_stage": record.get("lifecycle_stage"),
            "algorithm": record.get("algorithm"),
            "algorithm_stages": record.get("algorithm_stages"),
            "chunk_count": record.get("chunk_count"),
            "payload_per_chunk_bytes": record.get("payload_per_chunk_bytes"),
            "post_time_ms": record.get("post_time_ms"),
            "completion_time_ms": record.get("completion_time_ms"),
            "consumer_release_time_ms": record.get("consumer_release_time_ms"),
            "lifecycle": record.get("lifecycle"),
            "semantic_owner": record.get("semantic_owner"),
            "layer_id": record.get("layer_id"),
            "module_path": record.get("module_path"),
            "semantic_phase": record.get("semantic_phase"),
            "consumer_event": record.get("consumer_event"),
            "iteration_boundary": record.get("iteration_boundary"),
            "physical_decomposition": record.get("physical_decomposition"),
            "measured_duration_used": bool(
                record.get("measured_duration_used", False)),
            "fusion_scope": record.get("fusion_scope"),
            "physical_work_id": record.get("physical_work_id"),
            "memory_transaction_owner": record.get("memory_transaction_owner"),
            "physical_stage_role": record.get("physical_stage_role"),
            "layout_contract": record.get("layout_contract"),
        })
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema": "simumax_semantic_event_ledger_v2",
            "source": "model_structure_and_configuration",
            "provenance": ledger_provenance,
            "measured_data_used_as_parameters": bool(
                ledger_provenance.get(
                    "performance_observations_used_as_parameters", False)),
            "events": ledger_events,
        }, f, indent=2)
    print(f"Processed {len(records)} logs. Saved to {output_json_path}.")


if __name__ == "__main__":
    process_log_file("./tmp/log.log", "./tmp/tracing_logs.json")
