"""Various utilities"""

import json
import math
import os, subprocess


def get_chunk_idx(args):
    """Return per-model chunk index when present."""
    return getattr(args, "chunk_idx", None)


def format_scope_microbatch_tag(args, include_chunk=False):
    """Format a trace/scope-friendly microbatch tag.

    In VPP we expose the Megatron-like data microbatch id as ``microbatch``
    and keep uniqueness by appending ``chunk`` when requested.
    """
    tag = f"microbatch{args.microbatch}"
    chunk_idx = get_chunk_idx(args)
    if include_chunk and chunk_idx is not None:
        tag += f"-chunk{chunk_idx}"
    return tag


def format_model_info_microbatch_tag(args):
    """Format a model-info/debug tag with optional chunk identity."""
    tag = f"microbatch:{args.microbatch}"
    chunk_idx = get_chunk_idx(args)
    if chunk_idx is not None:
        tag += f"-chunk:{chunk_idx}"
    return tag

def wrap_name(src):
    return f"_orig_{src}"

def add_attr(module, name, target):
    setattr(module, name, target)

def wrap_attr(module, name, wrapper):
    target = getattr(module, name)
    setattr(module, wrap_name(name), target)
    setattr(module, name, wrapper)

def replace_attr(module, name, target):
    wrap_attr(module, name, target)
    
class HumanReadableSize:
    """Convert a size in bytes to a human-readable format."""

    BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"]
    NUM_UNITS = ["", "K", "M", "B", "T"]
    TIME_UNITS = ["ms", "s"]

    def __init__(
        self, value, base=1024, units=None, source_unit=None, target_unit=None
    ):
        """
        :param value: original value
        :param base: base: 1024 for byte conversion, 1000 for FLOPS or parameter conversion
        :param units:  ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        :param target_unit:  target unit, if specified, force conversion to this unit
        """
        self.original_value = float(value)
        self.base = base
        self.units = units or ["B", "KB", "MB", "GB", "TB", "PB"]
        self.source_unit = source_unit or self.units[0]
        self.target_unit = target_unit
        assert self.source_unit in self.units
        assert self.target_unit is None or self.target_unit in self.units
        self.converted_value, self.unit = self._convert()

    def _convert(self):
        size = self.original_value
        source_index = self.units.index(self.source_unit)
        size_in_base_unit = size * (self.base**source_index)

        # If the target unit is provided, convert to the specified unit
        if self.target_unit and self.target_unit in self.units:
            target_index = self.units.index(self.target_unit)
            size_in_target_unit = size_in_base_unit / (self.base**target_index)
            return size_in_target_unit, self.target_unit

        # Automatically select units
        unit_index = 0
        size_in_target_unit = size_in_base_unit
        while size_in_target_unit >= self.base and unit_index < len(self.units) - 1:
            size_in_target_unit /= self.base
            unit_index += 1

        return size_in_target_unit, self.units[unit_index]

    @staticmethod
    def from_string(size_str, units, base, target_unit=None):
        """Parse a size string like '500 MB' or '1 GB'."""
        value, source_unit = size_str.split(" ")
        if source_unit not in units:
            raise ValueError(f"Unknown unit: '{source_unit}'")
        return HumanReadableSize(
            float(value),
            base=base,
            units=units,
            source_unit=source_unit,
            target_unit=target_unit,
        )

    def __str__(self):
        return f"{self.converted_value:.4f} {self.unit}"

    def get_value(self):
        return self.converted_value

    def get_unit(self):
        return self.unit


def human_readable_bytes(value, target_unit=None):
    return str(
        HumanReadableSize(
            value,
            base=1024,
            units=HumanReadableSize.BYTE_UNITS,
            target_unit=target_unit,
        )
    )


def human_readable_nums(value, target_unit=None):
    return str(
        HumanReadableSize(
            value, base=1000, units=HumanReadableSize.NUM_UNITS, target_unit=target_unit
        )
    )


def human_readable_times(value, target_unit=None):
    return str(
        HumanReadableSize(
            value,
            base=1000,
            units=HumanReadableSize.TIME_UNITS,
            target_unit=target_unit,
        )
    )


def convert_final_result_to_human_format(result: dict):
    """：
    Based on the regularity of the key value of result,
    convert the value to a human-readable format.
    """
    if result is None:
        return

    for k, v in result.items():
        if not isinstance(v, (int, float, dict)):
            continue
        if isinstance(v, dict):
            convert_final_result_to_human_format(v)
            continue
        convert_func = None
        if "time" in k:
            convert_func = human_readable_times
        elif "mem" in k or "bytes" in k:
            convert_func = human_readable_bytes
        elif "flops" in k:
            convert_func = human_readable_nums
        if convert_func is None:
            continue
        result[k] = convert_func(v)
    return


def to_json_string(obj: dict):
    return json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False)


def get_point_name(parent, current, sep=" -> ") -> str:
    if parent and current:
        res = parent + sep + current
    else:
        res = parent if parent else current
    return res


def path_convert_to_str(path: list) -> str:
    path_name = ""
    if len(path) == 1:
        path_name = path[0]
    elif len(path) > 1:
        path_name = " -> ".join(path)
    return path_name

def get_pp_stage_representative_rank(pp_rank, strategy):
    """Pick the representative dense rank for one PP stage.

    The representative rank keeps `tp=0`, `cp=0`, and `dp=0`, and varies only
    along the pipeline dimension. pp is always the outermost dim — its stride
    is the product of all dense dims under any placement (see
    `_dense_strides`) — so with the inner coords fixed at 0 the member sits
    at `pp_rank * tp*cp*dp` whatever the inner permutation is (inner dims
    contribute 0 * stride = 0). The formula is placement-invariant.
    """

    return pp_rank * strategy.tp_size * strategy.cp_size * strategy.dp_size


def get_pp_p2p_comm_size(strategy, hidden_size, dtype_size):
    hidden_states_size = (
        strategy.micro_batch_size
        * strategy.seq_len
        * hidden_size
    )
    pp_comm_size = hidden_states_size * dtype_size / strategy.cp_size
    if strategy.enable_sequence_parallel:
        pp_comm_size = pp_comm_size / strategy.tp_size
    return pp_comm_size


# --- Placement parsing (design_simu_hierarchical_network.md, section 4) ---
#
# `strategy.order_of_paralielism` declares how the dense parallel dims are
# laid onto the physical hierarchy, innermost first (the default
# "tp-cp-ep-dp-pp" is the legacy hardcoded mesh). The MoE mesh (ep/etp/edp)
# keeps its fixed order and pp is always outermost (v1 constraint), so a
# placement reduces to a permutation of the dense dims tp/cp/dp.
DEFAULT_PLACEMENT = ["tp", "cp", "dp"]


def parse_placement(order_str):
    """Parse `order_of_paralielism` into the dense dim order (innermost first).

    Returns the dense dims as a list, e.g. "tp-cp-ep-dp-pp" ->
    ["tp", "cp", "dp"] and "cp-tp-ep-dp-pp" -> ["cp", "tp", "dp"].
    Rules (mirrors StrategyConfig._validate_order_of_paralielism):
    None/missing -> DEFAULT_PLACEMENT; "ep" tokens are dropped (the MoE mesh
    placement is fixed); the remaining tokens must be exactly one each of
    tp/cp/dp in any order with an optional trailing "pp" (pp, when present,
    must be outermost); anything else raises ValueError.
    """
    if order_str is None:
        return list(DEFAULT_PLACEMENT)
    tokens = str(order_str).split("-")
    if any(token == "" for token in tokens):
        raise ValueError(f"invalid placement {order_str!r}: empty token")
    tokens = [token for token in tokens if token != "ep"]
    if "pp" in tokens:
        if tokens[-1] != "pp":
            raise ValueError(
                f"invalid placement {order_str!r}: pp must be outermost (last)")
        tokens.pop()
    if sorted(tokens) != ["cp", "dp", "tp"]:
        raise ValueError(
            f"invalid placement {order_str!r}: dense dims must contain "
            "exactly one each of tp/cp/dp in any order")
    return tokens


def _dense_strides(strategy, placement=None):
    """Return per-dim member strides of the dense mesh from the placement.

    A dim's stride is the product of the sizes of the dims placed BEFORE it
    (the inner dims): with the default placement ["tp", "cp", "dp"] this
    yields exactly tp=1, cp=tp_size, dp=tp_size*cp_size. "pp" is always
    outermost (v1 constraint), so its stride is the product of all dense
    dims regardless of the declared order.
    """
    if placement is None:
        placement = parse_placement(getattr(strategy, "order_of_paralielism", None))
    strides = {}
    stride = 1
    for dim in placement:
        strides[dim] = stride
        stride *= getattr(strategy, f"{dim}_size")
    strides["pp"] = stride
    return strides


def get_rank_group(global_rank, strategy, placement=None):
    ## dense order: parsed from strategy.order_of_paralielism (default
    ## tp-cp-dp, pp always outermost); moe order remains ep-etp-edp-pp

    strides = _dense_strides(strategy, placement)
    tp_rank = (global_rank // strides["tp"]) % strategy.tp_size
    cp_rank = (global_rank // strides["cp"]) % strategy.cp_size
    dp_rank = (global_rank // strides["dp"]) % strategy.dp_size
    # dp_cp flattens the (cp, dp) plane with cp inner, the same enumeration
    # as the legacy default-order formula (rank // tp) % (cp*dp); the coords
    # above are placement-aware, so the flattening stays valid (a bijection
    # within the group) under any order.
    dp_cp_rank = cp_rank + dp_rank * strategy.cp_size
    dense_group_id_suffix = None
    # When fsdp_shard_size overrides the dense FSDP group, project the
    # rank into [0, fsdp_dense_group_size) so the comm op's local rank
    # is valid. The exact group composition is an approximation; the
    # levels router uses _group_size_and_stride for cross-node routing.
    if strategy.fsdp_shard_size is not None:
        dense_group_size = strategy.fsdp_dense_group_size
        dense_base_stride = min(strides["cp"], strides["dp"])
        dense_stride = dense_base_stride
        if dense_group_size * dense_base_stride > strategy.world_size:
            dense_stride = max(1, strategy.world_size // dense_group_size)
        dp_cp_rank = (global_rank // dense_stride) % dense_group_size
        dense_group_id_suffix = global_rank % dense_stride
    pp_rank = global_rank // strides["pp"]
    ep_rank = global_rank % strategy.ep_size
    logical_edp_rank = global_rank // strategy.ep_size % strategy.edp_size
    edp_rank = logical_edp_rank
    moe_group_id_suffix = None
    if strategy.oe_shard_size is not None:
        moe_group_size = strategy.fsdp_moe_group_size
        moe_base_stride = strategy.ep_size * strategy.etp_size
        moe_stride = moe_base_stride
        if moe_group_size * moe_base_stride > strategy.world_size:
            moe_stride = max(1, strategy.world_size // moe_group_size)
        edp_rank = (global_rank // moe_stride) % moe_group_size
        moe_group_id_suffix = global_rank % moe_stride
    tp_group_id = f"pp:{pp_rank}-cp:{cp_rank}-dp:{dp_rank}"
    pp_group_id = f"tp:{tp_rank}-cp:{cp_rank}-dp:{dp_rank}"
    dp_group_id = f"tp:{tp_rank}-pp:{pp_rank}"
    dp_cp_group_id = f"tp:{tp_rank}-pp:{pp_rank}"
    if dense_group_id_suffix is not None:
        dp_cp_group_id += f"-fsdp:{dense_group_id_suffix}"
    cp_group_id = f"tp:{tp_rank}-pp:{pp_rank}-dp:{dp_rank}"
    # EP collectives retain the logical EP×EDP mesh.  An OE/FSDP shard
    # override must not leak into the router EP group id.
    ep_group_id = f"tp:{tp_rank}-pp:{pp_rank}-edp:{logical_edp_rank}"
    if moe_group_id_suffix is None:
        edp_group_id = f"tp:{tp_rank}-pp:{pp_rank}-ep:{ep_rank}"
    else:
        # OE shard groups are formed by the configured arithmetic-progression
        # stride and therefore span the logical EP ranks.
        edp_group_id = f"tp:{tp_rank}-pp:{pp_rank}-oe:{moe_group_id_suffix}"
    dic = {
        "tp_group_id": tp_group_id,
        "tp_rank": tp_rank,
        "cp_group_id": cp_group_id,
        "cp_rank": cp_rank,
        "pp_group_id": pp_group_id,
        "pp_rank": pp_rank,
        "dp_group_id": dp_group_id,
        "dp_rank": dp_rank,
        "dp_cp_group_id": dp_cp_group_id,
        "dp_cp_rank": dp_cp_rank,
        "ep_group_id": ep_group_id,
        "ep_rank": ep_rank,
        "edp_group_id": edp_group_id,
        "edp_rank": edp_rank,
    }
    return dic


# --- Group -> node mapping (design_simu_network_fabric.md, Tier A) ---
#
# Members of a single-dimension group form an arithmetic progression
# `base + k*stride` in the rank mesh. The dense strides come from the
# placement (`_dense_strides`; default order tp-cp-dp with pp outermost,
# MoE order fixed ep-etp-edp-pp). The node count and cross-node traffic
# ratio of any collective therefore follow in O(1) from (group_size, stride).
def _group_size_and_stride(group_kind, strategy, placement=None):
    """Return (group_size, member_stride) for a parallelism group kind."""
    if group_kind in ("tp", "cp", "dp", "dp_cp", "pp"):
        strides = _dense_strides(strategy, placement)
        if group_kind in ("tp", "cp", "dp"):
            return getattr(strategy, f"{group_kind}_size"), strides[group_kind]
        if group_kind == "dp_cp":
            # dp_cp is Megatron's dense optimizer/data-parallel group: the
            # (cp, dp) plane with tp/pp fixed, size cp*dp. Derivation: the
            # plane is exactly the progression `base + k*min(stride_cp,
            # stride_dp)` whenever cp and dp are adjacent in the placement —
            # the inner of the two dims then tiles the plane contiguously
            # (default order: stride_cp=tp, stride_dp=tp*cp -> stride tp,
            # identical to the legacy hardcode). When tp sits between cp and
            # dp (orders cp-tp-dp / dp-tp-cp) the plane is not an exact
            # progression — the outer dim's blocks are strided by more than
            # the inner dim's span — and we keep the inner dim's stride as
            # the contiguous-plane approximation (v1 limitation).
            override = getattr(strategy, 'fsdp_shard_size', None)
            if override is not None:
                group_size = override
            else:
                group_size = strategy.dp_size * strategy.cp_size
            base_stride = min(strides["cp"], strides["dp"])
            world_size = getattr(strategy, 'world_size', group_size * base_stride)
            if override is not None and group_size * base_stride > world_size:
                stride = max(1, world_size // group_size)
            else:
                stride = base_stride
            return (group_size, stride)
        return strategy.pp_size, strides["pp"]
    if group_kind == "ep":
        return strategy.ep_size, 1
    if group_kind == "etp":
        return strategy.etp_size, strategy.ep_size
    if group_kind == "edp":
        override = getattr(strategy, 'oe_shard_size', None)
        if override is not None:
            group_size = override
        else:
            group_size = getattr(strategy, 'edp_size', 1)
        base_stride = getattr(strategy, 'ep_size', 1) * getattr(strategy, 'etp_size', 1)
        world_size = getattr(strategy, 'world_size', group_size * base_stride)
        if override is not None and group_size * base_stride > world_size:
            stride = max(1, world_size // group_size)
        else:
            stride = base_stride
        return (group_size, stride)
    raise ValueError(f"unknown group_kind: {group_kind!r}")


def group_node_stats(group_kind, strategy, num_per_node):
    """Return (group_size, node_count) spanned by a parallelism group.

    The first member is assumed node-aligned (rank 0 convention, matching
    the legacy ceil-based heuristics). With stride >= num_per_node every
    member sits on its own node; with stride < num_per_node members are
    packed contiguously so no intermediate node is skipped.
    """
    group_size, stride = _group_size_and_stride(group_kind, strategy)
    if group_size <= 1 or num_per_node <= 1:
        return group_size, 1
    if stride >= num_per_node:
        return group_size, group_size
    node_count = (group_size - 1) * stride // num_per_node + 1
    return group_size, node_count


def group_cross_node_ratio(group_kind, strategy, num_per_node):
    """Fraction of a collective's traffic that crosses node boundaries.

    Mirrors the legacy (k-1)/k convention but with the real node count:
    0.0 when the group fits in one node, (nodes-1)/nodes otherwise.
    """
    _, node_count = group_node_stats(group_kind, strategy, num_per_node)
    if node_count <= 1:
        return 0.0
    return (node_count - 1) / node_count


# --- Hierarchical level mapping (design_simu_hierarchical_network.md, T2) ---
#
# A "levels" topology is an ordered list of {"name", "size", "net"}
# entries, innermost first, where size counts how many units of the
# previous level this level contains (the first level's unit is one GPU).
class LevelSpan:
    """How one comm domain decomposes across one topology level."""

    def __init__(self, name, size, net, span, units_touched, members_per_unit,
                 kind="clos", convergence_ratio=1.0, port_num=None,
                 bandwidth_per_port_gbps=None):
        self.name = name
        self.size = size
        self.net = net
        self.span = span                    # cumulative GPUs per unit
        self.units_touched = units_touched  # distinct units of this level
        self.members_per_unit = members_per_unit
        # Physical topology type (design doc Part C): "fullmesh" = dedicated
        # per-pair links; "clos" = shared switch uplink with convergence.
        self.kind = kind
        self.convergence_ratio = convergence_ratio
        # Physical fabric params (完整物理描述，可选): per-device port count and
        # per-port effective bandwidth → 层有效带宽 = port_num × per_port ÷ conv.
        self.port_num = port_num
        self.bandwidth_per_port_gbps = bandwidth_per_port_gbps

    def __repr__(self):
        return (f"LevelSpan({self.name}: size={self.size} net={self.net} "
                f"units={self.units_touched} members/unit={self.members_per_unit:g} "
                f"kind={self.kind} conv={self.convergence_ratio})")


def group_level_span(group_kind, strategy, levels):
    """Decompose a parallelism group across a hierarchical topology.

    Returns (composition, spans):
    - composition: list of c_L = touched units of level L-1 per touched
      unit of level L (the user's [2, 8, 2]-style proportions);
      c_L > 1 means a communication phase exists at level L.
    - spans: list[LevelSpan] with units_touched / members_per_unit.

    levels: list of dicts {"name", "size", "net"} (innermost first).
    With a single level of size num_per_node this reduces to
    group_node_stats semantics (composition == [members per node]).
    """
    group_size, stride = _group_size_and_stride(group_kind, strategy)
    if not levels:
        return [group_size], []
    spans = []
    cumulative = 1
    for entry in levels:
        cumulative *= int(entry["size"])
        if group_size <= 1:
            units = 1
        elif stride >= cumulative:
            units = group_size
        else:
            units = (group_size - 1) * stride // cumulative + 1
        members = group_size / units if units else 0.0
        spans.append(LevelSpan(
            entry["name"], int(entry["size"]), entry["net"],
            cumulative, units, members,
            kind=entry.get("kind", "clos"),
            convergence_ratio=entry.get("convergence_ratio", 1.0),
            port_num=entry.get("port_num"),
            bandwidth_per_port_gbps=entry.get("bandwidth_per_port_gbps")))
    composition = []
    prev = group_size
    for span in spans:
        c = prev / span.units_touched if span.units_touched else 0.0
        composition.append(c)
        prev = span.units_touched
    return composition, spans


def all2all_level_fraction(group_kind, strategy, levels, level_index):
    """Fraction of an all2all member's traffic that uses the given level's links.

    A member's K-1 peers partition into bands by the innermost level whose
    unit contains them: peers in the same unit of level i but in different
    units of level i-1 use level i's fabric. The band fraction is
    (members_i - members_{i-1}) / (K - 1) with members_{-1} = 1 (self).

    Innermost level (index 0): the band is the intra-unit peers,
    (members_0 - 1) / (K - 1). When the group fits entirely within one
    level-0 unit (units_touched == 1), members_0 == K and the fraction is 1.0.
    Outermost level: carries the residual traffic — peers not in the same
    unit of the second-to-last level — (K - members_{n-2}) / (K - 1). This
    handles degenerate outer levels (size=1) whose members_per_unit does not
    grow, e.g. a so_clos fabric connecting whole nodes.
    The fractions across all levels sum to 1.0.
    """
    group_size, _ = _group_size_and_stride(group_kind, strategy)
    if group_size <= 1:
        return 0.0
    _, spans = group_level_span(group_kind, strategy, levels)
    n_levels = len(spans)
    span = spans[level_index]
    if n_levels == 1:
        # Single level: all traffic uses this fabric.
        return 1.0
    if level_index == 0:
        # Innermost level: only intra-unit peers use this fabric. When the
        # group fits entirely within one unit (e.g. EP=128 within a 128-GPU
        # pod), members_0 == K and the fraction is 1.0.
        return (span.members_per_unit - 1) / (group_size - 1)
    prev_members = spans[level_index - 1].members_per_unit
    if level_index == n_levels - 1:
        # Outermost level: residual traffic (peers in different units of the
        # second-to-last level), including peers split across this level's
        # own units. Handles degenerate size=1 outer levels.
        return (group_size - prev_members) / (group_size - 1)
    return (span.members_per_unit - prev_members) / (group_size - 1)


# --- Machine-level straggler model (design_simu_network_fabric.md, Phase C) ---
#
# Shared by the analytical path (PerfLLM, gated by
# strategy.enable_straggler_model) and the DES "virtual waiters" collective
# skew (strategy.collective_skew). Kept next to the group->node helpers
# above since both quantify node-granularity effects.
STRAGGLER_BASE_FACTOR = 0.09


def get_effective_straggler_sample_count(
    world_size: int,
    num_per_node: int,
    dp_size: int,
    edp_size: int,
) -> int:
    """Estimate the number of independent machine-level straggler samples.

    SimuMax assumes GPUs on the same node are performance-stable, while node-to-
    node runtime can fluctuate. Under that assumption, the effective sample
    count should be limited by:

    - how many nodes are present
    - how many dense-DP replicas are active
    - how many expert-DP replicas are active

    Using min(node_count, dp_size, edp_size) keeps single-node and small-scale
    runs from exaggerating straggler inflation.
    """

    safe_num_per_node = max(1, int(num_per_node))
    node_count = max(1, math.ceil(int(world_size) / safe_num_per_node))
    return max(1, min(node_count, int(dp_size), int(edp_size)))


def estimate_straggler_increase_ratio(worker_count: int) -> float:
    """Empirical machine-level straggler inflation ratio.

    The formula preserves the expected sqrt(log n) growth of the maximum over
    many machines, while damping small-n behavior to match local simulations.
    """

    n = max(1, int(worker_count))
    if n <= 1:
        return 1.0
    n_straggler = math.log2(n)
    return 1.0 + n_straggler / (n_straggler + 1.0) * STRAGGLER_BASE_FACTOR * math.sqrt(n_straggler)


def merge_dict(cur_data, merges_data):
    if len(merges_data) == 0:
        for key, value in cur_data.items():
            merges_data[key] = [value]
    else:
        for key, value in cur_data.items():
            merges_data[key].append(value)  
    return merges_data

def rm_tmp():
    if os.path.exists("./tmp"):
        subprocess.run(["rm", "-rf", "./tmp"])
