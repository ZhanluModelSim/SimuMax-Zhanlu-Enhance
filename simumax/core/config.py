"""Configuration classes for SimuMax """
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from collections import OrderedDict
import json
import copy
import math
import types
import warnings
import re

from simumax.core.utils import (
    to_json_string,
    group_cross_node_ratio,
    group_level_span,
    all2all_level_fraction,
)
from simumax.core.fusion import FUSION_POLICIES, build_fusion_policy
from simumax.core.cost_specs import get_block_template
from simumax.core.communication_plan import collective_algorithm as portable_collective_algorithm

capture_graph_only = False
ENABLE_SIMU_GRAPH = int(os.environ.get("ENABLE_SIMU_GRAPH", "0"))
SIMU_CHECK = int(os.environ.get("SIMU_CHECK", "0"))
SIMU_DEBUG = int(os.environ.get('SIMU_DEBUG', '0'))
SIMU_TMP_PATH_OVERRIDE = os.environ.get("SIMUMAX_TMP_PATH", "").strip()
if SIMU_TMP_PATH_OVERRIDE:
    TMP_PATH = SIMU_TMP_PATH_OVERRIDE
elif SIMU_CHECK:
    TMP_PATH = "tmp_check"
else:
    TMP_PATH = "tmp" + time.strftime("_%Y%m%d_%H%M%S", time.localtime())

kNetOp = (
    "all_reduce",
    "all_gather",
    "all_gatherv",
    "reduce_scatter",
    "reduce_scatterv",
    "p2p",
    "all2all",
    "alltoallv",
    "fsdp_all_gather",
    "fsdp_reduce_scatter",
    "model_embed_ag",
    "model_moe_ag",
    "model_moe_rs",
    "sync_all_reduce",
    "moe_small_a2a",
)

NET_OP_FALLBACK = {
    "alltoallv": "all2all",
    "all_gatherv": "all_gather",
    "reduce_scatterv": "reduce_scatter",
    "fsdp_all_gather": "all_gather",
    "fsdp_reduce_scatter": "reduce_scatter",
    "model_embed_ag": "all_gather",
    "model_moe_ag": "all_gather",
    "model_moe_rs": "reduce_scatter",
    "sync_all_reduce": "all_reduce",
    "moe_small_a2a": "all2all",
}


def set_capture_graph_only(value: bool):
    global capture_graph_only
    capture_graph_only = value

def get_capture_graph_only():
    return capture_graph_only

class ParameterExtractor:
    def __init__(self, param_patterns: Dict[str, Any]):
        # Parameter patterns and default values.
        self.param_patterns = param_patterns
    
    def extract_parameters(self, input_string):
        """Extract all configured parameters from input string."""
        parameters = {}
        
        for param_name, (pattern, default_value) in self.param_patterns.items():
            match = re.search(pattern, input_string)
            if match:
                parameters[param_name] = int(match.group(1))
            elif default_value is not None:
                parameters[param_name] = default_value
                print(f"Warning: parameter {param_name} not found, use default {default_value}")

        return parameters
    
    def extract_single_parameter(self, input_string, param_name, default_value=None):
        """Extract a single parameter by name."""
        if param_name not in self.param_patterns:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        pattern, default = self.param_patterns[param_name]
        if default_value is not None:
            default = default_value
        
        match = re.search(pattern, input_string)
        if match:
            return int(match.group(1))
        else:
            print(f"Warning: parameter {param_name} not found, use default {default}")
            return default


def _validate_efficiency_override_table(table, field_name):
    """Validate one per-operator efficiency table (cost-tunability design doc 4).

    Grammar per key (class_key or path_key): either a scalar efficiency in
    (0, 1], or a dict ``{"default": float, "shapes": {shape_desc: float}}``
    where "shapes" is optional. Raises AssertionError on invalid grammars.
    """
    if table is None:
        return
    assert isinstance(table, dict), (
        f"{field_name} must be a dict of key -> efficiency, but got {type(table)}"
    )

    def _check_eff(value, ctx):
        assert (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 < value <= 1
        ), f"{ctx} must be a number in (0, 1], but got {value!r}"

    for key, value in table.items():
        assert isinstance(key, str) and key, (
            f"{field_name} keys must be non-empty str, but got {key!r}"
        )
        ctx = f"{field_name}[{key!r}]"
        if not isinstance(value, dict):
            _check_eff(value, ctx)
            continue
        unknown_keys = set(value) - {"default", "shapes"}
        assert not unknown_keys, (
            f"{ctx} has unknown keys {sorted(unknown_keys)}, "
            "allowed keys are ['default', 'shapes']"
        )
        assert "default" in value, f"{ctx} must contain a 'default' entry"
        _check_eff(value["default"], f"{ctx}['default']")
        shapes = value.get("shapes")
        if shapes is not None:
            assert isinstance(shapes, dict), (
                f"{ctx}['shapes'] must be a dict of shape_desc -> efficiency, "
                f"but got {type(shapes)}"
            )
            for shape_desc, eff in shapes.items():
                assert isinstance(shape_desc, str), (
                    f"{ctx}['shapes'] keys must be str, but got {shape_desc!r}"
                )
                _check_eff(eff, f"{ctx}['shapes'][{shape_desc!r}]")

@dataclass
class Config:
    """
    Base class for all configuration
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.
        Automatically includes properties and fields.
        """
        def _normalize_jsonable(value):
            if isinstance(value, dict):
                return {
                    key: _normalize_jsonable(val)
                    for key, val in value.items()
                }
            if isinstance(value, list):
                return [_normalize_jsonable(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_normalize_jsonable(item) for item in value)
            if isinstance(value, set):
                return [_normalize_jsonable(item) for item in sorted(value)]
            return value

        # Start with the regular dataclass fields
        output = asdict(self)

        # Use reflection to automatically add all @property attributes
        for attr_name in dir(self):
            attr_value = getattr(self.__class__, attr_name, None)
            if isinstance(attr_value, property):
                output[attr_name] = _normalize_jsonable(getattr(self, attr_name))

        return _normalize_jsonable(output)

    def sanity_check(self) -> None:
        # Implement basic sanity checks here
        pass

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return to_json_string(self.to_dict())

    def __str__(self):
        return self.to_json_string()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"

    @classmethod
    def init_from_dict(cls, config_dict: Dict[str, Any]):
        """
        Initializes an instance from a dictionary.
        It handles nested dictionaries recursively.
        """
        return cls(**config_dict)

    @staticmethod
    def read_json_file(json_file: str) -> Dict[str, Any]:
        """Reads a JSON file and returns a dictionary."""
        with open(json_file, "r", encoding="utf-8") as reader:
            return json.load(reader)

    @classmethod
    def init_from_config_file(cls, config_file: str):
        """Initializes an instance from a JSON config file."""
        config_dict = cls.read_json_file(config_file)
        return cls.init_from_dict(config_dict)

@dataclass
class AttentionRecomputeConfig(Config):
    # input_norm_recompute:bool = False
    # qkv_norm_recompute:bool = False
    # qkv_recompute:bool = False
    # attn_recompute:bool = False
    # out_recompute:bool = False

    input_layernorm_recompute:bool = False

    q_down_recompute:bool = False
    kv_down_recompute:bool = False
    q_up_recompute:bool = False
    kv_up_recompute:bool = False

    q_layernorm_recompute:bool = False
    kv_layernorm_recompute:bool = False

    rope_recompute:bool = False
    core_attn_recompute:bool = False

    out_recompute:bool = False

    megatron_layernorm: bool = False
    megatron_mla_up_proj: bool = False

    def set_all_status(self, status:bool):
        self.input_layernorm_recompute = status
        self.q_down_recompute = status
        self.kv_down_recompute = status
        self.q_up_recompute = status
        self.kv_up_recompute = status
        self.q_layernorm_recompute = status
        self.kv_layernorm_recompute = status
        self.rope_recompute = status
        self.core_attn_recompute = status
        self.out_recompute = status

    @property
    def is_recompute_all(self):
        return all(self.__dict__.values())

@dataclass
class MLPRecomputeConfig(Config):
    pre_mlp_norm_recompute:bool = False
    shared_linear_recompute:bool = False
    linear_recompute:bool = False # Noraml MLP and grouped MLP
    router_recompute:bool = False
    permutation_recompute:bool = False

    megatron_layernorm: bool = False
    megatron_mlp: bool = False
    megatron_moe: bool = False
    megatron_moe_act: bool = False
    
    @property
    def is_recompute_all(self):
        return (self.pre_mlp_norm_recompute and 
                self.linear_recompute and 
                self.router_recompute and 
                self.permutation_recompute)
@dataclass
class StrategyConfig(Config):
    """
    Training strategy configuration
    """

    seq_len: Optional[int] = None
    micro_batch_size: Optional[int] = None
    micro_batch_num: Optional[int] = None
    dtype: Optional[int] = 'bf16'
    fp8: Optional[bool] = False
    
    # dist strategy
    world_size: Optional[int] = 8
    tp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    etp_size: int = 1
    cp_comm_type: str = "a2a"
    cp_a2a_mode: str = "async_cp"
    order_of_paralielism: str = "tp-cp-ep-dp-pp"
    moe_dispatcher_policy: str = "all2all"
    # Preserve variable-count collectives in the model graph/trace. This is a
    # framework implementation choice, not a performance calibration: AGV/RSV/
    # A2AV still use the same topology.levels beta/latency as their fixed-count
    # collective families.
    moe_variable_collectives: bool = False
    num_layers_in_first_pipeline_stage: Optional[int] = None
    num_layers_in_last_pipeline_stage: Optional[int] = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    # memory optimization
    grad_reduce_in_bf16: bool = False
    cache_groupgemm_col_fp8_inputs: Optional[bool] = False
    offload_groupgemm_col_inputs: Optional[bool] = False

    attn_recompute: bool = False
    mla_rms_recompute: bool = False 
    mlp_recompute: bool  = False
    mlp_rms_recompute: bool = False

    enable_sequence_parallel: bool = True
    interleaving_size: int = 1
    microbatch_group_size_per_vp_stage: Optional[int] = None
    pp_comm_async: bool = True
    enable_straggler_model: bool = True
    # DES-side collective skew switch (network-fabric design doc section 8,
    # Phase C). enable_straggler_model scales the analytical run_estimate()
    # result; collective_skew instead skews local collectives inside the
    # simulate() DES path and leaves the analytical estimate untouched.
    collective_skew: Optional[str] = None
    zero_state: int = 1
    # FSDP communication pattern (ZeRO-3 design doc section 3). Only
    # meaningful when zero_state >= 3; validated and warned below if set
    # with a lower zero_state.
    fsdp_mode: str = "model-wise"
    # FSDP sharding strategy (FSDP2 gap analysis doc section 3.1).
    # True  = FULL_SHARD (FSDP2 default): params resharded after forward,
    #          backward requires all-gather to re-unshard.
    # False = SHARD_GRAD_OP: params stay unsharded after forward, backward
    #          does not need all-gather. Only meaningful when zero_state >= 3
    #          and fsdp_mode == "layer-wise".
    reshard_after_forward: bool = True
    # Number of successor layers to prefetch in layer-wise FSDP (FSDP2 gap
    # analysis doc section 3.1). 0 = no successor prefetch (only the current
    # layer's AGs are posted). 1 = the historical implicit prefetch (the next
    # layer's AGs overlap with current compute). 2+ = explicit deeper
    # prefetch. The total structural AG depth is therefore 1 + this value.
    # Only meaningful when zero_state >= 3 and fsdp_mode == "layer-wise".
    fsdp_prefetch_layers: int = 1
    # FSDP forward AG consumer dependency. ``shared`` preserves the historical
    # one-barrier model in which dense and MoE AGs of a layer are released
    # together. ``split`` is a structural what-if/portable schedule option:
    # dense parameters are released before Attention and MoE parameters before
    # the MoE block. It changes dependency placement only; it does not alter
    # collective payloads or costs.
    fsdp_ag_consumer_dependency_mode: str = "shared"
    # Runtime queue depth for layer-wise FSDP gradient reduce-scatter. This is
    # a framework scheduling choice (not a measured overlap coefficient): the
    # producer waits for the oldest bucket before posting bucket N+depth.
    fsdp_max_inflight_reduce_scatters: int = 1
    # Optional semantic communication streams for layer-wise FSDP.  The
    # default keeps the historical single ``dp_comm`` queue.  A configured
    # mapping may declare independent framework streams for all-gather and
    # reduce-scatter; it changes only queue dependencies, never payload or
    # collective cost.  This is intentionally a strategy choice rather than
    # a hardware/CANN constant.
    fsdp_comm_streams: Optional[Dict[str, str]] = None
    # Override the dense FSDP shard group size. When set, the dense
    # all-gather/reduce-scatter and ZeRO-1/2/3 memory sharding use this
    # value instead of the default dp_size * cp_size. This models
    # frameworks whose FSDP group spans a different set of parallelism
    # dimensions than SimuMax's (dp, cp) plane.
    fsdp_shard_size: Optional[int] = None
    # Override the MoE (expert) FSDP shard group size. When set, the MoE
    # all-gather/reduce-scatter and ZeRO-1/2/3 memory sharding use this
    # value instead of the default edp_size.
    oe_shard_size: Optional[int] = None
    # Some training stacks shard embedding/lm-head and expert parameters in a
    # model-level unit in addition to transformer-block FSDP. Keep that graph
    # choice explicit instead of inferring it from a profiler trace.
    fsdp_sync_non_transformer_parameters: bool = False
    # Activation offload (training behavior, fully forward-derived cost).
    # Mirrors the real training flags `--model.activation_offload.*`:
    #   {"llm": "input", "single_block_mode": true, "block_size_in_gb": 20}
    # The DES injects one D2H transfer after each layer's forward and one H2D
    # transfer before that layer's backward/recompute. The transfer volume is
    # min(per-layer activation cache bytes, block_size_in_gb) — both structural
    # / configured facts, no measured fitting. The host-side transfer bandwidth
    # and latency are declared hardware facts in SystemConfig.activation_offload.
    # None (default) disables the behavior entirely, so configs that do not set
    # it (e.g. the 16p regression config) are bit-for-bit unaffected.
    activation_offload: Optional[dict] = None
    # DES trace granularity for the optimizer.  The analytical optimizer
    # model derives a Newton--Schulz/orthogonal update as one logical phase,
    # while ``detailed`` can be selected when callers need every structural
    # sub-step for debugging.  The default is semantic because the base
    # model is an offline cost model rather than a CANN kernel emulator.
    optimizer_trace_granularity: str = "semantic"
    # Fraction of the forward cost replayed by gradient checkpointing
    # (recompute). The 16p trace recomputes only part of the layer forward
    # (attention/vwn kernels, ~0.65s) while a full-block replay would re-run
    # every module. 1.0 = full forward replay.
    recompute_cost_ratio: float = 1.0

    attention_sparse_ratio: float = (
        0.0  # 0.0 means dense attention; 0.5 means compute optimize for causal attention
    )
    enable_dropout: bool = False
    use_fp32_accum_grad: bool = True
    use_accm_weight:bool = True # TODO(sherry): if True, No need to generate temporary variables of weight

    # recompute
    enable_recompute: bool = True
    recompute_granularity: Optional[str] = None
    recompute_layer_num: int = 0
    recompute_variance: bool = False
    megatron_recompute: bool = False
    megatron_recompute_modules: Optional[List[str]] = None

    # QKV projection recompute. 16p profiling: MatMulV3 5120 = 24 counts ~ 25
    # layers, 1 fwd/rank, no recompute -> the 16p strategy sets this False.
    # Config-driven so another run that does recompute the QKV proj is expressible.
    qkv_recompute: bool = False

    # CP all-to-all group size (explicit). None = node-local default
    # min(cp_size, num_per_node): 16p profiling shows the CP a2a runs as
    # 4-rank quads (num_per_node), not the full cp group. Set to cp_size for
    # full-group Ulysses.
    cp_a2a_group: Optional[int] = None

    # fused kernel
    use_flash_sdp: bool = True
    use_math_sdp: bool = False
    use_fused_norm: bool = True
    use_fused_swiglu: bool = True
    use_fused_grad_accumulation: bool = True
    cross_entropy_loss_fusion: bool = False
    overlap_grad_reduce: bool = True

    # TE release audit:
    # - regular linear starts using get_dummy_wgrad in release_v2.3
    # - CP A2A attention starts saving pre-PostA2A O in release_v2.8
    # - grouped linear starts using get_dummy_wgrad in release_v2.10
    te_version: Optional[str] = None
    te_dummy_wgrad_min_version: str = "2.3.0"
    te_cp_a2a_save_pre_posta2a_min_version: str = "2.8.0"
    te_grouped_linear_dummy_wgrad_min_version: str = "2.10.0"

    # network strategy
    # TODO: auto choose network strategy
    tp_net: Optional[str] = "auto"
    cp_net: Optional[str] = "auto"
    pp_net: Optional[str] = "auto"
    dp_net: Optional[str] = "auto"
    ep_net: Optional[str] = "auto"
    etp_net: Optional[str] = "auto"
    edp_net: Optional[str] = "auto"
    # FSDP/ZeRO-3 param unshard (all_gather) and grad reshard
    # (reduce_scatter) network selector. "auto" inherits the resolved
    # dp_net (dense) / edp_net (MoE expert), fully backward-compatible.
    # Only active when zero_state >= 3; zero_state < 3 is unaffected.
    fsdp_net: Optional[str] = "auto"
    fsdp_moe_net: Optional[str] = "auto"

    # Megatron related
    dispatch_probs: bool = False # The new version of Megatron combines probs in Silu after Groupgemm1 in ExpertMLP

    # Multi-resource fused ops (design doc 4.3/4.7, Phase 3 extension points).
    # compute_engine_map maps compute categories (e.g. "gemm", "elementwise")
    # to engine lane names; engine membership is validated when engine lanes
    # are consumed (system.engines wiring is future work).
    compute_engine_map: Optional[Dict[str, str]] = None
    # Each fused_ops entry: {"pattern": str, "policy": one of FUSION_POLICIES
    # (default "chunked_pipeline"), "chunks": int >= 1 (only meaningful for
    # the chunked_pipeline policy)}.
    fused_ops: Optional[List[dict]] = None
    # Fused-op memory accounting mode (design doc 4.7/9.2); "ramp" is reserved.
    fused_mem_mode: str = "steady_state"

    # Per-operator efficiency overrides (cost-tunability design doc section 4):
    # temporary what-if adjustments that win over SystemConfig.operator_efficiency
    # and lose to the API-level overrides. Same grammar as operator_efficiency.
    efficiency_overrides: Optional[Dict[str, Any]] = None

    mem_factor: float = 0.94

    # Layout transform overhead (µs) per all2all call in the levels cost
    # path. Models the _fused_dim01_transpose_kernel that flanks each
    # alltoallv in real profiling (e.g. Ulysses SP around attention).
    # Added to each all2all phase's base_latency in
    # _compute_net_op_time_levels. Default 0 (backward-compatible).
    layout_transform_overhead_us: float = 0.0
    
    valid_recompute_granularity = [
            "full_block",
            "attn_only",
            "mlp_only",
            "sdp_only",
            "selective_recompute"
        ]
    valid_megatron_recompute_modules = [
        "core_attn",
        "layernorm",
        "mla_up_proj",
        "moe_act",
        "mlp",
        "moe",
    ]
    valid_cp_a2a_modes = [
        "async_cp",
        "sync_cp",
    ]
    valid_collective_skew = [
        "virtual_waiters",
    ]
    valid_fsdp_modes = [
        "model-wise",
        "layer-wise",
    ]
    valid_fused_mem_modes = [
        "steady_state",
        "ramp",
    ]
    
    @classmethod
    def init_from_format_strings(cls, strs):
        """
        Docstring for init_from_format_strings
        parse format like:
        find
        seq{self.seq_len}.mbs{self.micro_batch_size}.mbc{self.micro_batch_num}.gbs{self.global_batch_size} tp{self.tp_size}.ep{self.ep_size}.pp{self.pp_size}.dp{self.dp_size}.etp{self.etp_size}.edp{self.edp_size}, world_size:{self.world_size}

        :param cls: Description
        :param strs: Description
        :return: Description
        :rtype: Any
        """
        param_patterns = {
            'seq_len': (r'seq(\d+)', 4096),
            'micro_batch_size': (r'mbs(\d+)', 1),
            'micro_batch_num': (r'mbc(\d+)', 1),
            'global_batch_size': (r'gbs(\d+)', 8),
            'tp_size': (r'tp(\d+)', 1),
            'cp_size': (r'cp(\d+)', 1),
            'ep_size': (r'ep(\d+)', 1),
            'pp_size': (r'pp(\d+)', 1),
            'world_size': (r'world_size:(\d+)', 8)
        }
        extractor = ParameterExtractor(param_patterns=param_patterns)
        params = extractor.extract_parameters(strs)
        global_batch_size = params.pop('global_batch_size')
        strategty = StrategyConfig(**params)
        strategty.reset_global_batch_size(global_batch_size)
        return strategty
        
    @property
    def shard_size(self):
        return self.pp_size * self.tp_size * self.cp_size

    @property
    def dp_size(self):
        assert self.world_size % self.shard_size == 0
        return self.world_size // self.shard_size

    @property
    def global_batch_size(self):
        global_batch_size = self.micro_batch_size * self.micro_batch_num * self.dp_size
        return global_batch_size

    @property
    def edp_size(self):
        return self.world_size // (self.ep_size * self.etp_size * self.pp_size)

    @property
    def fsdp_dense_group_size(self):
        """Effective dense FSDP shard group size.

        Defaults to ``dp_size * cp_size`` (the dense optimizer plane).
        When ``fsdp_shard_size`` is set, it overrides the default so the
        FSDP all-gather/reduce-scatter and ZeRO memory sharding use the
        framework's actual shard group.
        """
        if self.fsdp_shard_size is not None:
            return self.fsdp_shard_size
        return self.dp_size * self.cp_size

    @property
    def fsdp_moe_group_size(self):
        """Effective MoE FSDP shard group size.

        Defaults to ``edp_size``. When ``oe_shard_size`` is set, it
        overrides the default so the MoE FSDP all-gather/reduce-scatter
        and ZeRO memory sharding use the framework's actual expert shard
        group.
        """
        if self.oe_shard_size is not None:
            return self.oe_shard_size
        return self.edp_size
    
    @property
    def parallelism(self):
        sp_tag = f'sp{self.tp_size}.' if self.enable_sequence_parallel else ''
        return f'seq{self.seq_len}.mbs{self.micro_batch_size}.mbc{self.micro_batch_num}.gbs{self.global_batch_size} tp{self.tp_size}.{sp_tag}cp{self.cp_size}.ep{self.ep_size}.pp{self.pp_size}.dp{self.dp_size}.etp{self.etp_size}.edp{self.edp_size}, world_size:{self.world_size}'

    @property
    def megatron_recompute_module_set(self):
        return set(self.megatron_recompute_modules or [])

    @staticmethod
    def _version_tuple(version: Optional[str]):
        if not version:
            return None
        parts = re.findall(r"\d+", str(version))
        if not parts:
            return None
        nums = [int(part) for part in parts[:3]]
        while len(nums) < 3:
            nums.append(0)
        return tuple(nums)

    @property
    def te_dummy_wgrad_memory_enabled(self):
        cur = self._version_tuple(self.te_version)
        min_ver = self._version_tuple(self.te_dummy_wgrad_min_version)
        if cur is None or min_ver is None:
            return False
        return cur >= min_ver

    @property
    def te_grouped_linear_dummy_wgrad_memory_enabled(self):
        cur = self._version_tuple(self.te_version)
        min_ver = self._version_tuple(self.te_grouped_linear_dummy_wgrad_min_version)
        if cur is None or min_ver is None:
            return False
        return cur >= min_ver

    @property
    def te_cp_a2a_saves_pre_posta2a_output(self):
        cur = self._version_tuple(self.te_version)
        min_ver = self._version_tuple(self.te_cp_a2a_save_pre_posta2a_min_version)
        if cur is None or min_ver is None:
            return False
        return cur >= min_ver

    @property
    def use_variance_tail_model(self):
        return self.recompute_variance or (
            self.is_megatron_selective_recompute
            and bool(self.megatron_recompute_module_set & {"layernorm", "mla_up_proj", "moe_act"})
        )

    @property
    def is_megatron_selective_recompute(self):
        return (
            self.enable_recompute
            and self.recompute_layer_num > 0
            and self.recompute_granularity == "selective_recompute"
            and self.megatron_recompute
            and bool(self.megatron_recompute_module_set)
        )
    
    @property
    def is_recompute(self):
        is_full_recompute = self.recompute_layer_num > 0 and self.recompute_granularity == 'full_block'
        is_partial_recompute = self.recompute_layer_num > 0 and self.recompute_granularity in ['attn_only', 'mlp_only', 'sdp_only']
        is_selective_recompute = self.recompute_layer_num > 0 and self.recompute_granularity == 'selective_recompute' and any([self.attn_recompute, self.mla_rms_recompute, self.mlp_recompute, self.mlp_rms_recompute])
        return self.enable_recompute and (
            is_full_recompute
            or is_partial_recompute
            or is_selective_recompute
            or self.is_megatron_selective_recompute
        )
    
    @property
    def recompute_status(self):
        is_full_recompute = self.recompute_layer_num > 0 and self.recompute_granularity == 'full_block'
        is_partial_recompute = self.recompute_layer_num > 0 and self.recompute_granularity in ['attn_only', 'mlp_only', 'sdp_only']
        is_selective_recompute = self.recompute_layer_num > 0 and self.recompute_granularity == 'selective_recompute' and any([self.attn_recompute, self.mla_rms_recompute, self.mlp_recompute, self.mlp_rms_recompute])
        if not self.is_recompute:
            return 'No Recompute'
        if is_full_recompute:
            return f"{self.recompute_granularity}, recompute_layer_num={self.recompute_layer_num}"
        elif is_partial_recompute:
            return f"{self.recompute_granularity}, recompute_layer_num={self.recompute_layer_num}"
        elif self.is_megatron_selective_recompute:
            modules = ",".join(sorted(self.megatron_recompute_module_set))
            return (
                f"{self.recompute_granularity}, recompute_layer_num={self.recompute_layer_num}, "
                f"megatron_recompute=True, modules=[{modules}]"
            )
        elif is_selective_recompute:
            return f'{self.recompute_granularity}, recompute_layer_num={self.recompute_layer_num}, attn={self.attn_recompute}, attn_rms={self.mla_rms_recompute}, mlp={self.mlp_recompute}, mlp_rms={self.mlp_rms_recompute}, recompute_variance={self.recompute_variance}'
        else:
            return 'Unknown Recompute Status'
    @property
    def net(self):
        return f"pp_net={self.pp_net}, tp_net={self.tp_net}, cp_net={self.cp_net}, dp_net={self.dp_net}, ep_net={self.ep_net}, etp_net={self.etp_net}"
    
    def parse_attention_recompute(self, layer_idx):
        if self.recompute_granularity is None or layer_idx >= self.recompute_layer_num:
            return AttentionRecomputeConfig()
        conf = AttentionRecomputeConfig()
        if self.is_megatron_selective_recompute:
            modules = self.megatron_recompute_module_set
            conf.megatron_layernorm = "layernorm" in modules
            conf.megatron_mla_up_proj = "mla_up_proj" in modules
            conf.input_layernorm_recompute = conf.megatron_layernorm
            conf.q_down_recompute = conf.megatron_layernorm
            conf.kv_down_recompute = conf.megatron_layernorm
            conf.q_up_recompute = conf.megatron_mla_up_proj
            conf.kv_up_recompute = conf.megatron_mla_up_proj
            conf.q_layernorm_recompute = conf.megatron_mla_up_proj
            conf.kv_layernorm_recompute = conf.megatron_mla_up_proj
            conf.rope_recompute = conf.megatron_mla_up_proj
            conf.core_attn_recompute = conf.megatron_mla_up_proj
            return conf
        if self.recompute_granularity == "full_block":
            conf.set_all_status(True)
        elif self.recompute_granularity == "attn_only":
            conf.q_down_recompute = True
            conf.kv_down_recompute = True
            conf.q_up_recompute = True
            conf.kv_up_recompute = True
            conf.q_layernorm_recompute = True
            conf.kv_layernorm_recompute = True
            conf.rope_recompute = True
            conf.core_attn_recompute = True
            conf.out_recompute = True
        elif self.recompute_granularity == "sdp_only":
            conf.core_attn_recompute = True
        elif self.recompute_granularity == "mlp_only":
            pass

        elif self.recompute_granularity == "selective_recompute":
            if self.mla_rms_recompute:
                assert self.attn_recompute, "mla_rms_recompute requires attn_recompute"
            conf.input_layernorm_recompute =  self.mla_rms_recompute
            conf.q_down_recompute = self.mla_rms_recompute
            conf.kv_down_recompute = self.mla_rms_recompute
            conf.q_up_recompute = self.attn_recompute 
            conf.kv_up_recompute = self.attn_recompute 
            conf.q_layernorm_recompute = self.attn_recompute 
            conf.kv_layernorm_recompute = self.attn_recompute 
            conf.rope_recompute = self.attn_recompute
            conf.core_attn_recompute = self.attn_recompute 
            conf.out_recompute = False
        else:
            raise ValueError("Invalid recompute_granularity")

        return conf
    
    def parse_mlp_recompute(self, layer_idx):
        if self.recompute_granularity is None or layer_idx >= self.recompute_layer_num:
            return MLPRecomputeConfig()
        if self.is_megatron_selective_recompute:
            modules = self.megatron_recompute_module_set
            megatron_moe = "moe" in modules
            megatron_moe_act = "moe_act" in modules and not megatron_moe
            megatron_mlp = "mlp" in modules
            megatron_layernorm = "layernorm" in modules
            return MLPRecomputeConfig(
                pre_mlp_norm_recompute=megatron_layernorm,
                shared_linear_recompute=False,
                linear_recompute=False,
                router_recompute=False,
                permutation_recompute=False,
                megatron_layernorm=megatron_layernorm,
                megatron_mlp=megatron_mlp,
                megatron_moe=megatron_moe,
                megatron_moe_act=megatron_moe_act,
            )
        
        if self.recompute_granularity == "full_block":
            pre_mlp_norm_recompute = True 
            linear_recompute = True
            shared_linear_recompute = True
            router_recompute = True
            permutation_recompute = True
        elif self.recompute_granularity in ["attn_only", "sdp_only"]:
            pre_mlp_norm_recompute = False
            shared_linear_recompute = False
            linear_recompute = False
            router_recompute = False
            permutation_recompute = False
        elif self.recompute_granularity == "mlp_only":
            pre_mlp_norm_recompute = True
            shared_linear_recompute = True
            linear_recompute = True
            router_recompute = True
            permutation_recompute = True
        elif self.recompute_granularity == "selective_recompute":
            pre_mlp_norm_recompute = self.mlp_rms_recompute # normalization before mlp, after attention
            if self.mlp_rms_recompute:
                assert self.mlp_recompute, "mlp_rms_recompute requires mlp_recompute"
            shared_linear_recompute = self.mlp_rms_recompute 
            linear_recompute = self.mlp_recompute
            router_recompute = self.mlp_rms_recompute
            permutation_recompute = False
        else:
            raise ValueError("Invalid recompute_granularity")
        return MLPRecomputeConfig(pre_mlp_norm_recompute = pre_mlp_norm_recompute,
                                  shared_linear_recompute = shared_linear_recompute,
                                  linear_recompute = linear_recompute,
                                  router_recompute= router_recompute,
                                  permutation_recompute = permutation_recompute)

    def get_mesh_size(self, order="tp-dp-pp"):
        """According to the order to return the mesh size"""
        res = []
        for x in order.split("-"):
            assert x in (
                "tp",
                "dp",
                "pp",
                "ep",
                "etp",
                "edp",
            ), f"order {x} is not supported"
            res.append(getattr(self, f"{x}_size"))
        return res

    def _validate_order_of_paralielism(self):
        """Validate the placement string (hierarchical-network design doc, section 4).

        Grammar: '-'-separated tokens with exactly one each of tp/cp/dp in any
        order (innermost first), optional 'ep' tokens anywhere (dropped — the
        MoE mesh placement is fixed), and an optional trailing 'pp' (pp, when
        present, must be outermost). None falls back to the default placement.
        Mirrors parse_placement() in core/utils.py.
        """
        grammar = (
            "accepted grammar: '-'-separated tokens, exactly one each of "
            "tp/cp/dp in any order, optional 'ep' tokens anywhere (ignored, "
            "MoE mesh placement is fixed), optional trailing 'pp' (pp must be "
            "outermost when present), e.g. 'tp-cp-ep-dp-pp' (default) or "
            "'cp-tp-ep-dp-pp'"
        )
        order = self.order_of_paralielism
        if order is None:
            return
        tokens = str(order).split("-")

        def _invalid(reason):
            return ValueError(
                f"Invalid order_of_paralielism '{order}': {reason}; {grammar}"
            )

        if any(token == "" for token in tokens):
            raise _invalid("empty token")
        tokens = [token for token in tokens if token != "ep"]
        if "pp" in tokens:
            if tokens[-1] != "pp":
                raise _invalid("pp must be outermost (last)")
            tokens.pop()
        if sorted(tokens) != ["cp", "dp", "tp"]:
            raise _invalid(
                "dense dims must contain exactly one each of tp/cp/dp in any order"
            )

    def sanity_check(self):
        self._validate_order_of_paralielism()
        assert self.cp_a2a_mode in self.valid_cp_a2a_modes, (
            f"cp_a2a_mode {self.cp_a2a_mode} must be in [{','.join(self.valid_cp_a2a_modes)}]"
        )
        assert self.collective_skew is None or self.collective_skew in self.valid_collective_skew, (
            f"collective_skew {self.collective_skew} must be None or in [{','.join(self.valid_collective_skew)}]"
        )
        assert self.optimizer_trace_granularity in {"semantic", "detailed"}, (
            "optimizer_trace_granularity must be 'semantic' or 'detailed', "
            f"got {self.optimizer_trace_granularity!r}"
        )
        if self.cache_groupgemm_col_fp8_inputs:
            assert self.fp8, "cache_groupgemm_col_fp8_inputs requires fp8"
            
        if self.offload_groupgemm_col_inputs:
            assert self.recompute_granularity != 'full_block', "offload_groupgemm_col_inputs is not allowed when recompute_granularity = 'full_block'"

        assert self.seq_len % self.cp_size == 0, f"seq_len must be divisible by cp_size, but seq_len = {self.seq_len}, cp_size = {self.cp_size}"
        assert (
            self.world_size % self.shard_size == 0
        ), f"world_size must be divisible by pp_size * tp_size * cp_szie, but world_size = {self.world_size}, pp_size = {self.pp_size}, tp_size = {self.tp_size}, cp_size={self.cp_size}"
        assert self.zero_state in [0, 1, 2, 3], "zero_state must be in [0, 1, 2, 3]"
        assert self.fsdp_mode in self.valid_fsdp_modes, (
            f"fsdp_mode {self.fsdp_mode!r} must be in [{','.join(self.valid_fsdp_modes)}]"
        )
        if self.fsdp_mode != "model-wise" and self.zero_state < 3:
            warnings.warn(
                "fsdp_mode has no effect when zero_state < 3"
            )
        if self.zero_state >= 3:
            assert self.fsdp_prefetch_layers >= 0, (
                f"fsdp_prefetch_layers must be >= 0, got {self.fsdp_prefetch_layers}"
            )
            assert self.fsdp_ag_consumer_dependency_mode in {"shared", "split"}, (
                "fsdp_ag_consumer_dependency_mode must be 'shared' or 'split', "
                f"got {self.fsdp_ag_consumer_dependency_mode!r}"
            )
            assert self.fsdp_max_inflight_reduce_scatters >= 1, (
                "fsdp_max_inflight_reduce_scatters must be >= 1, got "
                f"{self.fsdp_max_inflight_reduce_scatters}"
            )
            if self.fsdp_prefetch_layers > 1 and self.fsdp_mode != "layer-wise":
                warnings.warn(
                    "fsdp_prefetch_layers > 1 has no effect when fsdp_mode != 'layer-wise'"
                )
        if self.fsdp_comm_streams is not None:
            assert isinstance(self.fsdp_comm_streams, dict), (
                "fsdp_comm_streams must be a mapping of collective role to stream"
            )
            for role, stream in self.fsdp_comm_streams.items():
                assert role in {"all_gather", "reduce_scatter"}, (
                    "fsdp_comm_streams supports only all_gather and "
                    f"reduce_scatter, got {role!r}"
                )
                assert isinstance(stream, str) and stream, (
                    f"fsdp_comm_streams[{role!r}] must be a non-empty string"
                )
        assert self.recompute_granularity is None or self.recompute_granularity in self.valid_recompute_granularity, f"recompute_granularity {self.recompute_granularity} must be in [{','.join(self.valid_recompute_granularity)}]"
        assert self.recompute_layer_num >= 0
        if not self.megatron_recompute:
            assert not self.megatron_recompute_module_set, (
                "megatron_recompute_modules requires megatron_recompute=True"
            )
        else:
            assert self.enable_recompute, "megatron_recompute requires enable_recompute=True"
            assert self.recompute_granularity == "selective_recompute", (
                "megatron_recompute requires recompute_granularity='selective_recompute'"
            )
            assert self.recompute_layer_num > 0, (
                "megatron_recompute requires recompute_layer_num > 0"
            )
            invalid_modules = self.megatron_recompute_module_set.difference(
                self.valid_megatron_recompute_modules
            )
            assert not invalid_modules, (
                f"invalid megatron_recompute_modules: {sorted(invalid_modules)}"
            )
            assert self.megatron_recompute_module_set, (
                "megatron_recompute requires non-empty megatron_recompute_modules"
            )
            assert "core_attn" not in self.megatron_recompute_module_set, (
                "megatron_recompute core_attn is not supported in SimuMax yet"
            )
            assert not any(
                [
                    self.attn_recompute,
                    self.mla_rms_recompute,
                    self.mlp_recompute,
                    self.mlp_rms_recompute,
                    self.recompute_variance,
                ]
            ), (
                "megatron_recompute is mutually exclusive with legacy selective flags "
                "and recompute_variance"
            )
        assert (
            self.world_size % (self.ep_size * self.etp_size * self.pp_size) == 0
        ), f"world_size must be divisible by ep_size * etp_size * pp_size, but world_size = {self.world_size}, ep_size = {self.ep_size}, etp_size = {self.etp_size}, pp_size = {self.pp_size}"
        assert self.moe_dispatcher_policy in [
            "all2all",
            "all2all-seq",
        ], "moe_dispatcher_policy must be 'all2all' (legacy alias 'all2all-seq' is accepted with warning)"
        if self.moe_dispatcher_policy == "all2all-seq":
            warnings.warn(
                "moe_dispatcher_policy='all2all-seq' is no longer supported. "
                "Falling back to 'all2all'."
            )
            self.moe_dispatcher_policy = "all2all"
        assert self.interleaving_size >= 1, "interleaving_size must be >= 1"
        if self.interleaving_size > 1:
            assert self.pp_size > 1, "interleaving_size > 1 requires pp_size > 1"
            assert self.pp_comm_async or self.pp_size > 2, (
                "When interleaved schedule is used and p2p communication overlap is disabled, "
                "pipeline-model-parallel size should be greater than 2 to avoid having multiple "
                "p2p sends and recvs between same 2 ranks per communication batch"
            )
            if self.microbatch_group_size_per_vp_stage is None:
                self.microbatch_group_size_per_vp_stage = self.pp_size
            assert self.microbatch_group_size_per_vp_stage >= self.pp_size, (
                "microbatch_group_size_per_vp_stage must be >= pp_size "
                f"(got {self.microbatch_group_size_per_vp_stage} < {self.pp_size})"
            )
            warnings.warn(
                "interleaving_size is enabled. VPP-aware timing/simulation paths are active; "
                "validate target configs with smoke/probe cases when introducing new schedules."
            )
        if self.enable_dropout:
            warnings.warn(
                "enable_dropout is not supported yet, the configuration will be ignored."
            )
        if self.enable_recompute:
            warnings.warn("Recompute is currently in experimental feature.")
        if self.zero_state == 2:
            warnings.warn(
                "zero_state 2 is not supported yet, the configuration will be ignored."
            )

        if self.recompute_granularity == "full_block":
            self.recompute_variance = False # megatron-LM's full recompute does not support variance

        if self.compute_engine_map is not None:
            assert isinstance(self.compute_engine_map, dict), (
                f"compute_engine_map must be a dict of str -> str, but got {type(self.compute_engine_map)}"
            )
            for category, engine in self.compute_engine_map.items():
                assert isinstance(category, str) and isinstance(engine, str), (
                    f"compute_engine_map must map str -> str, but got {category!r} -> {engine!r}"
                )

        if self.fused_ops is not None:
            assert isinstance(self.fused_ops, list), (
                f"fused_ops must be a list of dicts, but got {type(self.fused_ops)}"
            )
            for idx, fused_op in enumerate(self.fused_ops):
                assert isinstance(fused_op, dict), (
                    f"fused_ops[{idx}] must be a dict, but got {type(fused_op)}"
                )
                unknown_keys = set(fused_op) - {"pattern", "policy", "chunks"}
                assert not unknown_keys, (
                    f"fused_ops[{idx}] has unknown keys {sorted(unknown_keys)}, "
                    "allowed keys are ['chunks', 'pattern', 'policy']"
                )
                pattern = fused_op.get("pattern")
                assert isinstance(pattern, str) and pattern, (
                    f"fused_ops[{idx}]['pattern'] must be a non-empty str, but got {pattern!r}"
                )
                policy = fused_op.get("policy", "chunked_pipeline")
                assert policy in FUSION_POLICIES, (
                    f"fused_ops[{idx}]['policy'] {policy!r} must be one of {sorted(FUSION_POLICIES)}"
                )
                # chunks is only meaningful for the chunked_pipeline policy.
                chunks = fused_op.get("chunks", 1)
                assert isinstance(chunks, int) and not isinstance(chunks, bool) and chunks >= 1, (
                    f"fused_ops[{idx}]['chunks'] must be an int >= 1, but got {chunks!r}"
                )

        assert self.fused_mem_mode in self.valid_fused_mem_modes, (
            f"fused_mem_mode {self.fused_mem_mode} must be in [{','.join(self.valid_fused_mem_modes)}]"
        )
        if self.fused_mem_mode == "ramp":
            warnings.warn(
                "fused_mem_mode='ramp' is reserved but not yet implemented, "
                "steady_state is used."
            )
            self.fused_mem_mode = "steady_state"

        _validate_efficiency_override_table(self.efficiency_overrides, "efficiency_overrides")
    def reset_global_batch_size(self, global_batch_size):
        assert global_batch_size % (self.dp_size * self.micro_batch_size)==0, f"global_batch_size {global_batch_size} must be divisible by dp_size*miro_batch_size(dp_size={self.dp_size}, micro_batch_size={self.micro_batch_size})"
        self.micro_batch_num = global_batch_size // (self.dp_size * self.micro_batch_size)
        
@dataclass
class BandwidthConfig:
    gbps: int
    efficient_factor: float = 1.0
    latency_us: float = 0
    fixed_latency: float = 0
    fixed_latency_us_by_comm_num: Dict[str, float] = None
    # Per-card-count effective bandwidth (GiB/s), keyed by str(comm_num).
    # When present, compute_net_op_time uses gbps_by_comm_num[comm_num] instead
    # of the single `gbps` value — the bandwidth is a property of the comm
    # group size (physical link utilization), not of the net/domain name.
    # Absent = legacy single-gbps behavior (other models unaffected).
    gbps_by_comm_num: Dict[str, float] = None


@dataclass
class CompOpConfig:
    tflops: int
    efficient_factor: float = 1.0
    accurate_efficient_factor:dict = None


@dataclass
class AcceleratorConfig:
    backend: str
    mem_gbs: int
    bandwidth: Dict[str, BandwidthConfig]
    op: Dict[str, CompOpConfig]
    mode: str


@dataclass
class OpConfig:
    scale: float
    offset: float
    eff: float


@dataclass
class NetOpConfig:
    scale: float
    offset: float
    efficient_factor: float = None
    latency_us: float = None
    fixed_latency_us: float = None
    fixed_latency_us_by_comm_num: Dict[str, float] = None
    dp_fixed_bw: float = None


@dataclass
class NetworkConfig:
    processor_usage: float  # for overlap
    bandwidth: BandwidthConfig
    op: Dict[str, OpConfig]
    # Physical topology kind for this net profile (design doc
    # design_simu_system_net_ext.md Part C, section 5.2). Used when
    # topology.levels is not declared. "clos" (default) = shared uplink,
    # bandwidth is divided by num_per_node (legacy behavior);
    # "fullmesh" = dedicated per-pair links, bandwidth is not divided.
    # Overridden by topology.levels[i]["kind"] when the hierarchical
    # levels path is active.
    topology_kind: str = "clos"
    # Optional overlay bandwidth (GB/s) from a parallel fabric that can be
    # used simultaneously with this net's own bandwidth for p2p AND
    # collectives (all_reduce/all_gather/reduce_scatter/all2all) in the
    # levels cost path. Applied additively to `bandwidth.gbps`.
    # Example: UBLink mesh (56 GB/s) + SU Clos overlay (224 GB/s) = 280 GB/s
    # effective bandwidth within the SU domain for all communication patterns.
    overlay_bandwidth_gbps: float = 0


@dataclass
class SystemConfig(Config):
    """Accelerator system configuration"""

    sys_name: str = "null"
    num_per_node: int = 8
    accelerator: AcceleratorConfig = None
    networks: Dict[str, NetworkConfig] = None
    real_comm_bw: dict = field(default_factory=OrderedDict)
    FC8: bool = False
    intra_with_pcie: bool = False
    # Intra-node link type: "nvlink" (default), "pcie", or "ublink".
    # "ublink" is Huawei's UBLink high-speed interconnect, equivalent in
    # role to NVLink. Kept in sync with the legacy `intra_with_pcie`
    # boolean (True iff intra_link_type == "pcie") for backward compat.
    intra_link_type: str = "nvlink"
    miss_efficiency: dict = field(default_factory=OrderedDict)
    hit_efficiency: dict = field(default_factory=OrderedDict)
    # Extra hardware engine lanes (design doc 4.2), e.g.
    # {"cube": {"peak_tflops": 320}, "vector": {"peak_tflops": 80}}.
    # None means single-engine, which reproduces the current behavior.
    engines: Optional[Dict[str, dict]] = None
    # FSDP 通信-计算重叠效率（治本暴露模型，design doc 4.2 补充）：
    # 掩盖率 = fsdp_overlap_coefficient × (计算墙钟/总墙钟)。=1.0 时退化为
    # 计算占用率方程；>1 表示 fsdp 通信集中在计算密集段（layer-wise 交错），
    # 被计算覆盖的比例高于计算占墙钟比例（16p 实测 ~1.16）。
    fsdp_overlap_coefficient: float = 1.0
    # CP a2a 的 alltoall 有效带宽（GB/s，1e9 换算）。CP a2a 走 node 内全互联
    # 层带宽（16p = 49 GB/s），但 alltoall 为每对 rank 小包（Q 25.2MB/peer），
    # 实测有效带宽低于全互联（16p: 39.5 GB/s，30.2GB/764ms）。缺省 None =
    # 走 node 层推导；声明后对 alltoallv/all2all + cp 通信组生效。
    cp_a2a_bandwidth_gbps: Optional[float] = None
    # 内存写带宽（GB/s，1e9）。写量模型：all_gather 每 rank 收/写全量 W、
    # reduce_scatter 每 rank 收/写 W/N，时间 = 传输 + 写量/写带宽 →
    # RS 有效带宽 = 1/(1/β_tx + 1/(N·β_w))，AG = 1/(1/β_tx + 1/β_w) = per-card-count。
    # 16p 从 edp AG/RS transit 反推：β_w ≈ 88.9 GB/s（910B HBM 写路径）。
    # 缺省 None = 不启用写量模型（AG/RS 用同一 per-card-count 带宽）。
    write_bandwidth_gbps: Optional[float] = None
    # Network fabric model selection (network-fabric design doc section 6):
    # None = off (current behavior), "nic" = per-GPU NIC servers,
    # "nic+tor" = additionally activates ToR servers (Preview),
    # "nic+levels" = per-GPU NIC + per-level link servers (Preview,
    # hierarchical-network design doc section 8); requires topology["levels"].
    fabric_model: Optional[str] = None
    # Fabric topology knobs; reserved keys are "tor_capacity_gbps"
    # (number) and "tor_node_share" ("auto" or number >= 1).
    # Hierarchical-network keys (design_simu_hierarchical_network.md
    # section 3): "levels" (ordered list of {"name", "size", "net"},
    # innermost first; first level's size must equal num_per_node) and
    # "composition_policy" (per-op-type "max"/"serial" overrides).
    topology: Optional[Dict[str, Any]] = None
    # Machine-level per-operator efficiency table (cost-tunability design doc
    # section 4). Grammar per key (class_key or path_key): a scalar in (0, 1],
    # or {"default": float, "shapes": {shape_desc: float}} ("shapes" optional).
    operator_efficiency: Optional[Dict[str, Any]] = None
    # Optional, specification-only analytical model. When enabled, measured
    # efficiency/bandwidth tables remain loadable for regression, but are not
    # consulted by compute, HBM, or network timing.
    forward_derivation: Optional[Dict[str, Any]] = None
    # Separate measured-calibration branch. It is applied only when the
    # profile declares ``mode=measured_calibration`` and never changes the
    # structural FLOPs, shapes, topology, or event graph produced by the
    # forward-derived model.
    calibration_profile: Optional[Dict[str, Any]] = None
    # Versioned software implementation profiles.  These are deliberately
    # separate from hardware_spec: CANN tiling/engine facts and HCCL host/task
    # scheduling are software/runtime behavior, not physical device limits.
    cann_runtime: Optional[Dict[str, Any]] = None
    hccl_runtime: Optional[Dict[str, Any]] = None
    profile_sources: Optional[Dict[str, str]] = None
    # Activation-offload host transfer hardware facts (fully forward-derived
    # cost basis; declared, never fitted from measured times):
    #   {"host_bandwidth_gbps": 64.0, "latency_us": 10.0,
    #    "dma_channels": 1, "overlap_policy": "async"}
    # Used only when StrategyConfig.activation_offload is set. "host" here is
    # the device->host (D2H) / host->device (H2D) link the framework uses to
    # spill activation blocks (HCCS host port / PCIe-class link); the value is
    # a hardware declaration like cube_peak_tflops, to be confirmed against the
    # SKU sheet, not derived from any measured step time.
    activation_offload: Optional[Dict[str, Any]] = None
    # Traceable hardware identity/specification. Values may come from an
    # official product sheet or from aclrtGetDeviceInfo; performance traces are
    # deliberately not accepted here.
    hardware_spec: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Runtime override-chain slots, populated by PerfLLM.configure()
        # (cost-tunability design doc section 3). Plain attributes, not
        # dataclass fields: they never serialize into to_dict().
        self.efficiency_overrides_strategy = None
        self.efficiency_overrides_api = None
        self.operator_mfu_overrides = {}
        self._calibration_compute_index = {}
        self._calibration_memory_index = {}
        self._calibration_communication_index = {}
        self._calibration_compute_match_policy = "shape_then_stage_fallback"
        self._calibration_communication_strict_roles = False
        self._calibration_communication_bucket_tolerance = 0
        self._load_calibration_profile()
        self.forward_derivation_records = {
            "operators": {}, "network_layers": {}, "communications": {}}
        # The plan is a derived output, not a configuration input.  It is
        # rebuilt for each estimate/DES run and deliberately excluded from
        # dataclass serialization so no profiler artifact can become a model
        # parameter.
        self.communication_plan_document = None

    @staticmethod
    def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]):
        """Merge nested profile dictionaries without losing sibling fields."""
        for key, value in override.items():
            if (isinstance(value, dict) and isinstance(base.get(key), dict)):
                SystemConfig._deep_merge_dict(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
        return base

    @classmethod
    def _read_profile_ref(cls, reference: str, config_file: str):
        """Read a profile relative to the system config or current cwd."""
        candidates = []
        if os.path.isabs(reference):
            candidates.append(reference)
        else:
            candidates.extend([
                os.path.join(os.path.dirname(os.path.abspath(config_file)), reference),
                reference,
            ])
        for candidate in candidates:
            if os.path.isfile(candidate):
                return cls.read_json_file(candidate), os.path.normpath(candidate)
        raise FileNotFoundError(
            f"profile {reference!r} referenced by {config_file!r} was not found; "
            f"checked {candidates!r}")

    @classmethod
    def init_from_config_file(cls, config_file: str):
        """Load a system config and its separated hardware/software profiles.

        Forward-derived MXX configurations must declare hardware and topology
        profile references. CANN/HCCL references are optional; when omitted,
        portable default software profiles are loaded and marked in the audit.
        The former monolithic forward configuration path is intentionally
        removed; profile loading never imports calibrated alpha/beta tables
        into a forward run.
        """
        config_dict = cls.read_json_file(config_file)
        # Keep the system-file values separate from profile references.  The
        # merge order is deliberately explicit:
        #   hardware profile < topology profile < system-file overrides.
        # This keeps the profile precedence explicit and prevents a topology
        # profile from being silently mixed with an obsolete monolithic file.
        forward_requested = bool(
            (config_dict.get("forward_derivation") or {}).get("enabled", False))
        hardware_ref = config_dict.pop("hardware_profile", None)
        topology_ref = config_dict.pop("topology_profile", None)
        cann_ref = config_dict.pop("cann_profile", None)
        hccl_ref = config_dict.pop("hccl_runtime_profile", None)
        calibration_ref = config_dict.pop("calibration_profile_ref", None)
        cann_defaulted = False
        hccl_defaulted = False
        profile_sources = {}
        if forward_requested and not all((hardware_ref, topology_ref)):
            raise ValueError(
                "forward-derived system configs must declare hardware_profile, "
                "topology_profile")
        if forward_requested and not cann_ref:
            cann_ref = "../software/cann/default_cann_runtime.json"
            cann_defaulted = True
        if forward_requested and not hccl_ref:
            hccl_ref = "../software/hccl/default_hccl_runtime.json"
            hccl_defaulted = True
        system_overrides = config_dict
        merged_config = {}

        if hardware_ref:
            profile, resolved = cls._read_profile_ref(hardware_ref, config_file)
            cls._deep_merge_dict(merged_config, profile)
            profile_sources["hardware"] = resolved

        if topology_ref:
            profile, resolved = cls._read_profile_ref(topology_ref, config_file)
            cls._deep_merge_dict(merged_config, profile)
            profile_sources["topology"] = resolved

        cls._deep_merge_dict(merged_config, system_overrides)
        config_dict = merged_config

        if cann_ref:
            profile, resolved = cls._read_profile_ref(cann_ref, config_file)
            config_dict["cann_runtime"] = profile.get(
                "cann_runtime", profile)
            profile_sources["cann"] = (
                "default:portable_cann_runtime" if cann_defaulted else resolved)

        if hccl_ref:
            profile, resolved = cls._read_profile_ref(hccl_ref, config_file)
            config_dict["hccl_runtime"] = profile.get(
                "hccl_runtime", profile)
            profile_sources["hccl_runtime"] = (
                "default:portable_hccl_runtime" if hccl_defaulted else resolved)

        if calibration_ref:
            profile, resolved = cls._read_profile_ref(calibration_ref, config_file)
            config_dict["calibration_profile"] = profile.get(
                "calibration_profile", profile)
            # Preserve the hardware/topology/CANN/HCCL provenance collected
            # above when adding the calibration source.  Replacing this map
            # would make an explicitly selected software profile appear to be
            # a built-in default in calibrated-run audits.
            profile_sources = {
                **profile_sources,
                **dict(config_dict.get("profile_sources", {})),
            }
            profile_sources["measured_calibration"] = resolved

        if profile_sources:
            existing_sources = config_dict.get("profile_sources", {})
            config_dict["profile_sources"] = {
                **existing_sources, **profile_sources}
        return cls.init_from_dict(config_dict)

    @classmethod
    def init_from_dict(cls, config_dict: Dict[str, Any]):
        config_dict = copy.deepcopy(config_dict)
        # Profile references are resolved by init_from_config_file. Ignore
        # them here when callers construct a SystemConfig from an already
        # merged dictionary.
        config_dict.pop("hardware_profile", None)
        config_dict.pop("topology_profile", None)
        config_dict.pop("cann_profile", None)
        config_dict.pop("hccl_runtime_profile", None)
        accelerator = config_dict.pop("accelerator")
        sys_name = config_dict.pop("sys_name")
        num_per_node = config_dict.pop("num_per_node")
        networks = config_dict.pop("networks")
        intra_link_type = networks.pop('intra_link_type', None)
        if "intra_with_pcie" in networks:
            intra_with_pcie = networks.pop('intra_with_pcie')
            if intra_link_type is None:
                intra_link_type = "pcie" if intra_with_pcie else "nvlink"
        else:
            intra_with_pcie = False
        if intra_link_type is None:
            intra_link_type = "nvlink"
        intra_with_pcie = (intra_link_type == "pcie")
        accelerator = AcceleratorConfig(
            backend=accelerator["backend"],
            mem_gbs=accelerator["mem_gbs"],
            bandwidth={k: BandwidthConfig(**v) for k, v in accelerator["bandwidth"].items()},
            op={k: CompOpConfig(**v) for k, v in accelerator["op"].items()},
            mode=accelerator["mode"],
        )
        networks = {
            net_name: NetworkConfig(
                processor_usage=network["processor_usage"],
                bandwidth=BandwidthConfig(**network["bandwidth"]),
                op={k: NetOpConfig(**v) for k, v in network["op"].items()},
                overlay_bandwidth_gbps=network.get("overlay_bandwidth_gbps", 0),
                topology_kind=network.get("topology_kind", "clos"),
            )
            for net_name, network in networks.items()
        }
        FC8 = config_dict.pop("FC8", False)
        engines = config_dict.pop("engines", None)
        fabric_model = config_dict.pop("fabric_model", None)
        topology = config_dict.pop("topology", None)
        operator_efficiency = config_dict.pop("operator_efficiency", None)
        forward_derivation = config_dict.pop("forward_derivation", None)
        calibration_profile = config_dict.pop("calibration_profile", None)
        cann_runtime = config_dict.pop("cann_runtime", None)
        hccl_runtime = config_dict.pop("hccl_runtime", None)
        profile_sources = config_dict.pop("profile_sources", None)
        if (forward_derivation or {}).get("enabled", False):
            if cann_runtime is None:
                cann_runtime = {
                    "schema": "simumax_cann_runtime_v1",
                    "profile_kind": "portable_default",
                    "compute": {},
                }
            if hccl_runtime is None:
                hccl_runtime = {
                    "schema": "simumax_hccl_runtime_v1",
                    "profile_kind": "portable_default",
                    "network": {},
                }
            profile_sources = dict(profile_sources or {})
            profile_sources.setdefault("cann", "built_in:portable_cann_runtime")
            profile_sources.setdefault(
                "hccl_runtime", "built_in:portable_hccl_runtime")
        hardware_spec = config_dict.pop("hardware_spec", None)
        activation_offload = config_dict.pop("activation_offload", None)
        fsdp_overlap_coefficient = config_dict.pop("fsdp_overlap_coefficient", 1.0)
        cp_a2a_bandwidth_gbps = config_dict.pop("cp_a2a_bandwidth_gbps", None)
        write_bandwidth_gbps = config_dict.pop("write_bandwidth_gbps", None)
        return cls(
            sys_name=sys_name,
            num_per_node=num_per_node,
            accelerator=accelerator,
            networks=networks,
            FC8=FC8,
            intra_with_pcie = intra_with_pcie,
            intra_link_type = intra_link_type,
            engines=engines,
            fabric_model=fabric_model,
            topology=topology,
            operator_efficiency=operator_efficiency,
            forward_derivation=forward_derivation,
            calibration_profile=calibration_profile,
            cann_runtime=cann_runtime,
            hccl_runtime=hccl_runtime,
            profile_sources=profile_sources,
            hardware_spec=hardware_spec,
            activation_offload=activation_offload,
            fsdp_overlap_coefficient=fsdp_overlap_coefficient,
            cp_a2a_bandwidth_gbps=cp_a2a_bandwidth_gbps,
            write_bandwidth_gbps=write_bandwidth_gbps,
        )
    
    def record_miss_efficiency(self, op_name:str, flops:int, shape_desc:str, use_eff):
        if shape_desc:
            if op_name not in self.miss_efficiency:
                self.miss_efficiency[op_name] = {}
            self.miss_efficiency[op_name][f'shape={shape_desc}'] = {
                'flops': flops,
                'use_eff': use_eff
            }
    def record_net_bw(self, op_name:str, net, comm_num, comm_stage:str, base_bw, real_bw, eff_factor, total_time, comm_size, latency):
        if op_name not in self.real_comm_bw:
            self.real_comm_bw[op_name] = {}
        self.real_comm_bw[op_name][comm_stage.lower()] = {"net":net, "base_bw":base_bw, "real_bw":real_bw, "eff_factor":eff_factor, "comm_num":comm_num, "comm_size":comm_size, "total_time":total_time, "latency": latency, "FC8":self.FC8} 

    @property
    def forward_derivation_enabled(self):
        return bool((self.forward_derivation or {}).get("enabled", False))

    def _load_calibration_profile(self):
        """Index an explicitly requested measured-calibration profile.

        The index is intentionally separate from ``operator_efficiency`` and
        from the forward-derived hardware/software profiles.  A calibrated
        run therefore keeps the same structural formulas and applies only
        explicitly declared aggregate calibration components.  Compute
        entries adjust semantic operator utilization.  Communication entries
        may adjust only the pure transfer component through
        ``transfer_efficiency``; the newer profile never scales a complete
        Elapse/lifetime value.  The older ``time_multiplier`` form remains
        readable for backwards-compatible regression profiles.  Profiles
        generated by the calibration tool use list entries, but the parser
        accepts a missing/empty profile so baseline construction stays
        byte-compatible.
        """
        profile = self.calibration_profile
        if not isinstance(profile, dict):
            return
        if profile.get("mode") != "measured_calibration":
            return

        compute = profile.get("compute") or {}
        self._calibration_compute_match_policy = str(
            compute.get("match_policy") or "shape_then_stage_fallback")
        for entry in compute.get("entries", []):
            if not isinstance(entry, dict):
                continue
            op_name = entry.get("op_name")
            multiplier = entry.get("efficiency_multiplier")
            if not op_name or multiplier is None:
                continue
            try:
                multiplier = float(multiplier)
            except (TypeError, ValueError):
                continue
            if multiplier <= 0:
                continue
            key = (
                str(op_name),
                str(entry.get("stage") or ""),
                str(entry.get("shape_desc") or ""),
                self._normalize_calibration_context(entry.get("kernel_role")),
                self._normalize_calibration_context(entry.get("projection")),
            )
            self._calibration_compute_index[key] = {
                "multiplier": multiplier,
                "samples": entry.get("samples"),
                "statistic": entry.get("statistic"),
                "source": entry.get("source"),
                "parameter_name": entry.get(
                    "parameter_name", "operator_utilization"),
                "value": entry.get("value"),
                "source_type": entry.get("source_type"),
                "derivation_method": entry.get("derivation_method"),
                "confidence": entry.get("confidence"),
                "portable": entry.get("portable"),
                "case_specific": entry.get("case_specific"),
                "direct_duration_fill": bool(entry.get(
                    "direct_duration_fill", False)),
                "kernel_role": key[3],
                "projection": key[4],
            }

        memory = profile.get("memory") or {}
        for entry in memory.get("entries", []):
            if not isinstance(entry, dict):
                continue
            op_name = entry.get("op_name")
            multiplier = entry.get("memory_time_multiplier")
            if not op_name or multiplier is None:
                continue
            try:
                multiplier = float(multiplier)
            except (TypeError, ValueError):
                continue
            if multiplier <= 0:
                continue
            key = (
                str(op_name),
                str(entry.get("stage") or ""),
                str(entry.get("shape_desc") or ""),
                self._normalize_calibration_context(entry.get("kernel_role")),
                self._normalize_calibration_context(entry.get("projection")),
            )
            self._calibration_memory_index[key] = {
                "multiplier": multiplier,
                "samples": entry.get("samples"),
                "statistic": entry.get("statistic"),
                "source": entry.get("source"),
                "kernel_role": key[3],
                "projection": key[4],
            }

        communication = profile.get("communication") or {}
        self._calibration_communication_strict_roles = bool(
            communication.get("strict_semantic_roles", False))
        try:
            self._calibration_communication_bucket_tolerance = max(
                0, int(communication.get("size_bucket_tolerance", 0)))
        except (TypeError, ValueError):
            self._calibration_communication_bucket_tolerance = 0
        for entry in communication.get("entries", []):
            if not isinstance(entry, dict):
                continue
            op_name = entry.get("op_name")
            transfer_efficiency = entry.get("transfer_efficiency")
            multiplier = entry.get("time_multiplier")
            # ``efficiency_multiplier`` is accepted only as a compatibility
            # alias for older calibration artifacts.  New communication
            # profiles must use the component-specific transfer field.
            if transfer_efficiency is None and multiplier is None:
                transfer_efficiency = entry.get("efficiency_multiplier")
            if not op_name or (transfer_efficiency is None and multiplier is None):
                continue
            try:
                if transfer_efficiency is not None:
                    transfer_efficiency = float(transfer_efficiency)
                if multiplier is not None:
                    multiplier = float(multiplier)
            except (TypeError, ValueError):
                continue
            if ((transfer_efficiency is not None and transfer_efficiency <= 0)
                    or (transfer_efficiency is None
                        and (multiplier is None or multiplier <= 0))):
                continue
            comm_num = entry.get("comm_num")
            try:
                comm_num = int(comm_num) if comm_num is not None else None
            except (TypeError, ValueError):
                comm_num = None
            size_bucket = entry.get("size_bucket")
            try:
                size_bucket = int(size_bucket) if size_bucket is not None else None
            except (TypeError, ValueError):
                size_bucket = None
            key = (
                str(op_name),
                self._calibration_comm_role(entry.get("comm_role")),
                comm_num,
                size_bucket,
                self._calibration_direction(entry.get("direction")),
            )
            self._calibration_communication_index[key] = {
                "multiplier": multiplier,
                "transfer_efficiency": transfer_efficiency,
                "samples": entry.get("samples"),
                "statistic": entry.get("statistic"),
                "source": entry.get("source"),
                "parameter_name": entry.get(
                    "parameter_name",
                    "collective_transfer_efficiency"
                    if transfer_efficiency is not None
                    else "communication_time_multiplier"),
                "value": entry.get("value"),
                "source_type": entry.get("source_type"),
                "derivation_method": entry.get("derivation_method"),
                "confidence": entry.get("confidence"),
                "portable": entry.get("portable"),
                "case_specific": entry.get("case_specific"),
                "direct_duration_fill": bool(entry.get(
                    "direct_duration_fill", False)),
                "comm_role": key[1],
                "direction": key[4],
                "direct_observation": entry.get("direct_observation", True),
                "structural_extrapolation": entry.get(
                    "structural_extrapolation", False),
                "paired_from": entry.get("paired_from"),
            }

    @staticmethod
    def _normalize_calibration_context(value):
        """Normalize optional role/projection metadata without inventing it."""
        value = " ".join(str(value or "").strip().split())
        if value.lower() in {"", "未记录", "未提取", "n/a", "na", "none", "null"}:
            return ""
        return value

    @staticmethod
    def _calibration_direction(value):
        """Map a semantic stage label to the portable fwd/bwd/optimizer family."""
        value = str(value or "").lower().replace("-", "_")
        if value.startswith("bwd") or "backward" in value or "_bwd" in value:
            return "bwd"
        if value.startswith("opt") or "optimizer" in value or "_opt" in value:
            return "optimizer"
        if value.startswith("fwd") or "forward" in value or "_fwd" in value:
            return "fwd"
        return ""

    @staticmethod
    def _calibration_comm_role(value):
        """Normalize an optional semantic collective role for calibration.

        ``all_gather`` and ``reduce_scatter`` are reused by model-level sync,
        layer FSDP, and router paths.  A role is a structural model field
        (carried by the communication plan), not a measured identity.  Empty
        roles remain the portable fallback used by older profiles.
        """
        value = SystemConfig._normalize_calibration_context(value)
        return value.lower() if value else ""

    def _calibration_context(self, op_name, shape_desc, stage,
                             kernel_role=None, projection=None):
        """Resolve context available to the forward-derived call site.

        Most cost-model call sites pass the structural shape only.  GroupGEMM
        projection is therefore read from its model-generated shape descriptor,
        while the VWN phase role is read from its explicit structural stage.
        These are semantic model fields, not profiler timings.
        """
        role = self._normalize_calibration_context(kernel_role)
        projection = self._normalize_calibration_context(projection)
        shape_desc = str(shape_desc or "")
        if not projection:
            match = re.search(
                r"(?:^|[, ])projection=([A-Za-z0-9_]+)",
                shape_desc, re.IGNORECASE)
            if match:
                projection = self._normalize_calibration_context(match.group(1))
        if not role:
            match = re.search(
                r"(?:^|[, ])(?:kernel_role|role)=([A-Za-z0-9_]+)",
                shape_desc, re.IGNORECASE)
            if match:
                role = self._normalize_calibration_context(match.group(1))
        stage_text = str(stage or "").lower()
        if not role:
            for token in ("vwn_width", "vwn_depth", "vwn_out"):
                if token in stage_text:
                    role = token
                    break
        if not role and stage_text in {"bwd_grad_act", "bwd_grad_w"}:
            role = stage_text
        if not role and projection and str(op_name or "") in {
                "group_linear_col", "group_linear_row"}:
            role = projection
        return role, projection

    @staticmethod
    def _calibration_stage_family(stage):
        stage = str(stage or "")
        stage_lower = stage.lower().replace("-", "_")
        if stage_lower.startswith("opt") or "optimizer" in stage_lower:
            return "optimizer"
        if stage.startswith("bwd"):
            return "bwd"
        if stage == "fwd" or stage.endswith("_fwd"):
            return "fwd"
        return ""

    def _calibration_compute_multiplier(self, op_name, shape_desc, stage,
                                        kernel_role=None, projection=None):
        """Resolve one aggregate compute-efficiency multiplier."""
        if not self._calibration_compute_index:
            return None
        op_name = str(op_name or "")
        stage = str(stage or "")
        shape_desc = str(shape_desc or "")
        # The forward derivation records detailed backward stages, while
        # calibration groups may intentionally use the portable ``bwd``
        # family.  Keep exact semantic stages first (important for VWN and
        # other explicitly split phases), then fall back to the family.
        stage_family = self._calibration_stage_family(stage)
        role, projection = self._calibration_context(
            op_name, shape_desc, stage, kernel_role, projection)
        context_policy = (
            self._calibration_compute_match_policy
            == "role_projection_shape_then_shape")
        if context_policy:
            keys = []
            if role or projection:
                keys.append((op_name, stage, shape_desc, role, projection))
                if stage_family and stage_family != stage:
                    keys.append((op_name, stage_family, shape_desc,
                                 role, projection))
            # Context-free exact-shape entries are intentionally second tier.
            keys.append((op_name, stage, shape_desc, "", ""))
            if stage_family and stage_family != stage:
                keys.append((op_name, stage_family, shape_desc, "", ""))
            for key in keys:
                entry = self._calibration_compute_index.get(key)
                if entry is not None:
                    return entry
            return None
        if self._calibration_compute_match_policy == "exact_shape_stage_only":
            if not shape_desc:
                return None
            keys = [(op_name, stage, shape_desc, "", "")]
            if stage_family and stage_family != stage:
                keys.append((op_name, stage_family, shape_desc, "", ""))
            for key in keys:
                entry = self._calibration_compute_index.get(key)
                if entry is not None:
                    return entry
            return None
        keys = [
            (op_name, stage, shape_desc, "", ""),
            (op_name, stage, "", "", ""),
        ]
        if stage_family and stage_family != stage:
            keys.extend([
                (op_name, stage_family, shape_desc, "", ""),
                (op_name, stage_family, "", "", ""),
            ])
        keys.extend([
            (op_name, "", shape_desc, "", ""),
            (op_name, "", "", "", ""),
        ])
        for key in keys:
            entry = self._calibration_compute_index.get(key)
            if entry is not None:
                return entry
        return None

    def _calibration_memory_multiplier(self, op_name, shape_desc, stage,
                                       kernel_role=None, projection=None):
        """Resolve one memory-transfer multiplier for a forward-derived stage."""
        if not self._calibration_memory_index:
            return None
        op_name = str(op_name or "")
        stage = str(stage or "")
        shape_desc = str(shape_desc or "")
        stage_family = self._calibration_stage_family(stage)
        role, projection = self._calibration_context(
            op_name, shape_desc, stage, kernel_role, projection)
        context_policy = (
            self._calibration_compute_match_policy
            == "role_projection_shape_then_shape")
        keys = []
        if context_policy and (role or projection):
            keys.append((op_name, stage, shape_desc, role, projection))
            if stage_family and stage_family != stage:
                keys.append((op_name, stage_family, shape_desc,
                             role, projection))
        keys.append((op_name, stage, shape_desc, "", ""))
        if stage_family and stage_family != stage:
            keys.append((op_name, stage_family, shape_desc, "", ""))
        for key in keys:
            entry = self._calibration_memory_index.get(key)
            if entry is not None:
                return entry
        return None

    @staticmethod
    def _calibration_size_bucket(size):
        if not size or size <= 0:
            return None
        return int(round(math.log2(max(1, float(size)))))

    def _calibration_communication_multiplier(self, op_name, size, comm_num,
                                              direction=None, comm_role=None,
                                              comm_stage=None):
        """Resolve one structural communication calibration entry.

        Size buckets are logarithmic rather than exact payload keys, so the
        calibration is reusable for nearby shapes and is not a per-event
        duration table.
        """
        if not self._calibration_communication_index:
            return None
        op_name = str(op_name or "")
        # Runtime/model call sites may use semantic aliases such as
        # ``fsdp_all_gather`` and ``fsdp_reduce_scatter`` while calibration
        # profiles intentionally use the canonical collective family.  Keep
        # the requested name first, then fall back to the canonical network
        # operation without making the profile depend on one implementation's
        # event spelling.
        op_names = [op_name]
        canonical_op_name = NET_OP_FALLBACK.get(op_name)
        if canonical_op_name and canonical_op_name not in op_names:
            op_names.append(canonical_op_name)
        # ``all2all`` is the historical model spelling for the same
        # all-to-all-v semantic family used by the newer communication plan
        # and calibration profile.  Keep both spellings in the structural
        # lookup without changing the forward formula.
        if op_name == "all2all" and "alltoallv" not in op_names:
            op_names.append("alltoallv")
        try:
            comm_num = int(comm_num) if comm_num is not None else None
        except (TypeError, ValueError):
            comm_num = None
        size_bucket = self._calibration_size_bucket(size)
        raw_role = comm_role or self._calibration_role_from_stage(comm_stage)
        comm_role = self._calibration_comm_role(raw_role)
        direction = self._calibration_direction(direction)
        if not direction:
            direction = self._calibration_direction_from_role(comm_role)
        role_keys = (comm_role,) if (
            comm_role and self._calibration_communication_strict_roles) \
            else ((comm_role, "") if comm_role else ("",))
        direction_keys = (direction, "") if direction else ("",)
        keys = []
        for key_role in role_keys:
            for key_op_name in op_names:
                for key_direction in direction_keys:
                    keys.extend([
                        (key_op_name, key_role, comm_num, size_bucket,
                         key_direction),
                        (key_op_name, key_role, comm_num, None,
                         key_direction),
                        (key_op_name, key_role, None, size_bucket,
                         key_direction),
                        (key_op_name, key_role, None, None, key_direction),
                    ])
        for key in keys:
            entry = self._calibration_communication_index.get(key)
            if entry is not None:
                return entry
        # A payload may move by a small, structural logarithmic bucket when a
        # model/strategy changes a token count or dtype.  When the profile
        # explicitly opts in, reuse the nearest bucket within that tolerance
        # instead of silently dropping to a role-free multiplier.  The lookup
        # is based only on op/role/group/direction/payload structure; measured
        # durations never choose the identity of a communication event.
        tolerance = self._calibration_communication_bucket_tolerance
        if size_bucket is not None and tolerance > 0:
            nearest = []
            for key, entry in self._calibration_communication_index.items():
                key_op, key_role, key_num, key_bucket, key_direction = key
                if key_op not in op_names or key_role not in role_keys:
                    continue
                if key_num != comm_num or key_direction not in direction_keys:
                    continue
                if key_bucket is None:
                    continue
                delta = abs(int(key_bucket) - int(size_bucket))
                if delta <= tolerance:
                    nearest.append((delta, key, entry))
            if nearest:
                nearest.sort(key=lambda item: (item[0], str(item[1])))
                entry = dict(nearest[0][2])
                entry["lookup"] = "nearest_payload_bucket"
                entry["payload_bucket_delta"] = nearest[0][0]
                return entry
        return None

    @staticmethod
    def _calibration_role_from_stage(comm_stage):
        """Resolve known semantic role aliases from a model stage label.

        Some older operator call sites provide a semantic ``comm_stage`` but
        predate the optional ``comm_role`` argument.  These aliases describe
        the model graph (Router route fields, CP attention exchanges, and the
        two MoE physical exchanges); they do not inspect profiler timing or
        infer a runtime coefficient.
        """
        value = str(comm_stage or "").strip().lower().replace("-", "_")
        if value in {"moe_dispatch", "dispatch"}:
            return "dispatch_activation_a2av"
        if value in {"moe_combine", "combine"}:
            return "combine_ep"
        if value.startswith("router_"):
            if "route_grad" in value or "_rsv" in value:
                return "router_route_grad_rsv"
            if "topk_ids" in value or "topk_weights" in value or "_agv" in value:
                return "router_route_fields_agv"
        if value.startswith("attention_"):
            return value
        return ""

    @staticmethod
    def _calibration_direction_from_role(comm_role):
        value = str(comm_role or "").lower()
        if "bwd" in value or "grad" in value:
            return "bwd"
        if "fwd" in value or value in {
                "dispatch_activation_a2av", "combine_ep",
                "router_route_fields_agv"}:
            return "fwd"
        return ""

    def _apply_communication_calibration(self, op_name, size, comm_num,
                                         time_ms, direction=None,
                                         comm_role=None, transfer_time_ms=None,
                                         comm_stage=None):
        """Apply a declared communication calibration component.

        New profiles expose ``transfer_efficiency`` and therefore modify only
        the model-derived payload-transfer term:

        ``T = T_transfer / efficiency + T_physical_and_runtime``.

        The complete-call ``time_multiplier`` branch is retained solely for
        older regression profiles.  It is intentionally not used when a
        component-specific transfer efficiency is present, and no measured
        Elapse/Wait/Idle/Sync field is accepted here.
        """
        entry = self._calibration_communication_multiplier(
            op_name, size, comm_num, direction, comm_role, comm_stage)
        if entry is None:
            return time_ms, None
        transfer_efficiency = entry.get("transfer_efficiency")
        if transfer_efficiency is not None:
            if transfer_time_ms is None:
                # A caller that has not exposed the transfer component cannot
                # safely consume a transfer-only calibration.  Keep the
                # forward-derived time unchanged rather than scaling a whole
                # semantic lifetime by accident.
                return time_ms, None
            return max(
                0.0,
                time_ms - transfer_time_ms
                + transfer_time_ms / transfer_efficiency,
            ), entry
        multiplier = entry.get("multiplier")
        if multiplier is None:
            return time_ms, None
        # Legacy time_multiplier is measured/base. Keep the physical/topology
        # formula intact and scale only the final semantic call lifetime for
        # explicitly old profiles.
        return max(0.0, time_ms * multiplier), entry

    def _cann_compute_config(self, create=False):
        """Return the versioned CANN compute profile."""
        if self.cann_runtime is not None:
            if create:
                return self.cann_runtime.setdefault("compute", {})
            return self.cann_runtime.get("compute", {})
        # Direct dataclass construction can omit the optional software
        # profile. Keep the same portable-default semantics as init_from_dict.
        return {}

    def layout_pass_count(self, op_name, default):
        """Resolve a structural logical-pass count for a layout operation.

        ``Permutation``/``UnPermutation`` historically estimated an indexed
        reorder as ``1 + ceil(log2(local_expert_num))`` full-buffer passes.
        A software implementation profile may instead declare that the
        target library fuses the reorder into a smaller number of logical
        streaming passes.  This is a CANN/layout contract, not a measured
        duration or a per-event calibration value.  If no declaration exists,
        the caller's shape-derived fallback is preserved.
        """
        try:
            fallback = max(1, int(default))
        except (TypeError, ValueError):
            fallback = 1
        cfg = self._cann_compute_config()
        models = cfg.get("layout_resource_models", {}) or {}
        entry = models.get(op_name, {}) if isinstance(models, dict) else {}
        if not isinstance(entry, dict):
            return fallback
        value = entry.get("logical_passes", entry.get("pass_count"))
        if value is None:
            return fallback
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return fallback

    def resolve_layout_op_name(self, op_name, stage=None, path_key=None,
                               shape_desc=None):
        """Resolve a semantic layout path to its declared resource model.

        Layout operations can share one high-level name while using different
        physical paths.  For example, a CP ``*_redist`` stage is an indexed
        scatter into an existing buffer, whereas a Q/OUT stage is a full
        read--transpose--write pass.  The distinction is a software/library
        contract and must therefore live in the CANN profile, not in a
        measured-duration table.  Rules are optional and first-match; without
        a rule the requested name is preserved for backward compatibility.

        A rule has ``source_op`` (or ``op_name``), ``contains`` (a string or
        list of case-insensitive tokens), and ``resolved_op``.  Tokens are
        matched against the declared stage/path/shape description only; no
        profiler fields are inspected.
        """
        requested = str(op_name or "")
        cfg = self._cann_compute_config()
        rules = cfg.get("layout_path_rules", []) or []
        if isinstance(rules, dict):
            rules = [rules]
        context = " ".join(str(value or "") for value in (
            stage, path_key, shape_desc)).lower()
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            source = rule.get("source_op", rule.get("op_name"))
            if source not in (None, "", requested):
                continue
            tokens = rule.get("contains", rule.get("stage_contains", []))
            if isinstance(tokens, str):
                tokens = [tokens]
            if tokens and not all(str(token).lower() in context
                                  for token in tokens):
                continue
            resolved = rule.get("resolved_op", rule.get("profile"))
            if resolved:
                return str(resolved)
        return requested

    def _hccl_network_config(self):
        """Return the versioned HCCL/runtime network profile."""
        if self.hccl_runtime is not None:
            return self.hccl_runtime.get("network", self.hccl_runtime)
        return {}

    def provenance_audit(self):
        """Return an explicit separation of structural and performance inputs.

        A forward-derived run consumes only model/strategy/system facts.  The
        separate measured-calibration branch may additionally consume shape or
        role-indexed aggregate multipliers.  Keep these facts explicit so a
        report cannot accidentally describe a calibrated run as validation-only
        (or describe a structural mapping as a duration parameter).
        """
        profile = self.calibration_profile
        calibrated = (
            isinstance(profile, dict)
            and profile.get("mode") == "measured_calibration")
        profile = profile if calibrated else {}
        source = dict(profile.get("source") or {})
        statistics = dict(profile.get("statistics") or {})
        groups = {
            "compute": list((profile.get("compute") or {}).get("entries", []) or []),
            "memory": list((profile.get("memory") or {}).get("entries", []) or []),
            "communication": list(
                (profile.get("communication") or {}).get("entries", []) or []),
        }
        entries = [entry for rows in groups.values()
                   for entry in rows if isinstance(entry, dict)]
        policy_text = " ".join(
            str(value).lower()
            for value in (
                profile.get("description"),
                profile.get("calibration_scope"),
                statistics.get("compute_match_policy"),
                statistics.get("communication_calibration_scope"),
            )
            if value is not None)
        structural_fallback = bool(
            source.get("alignment_table")
            or source.get("trace_view_family_summary")
            or source.get("communication_sidecar"))
        shape_fallback = bool(
            "shape" in policy_text
            or any(entry.get("shape_desc") or entry.get("shape")
                   for entry in entries))
        role_fallback = bool(
            "role" in policy_text
            or any(entry.get("kernel_role") or entry.get("projection")
                   or entry.get("comm_role") for entry in entries))
        measured_token = (
            "measured", "duration", "exposed", "raw", "efficiency")
        duration_fallback = bool(
            calibrated and entries and any(
                any(token in str(value).lower() for token in measured_token)
                for value in list(statistics.values()) + [
                    entry.get("statistic") for entry in entries]
                    + [entry.get("source") for entry in entries]))
        explicit = dict(profile.get("provenance") or {})
        structural_used = bool(explicit.get(
            "structural_observations_used", structural_fallback))
        shape_used = bool(explicit.get(
            "shape_observations_used", shape_fallback))
        role_used = bool(explicit.get(
            "kernel_role_observations_used", role_fallback))
        duration_used = bool(explicit.get(
            "performance_duration_observations_used", duration_fallback))
        performance_used = bool(
            explicit.get("performance_observations_used_as_parameters",
                         calibrated and bool(entries)))

        fixed_scope = bool(
            profile.get("world_size") is not None
            or "fixed_configuration" in policy_text
            or "fixed configuration" in policy_text)
        parameter_sources = {
            "compute": (source.get("physical_windows")
                        or source.get("alignment_table")),
            "memory": source.get("alignment_table"),
            "communication": (
                source.get("trace_view_family_summary")
                or source.get("communication_sidecar")),
        }
        parameters = []
        for group, rows in groups.items():
            if not rows:
                continue
            parameter_names = sorted({
                str(entry.get("parameter_name") or "")
                for entry in rows
                if entry.get("parameter_name")
            })
            if group == "memory":
                parameter = "memory.time_multiplier"
            elif group == "compute":
                parameter = parameter_names[0] if len(parameter_names) == 1 \
                    else "compute.operator_utilization"
            else:
                parameter = parameter_names[0] if len(parameter_names) == 1 \
                    else "communication.transfer_efficiency"
            source_types = sorted({
                str(entry.get("source_type") or "")
                for entry in rows
                if entry.get("source_type")
            })
            parameters.append({
                "parameter": parameter,
                "source": parameter_sources.get(group),
                "source_type": (
                    source_types[0] if len(source_types) == 1
                    else "mixed" if source_types else
                    "measured_duration" if duration_used
                    else "structural_trace"),
                "case_specific": fixed_scope,
                "portable": not fixed_scope,
                "entries": len(rows),
                "parameter_names": parameter_names,
            })
        mode = "measured_calibration" if calibrated else "forward_derived"
        return {
            "mode": mode,
            "structural_observations_used": structural_used,
            "shape_observations_used": shape_used,
            "kernel_role_observations_used": role_used,
            "performance_duration_observations_used": duration_used,
            "performance_observations_used_as_parameters": performance_used,
            "model_structure_and_system_config_used": True,
            "calibration_parameter_count": len(parameters),
            "calibration_parameters": parameters,
            "measured_source": source or None,
            "inference_basis": (
                "explicit_profile_provenance_or_profile_fields"
                if calibrated else "no_measured_profile_loaded"),
        }

    def forward_profile_audit(self):
        """Describe profile provenance and legacy calibration isolation."""
        legacy_fields = []
        if self.cp_a2a_bandwidth_gbps is not None:
            legacy_fields.append("cp_a2a_bandwidth_gbps")
        if self.write_bandwidth_gbps is not None:
            legacy_fields.append("write_bandwidth_gbps")
        for net_name, net_cfg in (self.networks or {}).items():
            bandwidth = net_cfg.bandwidth
            if bandwidth.gbps_by_comm_num:
                legacy_fields.append(f"networks.{net_name}.bandwidth.gbps_by_comm_num")
            if bandwidth.fixed_latency or bandwidth.fixed_latency_us_by_comm_num:
                legacy_fields.append(f"networks.{net_name}.bandwidth.fixed_latency")
            for op_name, op_cfg in net_cfg.op.items():
                if op_cfg.fixed_latency_us is not None or op_cfg.fixed_latency_us_by_comm_num:
                    legacy_fields.append(
                        f"networks.{net_name}.op.{op_name}.fixed_latency_us")
        required_facts = list(
            ((self.hardware_spec or {}).get("runtime_discovered") or {})
            .get("required", []))
        memory_facts = (self.hardware_spec or {}).get("memory", {})
        compute_facts = (self.hardware_spec or {}).get("compute", {})
        missing_hardware_facts = [
            name for name in required_facts
            if not memory_facts.get(name) and not compute_facts.get(name)
            and not (self.hardware_spec or {}).get(name)]
        profile_sources = self.profile_sources or {}
        defaulted_profiles = sorted(
            name for name, source in profile_sources.items()
            if str(source).startswith(("default:", "built_in:")))
        cann_profile = self.cann_runtime or {}
        hccl_profile = self.hccl_runtime or {}
        cann_compute = self._cann_compute_config()
        cann_execution_contract = cann_compute.get(
            "public_execution_contract", {}) or {}
        stage_models = cann_compute.get("implementation_stage_models", {}) or {}
        if not isinstance(stage_models, dict):
            stage_models = {}
        mte_profile = cann_compute.get("mte", {}) or {}
        host_tiling = cann_compute.get("host_tiling", {}) or {}
        alignment_constraints = cann_compute.get("alignment_constraints", {}) or {}
        host_tiling_fields = (
            "tile", "block_dim", "tiling_key", "workspace_bytes")
        hccl_runtime = self._hccl_network_config().get("call_runtime", {}) or {}
        hccl_network = self._hccl_network_config()
        hccl_execution_contract = hccl_network.get(
            "public_execution_contract", {}) or {}
        algorithm_selection = hccl_network.get("algorithm_selection", {}) or {}
        communicator_defaults = hccl_network.get("communicator_defaults", {}) or {}
        completion = hccl_runtime.get("completion", {}) or {}
        completion_fields = (
            "completion_latency_us", "wait_latency_us", "barrier_latency_us")
        completion_complete = all(
            isinstance(completion.get(name), (int, float))
            and completion.get(name) >= 0
            for name in completion_fields)
        return {
            "forward_derivation_enabled": self.forward_derivation_enabled,
            "provenance": self.provenance_audit(),
            "measured_calibration_enabled": bool(
                isinstance(self.calibration_profile, dict)
                and self.calibration_profile.get("mode")
                == "measured_calibration"),
            "measured_calibration_world_size": (
                self.calibration_profile.get("world_size")
                if isinstance(self.calibration_profile, dict)
                else None),
            "measured_calibration_compute_match_policy": (
                self._calibration_compute_match_policy
                if isinstance(self.calibration_profile, dict)
                and self.calibration_profile.get("mode")
                == "measured_calibration"
                else None),
            "measured_calibration_source": (
                self.calibration_profile.get("source")
                if isinstance(self.calibration_profile, dict)
                else None),
            "measured_calibration_compute_entries": len(
                self._calibration_compute_index),
            "measured_calibration_communication_entries": len(
                self._calibration_communication_index),
            "measured_calibration_communication_component": (
                "pure_transfer_efficiency"
                if any(entry.get("transfer_efficiency") is not None
                       for entry in self._calibration_communication_index.values())
                else None),
            "profile_sources": profile_sources,
            "defaulted_software_profiles": defaulted_profiles,
            "cann_version": (self.cann_runtime or {}).get("version"),
            "hccl_runtime_version": (self.hccl_runtime or {}).get("version"),
            "cann_profile_kind": cann_profile.get("profile_kind"),
            "cann_spec_status": cann_profile.get("spec_status"),
            "cann_target_architecture": cann_profile.get("target_architecture"),
            "cann_source_refs": list(cann_profile.get("source_refs", []) or []),
            "cann_public_execution_contract": {
                "source_basis": cann_execution_contract.get("source_basis"),
                "host_inputs": list((cann_execution_contract.get(
                    "host_contract", {}) or {}).get("inputs", []) or []),
                "host_outputs": list((cann_execution_contract.get(
                    "host_contract", {}) or {}).get("outputs", []) or []),
                "api_symbols": list((cann_execution_contract.get(
                    "host_contract", {}) or {}).get("api_symbols", []) or []),
                "stage_contracts": cann_execution_contract.get(
                    "portable_stage_contract"),
                "exact_values_policy": (cann_execution_contract.get(
                    "host_contract", {}) or {}).get("exact_values_policy"),
            },
            "hccl_profile_kind": hccl_profile.get("profile_kind"),
            "hccl_spec_status": hccl_profile.get("spec_status"),
            "hccl_source_refs": list(hccl_profile.get("source_refs", []) or []),
            "hccl_public_execution_contract": {
                "source_basis": hccl_execution_contract.get("source_basis"),
                "call_inputs": list((hccl_execution_contract.get(
                    "call_contract", {}) or {}).get("inputs", []) or []),
                "executor_lifecycle": list(hccl_execution_contract.get(
                    "executor_lifecycle", []) or []),
                "resource_request_fields": list(hccl_execution_contract.get(
                    "resource_request_fields", []) or []),
                "topology_contract": hccl_execution_contract.get(
                    "topology_contract"),
                "runtime_values_not_publicly_fixed": list(
                    hccl_execution_contract.get(
                        "runtime_values_not_publicly_fixed", []) or []),
            },
            "legacy_calibration_fields_present": sorted(set(legacy_fields)),
            "legacy_calibration_fields_consumed": False
            if self.forward_derivation_enabled else None,
            "hardware_missing_facts": missing_hardware_facts,
            "cann_stage_models_declared": sorted(stage_models),
            "cann_stage_split_counts_known": sorted(
                name for name, value in stage_models.items()
                if isinstance(value, dict)
                and isinstance(value.get("materialized_kernel_count"), (int, float))),
            "mte_profile_declared": bool(mte_profile),
            "mte_profile_source": mte_profile.get("source"),
            "cann_host_tiling_declared": bool(host_tiling),
            "cann_host_tiling_unknown_fields": [
                name for name in host_tiling_fields
                if host_tiling.get(name) is None
            ],
            "cann_host_tiling_policies": {
                "policy": host_tiling.get("policy"),
                "input_fields": list(host_tiling.get("input_fields", []) or []),
                "output_fields": list(host_tiling.get("output_fields", []) or []),
                "tile_policy": host_tiling.get("tile_policy"),
                "block_dim_policy": host_tiling.get("block_dim_policy"),
                "tiling_key_policy": host_tiling.get("tiling_key_policy"),
                "workspace_policy": host_tiling.get("workspace_policy"),
                "stage_dependency_policy": host_tiling.get(
                    "stage_dependency_policy"),
            },
            "cann_alignment_constraints": alignment_constraints or None,
            "cann_alignment_constraints_known": sorted(
                name for name, value in alignment_constraints.items()
                if name.endswith("_bytes") and isinstance(value, (int, float))
            ),
            "cann_stage_split_count_policy": sorted(
                name for name, value in stage_models.items()
                if isinstance(value, dict)
                and value.get("materialized_kernel_count_policy")
            ),
            "hccl_runtime_execution_stages": hccl_runtime.get("execution_stages"),
            "hccl_completion_profile_complete": completion_complete,
            "hccl_completion_unknown_fields": [
                name for name in completion_fields
                if not isinstance(completion.get(name), (int, float))
            ],
            "hccl_runtime_structural_rules": {
                "completion_rule": hccl_runtime.get("completion_rule"),
                "wait_rule": hccl_runtime.get("wait_rule"),
                "barrier_rule": hccl_runtime.get("barrier_rule"),
                "descriptor_count": hccl_runtime.get("descriptor_count"),
                "task_count": hccl_runtime.get("task_count"),
            },
            "hccl_algorithm_selection": {
                "policy": algorithm_selection.get("policy"),
                "level_scope": algorithm_selection.get("level_scope"),
                "operator_override_count": len(
                    algorithm_selection.get("operator_overrides", {}) or {})
                if isinstance(algorithm_selection.get("operator_overrides", {}), dict)
                else None,
            },
            "hccl_supported_collectives": list(
                hccl_network.get("supported_collectives", []) or []),
            "hccl_runtime_policies": {
                "task_count_policy": hccl_runtime.get("task_count_policy"),
                "descriptor_policy": hccl_runtime.get("descriptor_policy"),
                "completion_policy": hccl_runtime.get("completion_policy"),
                "communicator_defaults_declared": bool(communicator_defaults),
                "resource_request": hccl_network.get("resource_request"),
            },
        }

    @staticmethod
    def _ceil_to(value, quantum):
        return int(math.ceil(value / quantum) * quantum) if value else 0

    @staticmethod
    def _canonical_compute_dtype(dtype):
        """Normalize legacy and framework dtype spellings for peak lookup."""
        aliases = {
            "float16": "fp16",
            "fp16": "fp16",
            "bfloat16": "bf16",
            "bf16": "bf16",
            "float8": "fp8",
            "fp8": "fp8",
            "float4": "fp4",
            "fp4": "fp4",
            "int8": "int8",
            "int4": "int4",
        }
        name = str(dtype or "bf16").strip().lower().replace("-", "")
        return aliases.get(name, name)

    @staticmethod
    def _dtype_size_bytes(dtype):
        """Return the packed element size used by structural memory formulas."""
        return {
            "fp4": 0.5,
            "int4": 0.5,
            "fp8": 1,
            "int8": 1,
            "bf16": 2,
            "fp16": 2,
            "fp32": 4,
        }.get(SystemConfig._canonical_compute_dtype(dtype), 2)

    @classmethod
    def _dtype_peak_tflops(cls, spec_compute, engine, dtype, fallback):
        """Select an optional engine/dtype peak while preserving old profiles."""
        tables = spec_compute.get("peak_tflops_by_dtype", {}) or {}
        if not isinstance(tables, dict):
            tables = {}
        table = tables.get(engine, {})
        if not isinstance(table, dict):
            table = {}
        # Also accept the explicit flat names in hand-authored normalized
        # profiles, while the legacy mapper emits the nested table above.
        if not table:
            table = spec_compute.get(f"{engine}_peak_tflops_by_dtype", {}) or {}
        if not isinstance(table, dict):
            table = {}
        canonical = cls._canonical_compute_dtype(dtype)
        peak = table.get(canonical)
        if peak is None and canonical == "bf16":
            peak = table.get("fp16")
        if isinstance(peak, (int, float)) and not isinstance(peak, bool) and peak > 0:
            return float(peak), (
                f"hardware_spec.compute.peak_tflops_by_dtype.{engine}.{canonical}")
        return fallback, None

    @classmethod
    def _declared_engine_utilization(cls, spec_compute, engine, flops):
        """Evaluate the optional legacy Cube/Vector utilization declaration.

        ``log_a``/``log_b`` use a natural-log linear fit over FLOP count.  A
        profile can opt into ``formula='log_flops_power'`` for the equivalent
        log-log/power form; the bare legacy format intentionally keeps the
        linear form used by the adapter audit.
        """
        profiles = spec_compute.get("utilization", {}) or {}
        if not isinstance(profiles, dict):
            return None, None
        profile = profiles.get(engine)
        if not isinstance(profile, dict):
            return None, None
        if profile.get("constant") is not None:
            value = float(profile["constant"])
            source = f"hardware_spec.compute.utilization.{engine}.constant"
        elif profile.get("log_a") is not None and profile.get("log_b") is not None:
            log_input = max(float(flops or 0), 1.0)
            log_base = str(profile.get("log_base", "e")).lower()
            logarithm = math.log10(log_input) if log_base in {"10", "log10"} \
                else math.log(log_input)
            value = float(profile["log_a"]) * logarithm + float(profile["log_b"])
            if str(profile.get("formula", "log_flops_linear")).lower() in {
                    "log_flops_power", "log_power", "power"}:
                value = math.exp(value)
            source = f"hardware_spec.compute.utilization.{engine}.log_fit"
        else:
            return None, None
        lower = float(profile.get("min_ratio", 0.0))
        upper = float(profile.get("max_ratio", 1.0))
        if lower < 0 or upper <= 0 or lower > upper:
            raise ValueError(
                f"invalid utilization bounds for {engine}: min={lower}, max={upper}")
        return max(lower, min(upper, value)), source

    def set_operator_mfu_override(self, op_name, mfu, shape_desc=None):
        """Set a customer what-if MFU for one operator, never baseline calibration."""
        if not 0 < mfu <= 1:
            raise ValueError(f"operator MFU must be in (0, 1], got {mfu}")
        self.operator_mfu_overrides[(op_name, shape_desc)] = float(mfu)

    def _operator_mfu_override(self, op_name, shape_desc):
        return self.operator_mfu_overrides.get(
            (op_name, shape_desc), self.operator_mfu_overrides.get((op_name, None)))

    def apply_hardware_probe(self, profile):
        """Merge non-performance device facts collected through AscendCL.

        The probe contains device identity, architecture, core counts, memory,
        and cache capacity only. It must not contain observed kernel duration,
        bandwidth, MFU, or operator efficiency.
        """
        forbidden = {
            "duration", "latency", "measured_bandwidth", "mfu", "efficiency",
            "kernel_time", "throughput",
        }
        bad = sorted(key for key in profile if key.lower() in forbidden)
        if bad:
            raise ValueError(f"hardware probe contains performance fields: {bad}")
        self.hardware_spec = copy.deepcopy(self.hardware_spec or {})
        runtime = self.hardware_spec.setdefault("runtime_discovered", {})
        runtime.update(copy.deepcopy(profile))
        mapping = {
            "aic_core_num": "aic_core_num",
            "aiv_core_num": "aiv_core_num",
            "l2_cache_bytes": "l2_cache_bytes",
            "npu_arch": "npu_arch",
        }
        spec_compute = self.hardware_spec.setdefault("compute", {})
        spec_memory = self.hardware_spec.setdefault("memory", {})
        for source, target in mapping.items():
            value = profile.get(source)
            if value is not None and value != 0:
                if target == "npu_arch":
                    self.hardware_spec[target] = value
                elif target == "l2_cache_bytes":
                    spec_memory[target] = value
                else:
                    spec_compute[target] = value
        if not spec_compute.get("aic_core_num") and profile.get("aicore_core_num"):
            spec_compute["aic_core_num"] = profile["aicore_core_num"]

    def apply_library_tiling_profile(self, profile):
        """Merge compiler/tiling facts, rejecting performance observations."""
        allowed = {"base_m", "base_n", "base_k", "block_dim"}
        result = {}
        for op_name, entry in profile.items():
            if op_name in {"schema", "source"}:
                continue
            if not isinstance(entry, dict):
                raise ValueError(f"tiling entry for {op_name} must be an object")
            unknown = set(entry) - allowed - {"default", "shapes"}
            if unknown:
                raise ValueError(
                    f"tiling entry for {op_name} has unsupported fields: "
                    f"{sorted(unknown)}")
            result[op_name] = copy.deepcopy(entry)
        compute = self._cann_compute_config(create=True)
        compute.setdefault("library_tiling", {}).update(result)

    @staticmethod
    def _operator_engine(op_name, cfg):
        mapping = cfg.get("operator_engines", {})
        return mapping.get(op_name, mapping.get("default", "cube"))

    @staticmethod
    def _tiling_for_operator(op_name, shape_desc, cfg):
        tiling = cfg.get("library_tiling", {})
        value = tiling.get(op_name, tiling.get("default", {}))
        if not isinstance(value, dict):
            return {}
        if "shapes" in value or "default" in value:
            return copy.deepcopy(
                (value.get("shapes") or {}).get(shape_desc, value.get("default", {})))
        return value

    @staticmethod
    def _implementation_stage_model(op_name, cfg, engine=None):
        """Return a generic CANN implementation-stage declaration.

        Stage declarations describe portable semantic phases (for example
        ``mte2_read -> vector_compute -> mte3_write``).  They are not CANN
        kernel names and do not imply that a semantic event was materialized
        as that many profiler kernels.  A concrete
        ``materialized_kernel_count`` is optional and remains unknown until
        the target CANN host tiling/export supplies it.
        """
        table = cfg.get("implementation_stage_models", {}) or {}
        if not isinstance(table, dict):
            return {}
        aliases = cfg.get("implementation_stage_aliases", {}) or {}
        alias = aliases.get(op_name) if isinstance(aliases, dict) else None
        for key in (op_name, alias, engine, "default"):
            if not key:
                continue
            value = table.get(key)
            if isinstance(value, dict):
                return copy.deepcopy(value)
        return {}

    @staticmethod
    def _implementation_stage_overhead_ms(stage_model, launch_us):
        """Charge only explicitly declared extra kernel launches.

        A stage list alone is metadata: internal pipeline phases do not imply
        extra wall time.  Extra launch time is charged only when a profile
        explicitly declares a numeric materialized-kernel count and opts into
        ``launch_per_materialized_kernel``.  This keeps an unknown CANN split
        from being inferred from measured durations.
        """
        if not isinstance(stage_model, dict):
            return 0.0, None
        raw_count = stage_model.get("materialized_kernel_count")
        try:
            count = int(raw_count) if raw_count is not None else None
        except (TypeError, ValueError):
            count = None
        if count is None:
            return 0.0, None
        count = max(1, count)
        charge = str(stage_model.get("timing_charge") or "none")
        if charge != "launch_per_materialized_kernel" or count <= 1:
            return 0.0, count
        return max(0.0, count - 1) * float(launch_us) / 1e3, count

    def _derive_compute_efficiency(
        self, op_name, flops, shape_desc, accessed_mem, stage=None, path_key=None,
        kernel_role=None, projection=None,
    ):
        """Derive theoretical, limiting, and achievable operator utilization."""
        cfg = self._cann_compute_config()
        tile = cfg.get("tensor_tile", {})
        tm = max(1, int(tile.get("m", 16)))
        tn = max(1, int(tile.get("n", 16)))
        tk = max(1, int(tile.get("k", 16)))
        dims = {}
        for name in ("b", "batch", "ng", "m", "n", "k"):
            match = re.search(
                rf"(?:^|[, ]){name}=([0-9]+)", shape_desc or "", re.IGNORECASE)
            if match:
                dims[name.lower()] = int(match.group(1))
        compute_dtype_match = re.search(
            r"(?:^|[, ])(?:compute_dtype|dtype|out_dtype)=([A-Za-z0-9_]+)",
            shape_desc or "", re.IGNORECASE)
        compute_dtype_name = (
            self._canonical_compute_dtype(
                compute_dtype_match.group(1) if compute_dtype_match else "bf16"))
        output_dtype_match = re.search(
            r"(?:^|[, ])out_dtype=([A-Za-z0-9_]+)",
            shape_desc or "", re.IGNORECASE)
        output_dtype_name = (
            self._canonical_compute_dtype(output_dtype_match.group(1))
            if output_dtype_match else compute_dtype_name)
        compute_dtype_bytes = self._dtype_size_bytes(compute_dtype_name)
        output_dtype_bytes = self._dtype_size_bytes(output_dtype_name)
        if output_dtype_match is None:
            output_dtype_bytes = compute_dtype_bytes
        if "accumulate=True" in (shape_desc or ""):
            output_dtype_bytes = max(output_dtype_bytes, 4)
        batch = dims.get("b", dims.get("batch", dims.get("ng", 1)))
        alignment = 1.0
        shape_bytes = None
        if all(name in dims for name in ("m", "n", "k")):
            m, n, k = dims["m"], dims["n"], dims["k"]
            padded = (self._ceil_to(m, tm) * self._ceil_to(n, tn)
                      * self._ceil_to(k, tk))
            alignment = (m * n * k / padded) if padded else 1.0
            shape_bytes = (
                batch * ((m * k + k * n) * compute_dtype_bytes
                         + m * n * output_dtype_bytes))
        memory_bytes = accessed_mem if accessed_mem and accessed_mem > 0 else shape_bytes
        mte_cfg = cfg.get("mte", {}) or {}
        host_tiling_cfg = cfg.get("host_tiling", {}) or {}
        alignment_constraints = cfg.get("alignment_constraints", {}) or {}
        transaction_bytes = max(1, int(cfg.get(
            "memory_transaction_bytes",
            mte_cfg.get("transaction_bytes", 256))))
        padded_memory_bytes = self._ceil_to(memory_bytes, transaction_bytes)
        memory_transaction_utilization = (
            memory_bytes / padded_memory_bytes if padded_memory_bytes else None)
        reference_peak_tflops = self.accelerator.op.get(
            op_name, self.accelerator.op["default"]).tflops
        engine = self._operator_engine(op_name, cfg)
        spec_compute = (self.hardware_spec or {}).get("compute", {})
        declared_utilization, declared_utilization_source = (
            self._declared_engine_utilization(spec_compute, engine, flops))
        peak_tflops, dtype_peak_source = self._dtype_peak_tflops(
            spec_compute, engine, compute_dtype_name, reference_peak_tflops)
        if dtype_peak_source is not None:
            peak_source = dtype_peak_source
        elif engine == "vector":
            peak_tflops = spec_compute.get(
                "vector_peak_tflops", cfg.get("vector_peak_tflops"))
            peak_source = "hardware_spec.vector_peak_tflops"
            if peak_tflops is None:
                peak_tflops = reference_peak_tflops
                peak_source = "unavailable:fallback_to_reference_peak"
        else:
            peak_tflops = reference_peak_tflops
            peak_source = "accelerator.op"
        peak_tflops = float(peak_tflops)
        hbm_gbps = self.accelerator.bandwidth["default"].gbps
        memory_spec = (self.hardware_spec or {}).get("memory", {})
        l2_gbps = memory_spec.get("l2_bandwidth_gbps")
        l2_capacity_bytes = memory_spec.get("l2_cache_bytes")
        hbm_roofline = None
        l2_roofline = None
        if memory_bytes and peak_tflops > 0:
            arithmetic_intensity = flops / memory_bytes
            hbm_roofline = min(1.0, arithmetic_intensity * hbm_gbps * 1e9
                               / (peak_tflops * 1e12))
            roofline = hbm_roofline
            # L2 is a second physical bandwidth ceiling when declared.  The
            # model does not invent cache-hit reuse; capacity is retained as
            # a hardware fact, while bandwidth participates conservatively as
            # an upper bound for traffic that traverses the cache.
            if isinstance(l2_gbps, (int, float)) and l2_gbps > 0:
                l2_roofline = min(1.0, arithmetic_intensity * l2_gbps * 1e9
                                  / (peak_tflops * 1e12))
                roofline = min(roofline, l2_roofline)
        else:
            arithmetic_intensity = None
            roofline = 1.0
        tiling = self._tiling_for_operator(op_name, shape_desc, cfg)
        stage_model = self._implementation_stage_model(op_name, cfg, engine)
        core_key = "aiv_core_num" if engine == "vector" else "aic_core_num"
        core_num = cfg.get(core_key) or spec_compute.get(core_key)
        active_core_num = (
            min(core_num, int(tiling["block_dim"]))
            if core_num and tiling.get("block_dim") else core_num)
        block_m = int(tiling.get("base_m", 0))
        block_n = int(tiling.get("base_n", 0))
        work_tiles = None
        wave_count = None
        wave_utilization = 1.0
        if active_core_num and block_m and block_n and "m" in dims and "n" in dims:
            work_tiles = (batch * math.ceil(dims["m"] / block_m)
                          * math.ceil(dims["n"] / block_n))
            wave_count = math.ceil(work_tiles / active_core_num)
            wave_utilization = work_tiles / (wave_count * active_core_num)
        limit_efficiency = max(
            1e-12, min(1.0, roofline * alignment * wave_utilization))
        ideal_compute_ms = flops / (peak_tflops * 1e12) * 1e3
        ideal_memory_ms = (
            padded_memory_bytes / (hbm_gbps * 1e9) * 1e3
            if memory_bytes else 0.0)
        launch_us = float(cfg.get("kernel_launch_latency_us", 0.0))
        padded_compute_ms = ideal_compute_ms / max(alignment, 1e-12)
        composite_batch_alignment = 1.0
        # Flash/SWA attention executes independent head batches on the Cube
        # pipeline.  M/N/K parsing is not available for these composite
        # kernels, but the head batch is still scheduled in tensor-tile
        # quanta.  Account for the final partially occupied head tile exactly
        # as GEMM accounts for a partially occupied M/N/K tile.  This uses
        # only the model head count and the declared hardware tensor tile.
        if (engine == "cube" and op_name in {
                "sdp_fwd", "sdp_bwd", "swa_fwd", "swa_bwd"}):
            head_match = re.search(
                r"(?:^|[, ])head_num=([0-9]+)", shape_desc or "",
                re.IGNORECASE)
            if head_match:
                head_num = max(1, int(head_match.group(1)))
                # Composite attention libraries distribute independent head
                # work as AIC-core waves. A partial final wave leaves the
                # remaining AICs idle even though each head's inner GEMMs use
                # the 16x16 tensor tile. Therefore the scheduling quantum is
                # the declared AIC count, not the inner matrix tile width.
                # A null/absent aic_core_num (unknown hardware) falls back to
                # the tensor-tile quantum exactly as the absent-key default did.
                head_wave = max(1, int(
                    cfg.get("aic_core_num")
                    or spec_compute.get("aic_core_num") or tm))
                padded_heads = self._ceil_to(head_num, head_wave)
                composite_batch_alignment = head_num / padded_heads
                padded_compute_ms /= composite_batch_alignment
        onchip_bytes = None
        onchip_time_ms = 0.0
        library_extra_time_ms = 0.0
        library_extra_detail = None
        inferred_core_frequency_ghz = None
        transfer_transaction_bytes = None
        resource_policy = cfg.get("resource_overlap_policy", "serial")

        # A Cube kernel must move every base tile through GM/L1/L0 before the
        # MMAD result can leave through FixPipe.  HBM-only Roofline misses this
        # traffic and therefore makes large GEMMs unrealistically approach the
        # device peak.  Derive an instruction-level movement lower bound from
        # the library base tile and architectural transfer granularity.  This
        # is a hardware/library rule, not a fitted operator coefficient.
        if (engine == "cube" and flops > 0 and active_core_num
                and block_m and block_n):
            block_k = max(1, int(tiling.get("base_k", tk)))
            dtype_bytes = compute_dtype_bytes
            output_bytes = output_dtype_bytes

            if all(name in dims for name in ("m", "n", "k")):
                m_tiles = math.ceil(dims["m"] / block_m)
                n_tiles = math.ceil(dims["n"] / block_n)
                k_tiles = math.ceil(dims["k"] / block_k)
                output_tiles = batch * m_tiles * n_tiles
                cube_units = output_tiles * k_tiles
                padded_flops = (
                    2 * output_tiles * k_tiles * block_m * block_n * block_k)
                # Some structural leaves use one M/N/K descriptor for a
                # multi-phase kernel (for example latent BMM).  Preserve all
                # declared FLOPs and scale its tile traffic by the same phase
                # multiplicity instead of accidentally making padded work
                # smaller than useful work.
                phase_multiplier = max(1.0, flops / max(1, padded_flops))
                cube_units *= phase_multiplier
                output_tiles *= phase_multiplier
                padded_flops *= phase_multiplier
                padded_compute_ms = padded_flops / (peak_tflops * 1e12) * 1e3
            else:
                # Composite Cube kernels (FA/SWA/LAT/VWN) do not expose one
                # GEMM M/N/K triple.  Their structural FLOPs still determine
                # the number of base-tile MMAD units without inventing a
                # shape-specific efficiency.
                unit_flops = 2 * block_m * block_n * block_k
                cube_units = max(1, math.ceil(flops / unit_flops))
                k_tiles = max(1, math.ceil(dims.get("k", tk) / block_k))
                output_tiles = max(1, math.ceil(cube_units / k_tiles))
                padded_flops = cube_units * unit_flops
                padded_compute_ms = (
                    padded_flops / (peak_tflops * 1e12) * 1e3
                    / composite_batch_alignment)

            input_tile_bytes = (block_m * block_k + block_n * block_k) * dtype_bytes
            output_tile_bytes = block_m * block_n * output_bytes
            read_hops = max(1, int(cfg.get("cube_operand_transfer_hops", 2)))
            onchip_bytes = (cube_units * input_tile_bytes * read_hops
                            + output_tiles * output_tile_bytes)
            if composite_batch_alignment < 1.0:
                onchip_bytes /= composite_batch_alignment
            transfer_transaction_bytes = max(1, int(cfg.get(
                "onchip_transfer_bytes_per_cycle",
                mte_cfg.get("gm_issue_bytes_per_cycle", 512))))

            # P = cores * frequency * 2*Tm*Tn*Tk for one BF16 MMAD/cycle.
            # This makes frequency a consequence of declared peak/core/tile
            # hardware facts rather than another performance-fit input.
            ops_per_core_cycle = 2 * tm * tn * tk
            core_num_for_peak = max(
                1, int(cfg.get("aic_core_num")
                       or spec_compute.get("aic_core_num")
                       or active_core_num))
            inferred_core_frequency_hz = (
                peak_tflops * 1e12 / (core_num_for_peak * ops_per_core_cycle))
            inferred_core_frequency_ghz = inferred_core_frequency_hz / 1e9
            transfer_cycles = math.ceil(onchip_bytes / transfer_transaction_bytes)
            onchip_time_ms = (
                transfer_cycles / (active_core_num * inferred_core_frequency_hz) * 1e3)

            work_tiles = output_tiles
            wave_count = math.ceil(work_tiles / active_core_num)
            wave_utilization = work_tiles / (wave_count * active_core_num)
            instruction_utilization = min(1.0, ideal_compute_ms / padded_compute_ms)
            limit_efficiency = max(
                1e-12,
                min(1.0, roofline * instruction_utilization * wave_utilization),
            )
        else:
            instruction_utilization = alignment

        # A grouped row GEMM reduces K-split partial outputs before emitting
        # the single hidden-state tensor. Column Gate/Up GEMMs do not have
        # this output reduction. Model the reduction tree from K tiles,
        # logical output bytes and group count; no measured efficiency is
        # involved.
        if (op_name == "group_linear_row" and engine == "cube"
                and all(name in dims for name in ("m", "n", "k"))
                and active_core_num and inferred_core_frequency_ghz
                and transfer_transaction_bytes):
            block_k = max(1, int(tiling.get("base_k", tk)))
            split_k = max(1, math.ceil(dims["k"] / block_k))
            reduction_steps = math.ceil(math.log2(split_k))
            group_count = max(1, int(dims.get("ng", 1)))
            partial_bytes = (
                group_count * dims["m"] * dims["n"]
                * max(4, output_dtype_bytes))
            onchip_bandwidth_bytes_s = (
                active_core_num * inferred_core_frequency_ghz * 1e9
                * transfer_transaction_bytes)
            reduction_transfer_ms = (
                2 * partial_bytes * reduction_steps
                / onchip_bandwidth_bytes_s * 1e3)
            reduction_barrier_ms = (
                group_count * reduction_steps * launch_us / 1e3)
            library_extra_time_ms = (
                reduction_transfer_ms + reduction_barrier_ms)
            library_extra_detail = {
                "kind": "grouped_row_split_k_reduction",
                "group_count": group_count,
                "split_k": split_k,
                "reduction_steps": reduction_steps,
                "partial_output_bytes": partial_bytes,
                "reduction_transfer_ms": reduction_transfer_ms,
                "reduction_barrier_ms": reduction_barrier_ms,
            }

        if resource_policy == "overlap":
            resource_time_ms = max(
                padded_compute_ms, ideal_memory_ms, onchip_time_ms)
        elif resource_policy == "serial":
            resource_time_ms = padded_compute_ms + ideal_memory_ms + onchip_time_ms
        else:
            raise ValueError(
                "forward_derivation.compute.resource_overlap_policy must be "
                f"'serial' or 'overlap', got {resource_policy!r}")
        implementation_stage_overhead_ms, materialized_kernel_count = (
            self._implementation_stage_overhead_ms(stage_model, launch_us))
        hbm_latency_us = (
            float(self.accelerator.bandwidth["default"].latency_us)
            if memory_bytes else 0.0)
        derived_time_ms = (
            resource_time_ms + library_extra_time_ms
            + implementation_stage_overhead_ms
            + (launch_us + hbm_latency_us) / 1e3)
        optimistic_time_ms = (max(
            padded_compute_ms, ideal_memory_ms, onchip_time_ms)
            + implementation_stage_overhead_ms
            + (launch_us + hbm_latency_us) / 1e3)
        conservative_time_ms = (
            padded_compute_ms + ideal_memory_ms + onchip_time_ms
            + implementation_stage_overhead_ms
            + (launch_us + hbm_latency_us) / 1e3)
        optimistic_utilization = (
            ideal_compute_ms / optimistic_time_ms if optimistic_time_ms else None)
        conservative_utilization = (
            ideal_compute_ms / conservative_time_ms if conservative_time_ms else None)
        achievable = (ideal_compute_ms / derived_time_ms
                      if derived_time_ms > 0 else limit_efficiency)
        achievable = max(1e-12, min(limit_efficiency, achievable))
        # ``attainable_efficiency`` is the explicit Roofline-to-runtime
        # metric.  It is intentionally bounded by the structural limit and
        # uses only forward-derived time: shape/tile padding, wave occupancy,
        # memory traffic, on-chip movement, library stages and declared
        # launch latency.  It is not a profiler counter and never consumes a
        # measured duration.
        resource_efficiency = (
            ideal_compute_ms / resource_time_ms
            if ideal_compute_ms > 0 and resource_time_ms > 0 else None)
        library_runtime_efficiency = (
            (resource_time_ms + implementation_stage_overhead_ms
             + (launch_us + hbm_latency_us) / 1e3)
            / derived_time_ms
            if derived_time_ms > 0 else None)
        attainable_efficiency = achievable if flops > 0 else None
        # compute_op_accuracy_time uses the configured Cube reference peak.
        # Translate an engine-relative utilization back to that reference.
        relative_peak = peak_tflops / reference_peak_tflops
        derived_reference_efficiency = achievable * relative_peak
        declared_reference_efficiency = (
            declared_utilization * relative_peak
            if declared_utilization is not None else None)
        override = self._operator_mfu_override(op_name, shape_desc)
        calibration_entry = self._calibration_compute_multiplier(
            op_name, shape_desc, stage, kernel_role, projection)
        memory_calibration_entry = self._calibration_memory_multiplier(
            op_name, shape_desc, stage, kernel_role, projection)
        calibrated_reference_efficiency = None
        if calibration_entry is not None:
            calibrated_reference_efficiency = max(
                1e-12,
                min(1.0, derived_reference_efficiency
                    * calibration_entry["multiplier"]),
            )
        used_efficiency = (
            override if override is not None
            else (calibrated_reference_efficiency
                  if calibrated_reference_efficiency is not None
                  else (declared_reference_efficiency
                        if declared_reference_efficiency is not None
                        else derived_reference_efficiency)))
        stage_key = stage or "unspecified"
        path = path_key or "path_unspecified"
        key = f"{path}|{op_name}|{stage_key}|{shape_desc or 'shape_unspecified'}"
        library_tiling_table = cfg.get("library_tiling", {})
        tiling_is_operator_specific = op_name in library_tiling_table
        generic_matmul_ops = {
            "matmul", "group_linear_col", "group_linear_row",
            "optimizer_orthogonal_bmm", "optimizer_orthogonal_matmul",
            "latent_bmm",
        }
        missing_facts = []
        dtype_tables = spec_compute.get("peak_tflops_by_dtype", {}) or {}
        engine_dtype_table = dtype_tables.get(engine, {}) if isinstance(
            dtype_tables, dict) else {}
        has_vector_peak = (
            spec_compute.get("vector_peak_tflops") is not None
            or (isinstance(engine_dtype_table, dict)
                and engine_dtype_table.get(compute_dtype_name) is not None))
        if engine == "vector" and not has_vector_peak:
            missing_facts.append("vector_peak_tflops")
        if (engine == "cube" and not tiling_is_operator_specific
                and op_name not in generic_matmul_ops):
            missing_facts.append("operator_specific_library_tiling")
        if engine == "cube" and not all(name in dims for name in ("m", "n", "k")):
            missing_facts.append("composite_cube_vector_mte_stage_decomposition")
        if not stage_model or materialized_kernel_count is None:
            missing_facts.append("cann_materialized_kernel_split_count")
        self.forward_derivation_records["operators"][key] = {
            "path": path_key,
            "stage": stage,
            "op_name": op_name,
            "shape": shape_desc,
            "flops": flops,
            "compute_utilization_applicable": flops > 0,
            "memory_bytes": memory_bytes,
            "compute_dtype": compute_dtype_name,
            "compute_dtype_bytes": compute_dtype_bytes,
            "output_dtype": output_dtype_name,
            "output_dtype_bytes": output_dtype_bytes,
            "engine": engine,
            "cann_runtime_spec_status": (
                (self.cann_runtime or {}).get("spec_status")),
            "peak_tflops": peak_tflops,
            "peak_source": peak_source,
            "reference_peak_tflops": reference_peak_tflops,
            "dtype_peak_tflops": peak_tflops,
            "dtype_peak_source": dtype_peak_source,
            "declared_utilization": declared_utilization,
            "declared_utilization_source": declared_utilization_source,
            "declared_reference_utilization": declared_reference_efficiency,
            "hbm_bandwidth_gbps": hbm_gbps,
            "hbm_roofline_utilization": hbm_roofline,
            "l2_cache_bytes": l2_capacity_bytes,
            "l2_bandwidth_gbps": l2_gbps,
            "l2_roofline_utilization": l2_roofline,
            "memory_transaction_bytes": transaction_bytes,
            "memory_transaction_utilization": memory_transaction_utilization,
            "tensor_tile": {"m": tm, "n": tn, "k": tk},
            "arithmetic_intensity_flop_per_byte": arithmetic_intensity,
            "theoretical_utilization": roofline,
            "shape_alignment": alignment,
            "composite_batch_alignment": composite_batch_alignment,
            "aic_core_num": spec_compute.get("aic_core_num"),
            "aiv_core_num": spec_compute.get("aiv_core_num"),
            "active_core_num": active_core_num,
            "library_tiling": tiling or None,
            "library_tiling_source": (
                tiling.get("source") if isinstance(tiling, dict) else None),
            "mte_profile": mte_cfg or None,
            "cann_host_tiling_policy": {
                "policy": host_tiling_cfg.get("policy"),
                "input_fields": list(
                    host_tiling_cfg.get("input_fields", []) or []),
                "output_fields": list(
                    host_tiling_cfg.get("output_fields", []) or []),
                "tile_policy": host_tiling_cfg.get("tile_policy"),
                "block_dim_policy": host_tiling_cfg.get("block_dim_policy"),
                "tiling_key_policy": host_tiling_cfg.get("tiling_key_policy"),
                "workspace_policy": host_tiling_cfg.get("workspace_policy"),
                "stage_dependency_policy": host_tiling_cfg.get(
                    "stage_dependency_policy"),
            },
            "cann_alignment_constraints": alignment_constraints or None,
            "tiling_is_operator_specific": tiling_is_operator_specific,
            "work_tiles": work_tiles,
            "wave_count": wave_count,
            "wave_utilization": wave_utilization,
            "instruction_utilization": instruction_utilization,
            "limit_utilization": limit_efficiency,
            "ideal_compute_time_ms": ideal_compute_ms,
            "padded_compute_time_ms": padded_compute_ms,
            "ideal_memory_time_ms": ideal_memory_ms,
            "onchip_transfer_bytes": onchip_bytes,
            "onchip_transfer_time_ms": onchip_time_ms,
            "onchip_transfer_bytes_per_cycle": transfer_transaction_bytes,
            "inferred_core_frequency_ghz": inferred_core_frequency_ghz,
            "resource_overlap_policy": resource_policy,
            "library_extra_time_ms": library_extra_time_ms,
            "library_extra_detail": library_extra_detail,
            "implementation_stage_model": stage_model or None,
            "materialized_kernel_count": materialized_kernel_count,
            "implementation_stage_overhead_ms": implementation_stage_overhead_ms,
            "derived_time_ms": derived_time_ms,
            "attainable_efficiency": attainable_efficiency,
            "attainable_efficiency_bound": (
                limit_efficiency if flops > 0 else None),
            "attainable_efficiency_factors": {
                "roofline_bound": roofline,
                "hbm_roofline_bound": hbm_roofline,
                "l2_roofline_bound": l2_roofline,
                "shape_tile_alignment": alignment,
                "composite_batch_alignment": composite_batch_alignment,
                "instruction_utilization": instruction_utilization,
                "wave_utilization": wave_utilization,
                "memory_transaction_utilization": memory_transaction_utilization,
                "mte_profile": mte_cfg or None,
                "resource_schedule_efficiency": resource_efficiency,
                "library_runtime_efficiency": library_runtime_efficiency,
                "implementation_stage_overhead": implementation_stage_overhead_ms,
                "formula": (
                    "U_attainable=min(U_bound,"
                    "ideal_compute_time/derived_time)"
                ),
                "source": "hardware_spec+cann_runtime+model_shape",
            },
            "semantic_effective_utilization": (
                flops / ((derived_time_ms / 1e3) * peak_tflops * 1e12)
                if flops > 0 and derived_time_ms > 0 and peak_tflops > 0 else None),
            "semantic_utilization_formula": (
                "semantic_flops/(simulated_duration_s*peak_tflops*1e12)"),
            "optimistic_overlap_utilization": optimistic_utilization,
            "conservative_serial_utilization": conservative_utilization,
            "implementation_facts_complete": not missing_facts,
            "missing_implementation_facts": missing_facts,
            "kernel_launch_latency_us": launch_us,
            "hbm_base_latency_us": hbm_latency_us,
            "derived_achievable_utilization": achievable,
            "derived_reference_peak_utilization": derived_reference_efficiency,
            "customer_mfu_override": self._operator_mfu_override(op_name, shape_desc),
            "calibration_efficiency_multiplier": (
                calibration_entry["multiplier"]
                if calibration_entry is not None else None),
            "calibrated_reference_utilization": calibrated_reference_efficiency,
            "calibration_samples": (
                calibration_entry.get("samples")
                if calibration_entry is not None else None),
            "calibration_comm_role": (
                calibration_entry.get("comm_role")
                if calibration_entry is not None else None),
            "calibration_kernel_role": (
                calibration_entry.get("kernel_role")
                if calibration_entry is not None else None),
            "calibration_projection": (
                calibration_entry.get("projection")
                if calibration_entry is not None else None),
            "calibration_memory_time_multiplier": (
                memory_calibration_entry["multiplier"]
                if memory_calibration_entry is not None else None),
            "calibration_memory_samples": (
                memory_calibration_entry.get("samples")
                if memory_calibration_entry is not None else None),
            "memory_performance_observations_used": (
                memory_calibration_entry is not None),
            "used_utilization": used_efficiency,
            "used_engine_utilization": (
                used_efficiency / relative_peak
                if relative_peak > 0 else None),
            "utilization_source": (
                "api_override" if override is not None else
                "measured_calibration" if calibrated_reference_efficiency is not None else
                declared_utilization_source if declared_utilization_source is not None else
                "forward_formula"),
            "performance_observations_used": calibration_entry is not None,
            "performance_observations_used_as_parameters": (
                calibration_entry is not None),
            "formula": (
                "theory=roofline; limit=theory*instruction_tile*wave; "
                "T_resource=serial_or_overlap(padded_cube,HBM,GM-L1-L0-FixPipe)"
                "+library_structural_stage; "
                "achievable=ideal_compute/(T_resource+launch); "
                "calibrated=derived_reference_efficiency*aggregate_efficiency_multiplier"
                if calibration_entry is not None else
                "declared=clamp(log_a*ln(FLOPs)+log_b|constant); "
                "used=declared_reference_utilization"
                if declared_utilization is not None else
                "theory=roofline; limit=theory*instruction_tile*wave; "
                "T_resource=serial_or_overlap(padded_cube,HBM,GM-L1-L0-FixPipe)"
                "+library_structural_stage; "
                "achievable=ideal_compute/(T_resource+launch)"
            ),
        }
        return used_efficiency

    @staticmethod
    def _collective_algorithm(op_name, comm_num):
        """Use the single portable collective definition for all outputs."""
        return portable_collective_algorithm(op_name, comm_num)

    def _collective_runtime_overhead(
            self, op_name, comm_num, message_bytes, active_level_count=1):
        """Derive call/runtime overhead separately from link propagation.

        A topology level's ``latency_us`` is a physical per-hop property. It
        must not absorb host dispatch, task construction, or collective
        schedule costs. Those costs belong to one collective call and are
        derived here from the runtime implementation specification.
        """
        network_cfg = self._hccl_network_config()
        runtime_cfg = network_cfg.get("call_runtime", {})
        algorithm_selection = network_cfg.get("algorithm_selection", {}) or {}
        if not isinstance(algorithm_selection, dict):
            algorithm_selection = {}
        supported_collectives = network_cfg.get("supported_collectives", []) or []
        resource_request = network_cfg.get("resource_request")
        compute_cfg = self._cann_compute_config()
        default_launch_us = float(
            compute_cfg.get("kernel_launch_latency_us", 0.0))
        call_launch_us = float(runtime_cfg.get(
            "call_launch_latency_us", default_launch_us))
        task_launch_us = float(runtime_cfg.get(
            "task_launch_latency_us", call_launch_us))
        tasks_per_stage = max(0, int(runtime_cfg.get("tasks_per_stage", 1)))
        chunk_bytes = max(0, int(runtime_cfg.get("descriptor_chunk_bytes", 0)))
        tasks_per_chunk = max(0, int(runtime_cfg.get(
            "tasks_per_additional_chunk", 0)))
        execution_stages = runtime_cfg.get(
            "execution_stages", ["post", "task", "completion", "wait"])
        if not isinstance(execution_stages, list):
            execution_stages = [str(execution_stages)]
        completion_cfg = runtime_cfg.get("completion", {}) or {}
        completion_fields = (
            "completion_latency_us", "wait_latency_us", "barrier_latency_us")
        completion_unknown_fields = [
            name for name in completion_fields
            if not isinstance(completion_cfg.get(name), (int, float))
        ]
        completion_overhead_us = sum(
            max(0.0, float(completion_cfg.get(name, 0.0)))
            for name in completion_fields
            if isinstance(completion_cfg.get(name), (int, float)))

        algorithm, stages = self._collective_algorithm(op_name, comm_num)
        if stages is None:
            return {
                "execution_engine": runtime_cfg.get(
                    "execution_engine", "host_cpu_ts"),
                "algorithm": algorithm,
                "algorithm_stages": None,
                "active_network_levels": max(1, int(active_level_count)),
                "payload_chunks": None,
                "stage_runtime_tasks": None,
                "descriptor_runtime_tasks": None,
                "runtime_task_count": None,
                "call_launch_latency_us": call_launch_us,
                "task_launch_latency_us": task_launch_us,
                "call_runtime_overhead_us": None,
                "execution_stages": execution_stages,
                "algorithm_selection_policy": algorithm_selection.get("policy"),
                "algorithm_level_scope": algorithm_selection.get("level_scope"),
                "supported_collectives": list(supported_collectives),
                "resource_request": resource_request,
                "task_count_policy": runtime_cfg.get("task_count_policy"),
                "descriptor_policy": runtime_cfg.get("descriptor_policy"),
                "completion_policy": runtime_cfg.get("completion_policy"),
                "runtime_profile_spec_status": (
                    (self.hccl_runtime or {}).get("spec_status")),
                "completion_latency_us": completion_cfg.get("completion_latency_us"),
                "wait_latency_us": completion_cfg.get("wait_latency_us"),
                "barrier_latency_us": completion_cfg.get("barrier_latency_us"),
                "completion_overhead_us": None,
                "runtime_profile_unknown_fields": completion_unknown_fields,
                "status": "unknown",
                "unknown_reason": "unsupported_collective_algorithm",
                "formula": (
                    "runtime unavailable until a generic collective algorithm "
                    "is declared"),
            }
        level_count = max(1, int(active_level_count))
        stage_tasks = stages * level_count * tasks_per_stage
        chunks = (math.ceil(message_bytes / chunk_bytes)
                  if chunk_bytes and message_bytes else 1)
        descriptor_tasks = max(0, chunks - 1) * tasks_per_chunk
        task_count = stage_tasks + descriptor_tasks
        runtime_us = call_launch_us + task_count * task_launch_us \
            + completion_overhead_us
        return {
            "execution_engine": runtime_cfg.get(
                "execution_engine", "host_cpu_ts"),
            "algorithm": algorithm,
            "algorithm_stages": stages,
            "active_network_levels": level_count,
            "payload_chunks": chunks,
            "stage_runtime_tasks": stage_tasks,
            "descriptor_runtime_tasks": descriptor_tasks,
            "runtime_task_count": task_count,
            "call_launch_latency_us": call_launch_us,
            "task_launch_latency_us": task_launch_us,
            "call_runtime_overhead_us": runtime_us,
            "execution_stages": execution_stages,
            "algorithm_selection_policy": algorithm_selection.get("policy"),
            "algorithm_level_scope": algorithm_selection.get("level_scope"),
            "supported_collectives": list(supported_collectives),
            "resource_request": resource_request,
            "task_count_policy": runtime_cfg.get("task_count_policy"),
            "descriptor_policy": runtime_cfg.get("descriptor_policy"),
            "completion_policy": runtime_cfg.get("completion_policy"),
            "runtime_profile_spec_status": (
                (self.hccl_runtime or {}).get("spec_status")),
            "completion_latency_us": completion_cfg.get("completion_latency_us"),
            "wait_latency_us": completion_cfg.get("wait_latency_us"),
            "barrier_latency_us": completion_cfg.get("barrier_latency_us"),
            "completion_overhead_us": completion_overhead_us,
            "runtime_profile_unknown_fields": completion_unknown_fields,
            "formula": (
                "T_runtime=L_call+(algorithm_stages*active_levels*"
                "tasks_per_stage+descriptor_tasks)*L_task+"
                "T_completion+T_wait+T_barrier"),
        }

    def _derive_network_time(self, op_name, actual_size, comm_num, net,
                             comm_stage, strategy, group_kind, topology_bw,
                             topology_latency, comm_direction=None,
                             comm_role=None):
        """Return alpha+payload/beta using only topology/hardware inputs."""
        del strategy  # topology/group decomposition has already been applied
        requested_op_name = op_name
        cfg = self._hccl_network_config()
        flit_bytes = max(1, int(cfg.get("flit_bytes", 256)))
        if topology_bw is None:
            topology_bw = self.networks[net].bandwidth.gbps
        if topology_latency is None:
            topology_latency = self.networks[net].bandwidth.latency_us
        padded_bytes = self._ceil_to(actual_size, flit_bytes)
        packet_eff = actual_size / padded_bytes if padded_bytes else 1.0
        beta = topology_bw * packet_eff
        algorithm, stages = self._collective_algorithm(op_name, comm_num)
        physical_latency_us = topology_latency
        runtime = self._collective_runtime_overhead(
            op_name, comm_num, actual_size, active_level_count=1)
        collective_latency_us = (
            physical_latency_us + runtime["call_runtime_overhead_us"])
        # Network configuration uses decimal GB/s (the hardware/link-rate
        # convention), unlike memory capacity fields that commonly use GiB.
        transfer_time_ms = actual_size / (beta * 1e9) * 1e3
        time_ms = transfer_time_ms + collective_latency_us / 1e3
        calibration_direction = comm_direction or comm_stage
        time_ms, calibration_entry = self._apply_communication_calibration(
            requested_op_name, actual_size, comm_num, time_ms,
            calibration_direction, comm_role,
            transfer_time_ms=transfer_time_ms, comm_stage=comm_stage)
        layer_key = f"{net}|bytes={int(actual_size)}"
        self.forward_derivation_records["network_layers"][layer_key] = {
            "network_level": net,
            "message_bytes": actual_size,
            "physical_bandwidth_gib_per_s": topology_bw,
            "packet_efficiency": packet_eff,
            "bandwidth_utilization": packet_eff,
            "effective_beta_gib_per_s": beta,
            # Canonical network-rate field.  Network profiles use decimal
            # GB/s; keep the historical *_gib_* alias for compatibility.
            "effective_beta_gb_per_s": beta,
            "base_latency_us": topology_latency,
            "algorithm_independent": True,
            "formula": "beta=B_physical*payload/ceil(payload/flit)",
        }
        key = (f"{op_name}|{comm_stage.lower()}"
               f"|direction={self._calibration_direction(calibration_direction)}"
               f"|n={comm_num}|bytes={int(actual_size)}")
        self.forward_derivation_records["communications"][key] = {
            "op_name": op_name,
            "stage": comm_stage.lower(),
            "direction": self._calibration_direction(calibration_direction),
            "group_kind": group_kind,
            "comm_num": comm_num,
            "message_bytes": actual_size,
            "topology_bandwidth_gbps": topology_bw,
            "flit_bytes": flit_bytes,
            "packet_efficiency": packet_eff,
            "derived_beta_gib_per_s": beta,
            "derived_beta_gb_per_s": beta,
            "algorithm": algorithm,
            "algorithm_stages": stages,
            "network_layer_latency_us": topology_latency,
            "physical_propagation_latency_us": physical_latency_us,
            "call_runtime_overhead_us": runtime["call_runtime_overhead_us"],
            "call_runtime": runtime,
            "collective_latency_us": collective_latency_us,
            "derived_transfer_time_ms": transfer_time_ms,
            "derived_time_ms": time_ms,
            "calibration_transfer_efficiency": (
                calibration_entry.get("transfer_efficiency")
                if calibration_entry is not None else None),
            "calibration_applied_to": (
                "pure_transfer_component"
                if calibration_entry is not None
                and calibration_entry.get("transfer_efficiency") is not None
                else "aggregate_call_lifetime"
                if calibration_entry is not None else None),
            "calibration_time_multiplier": (
                calibration_entry.get("multiplier")
                if calibration_entry is not None
                and calibration_entry.get("transfer_efficiency") is None
                else None),
            "calibration_samples": (
                calibration_entry.get("samples")
                if calibration_entry is not None else None),
            "calibration_parameter_name": (
                calibration_entry.get("parameter_name")
                if calibration_entry is not None else None),
            "calibration_source_type": (
                calibration_entry.get("source_type")
                if calibration_entry is not None else None),
            "performance_observations_used": calibration_entry is not None,
            "performance_observations_used_as_parameters": (
                calibration_entry is not None),
            "formula": (
                "T=D/beta_layer+T_physical_propagation+T_call_runtime"
                "+T_transfer*(1/transfer_efficiency-1)"
                if calibration_entry is not None
                and calibration_entry.get("transfer_efficiency") is not None
                else
                "T=D/beta_layer+T_physical_propagation+T_call_runtime"
                "+aggregate_runtime_multiplier"
                if calibration_entry is not None else
                "T=D/beta_layer+T_physical_propagation+T_call_runtime"),
        }
        self.record_net_bw(op_name, net, comm_num, comm_stage, topology_bw,
                           beta, packet_eff, time_ms * 1e3, actual_size,
                           collective_latency_us)
        return time_ms

    def record_hit_efficiency(
        self, op_name: str, flops: int, shape_desc: str, eff, path_key=None, level=None
    ):
        if op_name not in self.hit_efficiency:
            self.hit_efficiency[op_name] = {}
        if path_key is None and level is None:
            # Legacy record shape, kept byte-identical for the no-override path.
            self.hit_efficiency[op_name][shape_desc] = (flops, eff)
        else:
            # Override-chain hit (cost-tunability design doc section 3):
            # attribute the winning key level and source.
            self.hit_efficiency[op_name][shape_desc] = {
                'flops': flops,
                'eff': eff,
                'path_key': path_key,
                'level': level,
            }

    def _lookup_efficiency_override(self, class_key, path_key, shape_desc):
        """Resolve the per-operator efficiency override chain (design doc 3).

        Key levels are checked in order, first hit wins:
        (path_key, shape_desc) > path_key > (class_key, shape_desc) > class_key.
        Path keys use prefix semantics: "layer_0.mlp" covers the whole
        subtree, longest matching prefix wins.
        Within one key level the source precedence is
        efficiency_overrides_api > efficiency_overrides_strategy >
        operator_efficiency. A scalar entry applies at both the (key, shape)
        and the key level; a dict entry resolves to shapes[shape_desc] at the
        (key, shape) level (shape_desc may be "") and to its "default" at the
        key level.

        Returns (efficiency, level_label) on hit, (None, None) on miss.
        level_label is "<source>:<path|class>[+shape]", e.g. "api:path+shape",
        "strategy:class", "system:class+shape".
        """
        sources = (
            ("api", self.efficiency_overrides_api),
            ("strategy", self.efficiency_overrides_strategy),
            ("system", self.operator_efficiency),
        )
        # Path keys use prefix semantics: an override on "layer_0.mlp"
        # applies to the whole subtree (e.g. "layer_0.mlp.linear_fc1").
        # The longest matching prefix wins; ties break api > strategy >
        # system. Sub-levels keep the design order: (path, shape) first,
        # then the path-level default.
        if path_key is not None:
            matches = []
            for src_rank, (src_label, table) in enumerate(sources):
                if not table:
                    continue
                for key, value in table.items():
                    if path_key == key or path_key.startswith(key + "."):
                        matches.append((key, src_rank, src_label, value))
            if matches:
                matches.sort(key=lambda m: (-len(m[0]), m[1]))
                for _, _, src_label, value in matches:
                    eff = (value.get("shapes") or {}).get(shape_desc) \
                        if isinstance(value, dict) else value
                    if eff is not None:
                        return eff, f"{src_label}:path+shape"
                for _, _, src_label, value in matches:
                    eff = value.get("default") if isinstance(value, dict) else value
                    if eff is not None:
                        return eff, f"{src_label}:path"
        # Class keys are exact-match.
        for key, kind in ((class_key, "class"),):
            if key is None:
                continue
            entries = [
                (src_label, table[key])
                for src_label, table in sources
                if table and key in table
            ]
            if not entries:
                continue
            # (key, shape) level: scalar applies directly; dict needs a
            # matching shapes entry.
            for src_label, value in entries:
                if isinstance(value, dict):
                    eff = (value.get("shapes") or {}).get(shape_desc)
                else:
                    eff = value
                if eff is not None:
                    return eff, f"{src_label}:{kind}+shape"
            # key level: scalar applies directly; dict yields its default.
            for src_label, value in entries:
                if isinstance(value, dict):
                    eff = value.get("default")
                else:
                    eff = value
                if eff is not None:
                    return eff, f"{src_label}:{kind}"
        return None, None

    def validate_efficiency_override_keys(self, known_keys: set) -> list:
        """Return the sorted override keys that match no known class_key or
        path_key (design doc 4: unknown keys must raise at configure time).
        Path keys additionally match when they are an ancestor prefix of a
        known path (e.g. "layer_0.mlp" covers "layer_0.mlp.linear_fc1").
        Checks operator_efficiency and both runtime override dicts.
        """
        def _known(key):
            if key in known_keys:
                return True
            prefix = key + "."
            return any(k.startswith(prefix) for k in known_keys)

        unknown = set()
        for table in (
            self.operator_efficiency,
            self.efficiency_overrides_strategy,
            self.efficiency_overrides_api,
        ):
            if not table:
                continue
            unknown.update(key for key in table if not _known(key))
        return sorted(unknown)

    def reset_record_info(self):
        self.miss_efficiency.clear()
        self.hit_efficiency.clear()
        self.real_comm_bw.clear()
        self.forward_derivation_records = {
            "operators": {}, "network_layers": {}, "communications": {}}
        self.communication_plan_document = None

    def build_communication_plan_document(self, events=None, strategy=None):
        """Build the portable communication-plan output for this run.

        The import is local to keep the config module usable by the base
        package during its own import cycle.  ``events`` are SimuMax-generated
        DES events; measured profiler data is never accepted here.
        """
        from simumax.core.communication_plan import (
            build_communication_plan_document,
        )

        document = build_communication_plan_document(
            events=events,
            system=self,
            strategy=strategy,
            derivation_records=self.forward_derivation_records,
        )
        self.communication_plan_document = document
        return document

    @staticmethod
    def _lookup_accurate_eff(accurate_factor, shape_desc):
        """Cross-scale reuse: strip the m (seq) and b (batch) dims from the key
        and match a 16p-calibrated entry with identical remaining structure
        (k / n / layout / op-kind). Efficiency is per-FLOP for a fixed shape
        structure + hardware, so reusing the 16p value across 8p/32p is the
        validation hypothesis — whether it holds is answered by cross-config
        comparison. None when no entry matches after stripping.

        Deterministic collision handling (review P0b): several entries can share
        a stripped core (differ only in m=), e.g. the NT accumulate=True fp32
        group (m=1536 vs 4608/5120/6144 → eff 0.398 vs 0.589, 1.48x). The old
        code returned the first dict-iteration match — order-dependent and up to
        ~2x wrong for a query m not in the table. Now we return the entry whose
        m is closest to the query (best physical match for saturated GEMMs), and
        warn so the caller knows an exact entry is missing."""
        if not accurate_factor:
            return None

        def _strip(k):
            return re.sub(r'\bb=\d+', '', re.sub(r'\bm=\d+', '', k))

        def _extract_m(k):
            mt = re.search(r'\bm=(\d+)', k)
            return int(mt.group(1)) if mt else None

        core = _strip(shape_desc)
        q_m = _extract_m(shape_desc)
        matches = [(k, v, _extract_m(k))
                   for k, v in accurate_factor.items() if _strip(k) == core]
        if not matches:
            return None
        if q_m is not None:
            # Exact m present in the table -> exact hit (overrides any stripped-
            # core collision). 16p trained shapes always carry their exact m
            # entry, so the current config's collection stays an exact hit here.
            exact = [m for m in matches if m[2] == q_m]
            if exact:
                return exact[0][1]
        if len(matches) == 1:
            return matches[0][1]
        # Collision: pick the entry with m closest to the query (deterministic)
        # and warn so the missing exact entry / ambiguous core is surfaced.
        vals = sorted(v for _, v, _m in matches)
        warnings.warn(
            f"_lookup_accurate_eff: {len(matches)} entries share stripped core "
            f"'{core}' after removing m=/b= (eff {vals[0]:.4f}~{vals[-1]:.4f}, "
            f"{vals[-1] / vals[0]:.2f}x spread). Query '{shape_desc}' has no "
            f"exact match; using the m-closest entry. Add an exact entry or "
            f"split the core for a reliable value.",
            stacklevel=2)
        if q_m is not None:
            best = min(matches, key=lambda t: abs((t[2] or 0) - q_m))
            return best[1]
        return matches[0][1]

    def compute_op_accuracy_time(
        self, op_name: str, flops: int, shape_desc: str, reture_detail=False,
        class_key=None, path_key=None, accessed_mem=None, stage=None,
        kernel_role=None, projection=None,
    ):
        """
        compute float point operation time,
        return time in ms

        matmul_input_shapes: list of input shapes, e.g. "[1, 16384, 4096] x [1, 4096, 128256]"

        class_key/path_key enable the per-operator efficiency override chain
        (cost-tunability design doc section 3, levels 1-4). When both are None
        the override block is skipped entirely and the behavior (time and
        miss/hit records) is identical to the legacy lookup (levels 5-7).
        """
        if flops == 0:
            if self.forward_derivation_enabled:
                op = self.accelerator.op.get(
                    op_name, self.accelerator.op["default"])
                self._derive_compute_efficiency(
                    op_name, flops, shape_desc, accessed_mem, stage, path_key,
                    kernel_role, projection)
                record_key = (
                    f"{path_key or 'path_unspecified'}|{op_name}|"
                    f"{stage or 'unspecified'}|{shape_desc or 'shape_unspecified'}")
                record = self.forward_derivation_records["operators"].get(
                    record_key, {})
                detail = dict(
                    op_name=op_name,
                    tflops=op.tflops,
                    efficient_factor=None,
                    compute_only_time=0.0,
                    efficiency_source="forward_derived",
                    peak_tflops=record.get("peak_tflops", op.tflops),
                    peak_source=record.get("peak_source"),
                    compute_dtype=record.get("compute_dtype"),
                    declared_utilization=record.get("declared_utilization"),
                    utilization_source=record.get("utilization_source"),
                    calibration_efficiency_multiplier=(
                        record.get("calibration_efficiency_multiplier")),
                    performance_observations_used=bool(
                        record.get("performance_observations_used", False)),
                )
                return detail if reture_detail else 0
            if reture_detail:
                return dict(op_name=op_name,
                                tflops=None,
                                efficient_factor=None,
                                compute_only_time = 0.0)
            else:
                return 0

        op = self.accelerator.op.get(op_name, None)
        if op is None:
            if not self.forward_derivation_enabled:
                warnings.warn(
                    f"{op_name} not exist on {self.accelerator.op.keys()}, "
                    "use default value"
                )
            op = self.accelerator.op.get("default", None)
            assert op is not None, f"default not exist on {self.accelerator.op}"
            if not self.forward_derivation_enabled:
                self.record_miss_efficiency(op_name, flops, shape_desc, None)

        if self.forward_derivation_enabled:
            efficient_factor = self._derive_compute_efficiency(
                op_name, flops, shape_desc, accessed_mem, stage, path_key,
                kernel_role, projection)
            time = flops / (op.tflops * 1e12 * efficient_factor) * 1e3
            record_key = (
                f"{path_key or 'path_unspecified'}|{op_name}|"
                f"{stage or 'unspecified'}|{shape_desc or 'shape_unspecified'}")
            record = self.forward_derivation_records["operators"].get(
                record_key, {})
            detail = dict(op_name=op_name, tflops=op.tflops,
                          efficient_factor=efficient_factor,
                          compute_only_time=time,
                          efficiency_source="forward_derived",
                          peak_tflops=record.get("peak_tflops", op.tflops),
                          peak_source=record.get("peak_source"),
                          compute_dtype=record.get("compute_dtype"),
                          declared_utilization=record.get("declared_utilization"),
                          utilization_source=record.get("utilization_source"),
                          calibration_efficiency_multiplier=(
                              record.get("calibration_efficiency_multiplier")),
                          performance_observations_used=bool(
                              record.get("performance_observations_used", False)))
            return detail if reture_detail else time

        if class_key is not None or path_key is not None:
            override_eff, override_level = self._lookup_efficiency_override(
                class_key, path_key, shape_desc
            )
            if override_eff is not None:
                efficient_factor = override_eff
                # Key-grouped hit record: attribute the class_key (not the
                # coarse op_name) plus the winning path_key and chain level.
                self.record_hit_efficiency(
                    class_key if class_key is not None else path_key,
                    flops,
                    shape_desc,
                    efficient_factor,
                    path_key=path_key,
                    level=override_level,
                )
                if SIMU_DEBUG:
                    print(
                        f"=== \033[32m{op_name} ({class_key}/{path_key}) input shape "
                        f"{shape_desc} use override compute efficient factor "
                        f"{efficient_factor} [{override_level}]\033[0m, flops={flops}"
                    )
                time = flops / (op.tflops * 1e12 * efficient_factor) * 1e3
                if reture_detail:
                    return dict(op_name=op_name,
                                tflops=op.tflops,
                                efficient_factor=efficient_factor,
                                compute_only_time = time)
                else:
                    return time
            # Override miss: fall through to the legacy levels 5-7 below with
            # the existing record calls unchanged.

        if ( op.accurate_efficient_factor is not None ) and \
        (op.accurate_efficient_factor.get(shape_desc, None) is not None):
            # marmul use accurate efficient factor to get accurate time
            efficient_factor = op.accurate_efficient_factor[shape_desc]
            self.record_hit_efficiency(op_name, flops, shape_desc, efficient_factor)
            if SIMU_DEBUG:
                print(f"=== \033[32m{op_name} input shape {shape_desc} use accurate compute efficient factor {efficient_factor}\033[0m, flops={flops}")
        else:
            eff_w = (self._lookup_accurate_eff(op.accurate_efficient_factor, shape_desc)
                     if op.accurate_efficient_factor else None)
            if eff_w is not None:
                # 16p 校准 eff 跨规模复用（m/b 通配）——同 shape 结构 + 同硬件，
                # eff 每 FLOP 恒定，跨 8p/32p 复用是验证假设（是否成立由跨配置对照回答）。
                efficient_factor = eff_w
                self.record_hit_efficiency(op_name, flops, shape_desc, efficient_factor)
                if SIMU_DEBUG:
                    print(f"=== \033[32m{op_name} input shape {shape_desc} reuse 16p accurate eff {efficient_factor} (cross-scale)\033[0m, flops={flops}")
            else:
                efficient_factor = op.efficient_factor
                self.record_miss_efficiency(op_name, flops, shape_desc, efficient_factor)

                if SIMU_DEBUG:
                    print(f"{op_name} input shape {shape_desc} use default compute efficient factor {efficient_factor}, flops={flops}")

        time = flops / (op.tflops * 1e12 * efficient_factor) * 1e3
        if reture_detail:
            return dict(op_name=op_name, 
                            tflops=op.tflops, 
                            efficient_factor=efficient_factor,
                            compute_only_time = time)
        else:
            return time

    def compute_mem_access_time(
            self, op_name, mem_bytes: int, reture_detail=False,
            shape_desc=None, stage=None, kernel_role=None, projection=None):
        """
        compute memory access time,
        return time in ms
        """
        
        if self.forward_derivation_enabled:
            # Forward derivation starts from the physical HBM bandwidth (a
            # hardware property, identical for all operators).  The separate
            # calibration branch may scale only the derived HBM transfer term
            # with a shape/stage/profile multiplier; it never replaces the
            # event duration or the model-derived latency.
            op = self.accelerator.bandwidth["default"]
            cfg = self._cann_compute_config()
            mte_cfg = cfg.get("mte", {}) or {}
            transaction = max(1, int(cfg.get(
                "memory_transaction_bytes",
                mte_cfg.get("transaction_bytes", 256))))
            padded = self._ceil_to(mem_bytes, transaction)
            efficiency = mem_bytes / padded if padded else 1.0
            # Hardware bandwidth is declared in decimal GB/s throughout the
            # forward model (the same convention as topology levels).
            transfer_time = mem_bytes / (op.gbps * 1e9 * efficiency) * 1e3
            calibration_entry = self._calibration_memory_multiplier(
                op_name, shape_desc, stage, kernel_role, projection)
            if calibration_entry is not None:
                transfer_time *= calibration_entry["multiplier"]
            time = transfer_time
            if mem_bytes:
                time += op.latency_us / 1e3
            if reture_detail:
                return dict(gbps=op.gbps, efficient_factor=efficiency,
                            latency_us=op.latency_us, io_time=time,
                            efficiency_source="forward_derived",
                            calibration_time_multiplier=(
                                calibration_entry["multiplier"]
                                if calibration_entry is not None else None),
                            calibration_samples=(
                                calibration_entry.get("samples")
                                if calibration_entry is not None else None),
                            performance_observations_used=(
                                calibration_entry is not None))
            return time

        op = self.accelerator.bandwidth.get(op_name, None)
        if op is None:
            op = self.accelerator.bandwidth.get("default", None)
        else:
            if op_name != "default" and SIMU_DEBUG:
                print(f'{op_name} use accurate memory bw efficiency {op.efficient_factor}')
        
        time = (
            mem_bytes
            / (
                op.gbps
                * 1024**3
                * op.efficient_factor
            )
            * 1e3
        )
        time += op.latency_us / 1e3
        if mem_bytes == 0:
            time = 0
        if reture_detail:
            return dict(gbps=op.gbps, 
                            efficient_factor=op.efficient_factor,
                            latency_us=op.latency_us,
                            io_time = time)
        return time

    def compute_layout_time(self, op_name, input_bytes, output_bytes=None,
                            stage=None, path_key=None, shape_desc=None,
                            reture_detail=False):
        """Derive one materialized layout-kernel duration from byte traffic.

        ``input_bytes`` and ``output_bytes`` are structural tensor sizes. The
        duration uses only the configured HBM bandwidth/transaction size and
        kernel-launch parameter. This helper deliberately has no measured-time
        or efficiency argument.
        """
        output_bytes = input_bytes if output_bytes is None else output_bytes
        shape_desc = shape_desc or (
            f"input_bytes={int(input_bytes)}, output_bytes={int(output_bytes)}")
        resolved_op_name = self.resolve_layout_op_name(
            op_name, stage=stage, path_key=path_key, shape_desc=shape_desc)
        cfg = self._cann_compute_config()
        mte_cfg = cfg.get("mte", {}) or {}
        resource_profile = (cfg.get("layout_resource_models", {}) or {}).get(
            resolved_op_name, {})
        if not isinstance(resource_profile, dict):
            resource_profile = {}
        traffic_policy = str(resource_profile.get(
            "traffic_policy", "read_write")).lower()
        if traffic_policy in {"write_only", "indexed_write", "scatter_write"}:
            accessed_mem = max(0, output_bytes)
        elif traffic_policy in {"read_only", "indexed_read"}:
            accessed_mem = max(0, input_bytes)
        else:
            accessed_mem = max(0, input_bytes) + max(0, output_bytes)
        compute_detail = self.compute_op_accuracy_time(
            resolved_op_name, 0, shape_desc=shape_desc, reture_detail=True,
            accessed_mem=accessed_mem, stage=stage, path_key=path_key)
        io_detail = self.compute_mem_access_time(
            resolved_op_name, accessed_mem, reture_detail=True,
            shape_desc=shape_desc, stage=stage)
        launch_us = float(cfg.get("kernel_launch_latency_us", 0.0))
        stage_model = self._implementation_stage_model(
            resolved_op_name, cfg, "vector")
        stage_overhead_ms, materialized_kernel_count = (
            self._implementation_stage_overhead_ms(stage_model, launch_us))
        # Pure streaming kernels can overlap the first GM response with
        # later MTE2/MTE3 transfers through DoubleBuffer.  The bandwidth term
        # remains payable; only the non-overlapped first-response term is
        # removed.  This is an implementation property declared by the
        # library profile, not an operator efficiency fitted from duration.
        hidden_hbm_latency_us = (
            float(io_detail["latency_us"])
            if resource_profile.get("hide_hbm_base_latency", False) else 0.0)
        time_ms = (
            io_detail["io_time"]
            - hidden_hbm_latency_us / 1e3
            + launch_us / 1e3
            + stage_overhead_ms)
        # Layout events have zero FLOPs, but their calibration profile may
        # still carry an efficiency multiplier for the same structural
        # layout operation.  Apply it as the inverse time factor only in the
        # explicit measured_calibration branch; forward-derived runs keep the
        # pure HBM/UB/MTE formula unchanged.
        layout_efficiency_multiplier = compute_detail.get(
            "calibration_efficiency_multiplier")
        if (layout_efficiency_multiplier is not None
                and layout_efficiency_multiplier > 0):
            time_ms /= float(layout_efficiency_multiplier)
        layout_detail = None
        if resource_profile:
            memory_spec = (self.hardware_spec or {}).get("memory", {})
            ub_bytes = max(1, int(memory_spec.get(
                "ub_per_aiv_bytes", resource_profile.get("tile_bytes", 1))))
            ub_buffers = max(1, int(resource_profile.get("ub_buffer_count", 1)))
            transaction = max(1, int(cfg.get(
                "memory_transaction_bytes",
                mte_cfg.get("transaction_bytes", 256))))
            tile_bytes = max(
                transaction,
                (ub_bytes // ub_buffers // transaction) * transaction)
            hardware_compute = (self.hardware_spec or {}).get("compute", {})
            aiv_num = max(1, int(
                cfg.get("aiv_core_num")
                or hardware_compute.get("aiv_core_num", 1)))
            block_dim = min(
                aiv_num, max(1, int(resource_profile.get("block_dim", aiv_num))))
            work_bytes = max(0, input_bytes, output_bytes)
            work_blocks = math.ceil(work_bytes / tile_bytes) if work_bytes else 0
            waves = math.ceil(work_blocks / block_dim) if work_blocks else 0
            memory_stages = max(0, int(resource_profile.get(
                "dependent_memory_stages", 0)))
            vector_stages = max(0, int(resource_profile.get(
                "vector_pipeline_stages", 0)))
            wave_startup_us = (
                memory_stages * float(io_detail["latency_us"])
                + vector_stages * launch_us)
            extra_wave_time_ms = max(0, waves - 1) * wave_startup_us / 1e3
            time_ms += extra_wave_time_ms
            layout_detail = {
                "ub_per_aiv_bytes": ub_bytes,
                "ub_buffer_count": ub_buffers,
                "tile_bytes": tile_bytes,
                "block_dim": block_dim,
                "work_blocks": work_blocks,
                "wave_count": waves,
                "dependent_memory_stages": memory_stages,
                "vector_pipeline_stages": vector_stages,
                "hide_hbm_base_latency": bool(
                    resource_profile.get("hide_hbm_base_latency", False)),
                "hidden_hbm_base_latency_us": hidden_hbm_latency_us,
                "wave_startup_latency_us": wave_startup_us,
                "extra_wave_time_ms": extra_wave_time_ms,
                "calibration_efficiency_multiplier": (
                    layout_efficiency_multiplier),
            }

        if self.forward_derivation_enabled:
            key = (f"{path_key or 'path_unspecified'}|{resolved_op_name}|"
                   f"{stage or 'unspecified'}|{shape_desc or 'shape_unspecified'}")
            record = self.forward_derivation_records["operators"].get(key)
            if record is not None:
                transfer_expr = (
                    "D_write" if traffic_policy in {
                        "write_only", "indexed_write", "scatter_write"}
                    else "D_read" if traffic_policy in {
                        "read_only", "indexed_read"}
                    else "D_read+D_write")
                record.update({
                    "kernel_path_kind": "materialized_layout",
                    "structure_source": "model_graph_or_library_implementation_path",
                    "performance_observations_used": False,
                    "input_bytes": input_bytes,
                    "output_bytes": output_bytes,
                    "layout_resource_profile": resource_profile or None,
                    "resolved_layout_op_name": resolved_op_name,
                    "traffic_policy": traffic_policy,
                    "layout_resource_detail": layout_detail,
                    "mte_profile": mte_cfg or None,
                    "implementation_stage_model": stage_model or None,
                    "materialized_kernel_count": materialized_kernel_count,
                    "implementation_stage_overhead_ms": stage_overhead_ms,
                    "derived_time_ms": time_ms,
                    "formula": (
                        f"T_layout=T_launch+({transfer_expr})/(HBM_bw*"
                        "transaction_efficiency)+HBM_base_latency+"
                        "(waves-1)*pipeline_startup-hidden_streaming_latency+"
                        "declared_implementation_stage_overhead"
                    ),
                })
        if reture_detail:
            return {
                "op_name": op_name,
                "resolved_op_name": resolved_op_name,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "accessed_mem": accessed_mem,
                "compute_detail": compute_detail,
                "io_detail": io_detail,
                "kernel_launch_latency_us": launch_us,
                "mte_profile": mte_cfg or None,
                "implementation_stage_model": stage_model or None,
                "materialized_kernel_count": materialized_kernel_count,
                "implementation_stage_overhead_ms": stage_overhead_ms,
                "layout_resource_profile": resource_profile or None,
                "traffic_policy": traffic_policy,
                "layout_resource_detail": layout_detail,
                "time_ms": time_ms,
            }
        return time_ms

    @staticmethod
    def _lookup_comm_num_value(values: Dict[str, Any], comm_num: int, default=None):
        if not values:
            return default
        for key in (str(comm_num), comm_num):
            if key in values:
                return values[key]
        return default

    def compute_net_op_time(self, op_name: str, size: int, comm_num: int,
                            net="", comm_stage="unkonw",
                            strategy:StrategyConfig=None,
                            group_kind: str = None,
                            comm_direction: str = None,
                            comm_role: str = None):
        """
        compute network operation time,
        return time in ms

        Inter-node corrections follow Tier A of
        docs/design_simu_network_fabric.md: when `strategy` (and, for
        TP/CP/ETP collectives, `group_kind`) is provided, cross-node
        traffic ratios come from the real group->node mapping
        (`simumax.core.utils.group_cross_node_ratio`). Calls that pass no
        `strategy`/`group_kind` keep the legacy heuristics unchanged.

        ``comm_direction`` is an optional semantic fwd/bwd/optimizer label
        used only by the separate measured-calibration branch. Omitting it
        preserves the historical ``comm_stage``-based lookup.

        ``comm_role`` is an optional structural collective role (for example
        ``model_moe_ag`` or ``layer_moe_rs``).  It is used only to select a
        role-scoped entry in a separate measured-calibration profile; the
        forward-derived topology formula is unchanged when it is absent.
        """
        # Using ring alg for now
        assert op_name in kNetOp, f"{op_name} not exist on {kNetOp}"
        if net == self.LEVELS_NET:
            # Hierarchical levels path (design_simu_hierarchical_network.md
            # sections 5-7); fully separate from the single-net path below.
            return self._compute_net_op_time_levels(
                op_name, size, comm_num, comm_stage, strategy, group_kind,
                comm_direction, comm_role)
        net_data = self.networks.get(net, None)
        assert net_data is not None, f"{net} not exist on {self.networks.keys()}, op_name={op_name}"
        requested_op_name = op_name
        # Model-semantic collective names share the same physical algorithms.
        # Preserve an explicitly configured specialized op; otherwise resolve
        # to the generic collective. The levels path uses the same mapping.
        if op_name in NET_OP_FALLBACK and op_name not in net_data.op:
            op_name = NET_OP_FALLBACK[op_name]
        op:NetOpConfig = net_data.op.get(op_name, None)  # 0: scale 1: offset 2: efficient_factor
        assert op is not None, f"{op_name} not exist on {net_data}"
        scale, offset, eff_factor = op.scale, op.offset, op.efficient_factor
        
        # Calculate the actual communication data based on the scale and offset of the communication operator
        if eff_factor is None:
            eff_factor = net_data.bandwidth.efficient_factor
        actual_size = size * scale
        chunk_size = actual_size / comm_num
        actual_size += chunk_size * offset

        # Specially adapted to the dense-dp-family communication bandwidth of
        # A100 PCIe. `dp_cp` is Megatron's dense optimizer/data-parallel group
        # with context parallel folded in, so it should reuse the same dense-DP
        # bandwidth family here.
        is_dense_dp_stage = comm_stage in {"dp", "dp_cp"}

        if 'pcie' in net and is_dense_dp_stage and op.dp_fixed_bw and op.dp_fixed_bw.get(str(comm_num), None):
            dp_fixed_bw = op.dp_fixed_bw.get(str(comm_num))
            self.real_comm_bw[op_name + "_dp"] = {"net":net, "bw":f"{dp_fixed_bw} GB/S", "comm_num":comm_num, "latency": None} 
            return actual_size / (dp_fixed_bw * 1024**3)  * 1000

        # Intra Bandwidth decision
        bw = net_data.bandwidth.gbps
        # 基于互联拓扑：node/跨板层有效带宽（物理参数推导优先，level_bandwidth_gbps
        # 校准 fallback），通信组跨 node 流量比 r（group_cross_node_ratio）用层带宽
        # 加权求有效带宽。r=0（组内单 node）走板内；r>0 按跨板流量比例加权。
        _lv_bw = ((self.topology or {}).get('level_bandwidth_gbps') or {})
        _lv = (self.topology or {}).get('levels') or []
        _topo_bw = None
        _topo_latency = None
        if strategy is not None and group_kind and len(_lv) >= 2:
            # Forward mode must not fall back to level_bandwidth_gbps because
            # that optional table may contain calibrated effective bandwidths.
            _fallback0 = (None if self.forward_derivation_enabled
                          else _lv_bw.get(_lv[0].get('name')))
            _fallback1 = (None if self.forward_derivation_enabled
                          else _lv_bw.get(_lv[1].get('name')))
            _b0 = self._level_effective_bandwidth(_lv[0], _fallback0)
            _b1 = self._level_effective_bandwidth(_lv[1], _fallback1)
            if _b0 and _b1:
                # a2a（cp/ep）用层流量分解（all2all_level_fraction 的跨板占比）；
                # collective（AG/RS/AR）用组跨 node 比。二者跨板流量比例不同。
                # node 内 a2a（comm_num <= num_per_node，如 CP a2a 的
                # a2a_group = min(cp, num_per_node)）不走跨板层 → β = node 层。
                _r = group_cross_node_ratio(group_kind, strategy, self.num_per_node)
                if op_name.startswith('all2all') or op_name.startswith('alltoall'):
                    if group_kind in ('cp', 'ep'):
                        if comm_num <= self.num_per_node:
                            _r = 0.0  # node 内 a2a（a2a_group=min(cp, num_per_node)）
                        elif len(_lv) >= 2:
                            _f = all2all_level_fraction(group_kind, strategy, _lv, 1)
                            if _f > 0:
                                _r = _f
                if _r > 0:
                    _topo_bw = 1 / ((1 - _r) / _b0 + _r / _b1)
                else:
                    _topo_bw = _b0  # 同 node：板内带宽
                _lat0 = self.networks[_lv[0]['net']].bandwidth.latency_us
                _lat1 = self.networks[_lv[1]['net']].bandwidth.latency_us
                _topo_latency = (1 - _r) * _lat0 + _r * _lat1
        if self.forward_derivation_enabled:
            if comm_num == 1:
                return 0
            return self._derive_network_time(
                requested_op_name, actual_size, comm_num, net, comm_stage, strategy,
                group_kind, _topo_bw, _topo_latency, comm_direction, comm_role)
        # Per-card-count bandwidth (align by comm group size, not by net/domain
        # name): when the net profile declares gbps_by_comm_num, use the value
        # for this comm_num; the op's efficient_factor below still applies
        # (AG vs RS relative cost). Absent = legacy single-gbps.
        by_cn = getattr(net_data.bandwidth, 'gbps_by_comm_num', None)
        # FSDP collective 用 per-card-count 实测有效带宽（edp 2→18.9、dense
        # 16→30.9 GB/s），而非拓扑 node 全互联 49：2 卡对传只用部分链路，
        # 聚合带宽低于板内全互联（实测 fsdp 有效带宽 ~18.6 GB/s）。a2a/sync
        # 保留拓扑路径（现有对齐）。
        if op_name in ("fsdp_all_gather", "fsdp_reduce_scatter") and by_cn:
            bw = by_cn.get(str(comm_num), _topo_bw or bw)
            # 写量模型（AG/RS 带宽差，替代固定 op 因子）：all_gather 每 rank
            # 收/写全量 W、reduce_scatter 收/写 W/N（N=组规模）→ RS 写量少、
            # 有效带宽更高，且随 N 变（edp N=2: 22.9、dense N=16: 45.9 GB/s）。
            # β_tx（传输带宽）从 per-card-count(=AG 有效带宽) 与写带宽 β_w 反推。
            if op_name == "fsdp_reduce_scatter" and self.write_bandwidth_gbps:
                beta_w = self.write_bandwidth_gbps * 1e9 / (1024 ** 3)  # GB/s → GiB/s
                if 0 < bw < beta_w:
                    beta_tx = 1.0 / (1.0 / bw - 1.0 / beta_w)
                    bw = 1.0 / (1.0 / beta_tx + 1.0 / (comm_num * beta_w))
        elif _topo_bw is not None:
            bw = _topo_bw
            # 保留 op 的 efficient_factor（fsdp_all_gather 0.394 / model_* 等）
            # 作为相对成本：层带宽是跨板基准，op eff 区分 fsdp 与 collective。
            # β_eff 是纯带宽（GB/s），时间 = actual_size/(β_eff × op_eff)。
        elif by_cn:
            bw = by_cn.get(str(comm_num), bw)
        # CP a2a 的 alltoall 有效带宽（GiB/s，1024^3，与层带宽口径一致）：
        # 每对 rank 小包（Q 25.2MB/peer），有效带宽低于 node 全互联层带宽
        # （16p: node 49 GiB/s vs 实测 36.8 GiB/s = 39.5 GB/s）。
        if self.cp_a2a_bandwidth_gbps and op_name in ("alltoallv", "all2all") \
                and 'cp' in (comm_stage or '').lower():
            bw = self.cp_a2a_bandwidth_gbps
        if self.FC8 and net == "high_intra_node": # If the internal bandwidth is FC8 mode, the bandwidth changes according to the number of communications.
            bw *= (comm_num-1)/7

        # Inter Bandwidth decision
        if net == "inter_node":
            # Topology-kind-aware bandwidth (design doc Part C, section 5.4):
            # CLOS: shared uplink → divide by convergence_ratio (default
            #   num_per_node, preserving legacy behavior).
            # FullMesh: dedicated per-pair links → no division.
            topo_kind, conv_ratio = self._net_topology_kind(net)
            clos_divisor = conv_ratio if topo_kind == "clos" else 1

            # 1. pp
            if op_name == "p2p":
                bw /= clos_divisor
                
            # 2. ep & a2a cp
            if op_name == "all2all":
                if "ep" in comm_stage.lower():
                    # Only consider the case where ep is an integer multiple of num_per_node
                    # K machines cross ep, the total communication size = (k-1)/k *actual_size, 1 piece of data is sent to the self.
                    # At the same time, cross-machine a2a will use one network card, so the bw is the bw of the single network card

                    # decision comm_size
                    k = max(1, math.ceil(comm_num / self.num_per_node))
                    if k <= 1:
                        # ep group fully intra-node: a2a still moves the full payload
                        # on the node fabric (previously zeroed to latency-only).
                        node_net = self._node_level_net()
                        if node_net and node_net in self.networks:
                            net_cfg = self.networks[node_net]
                            nb = net_cfg.bandwidth
                            # node fabric effective bw = gbps + overlay
                            # (topology_skeleton_templates: 56 + 224 = 280 GB/s)
                            bw = nb.gbps + (getattr(net_cfg, 'overlay_bandwidth_gbps', 0) or 0)
                            eff_factor = nb.efficient_factor
                    else:
                        actual_size = (k-1)/k * actual_size
                        # decision bw
                        bw /= clos_divisor # bw of the single network card
                elif "cp" in comm_stage.lower():
                    # Similar to ep all2all: when cp spans multiple nodes, only cross-node
                    # traffic contributes to inter-node transfer and each group is limited by one NIC.
                    if strategy is not None:
                        # Tier A (docs/design_simu_network_fabric.md, section 4):
                        # use the real cross-node ratio from the cp group's
                        # arithmetic-progression mesh math; the legacy ceil-based
                        # (k-1)/k is wrong for non-contiguous strides (e.g. cp
                        # with tp=8 spans 2 nodes -> real ratio 0.5, legacy k=1 -> 0).
                        ratio = group_cross_node_ratio("cp", strategy, self.num_per_node)
                        if ratio <= 0:
                            # Fully intra-node cp group: a2a still moves the full
                            # payload on the node fabric (previously zeroed to
                            # latency-only).
                            node_net = self._node_level_net()
                            if node_net and node_net in self.networks:
                                net_cfg = self.networks[node_net]
                                nb = net_cfg.bandwidth
                                # node fabric effective bw = gbps + overlay
                                # (topology_skeleton_templates: 56 + 224 = 280 GB/s)
                                bw = nb.gbps + (getattr(net_cfg, 'overlay_bandwidth_gbps', 0) or 0)
                                eff_factor = nb.efficient_factor
                            # else: keep inter_node bw with full payload (conservative)
                        else:
                            actual_size *= ratio
                            bw /= clos_divisor
                    else:
                        k = max(1, math.ceil(comm_num / self.num_per_node))
                        actual_size = (k - 1) / k * actual_size
                        bw /= clos_divisor
            
            # 3. tp+sp & ag cp & dp
            if op_name in ["all_reduce", "all_gather", "reduce_scatter"]:
                # Tier A (docs/design_simu_network_fabric.md, section 4):
                # TP/CP/ETP collectives assigned to inter_node previously got
                # no cross-node correction at all; scale the payload by the
                # real cross-node ratio of the group. Purely additive — the
                # dp/dp_cp/edp NIC-contention divisions below are unchanged.
                if group_kind in ("tp", "cp", "etp") and strategy is not None:
                    actual_size *= group_cross_node_ratio(group_kind, strategy, self.num_per_node)
                if strategy is not None:
                    # Topology-kind: FullMesh skips the NIC-contention
                    # division (dedicated per-pair links, no sharing);
                    # CLOS keeps it (shared uplink). Design doc Part C,
                    # section 5.4.
                    if topo_kind == "clos":
                        if is_dense_dp_stage:
                            # zero0: all_reduce
                            # zero1: reduce_scatter & all_gather
                            # num_per_node = 8
                            # TP1, each DP group uses all 8 IBs
                            # TP2, each DP group uses 4 IBs, ...
                            #
                            # Distinguish two semantics:
                            # - `dp_cp`: dense optimizer group with CP folded
                            #   into the group itself, so per-node group
                            #   multiplicity is still driven by TP only.
                            # - `dp`: pure dense DP group. If CP is present,
                            #   each `(tp, cp)` slice owns its own DP group,
                            #   so the inter-node contention factor grows
                            #   with `tp * cp`.
                            dense_group_multiplicity = strategy.tp_size
                            if comm_stage == "dp":
                                dense_group_multiplicity *= strategy.cp_size
                            bw /= min(self.num_per_node, dense_group_multiplicity)
                        elif comm_stage == "edp":
                            # Same as dp
                            bw /= min(self.num_per_node, strategy.ep_size*strategy.etp_size)
                    

        base_latency = op.latency_us if op.latency_us is not None else net_data.bandwidth.latency_us
        fixed_latency = self._lookup_comm_num_value(
            op.fixed_latency_us_by_comm_num,
            comm_num,
            op.fixed_latency_us,
        )
        if fixed_latency is None:
            fixed_latency = self._lookup_comm_num_value(
                net_data.bandwidth.fixed_latency_us_by_comm_num,
                comm_num,
                net_data.bandwidth.fixed_latency,
            )
        latency = base_latency
        if comm_num == 1:
            return 0
        if self.num_per_node == 8 and op_name in ["all_reduce", "all_gather", "reduce_scatter", "all2all"]:
            latency = base_latency * (comm_num + offset) * scale
        time = (
            actual_size / (bw * 1024**3 * eff_factor) * 1e3
            + (latency+fixed_latency) / 1e3
        )
        if SIMU_DEBUG:
            if net == "high_intra_node" and op_name=="reduce_scatter":
                print(f"op_name={op_name}, comm_num={comm_num}, net={net}, bw={bw*eff_factor} GB/S, latency={latency} us size={size}")
        self.record_net_bw(op_name, net, comm_num, comm_stage, net_data.bandwidth.gbps, bw*eff_factor, eff_factor, time*1e3, actual_size, latency)
        return time

    def _composition_policy_for(self, op_name: str) -> str:
        """Composition policy of an op on the levels path (design doc 6).

        Defaults: all2all -> "max" (bottleneck level), collectives
        (all_reduce/all_gather/reduce_scatter) -> "serial" (phase sum),
        p2p -> "serial". topology["composition_policy"] overrides per key.
        """
        policies = (self.topology or {}).get("composition_policy") or {}
        if op_name == "all2all":
            return policies.get("all2all", "max")
        if op_name == "p2p":
            return policies.get("p2p", "serial")
        return policies.get("collectives", "serial")

    def _net_topology_kind(self, net: str):
        """Resolve (topology_kind, convergence_ratio) for a net on the
        legacy single-net path (design doc Part C, section 5.4).

        1. If topology.levels exists and `net` matches a level's `net`,
           return that level's kind / convergence_ratio.
        2. Otherwise return networks[net].topology_kind / num_per_node
           (default: "clos" with convergence_ratio = num_per_node,
           preserving the legacy bw /= num_per_node behavior).
        """
        levels = (self.topology or {}).get("levels")
        if levels:
            for entry in levels:
                if entry["net"] == net:
                    kind = entry.get("kind", "clos")
                    conv = entry.get("convergence_ratio", 1.0)
                    return kind, conv
        # Fallback: use the net profile's own topology_kind
        net_cfg = self.networks.get(net)
        if net_cfg is not None:
            kind = getattr(net_cfg, 'topology_kind', 'clos')
        else:
            kind = "clos"
        # Legacy default convergence_ratio = num_per_node for clos
        return kind, self.num_per_node

    def _node_level_net(self):
        """Net name of the innermost (node) topology level, if declared.

        Used to cost intra-node collectives (groups that fit entirely on one
        node) at the node fabric instead of collapsing to zero traffic.
        """
        levels = (self.topology or {}).get("levels")
        if levels:
            return levels[0].get("net")
        return None

    def _level_effective_bandwidth(self, level_entry: dict, fallback_bw=None):
        """层有效带宽：物理参数推导（port_num × bandwidth_per_port ÷ conv）优先。

        fullmesh: port_num × per_port（每 device 聚合出口带宽，无收敛折损）
        clos:     port_num × per_port ÷ convergence_ratio（共享上行收敛）
        未声明物理参数（port_num / bandwidth_per_port_gbps）→ fallback_bw
        （level_bandwidth_gbps 校准值 / 层 net 带宽）。
        """
        port = level_entry.get('port_num')
        ppb = level_entry.get('bandwidth_per_port_gbps')
        if port and ppb:
            kind = level_entry.get('kind', 'clos')
            conv = level_entry.get('convergence_ratio', 1.0)
            return port * ppb / (conv if kind == 'clos' else 1.0)
        return fallback_bw

    def _level_net_params(self, net: str, op_name: str, comm_num: int):
        """Resolve (scale, offset, eff_factor, bw_gbps, latency_us, fixed_latency_us)
        for one level's net entry.

        Same resolution rules as the single-net path: op-level overrides
        first, then the net bandwidth defaults. The num_per_node == 8
        latency scaling of the legacy path is intentionally NOT applied
        on the levels path.
        """
        net_data = self.networks.get(net, None)
        assert net_data is not None, f"{net} not exist on {self.networks.keys()}, op_name={op_name}"
        if op_name in NET_OP_FALLBACK and op_name not in net_data.op:
            op_name = NET_OP_FALLBACK[op_name]
        op: NetOpConfig = net_data.op.get(op_name, None)
        assert op is not None, f"{op_name} not exist on {net_data}"
        scale, offset, eff_factor = op.scale, op.offset, op.efficient_factor
        if eff_factor is None:
            eff_factor = net_data.bandwidth.efficient_factor
        base_latency = op.latency_us if op.latency_us is not None else net_data.bandwidth.latency_us
        fixed_latency = self._lookup_comm_num_value(
            op.fixed_latency_us_by_comm_num,
            comm_num,
            op.fixed_latency_us,
        )
        if fixed_latency is None:
            fixed_latency = self._lookup_comm_num_value(
                net_data.bandwidth.fixed_latency_us_by_comm_num,
                comm_num,
                net_data.bandwidth.fixed_latency,
            )
        bw_gbps = net_data.bandwidth.gbps
        # Overlay bandwidth: add a parallel fabric's bandwidth to model
        # simultaneous use of mesh links + Clos fabric (e.g. UBLink
        # mesh 56 GB/s + SU Clos overlay 224 GB/s = 280 GB/s effective).
        # Applied to p2p AND collectives (all_reduce/all_gather/
        # reduce_scatter/all2all) — the Clos fabric can be used in
        # parallel with the mesh for all communication patterns.
        bw_gbps += getattr(net_data, 'overlay_bandwidth_gbps', 0) or 0
        return scale, offset, eff_factor, bw_gbps, base_latency, fixed_latency

    def _compute_net_op_time_levels(self, op_name: str, size: int, comm_num: int,
                                    comm_stage: str, strategy: "StrategyConfig",
                                    group_kind: str,
                                    comm_direction: str = None,
                                    comm_role: str = None):
        """Hierarchical per-level cost composition (design doc sections 5-6).

        The group's traffic is decomposed across topology["levels"] via
        `group_level_span` (composition [c_0, c_1, ...]; a phase exists at
        level i iff c_i > 1) and each level is charged with its own net
        profile. Per op type:

        - all2all: time_i = (size * scale_i * all2all_level_fraction(i))
          / (bw_i * eff_i) + latency_i over the levels whose boundary the
          group crosses (fraction > 0); total = max (or sum when the
          "all2all" policy is overridden to "serial").
        - collectives: serial ring phases; for each level with c_i > 1,
          phase_size = actual_size_base_i * (c_i - 1) / c_i with
          actual_size_base_i = size*scale_i + size*scale_i/comm_num*offset_i
          (the legacy actual_size formula with that level's op params);
          phase_time = phase_size / (bw_i * eff_i) + latency_i;
          total = sum (or max when overridden).
        - p2p: serial over the levels the endpoint path crosses. The two
          endpoints are adjacent pipeline stages, so their path is computed
          from a 2-member pair at the group stride (not the whole group's
          span): level 0 is used only when both endpoints share one node
          (units_0 == 1); level i >= 1 is crossed when the pair sits in
          different units of level i-1 (units_{i-1} > 1). Each crossed
          level carries the full mirrored actual_size once (comm_num stays
          the caller's send/recv-pair convention, 2); total = sum (or max
          when overridden).

        Intentional differences vs the legacy single-net path:
        - the num_per_node == 8 latency scaling is NOT applied (each
          level contributes its fitted base latency + fixed latency);
        - the FC8 intra-node bandwidth scaling and the pcie dp_fixed_bw
          shortcut are NOT applied per level;
        - the p2p inter-node NIC-share division (bw /= num_per_node) is
          NOT applied; a level's net bandwidth is the link bandwidth;
        - collectives multiply the legacy actual_size (whose fitted
          offset already encodes a ring factor when offset = -1) by the
          per-phase (c_i - 1)/c_i factor, so with offset = -1 the
          degenerate single-phase case is a (K-1)/K factor below the
          legacy number (12.5% at K = 8); declare level nets with
          offset = 0 for byte-exact degenerate equivalence.
        """
        assert strategy is not None, (
            f"net='levels' requires strategy, op_name={op_name}, comm_stage={comm_stage}")
        assert group_kind is not None, (
            f"net='levels' requires group_kind, op_name={op_name}, comm_stage={comm_stage}")
        levels = (self.topology or {}).get("levels")
        assert levels, (
            f"net='levels' requires topology['levels'] to be declared, "
            f"op_name={op_name}, comm_stage={comm_stage}")
        requested_op_name = op_name
        # Resolve model-semantic aliases before phase-size/algorithm logic as
        # well as net-table lookup. Keep a specialized op only when at least
        # one declared level explicitly supports it.
        if op_name in NET_OP_FALLBACK and not any(
                op_name in self.networks[level["net"]].op for level in levels):
            op_name = NET_OP_FALLBACK[op_name]
        route_strategy = strategy
        # A collective may intentionally use a subgroup smaller than the
        # strategy dimension (for example node-local CP A2A with cp_size=8 and
        # cp_a2a_group=4). Route the declared call group, not the full logical
        # parallel dimension.
        dimension = {
            "tp": "tp_size", "cp": "cp_size", "pp": "pp_size",
            "dp": "dp_size", "ep": "ep_size", "etp": "etp_size",
            "edp": "edp_size",
        }.get(group_kind)
        if dimension and getattr(strategy, dimension, comm_num) != comm_num:
            values = {
                name: getattr(strategy, name, 1)
                for name in ("tp_size", "cp_size", "pp_size", "dp_size",
                             "ep_size", "etp_size", "edp_size")
            }
            values["world_size"] = strategy.world_size
            values[dimension] = comm_num
            # Preserve the configured FSDP/OE shard override when making a
            # subgroup routing view.  Without this, a 256p strategy with
            # ``oe_shard_size=128`` is copied with only ``edp_size=128``;
            # the router then falls back to the logical EP stride (8) even
            # though rank-group construction correctly uses the override
            # stride (world_size/group_size = 2).  The resulting level span
            # has the wrong members-per-node and therefore the wrong
            # per-level beta/time.  Keep the override and bind it to the
            # subgroup size so routing and emitted communication metadata
            # describe the same declared domain.  These are strategy facts,
            # not measured timing parameters.
            if group_kind == "dp_cp":
                values["fsdp_shard_size"] = comm_num
            elif group_kind == "edp":
                values["oe_shard_size"] = comm_num
            route_strategy = types.SimpleNamespace(**values)
        composition, spans = group_level_span(group_kind, route_strategy, levels)
        if op_name == "p2p":
            # p2p involves two adjacent stages, not the whole group: a
            # 2-member group at the same stride would give c_i == 1 at
            # every level, so the path is derived from the pair's
            # units_touched instead of the composition.
            pair = strategy
            if group_kind == "pp" and strategy.pp_size > 2:
                pair = types.SimpleNamespace(
                    pp_size=2, tp_size=strategy.tp_size,
                    cp_size=strategy.cp_size, dp_size=strategy.dp_size)
            _, spans = group_level_span(group_kind, pair, levels)
        policy = self._composition_policy_for(op_name)
        # (span, phase_size, bw, eff_factor, phase_time_ms,
        #  physical_propagation_latency_us)
        phases = []
        phase_facts = []
        for i, span in enumerate(spans):
            scale, offset, eff_factor, bw, base_latency, fixed_latency = \
                self._level_net_params(span.net, op_name, comm_num)
            # Topology-kind-aware bandwidth (design doc Part C, section 5.5):
            # CLOS levels divide by convergence_ratio; FullMesh levels keep
            # the full link bandwidth (no sharing). Default kind="clos" with
            # convergence_ratio=1.0 preserves the current levels behavior.
            if span.kind == "clos" and span.convergence_ratio > 1.0:
                bw /= span.convergence_ratio
            if self.forward_derivation_enabled:
                # Use only port-count/link-rate hardware fields.  Do not use
                # level net efficiency or fixed-latency calibration values.
                physical_bw = self._level_effective_bandwidth(levels[i], None)
                if physical_bw is None:
                    physical_bw = self.networks[span.net].bandwidth.gbps
                # A FullMesh endpoint can use only as many physical ports as
                # there are remote members in this topology phase. This is a
                # route-occupancy limit from group placement, independent of
                # the collective algorithm. CLOS uplinks remain shared at the
                # configured convergence-limited aggregate rate.
                port_num = max(1, int(levels[i].get("port_num", 1)))
                if span.kind == "fullmesh":
                    remote_members = max(1, math.ceil(composition[i]) - 1)
                    active_ports = min(port_num, remote_members)
                    port_utilization = active_ports / port_num
                else:
                    active_ports = port_num
                    port_utilization = 1.0
                bw = physical_bw * port_utilization
                eff_factor = 1.0
                base_latency = self.networks[span.net].bandwidth.latency_us
                fixed_latency = 0.0
            if op_name == "all2all":
                # Per-level share of each member's traffic; levels whose
                # boundary nobody crosses (fraction == 0) are skipped
                # entirely, latency included.
                fraction = all2all_level_fraction(
                    group_kind, route_strategy, levels, i)
                if fraction <= 0:
                    continue
                # ``size`` is the logical per-rank tensor. One K-way A2A
                # retains 1/K locally; only (K-1)/K enters network links.
                phase_size = (size * scale * (comm_num - 1) / comm_num
                              * fraction)
            elif op_name == "p2p":
                # Level 0 carries the pair only when both endpoints share
                # one node; level i >= 1 carries it when the endpoints sit
                # in different units of level i-1.
                crossed = spans[0].units_touched == 1 if i == 0 \
                    else spans[i - 1].units_touched > 1
                if not crossed:
                    continue
                phase_size = size * scale + size * scale / comm_num * offset
            else:
                if composition[i] <= 1:
                    continue
                actual_size_base = size * scale + size * scale / comm_num * offset
                phase_size = actual_size_base * (composition[i] - 1) / composition[i]
            if self.forward_derivation_enabled:
                flit_bytes = max(1, int(
                    self._hccl_network_config().get(
                        "flit_bytes", 256)))
                padded = self._ceil_to(phase_size, flit_bytes)
                eff_factor = phase_size / padded if padded else 1.0
                local_members = max(2, composition[i])
                algorithm, stage_count = self._collective_algorithm(
                    op_name, local_members)
                hop_count = max(1, int(levels[i].get("hop_count", 1)))
                physical_latency = base_latency * hop_count
                attainable_bandwidth_efficiency = max(
                    0.0, min(1.0, port_utilization * eff_factor))
                layer_record_key = f"{span.name}|bytes={int(phase_size)}"
                self.forward_derivation_records["network_layers"][
                    layer_record_key] = {
                    "network_level": span.name,
                    "net": span.net,
                    "op_name": requested_op_name,
                    "algorithm_family": op_name,
                    "comm_num": local_members,
                    "group_kind": group_kind,
                    "topology_kind": span.kind,
                    "units_touched": span.units_touched,
                    "message_bytes": phase_size,
                    "physical_bandwidth_gib_per_s": physical_bw,
                    "bandwidth_unit": "GB/s",
                        "port_count": port_num,
                        "active_ports": active_ports,
                        "port_utilization": port_utilization,
                        "routed_bandwidth_gib_per_s": bw,
                        "packet_efficiency": eff_factor,
                        "bandwidth_utilization": eff_factor,
                        "attainable_bandwidth_efficiency": (
                            attainable_bandwidth_efficiency),
                        "effective_beta_gib_per_s": bw * eff_factor,
                        "effective_beta_gb_per_s": bw * eff_factor,
                        "reachable_beta_gb_per_s": bw * eff_factor,
                        "base_latency_us": base_latency,
                        "hop_count": hop_count,
                        "physical_propagation_latency_us": physical_latency,
                        "latency_formula": "hop_count*per_hop_latency_us",
                        "algorithm_independent": True,
                        "formula": (
                            "beta=B_physical*port_utilization*"
                            "payload/ceil(payload/flit); "
                            "U_beta=port_utilization*packet_efficiency"),
                    }
                record_key = (f"{requested_op_name}|levels:{comm_stage.lower()}:{span.name}"
                              f"|n={local_members}|bytes={int(phase_size)}")
                self.forward_derivation_records["communications"][record_key] = {
                    "op_name": requested_op_name,
                    "algorithm_family": op_name,
                    "stage": f"levels:{comm_stage.lower()}:{span.name}",
                    "group_kind": group_kind,
                    "comm_num": local_members,
                    "message_bytes": phase_size,
                    "topology_bandwidth_gbps": physical_bw,
                    "bandwidth_unit": "GB/s",
                    "routed_bandwidth_gbps": bw,
                    "port_utilization": port_utilization,
                    "flit_bytes": flit_bytes,
                    "packet_efficiency": eff_factor,
                    "attainable_bandwidth_efficiency": (
                        attainable_bandwidth_efficiency),
                    "derived_beta_gib_per_s": bw * eff_factor,
                    "derived_beta_gb_per_s": bw * eff_factor,
                    "reachable_beta_gb_per_s": bw * eff_factor,
                    "algorithm": algorithm,
                    "algorithm_stages": stage_count,
                    "network_layer_latency_us": base_latency,
                    "physical_hop_count": hop_count,
                    "physical_propagation_latency_us": physical_latency,
                    "latency_formula": "hop_count*per_hop_latency_us",
                    "topology_kind": span.kind,
                    "units_touched": span.units_touched,
                    "call_runtime_overhead_us": 0.0,
                    "collective_latency_us": physical_latency,
                    "formula": (
                        "T=D/(B_topology*port_utilization*payload/"
                        "padded_payload)+hop_count*hop_latency; call runtime "
                        "is composed once after all physical levels"),
                }
                latency_time = physical_latency / 1e3
            else:
                latency_time = (base_latency + fixed_latency) / 1e3
            base_time = phase_size / (bw * 1e9 * eff_factor) * 1e3
            # Layout transform overhead (e.g. dim01_transpose around all2allv)
            layout_oh = 0.0
            if op_name == "all2all":
                layout_oh = (getattr(strategy, 'layout_transform_overhead_us', 0) or 0) / 1e3
            phase_time = base_time + latency_time + layout_oh
            if self.forward_derivation_enabled:
                phase_facts.append({
                    "level": span.name,
                    "net": span.net,
                    "topology_kind": span.kind,
                    "units_touched": span.units_touched,
                    "payload_bytes": phase_size,
                    "physical_bandwidth_gb_per_s": physical_bw,
                    "routed_bandwidth_gb_per_s": bw,
                    "beta_gb_per_s": bw * eff_factor,
                    "bandwidth_unit": "GB/s",
                    "port_count": port_num,
                    "active_ports": active_ports,
                    "port_utilization": port_utilization,
                    "packet_efficiency": eff_factor,
                    "attainable_bandwidth_efficiency": (
                        max(0.0, min(1.0, port_utilization * eff_factor))),
                    "hop_count": hop_count,
                    "physical_latency_us": physical_latency,
                    "latency_formula": "hop_count*per_hop_latency_us",
                    "derived_phase_time_ms": phase_time,
                    "source": "forward_formula",
                })
            phases.append((
                span, phase_size, bw, eff_factor, phase_time,
                physical_latency if self.forward_derivation_enabled
                else base_latency))
        if not phases:
            # Group of one (or no crossed level): no communication.
            return 0.0
        if policy == "max":
            base_phase_time = max(phase[4] for phase in phases)
        else:
            base_phase_time = sum(phase[4] for phase in phases)
        runtime = None
        calibration_entry = None
        effective_phase_times = [phase[4] for phase in phases]
        ideal_level_transfer_ms = [
            phase_size / (bw * 1e9 * eff_factor) * 1e3
            for _span, phase_size, bw, eff_factor, _phase_time, _latency
            in phases
            if bw > 0 and eff_factor > 0
        ]
        ideal_transfer_ms = (
            max(ideal_level_transfer_ms)
            if policy == "max" and ideal_level_transfer_ms
            else sum(ideal_level_transfer_ms)
            if ideal_level_transfer_ms else None
        )
        communication_attainable_efficiency = None
        if self.forward_derivation_enabled:
            runtime = self._collective_runtime_overhead(
                op_name, comm_num, size, active_level_count=len(phases))
            calibration_direction = comm_direction or comm_stage
            calibration_entry = self._calibration_communication_multiplier(
                requested_op_name, size, comm_num,
                calibration_direction, comm_role, comm_stage)
            if (calibration_entry is not None
                    and calibration_entry.get("transfer_efficiency") is not None):
                transfer_efficiency = calibration_entry["transfer_efficiency"]
                effective_phase_times = []
                for phase in phases:
                    _span, phase_size, bw, eff_factor, phase_time, _latency = phase
                    transfer_time = (
                        phase_size / (bw * 1e9 * eff_factor) * 1e3
                        if bw > 0 and eff_factor > 0 else 0.0)
                    effective_phase_times.append(
                        phase_time - transfer_time
                        + transfer_time / transfer_efficiency)
                if phase_facts:
                    for fact, effective_time in zip(
                            phase_facts, effective_phase_times):
                        fact["calibrated_phase_time_ms"] = effective_time
                        fact["calibration_applied_to"] = (
                            "pure_transfer_component")
                        fact["calibration_transfer_efficiency"] = (
                            transfer_efficiency)
            total_phase_time = (
                max(effective_phase_times)
                if policy == "max" else sum(effective_phase_times))
            total_time = total_phase_time + runtime["call_runtime_overhead_us"] / 1e3
            # Keep the legacy complete-call multiplier readable for old
            # profiles, but never apply it on top of a component calibration.
            if (calibration_entry is not None
                    and calibration_entry.get("transfer_efficiency") is None):
                multiplier = calibration_entry.get("multiplier")
                if multiplier is not None:
                    total_time = max(0.0, total_time * multiplier)
            communication_attainable_efficiency = (
                ideal_transfer_ms / total_time
                if ideal_transfer_ms is not None and total_time > 0 else None)
        else:
            total_time = base_phase_time
        if self.forward_derivation_enabled:
            reachable_betas = [
                bw * eff_factor for _span, _phase_size, bw, eff_factor,
                _phase_time, _latency in phases if bw > 0 and eff_factor > 0
            ]
            record_key = (
                f"{requested_op_name}|levels:{comm_stage.lower()}:call"
                f"|direction={self._calibration_direction(calibration_direction)}"
                f"|n={comm_num}|bytes={int(size)}")
            self.forward_derivation_records["communications"][record_key] = {
                "op_name": requested_op_name,
                "algorithm_family": op_name,
                "stage": f"levels:{comm_stage.lower()}:call",
                "direction": self._calibration_direction(calibration_direction),
                "group_kind": group_kind,
                "comm_num": comm_num,
                "message_bytes": size,
                "physical_level_count": len(phases),
                "physical_levels": phase_facts,
                "composition_policy": policy,
                "ideal_link_transfer_time_ms": ideal_transfer_ms,
                "communication_attainable_efficiency": (
                    communication_attainable_efficiency),
                "min_reachable_beta_gb_per_s": (
                    min(reachable_betas) if reachable_betas else None),
                "max_reachable_beta_gb_per_s": (
                    max(reachable_betas) if reachable_betas else None),
                "physical_propagation_latency_us": sum(
                    phase[5] for phase in phases),
                "call_runtime_overhead_us": runtime["call_runtime_overhead_us"],
                "call_runtime": runtime,
                "collective_latency_us": (
                    sum(phase[5] for phase in phases)
                    + runtime["call_runtime_overhead_us"]),
                "derived_time_ms": total_time,
                "calibration_transfer_efficiency": (
                    calibration_entry.get("transfer_efficiency")
                    if calibration_entry is not None else None),
                "calibration_applied_to": (
                    "pure_transfer_component"
                    if calibration_entry is not None
                    and calibration_entry.get("transfer_efficiency") is not None
                    else "aggregate_call_lifetime"
                    if calibration_entry is not None else None),
                "calibration_time_multiplier": (
                    calibration_entry.get("multiplier")
                    if calibration_entry is not None
                    and calibration_entry.get("transfer_efficiency") is None
                    else None),
                "calibration_samples": (
                    calibration_entry.get("samples")
                    if calibration_entry is not None else None),
                "calibration_comm_role": (
                    calibration_entry.get("comm_role")
                    if calibration_entry is not None else None),
                "calibration_parameter_name": (
                    calibration_entry.get("parameter_name")
                    if calibration_entry is not None else None),
                "calibration_source_type": (
                    calibration_entry.get("source_type")
                    if calibration_entry is not None else None),
                "performance_observations_used": calibration_entry is not None,
                "performance_observations_used_as_parameters": (
                    calibration_entry is not None),
                "formula": (
                    "T_call=compose_levels(D_i/beta_i+hop_i*L_i)"
                    "+T_runtime(call,group,payload,algorithm)"
                    "+T_transfer_i*(1/transfer_efficiency-1)"
                    if calibration_entry is not None
                    and calibration_entry.get("transfer_efficiency") is not None
                    else
                    "T_call=compose_levels(D_i/beta_i+hop_i*L_i)"
                    "+T_runtime(call,group,payload,algorithm)"
                    "+aggregate_runtime_multiplier"
                    if calibration_entry is not None else
                    "T_call=compose_levels(D_i/beta_i+hop_i*L_i)"
                    "+T_runtime(call,group,payload,algorithm)"),
            }
        # net_info.json decomposition: one record per level under
        # "levels:<stage>:<level>" plus the composed total under
        # "levels:<stage>". Records keep the legacy field set.
        stage_key = comm_stage.lower()
        for index, (span, phase_size, bw, eff_factor, phase_time, base_latency) in enumerate(phases):
            self.record_net_bw(
                requested_op_name, span.net, comm_num,
                f"levels:{stage_key}:{span.name}",
                bw, bw * eff_factor, eff_factor,
                effective_phase_times[index] * 1e3,
                phase_size, base_latency)
        self.record_net_bw(
            requested_op_name, self.LEVELS_NET, comm_num,
            f"levels:{stage_key}",
            None, None, None, total_time * 1e3, size,
            (sum(phase[5] for phase in phases)
             + (runtime["call_runtime_overhead_us"] if runtime else 0.0)))
        return total_time

    def compute_end2end_time(self, compute_time, mem_time):
        """
        According to the accelerator mode, return the end2end time.
        Users can plug in other methods here to simulate
        """
        assert self.accelerator.mode in ["only_compute", "roofline"]
        if self.accelerator.mode == "only_compute":
            # when compute time equal zero, backoff to mem_time
            total_time = compute_time
            if total_time == 0:
                total_time = mem_time
        elif self.accelerator.mode == "roofline":
            total_time = max(compute_time, mem_time)
        else:
            raise NotImplementedError(f"{self.accelerator.mode} is not supported")

        return total_time

    # Resource lanes that never collide with user-declared engine names
    # ("off" is the idle lane of SimuThread's lane clock, see design doc 4.2).
    RESERVED_RESOURCE_LANES = (
        "comp", "comm", "pp_fwd", "pp_bwd", "off", "offload")

    # Fabric model choices (network-fabric design doc section 6); None = off.
    # "nic+levels" (hierarchical-network design doc section 8) activates
    # per-level link servers on top of the per-GPU NIC servers and requires
    # topology["levels"].
    FABRIC_MODELS = ("nic", "nic+tor", "nic+levels")
    # Reserved keys of the `topology` dict.
    RESERVED_TOPOLOGY_KEYS = ("tor_capacity_gbps", "tor_node_share")
    # Reserved pseudo-net name selecting the hierarchical levels cost path
    # (hierarchical-network design doc sections 6-7). Never a real key of
    # `networks`; resolved to topology["levels"] at call time.
    LEVELS_NET = "levels"
    # Hierarchical-topology keys of the `topology` dict (design doc section 3).
    # `level_bandwidth_gbps` = per-level effective bandwidth (node 板内 hccs /
    # su_clos 跨板) for the topology-weighted collective cost path in
    # compute_net_op_time (based on the group's physical node spread).
    LEVELS_TOPOLOGY_KEYS = ("levels", "composition_policy", "level_bandwidth_gbps")
    # composition_policy keys and values (design doc sections 3/6).
    COMPOSITION_POLICY_KEYS = ("all2all", "collectives", "p2p")
    COMPOSITION_POLICIES = ("max", "serial")
    # Supported intra-node link types. "ublink" is Huawei's UBLink
    # high-speed interconnect (equivalent in role to NVLink).
    # "nvlink" and "ublink" share the same binary analysis path
    # (analysis_high_link_net); "pcie" uses analysis_pcie_net.
    INTRA_LINK_TYPES = ("nvlink", "pcie", "ublink")

    def simu_resource_lanes(self) -> list[str]:
        """Pinned resource-lane contract for the simulator (design doc 4.2).

        Returns the built-in lanes ["comp", "comm", "pp_fwd", "pp_bwd",
        "offload"] plus
        the sorted names of `engines` entries not already in that list.
        """
        lanes = ["comp", "comm", "pp_fwd", "pp_bwd", "offload"]
        if self.engines:
            lanes.extend(sorted(name for name in self.engines if name not in lanes))
        return lanes

    def compute_fused_op_cost(self, costs: Dict[str, float], policy_spec) -> float:
        """Dispatch entry for fused-op cost (design doc 4.3).

        ``costs`` maps each occupied resource lane to its busy cost (ms);
        ``policy_spec`` is a fusion policy name or a dict like
        ``{"policy": "chunked_pipeline", "chunks": 4}`` (see
        ``simumax.core.fusion``). Measured fused-kernel efficiency tables
        hanging off system.json are reserved future work; until they exist
        the fusion policy's analytic span is the cost.
        """
        return build_fusion_policy(policy_spec).span(costs)

    def sanity_check(self):
        self._sanity_check_intra_link_type()
        self._sanity_check_engines()
        self._sanity_check_fabric()
        self._sanity_check_operator_efficiency()

    def _sanity_check_intra_link_type(self):
        assert self.intra_link_type in self.INTRA_LINK_TYPES, (
            f"intra_link_type must be one of {list(self.INTRA_LINK_TYPES)}, "
            f"but got {self.intra_link_type!r}"
        )

    def _sanity_check_operator_efficiency(self):
        _validate_efficiency_override_table(self.operator_efficiency, "operator_efficiency")

    def _sanity_check_engines(self):
        if self.engines is None:
            return
        assert isinstance(self.engines, dict), (
            f"engines must be a dict of name -> dict, but got {type(self.engines)}"
        )
        reserved_lanes = set(self.RESERVED_RESOURCE_LANES)
        for name, engine in self.engines.items():
            assert isinstance(name, str) and name.isidentifier(), (
                f"engine name {name!r} must be a non-empty identifier"
            )
            assert name not in reserved_lanes, (
                f"engine name {name!r} collides with reserved resource lane, "
                f"reserved lanes are {sorted(reserved_lanes)}"
            )
            assert isinstance(engine, dict), (
                f"engines[{name!r}] must be a dict, but got {type(engine)}"
            )
            peak_tflops = engine.get("peak_tflops")
            if peak_tflops is not None:
                assert isinstance(peak_tflops, (int, float)) and not isinstance(peak_tflops, bool), (
                    f"engines[{name!r}]['peak_tflops'] must be numeric, but got {peak_tflops!r}"
                )

    def _sanity_check_fabric(self):
        assert self.fabric_model in (None, *self.FABRIC_MODELS), (
            f"fabric_model must be one of None, 'nic', 'nic+tor', "
            f"'nic+levels', but got {self.fabric_model!r}"
        )
        if self.fabric_model == "nic+levels":
            # The fabric builds one link server per (level, unit) from
            # topology["levels"] (hierarchical-network design doc section
            # 8); each level's net reference is validated in
            # _validate_topology_levels below.
            assert self.topology is not None and "levels" in self.topology, (
                "fabric_model 'nic+levels' requires topology['levels'] to be "
                "declared (hierarchical-network design doc section 8)"
            )
        if self.topology is None:
            return
        assert isinstance(self.topology, dict), (
            f"topology must be a dict, but got {type(self.topology)}"
        )
        if self.fabric_model is None and any(
            key in self.topology for key in self.RESERVED_TOPOLOGY_KEYS
        ):
            # The tor_* knobs only take effect inside the fabric model;
            # topology["levels"] is meaningful on its own (analytical
            # levels cost path), so it does not trigger this warning.
            warnings.warn(
                "topology is set but fabric_model is None; topology is only "
                "meaningful with fabric_model 'nic', 'nic+tor' or 'nic+levels'"
            )
        if "composition_policy" in self.topology and "levels" not in self.topology:
            warnings.warn(
                "topology['composition_policy'] is set but topology['levels'] "
                "is missing; the policy has no effect"
            )
        allowed_keys = set(self.RESERVED_TOPOLOGY_KEYS) | set(self.LEVELS_TOPOLOGY_KEYS)
        for key, value in self.topology.items():
            assert key in allowed_keys, (
                f"unknown topology key {key!r}, "
                f"reserved keys are {sorted(allowed_keys)}"
            )
            if key == "tor_capacity_gbps":
                assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                    f"topology['tor_capacity_gbps'] must be numeric, but got {value!r}"
                )
            elif key == "tor_node_share" and value != "auto":
                assert (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value >= 1
                ), (
                    "topology['tor_node_share'] must be 'auto' or a number >= 1, "
                    f"but got {value!r}"
                )
            elif key == "levels":
                self._validate_topology_levels(value)
            elif key == "composition_policy":
                self._validate_composition_policy(value)

    def _validate_topology_levels(self, levels):
        """Validate topology["levels"] (hierarchical-network design doc
        section 3, system-net-ext design doc Part C section 5.1).

        Levels are ordered innermost->outermost; each entry has required
        keys {"name", "size", "net"} and optional keys {"kind",
        "convergence_ratio"}. `size` = units of the previous level
        contained in this level (the first level's unit is one GPU, so its
        size must equal num_per_node). `net` must reference `networks`.
        `kind` is "fullmesh" or "clos" (default "clos");
        `convergence_ratio` is a positive float (default 1.0).
        """
        assert isinstance(levels, list) and len(levels) > 0, (
            f"topology['levels'] must be a non-empty list, but got {levels!r}"
        )
        names = set()
        for idx, entry in enumerate(levels):
            assert isinstance(entry, dict), (
                f"topology['levels'][{idx}] must be a dict, but got {type(entry)}"
            )
            required_keys = {"name", "size", "net"}
            # 完整物理描述（可选）：kind/clos、convergence_ratio（clos 收敛）、
            # port_num（每 device 端口数）、bandwidth_per_port_gbps（每端口有效带宽）、
            # latency_us（静态延迟）。层有效带宽 = port_num × per_port ÷ conv（clos）。
            optional_keys = {"kind", "convergence_ratio", "port_num",
                             "bandwidth_per_port_gbps", "latency_us",
                             "hop_count"}
            entry_keys = set(entry.keys())
            assert entry_keys.issuperset(required_keys), (
                f"topology['levels'][{idx}] must have keys "
                f"{sorted(required_keys)}, but got {sorted(entry_keys)}"
            )
            unknown = entry_keys - required_keys - optional_keys
            assert not unknown, (
                f"topology['levels'][{idx}] has unknown keys "
                f"{sorted(unknown)}, allowed optional keys are "
                f"{sorted(optional_keys)}"
            )
            name, size, net = entry["name"], entry["size"], entry["net"]
            assert isinstance(name, str) and name, (
                f"topology['levels'][{idx}]['name'] must be a non-empty str, "
                f"but got {name!r}"
            )
            assert isinstance(size, int) and not isinstance(size, bool) and size >= 1, (
                f"topology['levels'][{idx}]['size'] must be an int >= 1, but got {size!r}"
            )
            assert isinstance(net, str) and net in self.networks, (
                f"topology['levels'][{idx}]['net'] must be one of "
                f"{sorted(self.networks.keys())}, but got {net!r}"
            )
            kind = entry.get("kind", "clos")
            assert kind in ("fullmesh", "clos"), (
                f"topology['levels'][{idx}]['kind'] must be 'fullmesh' or "
                f"'clos', but got {kind!r}"
            )
            conv = entry.get("convergence_ratio", 1.0)
            assert (
                isinstance(conv, (int, float))
                and not isinstance(conv, bool)
                and conv > 0
            ), (
                f"topology['levels'][{idx}]['convergence_ratio'] must be a "
                f"positive number, but got {conv!r}"
            )
            hop_count = entry.get("hop_count", 1)
            assert (
                isinstance(hop_count, int)
                and not isinstance(hop_count, bool)
                and hop_count >= 1
            ), (
                f"topology['levels'][{idx}]['hop_count'] must be an int >= 1, "
                f"but got {hop_count!r}"
            )
            assert name not in names, (
                f"topology['levels'][{idx}]['name'] {name!r} is duplicated"
            )
            names.add(name)
        first_size = levels[0]["size"]
        assert first_size == self.num_per_node, (
            f"topology['levels'][0]['size'] must equal num_per_node "
            f"({self.num_per_node}), but got {first_size}"
        )

    def _validate_composition_policy(self, policy):
        """Validate topology["composition_policy"] (design doc sections 3/6)."""
        assert isinstance(policy, dict), (
            f"topology['composition_policy'] must be a dict, but got {type(policy)}"
        )
        for key, value in policy.items():
            assert key in self.COMPOSITION_POLICY_KEYS, (
                f"unknown topology['composition_policy'] key {key!r}, "
                f"allowed keys are {list(self.COMPOSITION_POLICY_KEYS)}"
            )
            assert value in self.COMPOSITION_POLICIES, (
                f"topology['composition_policy'][{key!r}] must be one of "
                f"{list(self.COMPOSITION_POLICIES)}, but got {value!r}"
            )


@dataclass
class ModelConfig(Config):
    """Transformer model(decode-only) configuration"""
    hidden_size: int
    head_num: int
    kv_head_num: int
    model_type:str = None
    model_name:str = None
    head_size: int = None
    intermediate_size: int = None
    layer_num: int = None
    vocab_size: int = None
    orig_vocab_size: int = None
    use_swiglu: bool = None
    expert_num: int = 1
    topk: int = None
    attention_type: str = 'mha'
    # ───  SWA (Sliding Window Attention) config  ───
    swa_head_num: int = 0               # SWA query head count (0 = no SWA)
    swa_kv_head_num: int = None         # SWA KV head count (None = same as swa_head_num)
    swa_head_dim: int = None            # SWA head dim (None = use head_size)
    swa_window_size: int = 1028         # sliding window size (from op_define)
    # FA (global) window size for windowed FlashAttention. 16p profiling report
    # docs/mxx_profiling_alignment.md §1.2 gives win=(2048,2048); overridable per
    # model so a different FA window doesn't require code changes.
    fa_window_size: int = 2048
    # ───  Trunk CP divisor (decouples trunk from attention CP sharding)  ───
    trunk_cp_divisor: int = 1           # trunk seq = seq_len / (cp_size // divisor)
                                        # e.g. divisor=2 -> trunk uses cp/2, attn uses cp
    # Per-layer SWA head count override. When set (list of length == layer_num),
    # each layer uses layers_swa_head_num[i] instead of swa_head_num, so that
    # the SWA/GQA ratio can vary across layers. None = backward-compatible,
    # all layers use the global swa_head_num.
    layers_swa_head_num: List[int] = None
    # Per-layer SWA KV head count override (optional, requires layers_swa_head_num).
    layers_swa_kv_head_num: List[int] = None
    # Per-layer SWA head dim override (optional, requires layers_swa_head_num).
    layers_swa_head_dim: List[int] = None
    # Per-layer GQA KV head count override. When set (list of length == layer_num),
    # each layer uses layers_kv_head_num[i] instead of kv_head_num, so that the GQA
    # compression ratio (head_num // kv_head_num) can vary across layers independently
    # of the SWA split. None = backward-compatible, all layers use the global kv_head_num.
    layers_kv_head_num: List[int] = None
    # MQA-per-rank structure: the dense (FA) branch holds one KV head per rank
    # (kv_head_num does not divide cp_size) — skip the Ulysses-style KV-head-
    # split check in CP-a2a validation. Declared as a structure flag, not a
    # model_type string special-case.
    mqa_per_rank: bool = False
    # lm_head effective-seq ratio (0,1]: fraction of the global sequence the
    # lm_head/CE actually process. 16p profiling: embedding/transformer run the
    # full 131072 but lm_head only processes half (65536 = 32 chunks), so
    # ratio=0.5. Declared as model structure (like head_num/seq_len), not a
    # per-model patch. 1.0 = full sequence (default, other models unaffected).
    lm_head_seq_ratio: float = 1.0
    # SWA per-head Q/KV RMSNorm pass count per layer (profiling: kernel count
    # per layer = q_ops for the Q norm + kv_ops for the KV norm, 16p = 9 + 3).
    # These are independent fused rmsnorm kernels (393210x128 Q / 131064x128
    # KV on the global SWA sequence) not covered by NormRoPE. 0/0 = no
    # standalone SWA norm (default, other models unaffected).
    swa_norm_q_ops: int = 0
    swa_norm_kv_ops: int = 0
    # ───  Second attention group (16p fused-QKV + latent BMM)  ───
    # 16p profiling (docs/16p算子shape对齐报告.md) shows the fused QKV proj is
    # heterogeneous per layer (4608/5120) instead of a uniform 5120, the deep
    # layers carry a second QKV projection (count=2), L18 has an extra 1536,
    # L24 feeds a standalone 48-head 6144 block, and every layer runs a latent
    # BatchMatMul (8,m,1536)x(8,1536,n)->(8,m,n). These fields carry those
    # profiling-derived per-layer values. None/[] = backward-compatible, no
    # second group (512p / 16k models unaffected).
    enable_second_attn_group: bool = False
    # 25-entry list: per-layer FA-branch KV head count (16p trace: 2 on the
    # 4608-width layers / 4 on the 5120-width layers; flash_attn_num_kvheads).
    # Drives the fused-QKV width via the two-QKV structure formula, so the
    # width is derived from structure rather than declared (CostShape-free).
    layers_fa_kv_head_num: List[int] = None
    # 25-entry list: real first-QKV-proj width per layer (4608/5120).
    layers_qkv_width: List[int] = None
    # 25-entry list: number of first-QKV projections per layer (1 / deep=2).
    layers_qkv_count: List[int] = None
    # 25-entry list: extra QKV-proj widths per layer beyond count (L18 -> [1536]).
    layers_qkv_extra_widths: List[List[int]] = None
    # Standalone wide block after the last layer (L24 -> [6144, 6144]), NOT
    # recomputed (profiling: fwd count == 2, no rc pass).
    second_qkv_block_widths: List[int] = None
    # 25-entry list of [m, n] for the per-layer latent BatchMatMul. n follows
    # cycle[layer%8] = {4099,4109,4118,4127,4136,4144,4152,4161}; m = 1024 on
    # the F5120 layers, else 512.
    layers_latent_bmm: List[List[int]] = None
    # Portable form: m = 2*head_size*FA_KV_heads and
    # n = ceil(seq_len/vwn_n) + offsets[layer % len(offsets)]. The legacy
    # explicit [m,n] list remains supported for older model declarations.
    latent_bmm_n_offsets: List[int] = None
    latent_bmm_batch: int = 0    # 0 = 未设 → 用 strategy.cp_size（LAT 对 CP 切分的 seq 块注意力，batch=cp）
    latent_bmm_hidden: int = None          # latent BMM inner dim (None = hidden_size)
    # ───  BMMV2 族（latent 内部自注意力，aclnnMatmul_BatchMatMulNd_BatchMatMulV2）  ───
    # 1125 核 / 45.92ms（16p step 9），纯 fwd（无 rc/bwd，实测只发现 fwd 形态）。
    # 每层每相位两个 480 维实例（batch 10/20）+ 一个 128 维实例（batch 4/6，绑定 F 族），
    # 每实例 3 个 BMM（QKᵀ / PV / 内部）。全部 profiling 导出，非硬编码。
    latent_attn_phase_num: int = 0             # 每层相位数（实跑 5）
    latent_attn_dim480: int = 0                # 480 族维度（480）
    latent_attn_dim128: int = 0                # 128 族维度（128）
    latent_attn_batches_480: List[int] = None  # [10, 20] 两个 480 实例的 batch
    latent_attn_batches_128: List[int] = None  # 25 项逐层 128 实例 batch（4/6）
    # ───  BT Model config  ───
    enable_vwn: bool = False            # True = VWN 层, False = 标准层
    use_attn_gate: bool = False         # True = AttnGate, False = ContextNorm
    # ───  VWN (Variable Window Network) config  ───
    vwn_n: int = 1                      # residual streams count
    vwn_m: int = 1                      # block output streams count
    vwn_layer_indices: list = None      # layer indices using VWN (None = none)
    # ───  Quantized training config  ───
    quant_dtype: str = "bf16"                    # activation quant dtype: "bf16" | "int8" | "fp8"
    quant_mode: str = None                       # quant mode: None(no quant) | "dynamic" | "static"
    attn_gate_quant: bool = False                # use ATTN_GATE_QUANT (vs CONTEXT_RMSNORM_QUANT)
    context_rmsnorm_quant: bool = False          # use CONTEXT_RMSNORM_QUANT
    moe_dispatch_quant: bool = False             # quantize MoE dispatch activations
    moe_ffn_hidden_size: int = None
    moe_shared_expert_intermediate_size: int = None
    v_head_dim: int = None
    qk_head_dim: int = None
    qk_pos_emb_head_dim: int = None
    q_lora_rank: int = None
    kv_lora_rank: int = None
    dense_layers: int = 0 # number of dense layers in moe model
    moe_pad_expert_input_to_capacity:bool = True
    capacity:int = 1
    group_linear_mode:str = "parallel"
    # Declarative block recipe (cost-tunability design doc section 6): an
    # optional {"blocks": [{"template": <name>, "count": <int>}, ...]}
    # composition, expanded into layer_num / dense_layers by apply_recipe().
    recipe: Optional[Dict[str, Any]] = None
    make_vocab_size_divisible_by = 128 # default is 128 in megatron
    padded_vocab_size = True # When tokinzer is NullTokenizer, pad vocab size to make it divisible by make_vocab_size_divisible_by * tp_size in Megatron
    

    def __post_init__(self):
        if self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.intermediate_size
        if self.model_type is None:
            if self.expert_num > 1:
                self.model_type = 'moe'
            else:
                self.model_type = 'dense'
        # SWA defaults: convenient shorthand for full-SWA models
        if self.swa_head_num > 0:
            if self.swa_kv_head_num is None:
                self.swa_kv_head_num = self.swa_head_num
            if self.swa_head_dim is None:
                self.swa_head_dim = self.head_size
        # Quant: attn_gate_quant and context_rmsnorm_quant are mutually exclusive
        # (op_define: ATTN_GATE_QUANT 和 CONTEXT_RMSNORM_QUANT 二选一)
        if self.attn_gate_quant and self.context_rmsnorm_quant:
            raise ValueError(
                "attn_gate_quant and context_rmsnorm_quant are mutually exclusive "
                "(op_define: ATTN_GATE_QUANT 和 CONTEXT_RMSNORM_QUANT 二选一)"
            )

    @classmethod
    def init_from_config_file(cls, config_file: str):
        """Initializes an instance from a JSON config file."""
        config_dict = cls.read_json_file(config_file)
        if config_dict.get('moe_ffn_hidden_size') is None:
            config_dict['moe_ffn_hidden_size'] = config_dict['intermediate_size']
        return cls.init_from_dict(config_dict)
    
 
    def maybe_pad_vocab_size(self, tp_size, log=False):
        """ref Megatron-LM: Megatron-LM/megatron/training/tokenizer/tokenizer.py:105
        Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""
        if self.padded_vocab_size:
            if self.orig_vocab_size is None:
                self.orig_vocab_size = self.vocab_size
            multiple = self.make_vocab_size_divisible_by * tp_size
            after = int(math.ceil(self.orig_vocab_size / multiple) * multiple)
            if log:
                print(
                    ' > padded vocab (size: {}) with {} dummy tokens '
                    '(new size: {})'.format(self.orig_vocab_size, after - self.orig_vocab_size, after),
                    flush=True,
                )
            self.vocab_size = after
    
    def set_vocab_size(self, vocab_size):
        self.orig_vocab_size = vocab_size 
        self.vocab_size = vocab_size
        
    @property
    def param_numel(self):
        return (
            2 * self.vocab_elements
            + self.layer_elements * self.layer_num
            + self.norm_elements
        )

    @property
    def activated_param_numel(self):
        return (
            2 * self.vocab_elements
            + self.layer_act_elements * self.layer_num
            + self.norm_elements
        )

    def flops_per_token(self, context_seq_len, with_attn=True):
        """compute theoretical FLOPs per token"""
        attn_matmul = (
            3 * 2 * self.layer_num * (self.qkv_proj_elements + self.attn_proj_elements)
        )
        factor = 1
        res = 0
        if self.topk is not None and self.topk > 1:
            factor += self.topk - 1
            attn_router = 3 * 2 * self.layer_num * self.hidden_size * self.expert_num
            res += attn_router
        if self.moe_shared_expert_intermediate_size is not None:
            factor += self.moe_shared_expert_intermediate_size / self.moe_ffn_hidden_size
        mlp_matmul = 3 * 2 * self.layer_num * self.mlp_elements * factor
        res += attn_matmul + mlp_matmul
        if with_attn:
            attn_sdp = 3 * 2 * self.layer_num * (2 * context_seq_len * self.hidden_size)
            if self.attention_type == 'mla':
                attn_sdp = 3 * 2 * self.layer_num * (context_seq_len * (self.qk_head_dim+self.qk_pos_emb_head_dim) * self.head_num+
                                                     context_seq_len * self.v_head_dim * self.head_num)
            res += attn_sdp
            if SIMU_DEBUG:
                print(f"1layer mlp_matmul={mlp_matmul/self.layer_num}; 1layer attn_matmul={attn_matmul/self.layer_num}; 1layer attn_sdp={attn_sdp/self.layer_num}")

            # res += attn_sdp*7/6  #for fa addition bmm; in this case mfu_6nd_with_attn is equal to mean mfu bwtween pp stages
        if SIMU_DEBUG:
            print(f"1layer={res/self.layer_num}; embdedding={3 * 2 * (self.hidden_size * self.vocab_size)}")
        res += 3 * 2 * (self.hidden_size * self.vocab_size)  #for linear in ce
        return res

    @property
    def mlp_elements(self):
        mlp_weight_factor = 3 if self.use_swiglu else 2
        mlp_elements = mlp_weight_factor * self.hidden_size * self.moe_ffn_hidden_size
        return mlp_elements

    @property
    def base_proj_elements(self):
        if self.attention_type=='mla':
            return self.v_head_dim * self.head_num * self.hidden_size
        attn_proj_elements = self.hidden_size * self.hidden_size
        return attn_proj_elements

    @property
    def attn_proj_elements(self):
        return self.base_proj_elements

    @property
    def norm_elements(self):
        # consider rms norm for now
        return self.hidden_size

    @property
    def qkv_proj_elements(self):
        assert self.head_num is not None

        kv_head_num = self.head_num if self.kv_head_num is None else self.kv_head_num
        if self.attention_type=='mla':
            if self.q_lora_rank is None:
                elements = self.hidden_size * self.head_num * (self.qk_head_dim + self.qk_pos_emb_head_dim)
            else:
                elements = self.hidden_size * self.q_lora_rank  #q_down
                elements += self.q_lora_rank * self.head_num * (self.qk_head_dim + self.qk_pos_emb_head_dim) #q_up
            elements += self.hidden_size * (self.kv_lora_rank + self.qk_pos_emb_head_dim)  #kv_down
            elements += self.kv_lora_rank * self.head_num * (self.qk_head_dim + self.v_head_dim) #kv_up
            return elements
        else:
            proj_size = self.head_size * self.head_num + 2 * self.head_size * kv_head_num
            return self.hidden_size * proj_size

    @property
    def vocab_elements(self):
        return self.vocab_size * self.hidden_size

    @property
    def layer_elements(self):
        return (
            self.qkv_proj_elements
            + 2 * self.norm_elements
            + self.attn_proj_elements
            + self.expert_num * self.mlp_elements
        )

    @property
    def layer_act_elements(self):
        factor = 1
        if self.topk is not None and self.topk > 1:
            factor += self.topk - 1
        return (
            self.qkv_proj_elements
            + 2 * self.norm_elements
            + self.attn_proj_elements
            + factor * self.mlp_elements
        )

    def apply_recipe(self):
        """Expand the optional declarative block recipe (design doc section 6).

        v1 supports a flat "blocks" list over the registered BLOCK_TEMPLATES.
        The current LLMModel only supports a dense prefix, so DenseLLMBlock
        entries must lead; a dense block after a MoE block is an error. The
        recipe expands into layer_num (sum of counts) and dense_layers (sum
        of the leading dense counts), which win over explicitly set
        conflicting values (with a warning). Absent a recipe, nothing changes.
        """
        if self.recipe is None:
            return
        assert isinstance(self.recipe, dict), (
            f"recipe must be a dict, but got {type(self.recipe)}"
        )
        unknown_keys = set(self.recipe) - {"blocks"}
        assert not unknown_keys, (
            f"recipe has unknown keys {sorted(unknown_keys)}, "
            "allowed keys are ['blocks']"
        )
        blocks = self.recipe.get("blocks")
        assert isinstance(blocks, list) and blocks, (
            "recipe['blocks'] must be a non-empty list of "
            "{'template': <name>, 'count': <int>=1} entries"
        )
        layer_num = 0
        dense_layers = 0
        seen_moe = False
        for i, block in enumerate(blocks):
            ctx = f"recipe['blocks'][{i}]"
            assert isinstance(block, dict), (
                f"{ctx} must be a dict, but got {type(block)}"
            )
            assert set(block) == {"template", "count"}, (
                f"{ctx} must have exactly the keys ['template', 'count'], "
                f"but got {sorted(block)}"
            )
            template = get_block_template(block["template"])
            count = block["count"]
            assert (
                isinstance(count, int) and not isinstance(count, bool) and count >= 1
            ), f"{ctx}['count'] must be an int >= 1, but got {count!r}"
            layer_num += count
            if template.family == "moe":
                seen_moe = True
            else:
                assert not seen_moe, (
                    f"{ctx}: dense blocks must lead the recipe; a "
                    "DenseLLMBlock after a MoELLMBlock is not supported "
                    "(LLMModel only supports a dense prefix)"
                )
                dense_layers += count
        if self.layer_num is not None and self.layer_num != layer_num:
            warnings.warn(
                f"recipe expands to layer_num={layer_num}, but layer_num="
                f"{self.layer_num} was also set explicitly; the recipe wins."
            )
        if self.dense_layers and self.dense_layers != dense_layers:
            warnings.warn(
                f"recipe expands to dense_layers={dense_layers}, but "
                f"dense_layers={self.dense_layers} was also set explicitly; "
                "the recipe wins."
            )
        self.layer_num = layer_num
        self.dense_layers = dense_layers

    def sanity_check(self):
        self.apply_recipe()
        if not self.v_head_dim: 
            # not used for MLA
            # assert self.head_num * self.head_size == self.hidden_size
            ...
