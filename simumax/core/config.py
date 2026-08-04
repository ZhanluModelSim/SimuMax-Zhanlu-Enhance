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
)


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
    # Number of layers to prefetch in layer-wise FSDP (FSDP2 gap analysis doc
    # section 3.1). 1 = implicit prefetch (AG of next layer overlaps with
    # current compute). 2+ = explicit prefetch (multiple AGs fly simultaneously).
    # Only meaningful when zero_state >= 3 and fsdp_mode == "layer-wise".
    fsdp_prefetch_layers: int = 1

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
            assert self.fsdp_prefetch_layers >= 1, (
                f"fsdp_prefetch_layers must be >= 1, got {self.fsdp_prefetch_layers}"
            )
            if self.fsdp_prefetch_layers > 1 and self.fsdp_mode != "layer-wise":
                warnings.warn(
                    "fsdp_prefetch_layers > 1 has no effect when fsdp_mode != 'layer-wise'"
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
    efficient_factor: int
    latency_us: int
    fixed_latency: float = 0
    fixed_latency_us_by_comm_num: Dict[str, float] = None


@dataclass
class CompOpConfig:
    tflops: int
    efficient_factor: int
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

    def __post_init__(self):
        # Runtime override-chain slots, populated by PerfLLM.configure()
        # (cost-tunability design doc section 3). Plain attributes, not
        # dataclass fields: they never serialize into to_dict().
        self.efficiency_overrides_strategy = None
        self.efficiency_overrides_api = None

    @classmethod
    def init_from_dict(cls, config_dict: Dict[str, Any]):
        config_dict = copy.deepcopy(config_dict)
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

    def compute_op_accuracy_time(
        self, op_name: str, flops: int, shape_desc: str, reture_detail=False,
        class_key=None, path_key=None,
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
            if reture_detail:
                return dict(op_name=op_name,
                                tflops=None,
                                efficient_factor=None,
                                compute_only_time = 0.0)
            else:
                return 0

        op = self.accelerator.op.get(op_name, None)
        if op is None:
            warnings.warn(
                f"{op_name} not exist on {self.accelerator.op.keys()}, use default value"
            )
            op = self.accelerator.op.get("default", None)
            assert op is not None, f"default not exist on {self.accelerator.op}"
            self.record_miss_efficiency(op_name, flops, shape_desc, None)

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

    def compute_mem_access_time(self, op_name, mem_bytes: int, reture_detail=False):
        """
        compute memory access time,
        return time in ms
        """
        
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

    @staticmethod
    def _lookup_comm_num_value(values: Dict[str, Any], comm_num: int, default=None):
        if not values:
            return default
        for key in (str(comm_num), comm_num):
            if key in values:
                return values[key]
        return default

    def compute_net_op_time(self, op_name: str, size: int, comm_num: int, net="", comm_stage="unkonw", strategy:StrategyConfig=None, group_kind: str = None):
        """
        compute network operation time,
        return time in ms

        Inter-node corrections follow Tier A of
        docs/design_simu_network_fabric.md: when `strategy` (and, for
        TP/CP/ETP collectives, `group_kind`) is provided, cross-node
        traffic ratios come from the real group->node mapping
        (`simumax.core.utils.group_cross_node_ratio`). Calls that pass no
        `strategy`/`group_kind` keep the legacy heuristics unchanged.
        """
        # Using ring alg for now
        assert op_name in kNetOp, f"{op_name} not exist on {kNetOp}"
        if net == self.LEVELS_NET:
            # Hierarchical levels path (design_simu_hierarchical_network.md
            # sections 5-7); fully separate from the single-net path below.
            return self._compute_net_op_time_levels(
                op_name, size, comm_num, comm_stage, strategy, group_kind)
        net_data = self.networks.get(net, None)
        assert net_data is not None, f"{net} not exist on {self.networks.keys()}, op_name={op_name}"
        # alltoallv falls back to all2all when the network config only defines all2all
        if op_name == "alltoallv" and "alltoallv" not in net_data.op and "all2all" in net_data.op:
            op_name = "all2all"
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
                                    group_kind: str):
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
        # alltoallv falls back to all2all, mirroring the single-net path
        # (compute_net_op_time): level nets only define all2all, and an
        # alltoallv on the ep/cp group IS an all2all (context-parallel
        # sequence exchange / MoE dispatch). Without this mapping the
        # levels path cannot express alltoallv at all.
        if op_name == "alltoallv":
            op_name = "all2all"
        composition, spans = group_level_span(group_kind, strategy, levels)
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
        # (span, phase_size, bw, eff_factor, phase_time_ms, base_latency_us)
        phases = []
        for i, span in enumerate(spans):
            scale, offset, eff_factor, bw, base_latency, fixed_latency = \
                self._level_net_params(span.net, op_name, comm_num)
            # Topology-kind-aware bandwidth (design doc Part C, section 5.5):
            # CLOS levels divide by convergence_ratio; FullMesh levels keep
            # the full link bandwidth (no sharing). Default kind="clos" with
            # convergence_ratio=1.0 preserves the current levels behavior.
            if span.kind == "clos" and span.convergence_ratio > 1.0:
                bw /= span.convergence_ratio
            if op_name == "all2all":
                # Per-level share of each member's traffic; levels whose
                # boundary nobody crosses (fraction == 0) are skipped
                # entirely, latency included.
                fraction = all2all_level_fraction(group_kind, strategy, levels, i)
                if fraction <= 0:
                    continue
                phase_size = size * scale * fraction
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
            base_time = phase_size / (bw * 1024**3 * eff_factor) * 1e3
            latency_time = (base_latency + fixed_latency) / 1e3
            # Layout transform overhead (e.g. dim01_transpose around all2allv)
            layout_oh = 0.0
            if op_name == "all2all":
                layout_oh = (getattr(strategy, 'layout_transform_overhead_us', 0) or 0) / 1e3
            phase_time = base_time + latency_time + layout_oh
            phases.append((span, phase_size, bw, eff_factor, phase_time, base_latency))
        if not phases:
            # Group of one (or no crossed level): no communication.
            return 0.0
        if policy == "max":
            total_time = max(phase[4] for phase in phases)
        else:
            total_time = sum(phase[4] for phase in phases)
        # net_info.json decomposition: one record per level under
        # "levels:<stage>:<level>" plus the composed total under
        # "levels:<stage>". Records keep the legacy field set.
        stage_key = comm_stage.lower()
        for span, phase_size, bw, eff_factor, phase_time, base_latency in phases:
            self.record_net_bw(
                op_name, span.net, comm_num, f"levels:{stage_key}:{span.name}",
                bw, bw * eff_factor, eff_factor, phase_time * 1e3,
                phase_size, base_latency)
        self.record_net_bw(
            op_name, self.LEVELS_NET, comm_num, f"levels:{stage_key}",
            None, None, None, total_time * 1e3, size,
            sum(phase[5] for phase in phases))
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
    RESERVED_RESOURCE_LANES = ("comp", "comm", "pp_fwd", "pp_bwd", "off")

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
    LEVELS_TOPOLOGY_KEYS = ("levels", "composition_policy")
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

        Returns the built-in lanes ["comp", "comm", "pp_fwd", "pp_bwd"] plus
        the sorted names of `engines` entries not already in that list.
        """
        lanes = ["comp", "comm", "pp_fwd", "pp_bwd"]
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
            optional_keys = {"kind", "convergence_ratio"}
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
