"""basic moe transformer module"""
import math
from copy import deepcopy
from simumax.core.base_struct import (
    MetaModule,
    TensorSize,
    InputOutputInfo,
    GroupLinearBase
)
from simumax.core.base_struct import (all_gather, all_gatherv, reduce_scatter,
                           reduce_scatterv,
                           all_reduce, all_gather_bwd, all2all, alltoallv,
                           AtomModel, LeafModel, FwdQue,PathDebugContext,
                           COM_BUFF)
from simumax.core.config import StrategyConfig, SystemConfig, ModelConfig, MLPRecomputeConfig
import simumax.core.transformer.simu_ops as simu_ops
from simumax.core.transformer.dense_module import Swiglu, Gelu, MLP, Float8Quantizer, LinearCol, LinearRow, LinearBase
from simumax.core.utils import format_model_info_microbatch_tag, get_rank_group
from simumax.core.transformer.function import AddFunction
Input = InputOutputInfo


def _layout_logical_passes(system, op_name, local_expert_num):
    """Return the declared logical layout-pass count for a fused route op.

    The historical fallback models an index sort as ``1 + ceil(log2(E))``
    memory passes.  A target CANN structural profile may declare a fused
    implementation's total logical passes instead.  This keeps the model
    portable and shape-driven while avoiding a trace-derived per-kernel time
    or a hard-coded event duration.
    """
    fallback = 1 + math.ceil(math.log2(max(1, local_expert_num)))
    resolver = getattr(system, "layout_pass_count", None)
    if resolver is None:
        return fallback
    return resolver(op_name, fallback)


def _layout_kernel_time(system, op_name, read_bytes, write_bytes, stage,
                        path_key=None):
    """Derive a materialized layout-kernel time from tensor traffic.

    Layout operations are memory-bound, but they are still distinct kernels in
    the DES graph.  Keeping the read/write split here lets the same
    HBM/transaction/launch/MTE model serve both ``prefill`` and the analytical
    parent cost.  A legacy SystemConfig without ``compute_layout_time`` keeps
    the old byte-only fallback, so this helper does not introduce a measured
    duration or an implementation-specific kernel name.
    """
    read_bytes = max(0, int(read_bytes))
    write_bytes = max(0, int(write_bytes))
    layout_time = getattr(system, "compute_layout_time", None)
    if layout_time is not None:
        return layout_time(
            op_name,
            read_bytes,
            write_bytes,
            stage=stage,
            path_key=path_key,
            shape_desc=(f"input_bytes={read_bytes},output_bytes={write_bytes}"),
        )
    return system.compute_mem_access_time(op_name, read_bytes + write_bytes)
#region ------------------ Atomic module ------------------
class Router(LinearBase):
    """
    Megatron alltoall impl (fwd)
    1.apply jitter
    2.linear gating
    3.rounting:
      - z_loss for local logits
      - aux loss: input logits, output scores and indexs
        - topk_softmax_with_capacity
        - softmax
        - apply_load_balancing_loss
    """

    def __init__(
        self,
        layer_idx,
        hidden_size: int,
        expert_num: int,
        topk: int,
        moe_dispatcher_policy: str,
        has_cached_inputs: bool,
        enable_recompute: bool,
        is_last_recompute: bool,
        use_variance_tail_model: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
        specific_name: str = 'Router',
    ) -> None:
        super().__init__(hidden_size, expert_num, strategy, system, specific_name)
        self.layer_idx = layer_idx
        self.expert_num = expert_num
        self.local_expert_num = expert_num // self.strategy.ep_size
        self.topk = topk
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.is_last_recompute = is_last_recompute
        self.use_variance_tail_model = self.use_variance_tail_model or use_variance_tail_model
        if self.is_last_recompute and self.enable_recompute:
            self.set_variance_node(True)
        self.hidden_size = hidden_size
        self.moe_dispatcher_policy = moe_dispatcher_policy
        self._gating_fwd_time = 0.0
        self._router_local_stages = []
        # TODO: consider z-loss、aux-loss etc.

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        model_info = f"{format_model_info_microbatch_tag(args)}-layer:{self.layer_idx}-name:{self.__class__.__name__}"
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)

        # Gating GEMM
        self.layers.append(AtomModel(fwd_cost=self._gating_fwd_time,
                                  bwd_cost=self._cost_info.bwd_grad_act_time+self._cost_info.bwd_grad_w_time,
                                  recompute_cost=(self._gating_fwd_time
                                                  if self.enable_recompute else 0),
                                  specific_name='MoEGating'))

        # The score tensor must be normalized, softmaxed and reduced to top-k
        # route fields before its metadata collectives can launch. These are
        # stable algorithm stages; each duration is derived from tensor bytes,
        # HBM and launch facts rather than profiler timing.
        for name, cost in self._router_local_stages:
            self.layers.append(AtomModel(
                fwd_cost=cost, bwd_cost=0,
                recompute_cost=cost if self.enable_recompute else 0,
                specific_name=name))

        # Router metadata is synchronized in the EP domain before dispatch.
        # IDs and weights are materialized as distinct tensors; expert counts
        # and prefix offsets are likewise distinct int32 arrays. Keeping these
        # four graph edges explicit preserves launch latency without inventing
        # an aggregate empirical coefficient.
        if self.strategy.ep_size > 1:
            batch_size = self.input_info.tensors[0].size(0)
            seq_len = self.input_info.tensors[0].size(1)
            route_field_size = (batch_size * seq_len * self.topk
                                * self.dtype_to_element_size["fp32"])
            expert_field_size = self.expert_num * 4  # int32
            ag_op = ("all_gatherv" if self.strategy.moe_variable_collectives
                     else "all_gather")
            ag_cls = (all_gatherv if self.strategy.moe_variable_collectives
                      else all_gather)
            # Counts and offsets are one contiguous [2, E] header in the
            # implementation graph. Synchronize it before gathering the two
            # variable route fields so the dependency order is explicit.
            expert_header_size = 2 * expert_field_size
            ar_cost = self.system.compute_net_op_time(
                "all_reduce", expert_header_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
                strategy=self.strategy,
                group_kind="ep",
                comm_stage="Router_FWD_expert_header_AR",
                comm_direction="fwd",
            )
            ar_bwd_cost = self.system.compute_net_op_time(
                "all_reduce", expert_header_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
                strategy=self.strategy,
                group_kind="ep",
                comm_stage="Router_BWD_expert_header_AR",
                comm_direction="bwd",
            )
            self.layers.append(all_reduce(
                f"{state.comm_order}-{model_info}-ep_group:{rank_info['ep_group_id']}-expert_header_ar",
                rank_info['ep_rank'], self.strategy.ep_size,
                com_buff=com_buff, fwd_cost=ar_cost, bwd_cost=ar_bwd_cost,
                global_rank=args.rank, net=self.strategy.ep_net,
                size_bytes=expert_header_size,
                group_kind="ep", comm_stage="Router_expert_header_AR",
            ))
            state.comm_order += 1

            for field in ("topk_ids", "topk_weights"):
                ag_cost = self.system.compute_net_op_time(
                    ag_op, route_field_size,
                    comm_num=self.strategy.ep_size,
                    net=self.strategy.ep_net,
                    strategy=self.strategy,
                    group_kind="ep",
                    comm_stage=f"Router_FWD_{field}_AGV",
                    comm_direction="fwd",
                )
                self.layers.append(ag_cls(
                    f"{state.comm_order}-{model_info}-ep_group:{rank_info['ep_group_id']}-{field}_agv",
                    rank_info['ep_rank'], self.strategy.ep_size,
                    com_buff=com_buff, fwd_cost=ag_cost, bwd_cost=0,
                    global_rank=args.rank, net=self.strategy.ep_net,
                    size_bytes=route_field_size,
                    group_kind="ep", comm_stage="Router_route_fields_AGV",
                ))
                state.comm_order += 1

            rsv_cost = self.system.compute_net_op_time(
                "reduce_scatterv", route_field_size,
                comm_num=self.strategy.ep_size, net=self.strategy.ep_net,
                strategy=self.strategy, group_kind="ep",
                comm_stage="Router_BWD_route_grad_RSV",
                comm_direction="bwd")
            self.layers.append(reduce_scatterv(
                f"{state.comm_order}-{model_info}-ep_group:{rank_info['ep_group_id']}-route_grad_rsv",
                rank_info['ep_rank'], self.strategy.ep_size,
                com_buff=com_buff, fwd_cost=0, bwd_cost=rsv_cost,
                global_rank=args.rank, net=self.strategy.ep_net,
                size_bytes=route_field_size, group_kind="ep",
                comm_stage="Router_route_grad_RSV"))
            state.comm_order += 1

        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff)
    
    @property
    def micro_input_tensor(self):
        assert self.input_info is not None, "Please set input info"
        # [B, S, H]
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        if self.strategy.enable_sequence_parallel:
            # collect the full sequence data by all-gather, the seq_size is seq_len * tp_size 
            seq_len *= self.strategy.tp_size
        hidden_size = self.input_info.tensors[0].size(2)
        return TensorSize(shape = [batch_size, seq_len, hidden_size], dtype=self.input_info.tensors[0].dtype)
    
    @property
    def local_logits_size(self):
        assert self.input_info is not None, "Please set input info"
        b = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        ep_num = self.expert_num
        return b * seq_len * ep_num

    def create_output_info(self):
        # FIXME(sherry): check this, return [hidden_states, scores, routting_map]
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        # hidden_size = self.input_info.tensors[0].size(2)
        hidden_states = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, self.expert_num), dtype="int32")]
        )
        return hidden_states
    
    @property
    def weight(self):
        return TensorSize(shape=(self.hidden_size, self.expert_num))
    
    def _pre_op(self): 
        assert self.hidden_size == self.input_info.tensors[0].size(2)

    def _comp_leaf_intra_net_info(self):
        """Router does not model extra TP full-logit gather in the all2all path.

        DP topk communication (allGatherv + allReduce across edp group) is
        modeled here for cost estimation.
        """
        if self.strategy.ep_size > 1:
            batch_size = self.input_info.tensors[0].size(0)
            seq_len = self.input_info.tensors[0].size(1)
            route_field_size = (batch_size * seq_len * self.topk
                                * self.dtype_to_element_size["fp32"])
            expert_field_size = self.expert_num * 4  # int32
            ag_op = ("all_gatherv" if self.strategy.moe_variable_collectives
                     else "all_gather")
            for field in ("topk_ids", "topk_weights"):
                self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                    ag_op, route_field_size,
                    comm_num=self.strategy.ep_size, net=self.strategy.ep_net,
                    strategy=self.strategy, group_kind="ep",
                    comm_stage=f"Router_FWD_{field}_AGV",
                    comm_direction="fwd")
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_reduce", 2 * expert_field_size,
                comm_num=self.strategy.ep_size, net=self.strategy.ep_net,
                strategy=self.strategy, group_kind="ep",
                comm_stage="Router_FWD_expert_header_AR",
                comm_direction="fwd")
            self._cost_info.bwd_grad_act_net_time += \
                self.system.compute_net_op_time(
                    "all_reduce", 2 * expert_field_size,
                    comm_num=self.strategy.ep_size,
                    net=self.strategy.ep_net,
                    strategy=self.strategy, group_kind="ep",
                    comm_stage="Router_BWD_expert_header_AR",
                    comm_direction="bwd")
            self._cost_info.bwd_grad_act_net_time += \
                self.system.compute_net_op_time(
                    "reduce_scatterv", route_field_size,
                    comm_num=self.strategy.ep_size, net=self.strategy.ep_net,
                    strategy=self.strategy, group_kind="ep",
                    comm_stage="Router_BWD_route_grad_RSV",
                    comm_direction="bwd")

    def _comp_leaf_act_info_impl(self):
        """
        activation_mem_cache = input(linear), scores([S,K], softmax)
        """
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        input_size = batch_size * seq_len * hidden_size
        self._act_info.activation_mem_cache = input_size * self.element_size
        
        if self.has_cached_inputs:
            self._act_info.activation_mem_cache = 0
        # Gating, The tensor processed by the softmax is relatively small,
        # so the gating here is used as the operator that appears peak in this module
        gating_weight_size = self.hidden_size * self.expert_num * self.element_size
        input_size = input_size * self.element_size
        output_size = self.local_logits_size * self.element_size
        self._act_info.fwd_peak_mem_no_cache = (
            input_size + output_size + gating_weight_size
        )
        self._act_info.bwd_peak_mem_no_cache = (
            input_size + output_size + gating_weight_size
        )

    def _comp_leaf_model_info_impl(self):
        """
        weight = input(linear), scores([S,K], softmax)
        """
        weight_numel = self.hidden_size * self.expert_num
        self._model_info.weight_numel = weight_numel
        self._model_info.dense_weight_bytes = weight_numel * self.element_size
        self._model_info.dense_grad_bytes = weight_numel * self.main_grad_element_size
        self._model_info.dense_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        optimizer_group_size = self.strategy.fsdp_dense_group_size
        if self.strategy.zero_state >= 1:
            self._model_info.dense_state_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 2:
            self._model_info.dense_grad_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 3:
            self._model_info.dense_weight_bytes /= optimizer_group_size

    def _comp_leaf_flops_info(self):
        # Count Gating
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        input_size = batch_size * seq_len * hidden_size

        base_flops = 2 * input_size * self.expert_num
        self._compute_info.fwd_flops = base_flops
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = base_flops
        self._compute_info.bwd_grad_w_flops = base_flops

    def _comp_leaf_mem_accessed_info(self):
        """
        linear + softmax
        """
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        input_size = batch_size * seq_len * hidden_size
        # linear
        gating_weight_size = self.hidden_size * self.expert_num * self.element_size
        linear_input_size = input_size * self.element_size
        linear_output_size = self.local_logits_size * self.element_size
        linear_mem_accessed = (
            gating_weight_size + linear_input_size + linear_output_size
        )
        # softmax
        softmax_input_size = linear_output_size
        if self.strategy.enable_sequence_parallel and self.strategy.tp_size > 1:
            softmax_input_size *= self.strategy.tp_size
        # output_size = self.local_logits_size * self.element_size
        softmax_fwd_mem_accessed = 2 * softmax_input_size
        softmax_bwd_mem_accessed = 3 * softmax_input_size

        self._compute_info.fwd_accessed_mem = (
            linear_mem_accessed + softmax_fwd_mem_accessed
        )
        self._compute_info.bwd_grad_act_accessed_mem = (
            linear_mem_accessed + softmax_bwd_mem_accessed
        )
        self._compute_info.bwd_grad_w_accessed_mem = linear_mem_accessed

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="router",
            bwd_grad_act_op="router",
            bwd_grad_w_op="router",
            enable_recompute=self.enable_recompute,
        )
        batch = self.input_info.tensors[0].size(0)
        seq = self.input_info.tensors[0].size(1)
        hidden = self.input_info.tensors[0].size(2)
        logits = batch * seq * self.expert_num
        routes = batch * seq * self.topk
        logits_bytes = logits * self.element_size
        route_bytes = routes * (4 + 4)  # int32 expert id + fp32 weight
        weight_bytes = self.hidden_size * self.expert_num * self.element_size
        input_bytes = batch * seq * hidden * self.element_size
        linear_mem = input_bytes + weight_bytes + logits_bytes
        linear_flops = 2 * batch * seq * hidden * self.expert_num
        class_key, path_key = self.get_cost_keys()
        self._gating_fwd_time = self.system.compute_op_accuracy_time(
            "router", linear_flops,
            shape_desc=(f"m={batch * seq}, n={self.expert_num}, k={hidden}, "
                        f"dtype={self.strategy.dtype}, stage=gating_gemm"),
            accessed_mem=linear_mem, stage="fwd",
            class_key=class_key, path_key=path_key)

        local_specs = (
            ("RouterLogitNorm", 2 * logits_bytes, logits_bytes),
            ("RouterSoftmax", 2 * logits_bytes, logits_bytes),
            ("RouterTopK", logits_bytes, route_bytes),
            ("RouterAuxReduce", logits_bytes + route_bytes,
             self.expert_num * 4),
        )
        self._router_local_stages = []
        for name, read_bytes, write_bytes in local_specs:
            cost = self.system.compute_layout_time(
                name, read_bytes, write_bytes, stage="fwd",
                path_key=path_key,
                shape_desc=(f"batch={batch},seq={seq},experts={self.expert_num},"
                            f"topk={self.topk},read={read_bytes},write={write_bytes}"))
            self._router_local_stages.append((name, cost))
        self._cost_info.fwd_compute_time = (
            self._gating_fwd_time
            + sum(cost for _, cost in self._router_local_stages))
        self._cost_info.recompute_compute_time = (
            self._cost_info.fwd_compute_time if self.enable_recompute else 0)

class Permutation(MetaModule):
    """
    Permutation Impl
    1.permute1([S, M] -> [E, C, M] or unbalance [(E1T1, E1T2, ..., E2T1, ...)])
    2.all2all on ep group
    3.permutate2: when local_expert_num > 1,
      arranged according to the batch that the expert needs to process
    4.all_gather feat-dim on tp group or token-dim etp group (fwd: all gather, bwd: reduce_scatter)
    """

    def __init__(
        self,
        layer_idx,
        expert_num: int,
        local_expert_num: int,
        topk: int,
        moe_pad_expert_input_to_capacity:bool,
        capacity:int,
        moe_dispatcher_policy: str,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
        stage_partition: str = 'all',
    ) -> None:
        super().__init__(strategy, system)
        self.layer_idx = layer_idx
        self.expert_num = expert_num
        self.local_expert_num = local_expert_num
        self.topk = topk
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.moe_dispatcher_policy = moe_dispatcher_policy
        self.moe_pad_expert_input_to_capacity = moe_pad_expert_input_to_capacity
        self.capacity = capacity
        if stage_partition not in ('all', 'pre_metadata', 'post_metadata'):
            raise ValueError(
                "stage_partition must be all, pre_metadata, or post_metadata"
            )
        self.stage_partition = stage_partition

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        model_info = f"{format_model_info_microbatch_tag(args)}-layer:{self.layer_idx}-name:{self.__class__.__name__}"
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)
        

    
        include_pre = self.stage_partition in ('all', 'pre_metadata')
        include_post = self.stage_partition in ('all', 'post_metadata')

        # permutate1 creates the metadata/activation buffers consumed by the
        # first EP exchange. It belongs to the pre-metadata partition.
        # concat (sort_by_expert chunks) writes the permuted buffer once more
        # than the plain permute read+write above (real ConcatD kernel at
        # 2087/step9 on the 16p trace).
        permutate1_mem_accessed = (
            self.input_act_size + 2 * self.permuted_act_size
        ) * self.dtype_to_element_size[self.strategy.dtype]
        element_size = self.dtype_to_element_size[self.strategy.dtype]
        permutate1_read = (
            self.input_act_size + self.permuted_act_size) * element_size
        permutate1_write = self.permuted_act_size * element_size
        fwd_compute_time = _layout_kernel_time(
            self.system, "permute1", permutate1_read, permutate1_write,
            "fwd", self.call_stk)
        bwd_mem_time = _layout_kernel_time(
            self.system, "permute1", permutate1_read, permutate1_write,
            "bwd", self.call_stk)
        bwd_grad_w_accessed_mem = 0
        bwd_grad_act_accessed_mem = bwd_mem_time
        bwd_grad_act_time = bwd_mem_time
        bwd_grad_w_time = self.system.compute_end2end_time(0, bwd_grad_w_accessed_mem)
        if include_pre:
            self.layers.append(AtomModel(fwd_cost=fwd_compute_time,
                                     bwd_cost=bwd_grad_act_time+bwd_grad_w_time,
                                     specific_name='permute1'))
        


        if self.strategy.ep_size > 1:
            comm_size = self.dispatch_comm_size
            main_a2a_op = ("alltoallv" if self.strategy.moe_variable_collectives
                           else "all2all")
            main_a2a_cls = (alltoallv if self.strategy.moe_variable_collectives
                            else all2all)
            # Fixed expert metadata is exchanged first. The activation buffer
            # is then dispatched before its variable route-field sidecar; the
            # receiver consumes the pair only after both collectives complete.
            # This is a framework dependency, independent of their durations.
            metadata_size = self.expert_num * 4  # int32 expert counts
            metadata_cost = self.system.compute_net_op_time(
                "moe_small_a2a", metadata_size, self.strategy.ep_size,
                net=self.strategy.ep_net, strategy=self.strategy, group_kind="ep",
            )
            if include_pre:
                self.layers.append(all2all(f"{state.comm_order}-{model_info}-ep_group:{rank_info['ep_group_id']}-prob",
                                             rank_info['ep_rank'], self.strategy.ep_size, com_buff=com_buff,
                                             fwd_cost=metadata_cost, bwd_cost=0,
                                             global_rank=args.rank, net=self.strategy.ep_net,
                                             size_bytes=metadata_size, group_kind="ep",
                                             comm_stage="Dispatch_expert_counts_A2A"))
                state.comm_order += 1
            cost = self.system.compute_net_op_time(
                main_a2a_op,
                comm_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
                strategy=self.strategy,
                group_kind="ep",
                comm_stage="Dispatch_activation_A2AV",
            )
            if include_post:
                self.layers.append(main_a2a_cls(
                    f"{state.comm_order}-{model_info}-ep_group:{rank_info['ep_group_id']}",
                    rank_info['ep_rank'], self.strategy.ep_size,
                    com_buff=com_buff, fwd_cost=cost, bwd_cost=cost,
                    global_rank=args.rank, net=self.strategy.ep_net,
                    size_bytes=comm_size, group_kind="ep",
                    comm_stage="Dispatch_activation_A2AV"))
                state.comm_order += 1
            batch_size = self.input_info.tensors[0].size(0)
            route_size = (batch_size * self.topk * self._per_rank_seq()
                          * (4 + 4 + 4))
            route_cost = self.system.compute_net_op_time(
                "alltoallv", route_size, self.strategy.ep_size,
                net=self.strategy.ep_net, strategy=self.strategy,
                group_kind="ep", comm_stage="Dispatch_route_fields_A2AV")
            if include_post:
                self.layers.append(alltoallv(
                    f"{state.comm_order}-{model_info}-ep_group:{rank_info['ep_group_id']}-route_fields",
                    rank_info['ep_rank'], self.strategy.ep_size,
                    com_buff=com_buff, fwd_cost=route_cost, bwd_cost=0,
                    global_rank=args.rank, net=self.strategy.ep_net,
                    size_bytes=route_size, group_kind="ep",
                    comm_stage="Dispatch_route_fields_A2AV"))
                state.comm_order += 1
        if include_post and self.strategy.etp_size > 1:
            comm_size = (
                self.permuted_act_size
                * self.dtype_to_element_size[self.strategy.dtype]
                * self.strategy.etp_size
            )
            cost = self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                strategy=self.strategy,
                group_kind="tp",
            ) 
            self.layers.append(all_gather(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost, global_rank=args.rank, net=self.strategy.tp_net, size_bytes=comm_size,))
            state.comm_order += 1

        #permutate2 after ep all2all and tp
        concat_depth = _layout_logical_passes(
            self.system, "permute2", self.local_expert_num)
        permutate2_mem_accessed = (
            2 * self.permuted_act_size * concat_depth
        ) * self.dtype_to_element_size[self.strategy.dtype]
        fwd_compute_time = _layout_kernel_time(
            self.system, "permute2", permutate2_mem_accessed // 2,
            permutate2_mem_accessed // 2, "fwd", self.call_stk)
        bwd_mem_time = _layout_kernel_time(
            self.system, "permute2", permutate2_mem_accessed // 2,
            permutate2_mem_accessed // 2, "bwd", self.call_stk)
        bwd_grad_w_accessed_mem = 0
        bwd_grad_act_accessed_mem = bwd_mem_time
        bwd_grad_act_time = bwd_mem_time
        bwd_grad_w_time = self.system.compute_end2end_time(0, bwd_grad_w_accessed_mem)
        if include_post:
            self.layers.append(AtomModel(fwd_cost=fwd_compute_time,
                                     bwd_cost=bwd_grad_act_time+bwd_grad_w_time,
                                     specific_name='permute2'))
        
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff)

    @property
    def permuted_act_size(self):
        # only consider balanced case for now
        # Use the per-rank full token count, not input_info's seq: input_info
        # seq is CP-split (seq//cp) on the DES path, which halves the MoE
        # dispatch payload vs the manual/prefill path (per-rank full token =
        # seq_len // (cp//td)). Same root cause as CP a2a volume. MoE dispatch
        # moves per-rank tokens to expert-holding ranks regardless of CP split.
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self._per_rank_seq()
        hidden_size = self.input_info.tensors[0].size(2)
        token_num = self.topk * batch_size * seq_len
        if self.moe_pad_expert_input_to_capacity:
            token_num = math.ceil(token_num/self.expert_num) * self.expert_num * self.capacity
        return token_num * hidden_size

    def _per_rank_seq(self):
        """Per-rank full sequence (not CP-split): seq_len // (cp//td).
        input_info seq is CP-split (seq//cp) on the DES path; MoE dispatch
        moves per-rank tokens regardless of CP split, so derive from strategy.
        td comes from the model config, exposed on the strategy by the mxx
        model builder (MxxModelLayer sets strategy.mxx_trunk_cp_divisor).
        """
        td = getattr(self.strategy, 'mxx_trunk_cp_divisor', None)
        if td is None:
            td = 1
        per_rank_cp = max(1, self.strategy.cp_size // td)
        return max(1, self.strategy.seq_len // per_rank_cp)

    @property
    def input_act_size(self):
        # only consider balanced case for now
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self._per_rank_seq()
        hidden_size = self.input_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    @property
    def dispatch_comm_size(self):
        """MoE dispatch payload (external all-to-all transfers only).

        Without dispatch quantization, every route carries the configured
        activation dtype. With fused FP8 dispatch enabled, it instead carries
        one FP8 hidden vector, one fp32 scale per 128 hidden elements, and
        three 4-byte metadata fields. The mode is a model/runtime setting.

        This property returns the logical per-rank tensor size. The levels
        router applies the balanced EP peer fraction (ep-1)/ep exactly once;
        local retention is a memory operation rather than network traffic.
        """
        batch_size = self.input_info.tensors[0].size(0)
        route_count = batch_size * self._per_rank_seq() * self.topk
        hidden_size = self.input_info.tensors[0].size(2)
        if getattr(self.strategy, 'mxx_moe_dispatch_quant', False):
            scale_count = math.ceil(hidden_size / 128)
            record_bytes = hidden_size + scale_count * 4 + 3 * 4
        else:
            record_bytes = (hidden_size
                            * self.dtype_to_element_size[self.strategy.dtype])
        return route_count * record_bytes

    def _pre_op(self):
        super()._pre_op()
        # if self.strategy.dispatch_probs:
        #     assert len(self.input_info.tensors) == 2, "dispatch_probs=True requires two inputs in Permutation, [x, probs]"
        # else:
        #     assert len(self.input_info.tensors) == 1, "dispatch_probs=False requires one inputs in Permutation, [x]"

    def create_output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        part_seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        if self.strategy.enable_sequence_parallel and self.strategy.etp_size > 1:  
            seq_len = part_seq_len * self.strategy.etp_size
            # part_hidden_size = hidden_size // self.strategy.tp_size
        else:
            seq_len = part_seq_len
            # part_hidden_size = hidden_size
        balance_token_num = batch_size * seq_len * self.topk
        if self.moe_pad_expert_input_to_capacity:
            balance_token_num = math.ceil(balance_token_num/self.expert_num) * self.expert_num * self.capacity
        output_info = InputOutputInfo(
            tensors=[
                TensorSize(
                    shape=(balance_token_num, hidden_size)
                ),  # permuted moe input
            ]
        )
        return output_info

    def _comp_leaf_intra_net_info(self):
        include_pre = self.stage_partition in ('all', 'pre_metadata')
        include_post = self.stage_partition in ('all', 'post_metadata')
        if self.strategy.ep_size > 1:
            comm_size = self.dispatch_comm_size
            main_a2a_op = ("alltoallv" if self.strategy.moe_variable_collectives
                           else "all2all")
            if include_post:
                # fwd
                self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                    main_a2a_op,
                    comm_size,
                    comm_num=self.strategy.ep_size,
                    net=self.strategy.ep_net,
                    strategy=self.strategy,
                    group_kind="ep",
                    comm_stage="Dispatch_FWD_EP"
                )

                # bwd
                self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                    main_a2a_op,
                    comm_size,
                    comm_num=self.strategy.ep_size,
                    net=self.strategy.ep_net,
                    strategy=self.strategy,
                    group_kind="ep",
                    comm_stage="Dispatch_BWD_EP"
                )

            # HACK(sherry): all2all the router probs to expert, and fused combined probs to SiluOp in ExpertMLP, to avoid the activation_mem_cache in Unpermutaion
            if include_post and self.strategy.dispatch_probs:
                prob_comm_size = self.input_info.tensors[1].numel() * self.dtype_to_element_size[self.strategy.dtype]
                self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                    "all2all",
                    prob_comm_size,
                    comm_num=self.strategy.ep_size,
                    net=self.strategy.ep_net,
                    strategy=self.strategy,
                    group_kind="ep",
                    comm_stage="Dispatch_PROB_FWD_EP"
                )
                self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                    "all2all",
                    prob_comm_size,
                    comm_num=self.strategy.ep_size,
                    net=self.strategy.ep_net,
                    strategy=self.strategy,
                    group_kind="ep",
                    comm_stage="Dispatch_PROB_BWD_EP"
                )
            # HACK(sherry)

            metadata_size = self.expert_num * 4
            if include_pre:
                self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                    "moe_small_a2a", metadata_size,
                    comm_num=self.strategy.ep_size, net=self.strategy.ep_net,
                    strategy=self.strategy, group_kind="ep",
                    comm_stage="Dispatch_expert_counts_A2A")
            # Each routed token carries expert id, destination offset and
            # combine weight (two int32 fields + one fp32 field).
            batch_size = self.input_info.tensors[0].size(0)
            route_size = batch_size * self.topk * self._per_rank_seq() * (4 + 4 + 4)
            if include_post:
                self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                    "alltoallv", route_size,
                    comm_num=self.strategy.ep_size, net=self.strategy.ep_net,
                    strategy=self.strategy, group_kind="ep",
                    comm_stage="Dispatch_route_fields_A2AV")

        if include_post and self.strategy.etp_size > 1:
            comm_size = (
                self.permuted_act_size
                * self.dtype_to_element_size[self.strategy.dtype]
                * self.strategy.etp_size
            )
            # fwd
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.etp_size,
                net=self.strategy.etp_net,
                strategy=self.strategy,
                group_kind="etp",
                comm_stage="Permutation_FWD_ETP"
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.etp_size,
                net=self.strategy.etp_net,
                strategy=self.strategy,
                group_kind="etp",
                comm_stage="Permutation_BWD_ETP"
            )
        if self.enable_recompute:
            self._cost_info.recompute_net_time = self._cost_info.fwd_net_time

    def _comp_leaf_act_info_impl(self):
        probs_mem = self.input_info.tensors[1].numel() * 8
        self._act_info.activation_mem_cache = probs_mem
        self._act_info.fwd_peak_mem_no_cache = 0
        self._act_info.bwd_peak_mem_no_cache = 0

    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    def _comp_leaf_flops_info(self):
        """
        ignore memory bound operation's flops for now
        """
        self._compute_info.fwd_flops = 0
        self._compute_info.recompute_flops = 0
        self._compute_info.bwd_grad_act_flops = 0
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_mem_accessed_info(self):
        """
        permutate1 for ep all2all, scatter
        permutate2 for mlp compute, drop_and_pad=True: transpose + contiuous memory, drop_and_pad=False: sort_chunks_by_idx
        """
        permutate1_mem_accessed, permutate2_mem_accessed = 0, 0
        permutate1_mem_accessed = (
            self.input_act_size + self.permuted_act_size
        ) * self.dtype_to_element_size[self.strategy.dtype] # fused: scatter
        permutate2_passes = _layout_logical_passes(
            self.system, "permute2", self.local_expert_num)
        permutate2_mem_accessed = (
            2 * self.permuted_act_size * permutate2_passes
        ) * self.dtype_to_element_size[self.strategy.dtype]

        include_pre = self.stage_partition in ('all', 'pre_metadata')
        include_post = self.stage_partition in ('all', 'post_metadata')
        selected_mem = ((permutate1_mem_accessed if include_pre else 0)
                        + (permutate2_mem_accessed if include_post else 0))
        self._compute_info.fwd_accessed_mem = selected_mem
        self._compute_info.bwd_grad_act_accessed_mem = selected_mem
        self._compute_info.bwd_grad_w_accessed_mem = 0

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        # Keep perf-side parent cost aligned with the simulator trace.
        # `Permutation` executes two memory-bound layout kernels (`permute1`,
        # `permute2`) as separate leaf ops. Since the memory model includes a
        # fixed launch latency, collapsing them into one aggregated mem-access
        # estimate systematically underestimates the stage time.
        permutate1_mem_accessed = (
            self.input_act_size + self.permuted_act_size
        ) * self.dtype_to_element_size[self.strategy.dtype]
        permutate2_passes = _layout_logical_passes(
            self.system, "permute2", self.local_expert_num)
        permutate2_mem_accessed = (
            2 * self.permuted_act_size * permutate2_passes
        ) * self.dtype_to_element_size[self.strategy.dtype]

        element_size = self.dtype_to_element_size[self.strategy.dtype]
        permutate1_read = (
            self.input_act_size + self.permuted_act_size) * element_size
        permutate1_write = self.permuted_act_size * element_size
        permutate2_read = permutate2_mem_accessed // 2
        permutate2_write = permutate2_mem_accessed // 2

        def split_stage_time(stage):
            # ``compute_layout_time`` returns milliseconds and includes the
            # declared launch/MTE/layout resource costs for each materialized
            # kernel.  Keep the parent cost in the same units as net times.
            return (
                _layout_kernel_time(
                    self.system, "permute1", permutate1_read,
                    permutate1_write, stage, self.call_stk)
                + _layout_kernel_time(
                    self.system, "permute2", permutate2_read,
                    permutate2_write, stage, self.call_stk)
            )

        include_pre = self.stage_partition in ('all', 'pre_metadata')
        include_post = self.stage_partition in ('all', 'post_metadata')
        if include_pre and include_post:
            self._cost_info.fwd_compute_time = split_stage_time("fwd")
            self._cost_info.bwd_grad_act_time = split_stage_time("bwd")
        elif include_pre:
            self._cost_info.fwd_compute_time = _layout_kernel_time(
                self.system, "permute1", permutate1_read, permutate1_write,
                "fwd", self.call_stk)
            self._cost_info.bwd_grad_act_time = _layout_kernel_time(
                self.system, "permute1", permutate1_read, permutate1_write,
                "bwd", self.call_stk)
        elif include_post:
            self._cost_info.fwd_compute_time = _layout_kernel_time(
                self.system, "permute2", permutate2_read, permutate2_write,
                "fwd", self.call_stk)
            self._cost_info.bwd_grad_act_time = _layout_kernel_time(
                self.system, "permute2", permutate2_read, permutate2_write,
                "bwd", self.call_stk)
        else:
            self._cost_info.fwd_compute_time = 0
            self._cost_info.bwd_grad_act_time = 0
        self._cost_info.bwd_grad_w_time = 0
        self._cost_info.recompute_compute_time = (
            self._cost_info.fwd_compute_time if self.enable_recompute else 0
        )


class UnPermutation(MetaModule):
    """
    Reverse permutation
    1.reduce_scatter feat-dim on tp group or token-dim etp group
    2.Unpermuation1: when local_expert_num > 1, rearraange for all2all
    3.all2all on ep group
    4.Unpermuation2:
        - no (padding and drop):
          - 通过argsort的sorted_indices反向unpermutate,然后根据probs进行combine，但是drop的没有残差连接
        - padding and drop：
          - 通过final_indices(token indices, [E, C])和scatter_add实现恢复和combine weight
    """

    def __init__(
        self,
        layer_idx,
        expert_num: int,
        local_expert_num: int,
        topk: int,
        # moe_pad_expert_input_to_capacity:bool,
        moe_dispatcher_policy: str,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
    ) -> None:
        super().__init__(strategy, system)
        self.layer_idx = layer_idx
        self.expert_num = expert_num
        self.local_expert_num = local_expert_num
        self.topk = topk
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.moe_dispatcher_policy = moe_dispatcher_policy
        self.ori_shape = None
        # capacity-based padding not passed to UnPermutation ctor; default to
        # off (balanced) so act_size_before_combined's topk expansion is exact.
        self.moe_pad_expert_input_to_capacity = getattr(self, 'moe_pad_expert_input_to_capacity', False)
        self.capacity = getattr(self, 'capacity', 0)

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        model_info = f"{format_model_info_microbatch_tag(args)}-layer:{self.layer_idx}-name:{self.__class__.__name__}"
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)
        

    
        #unpermutate1 before tp and ep all2all
        unpermutate1_passes = _layout_logical_passes(
            self.system, "unpermute1", self.local_expert_num)
        unpermutate1_mem_accessed = ( # none-fused: contiguous memory(drop_and_pad) or sort_chunks_by_idxs
            2 * self.act_size_before_combined * unpermutate1_passes
        ) * self.dtype_to_element_size[self.strategy.dtype]

        unpermutate1_read = unpermutate1_mem_accessed // 2
        unpermutate1_write = unpermutate1_mem_accessed // 2
        fwd_compute_time = _layout_kernel_time(
            self.system, "unpermute1", unpermutate1_read,
            unpermutate1_write, "fwd", self.call_stk)
        bwd_mem_time = _layout_kernel_time(
            self.system, "unpermute1", unpermutate1_read,
            unpermutate1_write, "bwd", self.call_stk)
        bwd_grad_w_accessed_mem = 0
        bwd_grad_act_accessed_mem = bwd_mem_time
        bwd_grad_act_time = bwd_mem_time
        bwd_grad_w_time = self.system.compute_end2end_time(0, bwd_grad_w_accessed_mem)
        self.layers.append(AtomModel(fwd_cost=fwd_compute_time,
                                 bwd_cost=bwd_grad_act_time+bwd_grad_w_time,
                                 specific_name='unpermute1'))
        
        if self.strategy.etp_size > 1:
            comm_size = (
                self.act_size_before_combined
                * self.dtype_to_element_size[self.strategy.dtype]
                * self.strategy.etp_size
            )
            cost = self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.tp_size,
                net=self.strategy.tp_net,
                strategy=self.strategy,
                group_kind="tp",
            ) 
            self.layers.append(reduce_scatter(f"{state.comm_order}-{model_info}-tp_group:{rank_info['tp_group_id']}", 
                                         rank_info['tp_rank'], self.strategy.tp_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost, global_rank=args.rank, net=self.strategy.tp_net, size_bytes=comm_size,))
            state.comm_order += 1


        if self.strategy.ep_size > 1:
            comm_size = (
                self.act_size_before_combined
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            main_a2a_op = ("alltoallv" if self.strategy.moe_variable_collectives
                           else "all2all")
            main_a2a_cls = (alltoallv if self.strategy.moe_variable_collectives
                            else all2all)
            cost = self.system.compute_net_op_time(
                main_a2a_op,
                comm_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
                strategy=self.strategy,
                group_kind="ep",
            )
            self.layers.append(main_a2a_cls(f"{state.comm_order}-{model_info}-ep_group:{rank_info['ep_group_id']}",
                                         rank_info['ep_rank'], self.strategy.ep_size, com_buff=com_buff,
                                         fwd_cost=cost, bwd_cost=cost, global_rank=args.rank,
                                         net=self.strategy.ep_net, size_bytes=comm_size,
                                         group_kind="ep", comm_stage="Combine_EP"))
            state.comm_order += 1
            route_size = self.topk * self._per_rank_seq() * (4 + 4 + 4)
            route_cost = self.system.compute_net_op_time(
                "alltoallv", route_size, self.strategy.ep_size,
                net=self.strategy.ep_net, strategy=self.strategy,
                group_kind="ep", comm_stage="Combine_route_grad_A2AV")
            self.layers.append(alltoallv(
                f"{state.comm_order}-{model_info}-ep_group:{rank_info['ep_group_id']}-route_grad",
                rank_info['ep_rank'], self.strategy.ep_size,
                com_buff=com_buff, fwd_cost=0, bwd_cost=route_cost,
                global_rank=args.rank, net=self.strategy.ep_net,
                size_bytes=route_size, group_kind="ep",
                comm_stage="Combine_route_grad_A2AV"))
            state.comm_order += 1

        #permutate2 and combine
        unpermutate2_and_combine_mem_accessed = (
            self.act_size_before_combined + self.act_size_after_combined
        ) * self.dtype_to_element_size[self.strategy.dtype]
        unpermutate2_read = self.act_size_before_combined * self.dtype_to_element_size[self.strategy.dtype]
        unpermutate2_write = self.act_size_after_combined * self.dtype_to_element_size[self.strategy.dtype]
        fwd_compute_time = _layout_kernel_time(
            self.system, "unpermutate2_and_combine", unpermutate2_read,
            unpermutate2_write, "fwd", self.call_stk)
        bwd_mem_time = _layout_kernel_time(
            self.system, "unpermutate2_and_combine", unpermutate2_read,
            unpermutate2_write, "bwd", self.call_stk)
        bwd_grad_w_accessed_mem = 0
        bwd_grad_act_accessed_mem = bwd_mem_time
        bwd_grad_act_time = bwd_mem_time
        bwd_grad_w_time = self.system.compute_end2end_time(0, bwd_grad_w_accessed_mem)
        self.layers.append(AtomModel(fwd_cost=fwd_compute_time,
                                 bwd_cost=bwd_grad_act_time+bwd_grad_w_time,
                                 specific_name='unpermutate2_and_combine'))
        
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff)

    @property
    def act_size_before_combined(self):
        # Combine a2a is symmetric with dispatch: it moves the topk-expanded
        # expert output back to the source ranks, so the payload is the
        # per-rank token count x topk x hidden (mirror of permuted_act_size),
        # NOT the plain input token x hidden (which would under-size by topk).
        # Per-rank full seq (not CP-split) — same fix as dispatch.
        batch_size = self.input_info.tensors[0].size(0) if self.input_info.tensors[0].ndim == 3 else 1
        seq_len = self._per_rank_seq()
        hidden_size = self.input_info.tensors[0].size(-1)
        token_num = self.topk * batch_size * seq_len
        if self.moe_pad_expert_input_to_capacity:
            token_num = math.ceil(token_num/self.expert_num) * self.expert_num * self.capacity
        return token_num * hidden_size

    def _per_rank_seq(self):
        """Per-rank full sequence (not CP-split): seq_len // (cp//td).
        See Permutation._per_rank_seq — same fix, UnPermutation combine payload."""
        td = getattr(self.strategy, 'mxx_trunk_cp_divisor', None)
        if td is None:
            td = 1
        per_rank_cp = max(1, self.strategy.cp_size // td)
        return max(1, self.strategy.seq_len // per_rank_cp)

    @property
    def act_size_after_combined(self):
        # only consider balanced case
        act_size = self.output_info_.tensors[0].numel()
        return act_size
    
    def _pre_op(self):
        super()._pre_op()
        if not self.strategy.dispatch_probs:
            assert len(self.input_info.tensors) == 2, "dispatch_probs=False requires two inputs in Permutation, [x, probs]"
        else:
            assert len(self.input_info.tensors) == 1, "dispatch_probs=True requires one inputs in Permutation, [x]"
    def set_ori_shape(self, shape):
        self.ori_shape = shape

    def create_output_info(self):
        # recover the original input
        assert self.output_info_ is None
        assert self.ori_shape is not None
        output_info = InputOutputInfo(tensors=[TensorSize(shape=self.ori_shape)])
        # print('-- unpermute output_info', output_info)
        return output_info

    def _comp_leaf_intra_net_info(self):
        if self.strategy.etp_size > 1:
            comm_size = (
                self.act_size_before_combined
                * self.dtype_to_element_size[self.strategy.dtype]
                * self.strategy.etp_size
            )
            # fwd
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                "reduce_scatter",
                comm_size,
                comm_num=self.strategy.etp_size,
                net=self.strategy.etp_net,
                strategy=self.strategy,
                group_kind="etp",
                comm_stage="Combine_FWD_ETP"
            )
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                "all_gather",
                comm_size,
                comm_num=self.strategy.etp_size,
                net=self.strategy.etp_net,
                strategy=self.strategy,
                group_kind="etp",
                comm_stage="Combine_BWD_ETP"
            )

        # all2all on ep group
        if self.strategy.ep_size > 1:
            comm_size = (
                self.act_size_before_combined
                * self.dtype_to_element_size[self.strategy.dtype]
            )
            main_a2a_op = ("alltoallv" if self.strategy.moe_variable_collectives
                           else "all2all")
            # fwd
            self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                main_a2a_op,
                comm_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
                strategy=self.strategy,
                group_kind="ep",
                comm_stage="Combine_FWD_EP"
            )
            # bwd
            self._cost_info.bwd_grad_act_net_time += self.system.compute_net_op_time(
                main_a2a_op,
                comm_size,
                comm_num=self.strategy.ep_size,
                net=self.strategy.ep_net,
                strategy=self.strategy,
                group_kind="ep",
                comm_stage="Combine_BWD_EP"
            )
            route_size = self.topk * self._per_rank_seq() * (4 + 4 + 4)
            self._cost_info.bwd_grad_act_net_time += \
                self.system.compute_net_op_time(
                    "alltoallv", route_size,
                    comm_num=self.strategy.ep_size, net=self.strategy.ep_net,
                    strategy=self.strategy, group_kind="ep",
                    comm_stage="Combine_route_grad_A2AV")
        if self.enable_recompute:
            self._cost_info.recompute_net_time = self._cost_info.fwd_net_time

    def _comp_leaf_act_info_impl(self):
        """
        Mainly layout operators, ignore for now
        """
        # HACK(sherry): the weighted_probs is fused in SiluOP.
        # if dispatch_probs=True, no cache.
        # if dispatch_probs=False, cache the unpermute_before_hidden_states and probs (for mul op).
        if self.strategy.dispatch_probs:
            self._act_info.activation_mem_cache = 0
            self._act_info.fwd_peak_mem_no_cache = max(self.act_size_before_combined, self.act_size_after_combined) * self.element_size
            self._act_info.bwd_peak_mem_no_cache = 0
        else:
            # mul
            self._act_info.activation_mem_cache =  self.act_size_before_combined * self.element_size # Cache hidden states, probs are cache in Permutation
            self._act_info.fwd_peak_mem_no_cache = self.act_size_before_combined * self.element_size + self.act_size_after_combined * self.element_size
            self._act_info.bwd_peak_mem_no_cache = self.act_size_before_combined * self.element_size + self.act_size_after_combined * self.element_size
        # HACK(sherry)

    def _comp_leaf_model_info_impl(self):
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    def _comp_leaf_flops_info(self):
        """
        Mainly layout operators, ignore for now
        """
        self._compute_info.fwd_flops = 0
        self._compute_info.recompute_flops = 0
        self._compute_info.bwd_grad_act_flops = 0
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_mem_accessed_info(self):
        """
        4.Unpermuation2:
        - no (padding and drop):
          - 通过argsort的sorted_indices反向unpermutate,然后根据probs进行combine，但是drop的没有残差连接
        - padding and drop：
          - 通过final_indices(token indices, [E, C])和scatter_add实现恢复和combine weight

        1.permutate1 for ep all2all
        2.permutate2 for mlp compute
        3.combine scores
        """
        # pylint: disable=invalid-name
        unpermutate1_passes = _layout_logical_passes(
            self.system, "unpermute1", self.local_expert_num)
        permutate1_mem_accessed = ( # none-fused: contiguous memory(drop_and_pad) or sort_chunks_by_idxs
            2 * self.act_size_before_combined * unpermutate1_passes
        ) * self.dtype_to_element_size[self.strategy.dtype]
        
        permutate2_and_combine_mem_accessed = ( # fused-op: combine permuted_features by probs and scatter_add
            self.act_size_before_combined + self.act_size_after_combined
        ) * self.dtype_to_element_size[self.strategy.dtype]

        self._compute_info.fwd_accessed_mem = (
            permutate1_mem_accessed + permutate2_and_combine_mem_accessed
        )
        self._compute_info.bwd_grad_act_accessed_mem = (
            permutate1_mem_accessed + permutate2_and_combine_mem_accessed
        )
        self._compute_info.bwd_grad_w_accessed_mem = 0

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )
        # pylint: enable=invalid-name

    def _comp_cost_info(self):
        # Keep perf-side parent cost aligned with the simulator trace.
        # `UnPermutation` executes two layout kernels (`unpermute1`,
        # `unpermutate2_and_combine`) as separate leaf ops. Summing the two
        # kernel times matches simulator timing better than aggregating the
        # total bytes into one memory-bound estimate, because the bandwidth
        # model includes a fixed launch latency per kernel.
        unpermutate1_passes = _layout_logical_passes(
            self.system, "unpermute1", self.local_expert_num)
        unpermutate1_mem_accessed = (
            2 * self.act_size_before_combined * unpermutate1_passes
        ) * self.dtype_to_element_size[self.strategy.dtype]
        unpermutate2_and_combine_mem_accessed = (
            self.act_size_before_combined + self.act_size_after_combined
        ) * self.dtype_to_element_size[self.strategy.dtype]

        unpermutate1_read = unpermutate1_mem_accessed // 2
        unpermutate1_write = unpermutate1_mem_accessed // 2
        unpermutate2_read = (
            self.act_size_before_combined
            * self.dtype_to_element_size[self.strategy.dtype])
        unpermutate2_write = (
            self.act_size_after_combined
            * self.dtype_to_element_size[self.strategy.dtype])

        self._cost_info.fwd_compute_time = (
            _layout_kernel_time(
                self.system, "unpermute1", unpermutate1_read,
                unpermutate1_write, "fwd", self.call_stk)
            + _layout_kernel_time(
                self.system, "unpermutate2_and_combine", unpermutate2_read,
                unpermutate2_write, "fwd", self.call_stk))
        self._cost_info.bwd_grad_act_time = (
            _layout_kernel_time(
                self.system, "unpermute1", unpermutate1_read,
                unpermutate1_write, "bwd", self.call_stk)
            + _layout_kernel_time(
                self.system, "unpermutate2_and_combine", unpermutate2_read,
                unpermutate2_write, "bwd", self.call_stk))
        self._cost_info.bwd_grad_w_time = 0
        self._cost_info.recompute_compute_time = (
            self._cost_info.fwd_time if self.enable_recompute else 0
        )


class GroupLinearCol(GroupLinearBase):
    """Multi Expert Linear Layer, Suport column parallelism"""

    def __init__(
        self,
        layer_idx,
        input_size: int,
        output_size: int,
        local_expert_num: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        mode:str,
        strategy: StrategyConfig,
        system: SystemConfig,
        is_last_recompute: bool = False,
        use_variance_tail_model: bool = False,
        specific_name: str = 'GroupLinearCol',
    ) -> None:
        super().__init__(local_expert_num, input_size, output_size, strategy, system,
                         specific_name)
        assert mode in ['parallel', 'serial']
        assert output_size % self.strategy.etp_size == 0
        self.layer_idx = layer_idx
        self.local_expert_num = local_expert_num
        self.input_size = input_size
        self.output_size = output_size // self.strategy.etp_size
        self.use_bias = use_bias  # for now unless
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.is_last_recompute = is_last_recompute
        self.use_variance_tail_model = self.use_variance_tail_model or use_variance_tail_model
        
        if self.is_last_recompute and self.enable_recompute:
            self.set_variance_node(True)
        if self.strategy.fp8:
            self.w_dtype = "fp8"
            self.a_dtype = "fp8"
        else:
            self.w_dtype = self.strategy.dtype
            self.a_dtype = self.strategy.dtype

        self.w_element_size = self.dtype_to_element_size[self.w_dtype]
        self.a_element_size = self.dtype_to_element_size[self.a_dtype]

        if mode == "serial":
            import types
            for i in range(self.local_expert_num):
                setattr(self, f"linear_{i}", LinearCol(layer_idx=layer_idx,
                                                    input_size=input_size, 
                                                    output_size=output_size,
                                                    use_bias=use_bias,
                                                    has_cached_inputs=False,
                                                    enable_recompute=enable_recompute,
                                                    strategy=strategy,
                                                    system=system)
                )   
            def forward(self, input_output_info: InputOutputInfo, path_debug_context:PathDebugContext):
                input = simu_ops.split(input_output_info.tensors[0], self.local_expert_num, 0)
                out = []
                for i in range(self.local_expert_num):
                    linear_i = getattr(self, f"linear_{i}")
                    x = simu_ops.unsqueeze(input[i], 0)
                    x = linear_i(x, path_debug_context)
                    out.append(simu_ops.squeeze(x, 0))
                out = simu_ops.cat(out, 0)
                return out
            # Methods to bind functions as instances
            self.forward = types.MethodType(forward, self)

            
    def prefill(self, args, call_stk='', com_buff=None):
        # tp comm is in Permuation
        self.call_stk = call_stk + self.call_stk
        # SwiGLU gate/up projections are separate grouped GEMM launches, and
        # the measured kernel stream splits each projection's backward into
        # separate bwd_grad_act and bwd_grad_w kernels (bwd_act ~3.5x fwd,
        # bwd_w ~1x fwd on the reference device). Emit them as distinct
        # AtomModels so event count, per-kernel cost, and the serial chain
        # match the measured kernel structure.
        fwd = self._cost_info.fwd_compute_time / 2
        bwd_act = self._cost_info.bwd_grad_act_time / 2
        bwd_w = self._cost_info.bwd_grad_w_time / 2
        for projection in ('Gate', 'Up'):
            self.layers.append(AtomModel(
                fwd_cost=fwd, bwd_cost=0.0,
                specific_name=f'GroupGemmTraining_{projection}'))
            self.layers.append(AtomModel(
                fwd_cost=0.0, bwd_cost=bwd_act,
                specific_name=f'GroupGemmTraining_{projection}_bwd_act',
                skip_recompute=True))
            self.layers.append(AtomModel(
                fwd_cost=0.0, bwd_cost=bwd_w,
                specific_name=f'GroupGemmTraining_{projection}_bwd_w',
                skip_recompute=True))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff)

    @property
    def micro_input_tensor(self):
        assert self.input_info is not None, "Please set input info"
        # [ep_size * local_expert_num, H]
        token_num = self.input_info.tensors[0].size(0)
        hidden_size = self.input_info.tensors[0].size(1)
        return TensorSize(shape = [token_num, hidden_size], dtype=self.input_info.tensors[0].dtype)
    
    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [ep_size * local_expert_num, H]
        # token_num = self.input_info.tensors[1].size(0)
        # hidden_size = self.input_info.tensors[1].size(1)
        token_num = self.input_info.tensors[0].size(0)
        hidden_size = self.input_info.tensors[0].size(1)
        # if self.strategy.enable_sequence_parallel:
        #     hidden_size *= self.strategy.etp_size
        return token_num * hidden_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        token_num = self.output_info_.tensors[0].size(0)
        return token_num * self.output_size

    def create_output_info(self):
        token_num = self.input_info.tensors[0].size(0)
        origin_input_info = self.input_info.tensors[1:]
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(token_num, self.output_size))]
            + origin_input_info
        )
        return output_info
    
    def _pre_op(self):
        hidden_size = self.input_info.tensors[0].size(1)
        assert self.input_size == hidden_size

    def _comp_leaf_intra_net_info(self):
        # tp comm is in Permuation
        pass

    def _comp_leaf_act_info_impl(self):
        
        self._act_info.activation_mem_cache = (
            self.micro_hidden_state_size * self.a_element_size # fp8
        )
        if self.has_cached_inputs or self.offload_inputs:
            self._act_info.activation_mem_cache = 0
        weight_size = (
            self.local_expert_num
            * self.input_size
            * self.output_size
            * self.w_element_size # fp8
        )
        grad_size = (
            self.local_expert_num
            * self.input_size
            * self.output_size
            * self.dtype_to_element_size['fp32'] # fp8
        )
        input_size = self.micro_hidden_state_size * self.a_element_size # fp8
        output_size = self.micro_output_grad_size * self.element_size   # bf16
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + (0 if self.strategy.use_accm_weight else weight_size)
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size + (grad_size if self.strategy.fp8 else 0) + (input_size if self.offload_inputs else 0)

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.local_expert_num * self.input_size * self.output_size
        self._model_info.moe_weight_numel = weight_numel * self.strategy.ep_size * self.strategy.etp_size # Statistics the parameters of all etp ranks and ep ranks
        self._model_info.moe_weight_bytes = weight_numel * self.w_element_size # fp8
        self._model_info.moe_grad_bytes = weight_numel * self.main_grad_element_size
        self._model_info.moe_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        
        optimizer_group_size = self.strategy.fsdp_moe_group_size
        if self.strategy.zero_state >= 1:
            self._model_info.moe_state_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 2:
            self._model_info.moe_grad_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 3:
            self._model_info.moe_weight_bytes /= optimizer_group_size
        self._record_te_dummy_wgrad_shape(grouped_linear=True)

    def _comp_leaf_flops_info(self):
        token_num = self.input_info.tensors[0].size(0)
        base_flops = 2 * token_num * self.input_size * self.output_size
        self._compute_info.fwd_flops = base_flops
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = base_flops
        self._compute_info.bwd_grad_w_flops = base_flops

    def _comp_leaf_mem_accessed_info(self):
        weight_size = (
            self.input_size
            * self.output_size
            * self.w_element_size  # fp8
            * self.local_expert_num
        )
        input_size = self.micro_hidden_state_size * self.a_element_size # fp8
        output_size = self.micro_output_grad_size * self.element_size # bf16

        self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
        self._compute_info.bwd_grad_act_accessed_mem = (
            weight_size + output_size + input_size
        )
        main_grad_size = self.input_size * self.output_size * 4 # fp32
        self._compute_info.bwd_grad_w_accessed_mem = (
            output_size + input_size + weight_size + (main_grad_size if self.strategy.use_fused_grad_accumulation else 0)
        )

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

        # SwiGLU gate and up projections are two independent grouped GEMMs.
        # They share the logical concatenated output but each reads the input;
        # dX likewise materializes two partial gradients before accumulation.
        # Account for that compulsory second input pass without using a
        # profiler-derived efficiency multiplier.
        self._compute_info.fwd_accessed_mem += input_size
        self._compute_info.bwd_grad_act_accessed_mem += input_size
        self._compute_info.bwd_grad_w_accessed_mem += input_size
        if self.enable_recompute:
            self._compute_info.recompute_accessed_mem += input_size

    def _comp_cost_info(self):
        if self.strategy.fp8:
            self._comp_cost_info_impl(
                fwd_op="fp8_group_linear_col",
                bwd_grad_act_op="fp8_group_linear_col",
                bwd_grad_w_op="fp8_group_linear_col",
                enable_recompute=self.enable_recompute,
            )
        else:
            token_num = int(self.input_info.tensors[0].size(0))
            half_output = self.output_size // 2
            assert self.output_size % 2 == 0, (
                "SwiGLU grouped column output must split into gate/up halves")
            avg_m = max(1, math.ceil(token_num / self.local_expert_num))
            desc_base = (
                f"ng={self.local_expert_num}, M={avg_m}, N={half_output}, "
                f"K={self.input_size}, dtype={self.a_dtype}, "
                f"out_dtype={self.strategy.dtype}")
            half_flops = self._compute_info.fwd_flops / 2

            input_bytes = token_num * self.input_size * self.a_element_size
            weight_bytes = (self.local_expert_num * self.input_size
                            * half_output * self.w_element_size)
            output_bytes = token_num * half_output * self.element_size
            fwd_mem = input_bytes + weight_bytes + output_bytes
            bwd_act_mem = weight_bytes + output_bytes + input_bytes
            bwd_w_mem = input_bytes + output_bytes + weight_bytes

            def two_kernel_time(stage, mem_bytes, suffix):
                class_key, path_key = self.get_cost_keys()
                return sum(
                    self.system.compute_op_accuracy_time(
                        "group_linear_col", half_flops,
                        shape_desc=(f"{desc_base}, stage={stage}, "
                                    f"projection={projection}{suffix}"),
                        accessed_mem=mem_bytes, stage=stage,
                        class_key=class_key, path_key=path_key)
                    for projection in ("gate", "up")
                )

            self._cost_info.fwd_compute_time = two_kernel_time(
                "fwd", fwd_mem, ", accumulate=False")
            self._cost_info.bwd_grad_act_time = two_kernel_time(
                "bwd_grad_act", bwd_act_mem, ", accumulate=True")
            self._cost_info.bwd_grad_w_time = two_kernel_time(
                "bwd_grad_w", bwd_w_mem, ", accumulate=True")
            self._cost_info.recompute_compute_time = (
                self._cost_info.fwd_compute_time
                if self.enable_recompute else 0)


    def extra_repr(self) -> str:
        repr_info = (
            f"input_size={self.input_size},"
            f"output_size={self.output_size},"
            f"local_expert_num={self.local_expert_num},"
            f"use_bias={self.use_bias}"
        )
        return repr_info


class GroupLinearRow(GroupLinearBase):
    """Multi Expert Linear Layer, Suport row parallelism"""

    def __init__(
        self,
        layer_idx,
        input_size: int,
        output_size: int,
        local_expert_num: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        mode:str,
        strategy: StrategyConfig,
        system: SystemConfig,
        is_last_recompute: bool = False,
        use_variance_tail_model: bool = False,
        specific_name: str = 'GroupLinearRow',
    ) -> None:
        super().__init__(local_expert_num, input_size, output_size, strategy, system,
                         specific_name)
        assert mode in ['parallel', 'serial']
        assert input_size % self.strategy.etp_size == 0
        self.layer_idx = layer_idx
        self.local_expert_num = local_expert_num
        self.input_size = input_size // self.strategy.etp_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.is_last_recompute = is_last_recompute
        self.use_variance_tail_model = self.use_variance_tail_model or use_variance_tail_model
        if self.is_last_recompute and self.enable_recompute:
            self.set_variance_node(True)
        if self.strategy.fp8:
            self.w_dtype = "fp8"
            self.a_dtype = "fp8"
        else:
            self.w_dtype = self.strategy.dtype
            self.a_dtype = self.strategy.dtype

        self.w_element_size = self.dtype_to_element_size[self.w_dtype]
        self.a_element_size = self.dtype_to_element_size[self.a_dtype]

        if mode == "serial":
            import types
            for i in range(self.local_expert_num):
                setattr(self, f"linear_{i}", LinearCol(layer_idx=layer_idx,
                                                    input_size=input_size, 
                                                    output_size=output_size,
                                                    use_bias=use_bias,
                                                    has_cached_inputs=False,
                                                    enable_recompute=enable_recompute,
                                                    strategy=strategy,
                                                    system=system)
                )   
            def forward(self, input_output_info: InputOutputInfo, path_debug_context:PathDebugContext):
                input = simu_ops.split(input_output_info.tensors[0], self.local_expert_num, 0)
                out = []
                for i in range(self.local_expert_num):
                    linear_i = getattr(self, f"linear_{i}")
                    x = simu_ops.unsqueeze(input[i], 0)
                    x = linear_i(x, path_debug_context)
                    out.append(simu_ops.squeeze(x, 0))
                out = simu_ops.cat(out, 0)
                return out
            # Methods to bind functions as instances
            self.forward = types.MethodType(forward, self)
            
    def forward(self, input_output_info: InputOutputInfo, path_debug_context:PathDebugContext):
        input = simu_ops.split(input_output_info.tensors[0], self.local_expert_num, 0)
        out = []
        for i in range(self.local_expert_num):
            linear_i = getattr(self, f"linear_{i}")
            x = simu_ops.unsqueeze(input[i], 0)
            x = linear_i(x, path_debug_context)
            out.append(simu_ops.squeeze(x, 0))
        out = simu_ops.cat(out, 0)
        return out
    
    def prefill(self, args, call_stk='', com_buff=None):
        # tp comm is in UnPermuation
        self.call_stk = call_stk + self.call_stk
        # The measured kernel stream splits the row backward into separate
        # bwd_grad_act and bwd_grad_w kernels; emit them as distinct
        # AtomModels (bwd_act ~3.5x fwd, bwd_w ~1x fwd on the reference
        # device) so event count and per-kernel cost match the target.
        self.layers.append(AtomModel(
            fwd_cost=self._cost_info.fwd_compute_time, bwd_cost=0.0,
            specific_name='GroupGemmTraining_Down'))
        self.layers.append(AtomModel(
            fwd_cost=0.0, bwd_cost=self._cost_info.bwd_grad_act_time,
            specific_name='GroupGemmTraining_Down_bwd_act',
            skip_recompute=True))
        self.layers.append(AtomModel(
            fwd_cost=0.0, bwd_cost=self._cost_info.bwd_grad_w_time,
            specific_name='GroupGemmTraining_Down_bwd_w',
            skip_recompute=True))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff)

    @property
    def micro_input_tensor(self):
        assert self.input_info is not None, "Please set input info"
        # [ep_size * local_expert_num, H]
        token_num = self.input_info.tensors[0].size(0)
        hidden_size = self.input_info.tensors[0].size(1)
        return TensorSize(shape = [token_num, hidden_size], dtype=self.input_info.tensors[0].dtype)
    
    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        # [ep_size * local_expert_num, H]
        token_num = self.input_info.tensors[0].size(0)
        hidden_size = self.input_info.tensors[0].size(1)
        return token_num * hidden_size

    @property
    def micro_output_grad_size(self):
        # [B, S, H]
        token_num = self.output_info_.tensors[0].size(0)
        hidden_size = self.output_info_.tensors[0].size(1)
        # hidden_size = self.output_info.tensors[0].size(2)
        return token_num * hidden_size

    def create_output_info(self):
        token_num = self.input_info.tensors[0].size(0)
        origin_input_info = self.input_info.tensors[1:]

        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(token_num, self.output_size))]
            + origin_input_info
        )
        return output_info

    def _pre_op(self):
        hidden_size = self.input_info.tensors[0].size(1)
        assert self.input_size == hidden_size, f"input_size {self.input_size} != hidden_size {hidden_size}"

    def _comp_leaf_intra_net_info(self):
        # tp comm is in UnPermuation
        pass

    def _comp_leaf_act_info_impl(self):
        self._act_info.activation_mem_cache = (
            self.micro_hidden_state_size * self.a_element_size # fp8
        )
        if self.has_cached_inputs:
            self._act_info.activation_mem_cache = 0
        weight_size = (
            self.input_size
            * self.output_size
            * self.local_expert_num
            * self.w_element_size # fp8
        )
        grad_size =  (
            self.input_size
            * self.output_size
            * self.local_expert_num
            * self.dtype_to_element_size['fp32'] # fp8
        )
        input_size = self.micro_hidden_state_size * self.a_element_size # fp8
        output_size = self.micro_output_grad_size * self.element_size # bf16
        self._act_info.fwd_peak_mem_no_cache = input_size + output_size + (0 if self.strategy.use_accm_weight else weight_size)
        self._act_info.bwd_peak_mem_no_cache = input_size + output_size +  (grad_size if self.strategy.fp8 else 0)

    def _comp_leaf_model_info_impl(self):
        weight_numel = self.input_size * self.output_size * self.local_expert_num
        self._model_info.moe_weight_numel = weight_numel * self.strategy.ep_size * self.strategy.etp_size # Statistics the parameters of all etp ranks and ep ranks
        self._model_info.moe_weight_bytes = weight_numel * self.w_element_size # fp8
        self._model_info.moe_grad_bytes = weight_numel * self.main_grad_element_size
        self._model_info.moe_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * weight_numel
        )
        
        optimizer_group_size = self.strategy.fsdp_moe_group_size
        if self.strategy.zero_state >= 1:
            self._model_info.moe_state_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 2:
            self._model_info.moe_grad_bytes /= optimizer_group_size
        if self.strategy.zero_state >= 3:
            self._model_info.moe_weight_bytes /= optimizer_group_size
        self._record_te_dummy_wgrad_shape(grouped_linear=True)

    def _comp_leaf_flops_info(self):
        token_num = self.input_info.tensors[0].size(0)
        base_flops = 2 * token_num * self.input_size * self.output_size
        self._compute_info.fwd_flops = base_flops
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = base_flops
        self._compute_info.bwd_grad_w_flops = base_flops

    def _comp_leaf_mem_accessed_info(self):
        weight_size = (
            self.input_size
            * self.output_size
            * self.w_element_size # fp8
            * self.local_expert_num
        )
        input_size = self.micro_hidden_state_size * self.a_element_size  # fp8
        output_size = self.micro_output_grad_size * self.element_size   # bf16

        self._compute_info.fwd_accessed_mem = input_size + weight_size + output_size
        self._compute_info.bwd_grad_act_accessed_mem = (
            weight_size + output_size + input_size
        )
        main_grad_size = self.input_size * self.output_size * 4 # fp32
        self._compute_info.bwd_grad_w_accessed_mem = (
            output_size + input_size + weight_size + (main_grad_size if self.strategy.use_fused_grad_accumulation else 0)
        )

        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )

    def _comp_cost_info(self):
        if self.strategy.fp8:
            self._comp_cost_info_impl(
                fwd_op="fp8_group_linear_row",
                bwd_grad_act_op="fp8_group_linear_row",
                bwd_grad_w_op="fp8_group_linear_row",
                enable_recompute=self.enable_recompute,
            )
        else:
            self._comp_cost_info_impl(
                fwd_op="group_linear_row",
                bwd_grad_act_op="group_linear_row",
                bwd_grad_w_op="group_linear_row",
                enable_recompute=self.enable_recompute,
            )

    def extra_repr(self) -> str:
        repr_info = (
            f"input_size={self.input_size},"
            f"output_size={self.output_size},"
            f"local_expert_num={self.local_expert_num},"
            f"use_bias={self.use_bias}"
        )
        return repr_info
#endregion 

#region ----------------- Composite module ----------------
class QuantizedGroupLinearCol(MetaModule):
    def __init__(self,
        layer_idx,
        input_size: int,
        output_size: int,
        local_expert_num: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        mode:str,
        strategy: StrategyConfig,
        system: SystemConfig,
        is_last_recompute: bool = False,
        use_variance_tail_model: bool = False,
        ):
        super().__init__(strategy, system)
        quantizer_recompute = False if strategy.cache_groupgemm_col_fp8_inputs else enable_recompute
        self.quantizer = Float8Quantizer(enable_recompute=quantizer_recompute, strategy=strategy, system=system)
        enable_cahce_bf16_inputs = not self.strategy.cache_groupgemm_col_fp8_inputs
        if enable_cahce_bf16_inputs:
            self.quantizer.offload_inputs = self.strategy.offload_groupgemm_col_inputs  # the quantizer can perform offload When the input of bf16 needs to be cached
       
        self.linear = GroupLinearCol(
            layer_idx,
            input_size,
            output_size,
            local_expert_num,
            use_bias,
            has_cached_inputs,
            enable_recompute,
            mode,
            strategy,
            system,
            is_last_recompute,
            use_variance_tail_model,
        )
    def forward(self, hidden_states, path_debug_context=None):
        hidden_states = self.quantizer(hidden_states, path_debug_context)
        hidden_states = self.linear(hidden_states, path_debug_context)
        return hidden_states
    

class QuantizedGroupLinearRow(MetaModule):
    def __init__(self,
        layer_idx,
        input_size: int,
        output_size: int,
        local_expert_num: int,
        use_bias: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        mode:str,
        strategy: StrategyConfig,
        system: SystemConfig,
        if_first_recompute: bool = False,
        is_last_recompute: bool = False,
        use_variance_tail_model: bool = False,
    ):
        super().__init__(strategy, system)
        self.quantizer = Float8Quantizer(enable_recompute=enable_recompute, strategy=strategy, system=system)
        self.linear = GroupLinearRow(
                    layer_idx,
                    input_size,
                    output_size,
                    local_expert_num,
                    use_bias,
                    has_cached_inputs,
                    enable_recompute,
                    mode,
                    strategy,
                    system,
                    is_last_recompute,
                    use_variance_tail_model,
                )

    def forward(self, hidden_states, path_debug_context=None):
        hidden_states = self.quantizer(hidden_states, path_debug_context)
        hidden_states = self.linear(hidden_states, path_debug_context)
        return hidden_states

class ExpertMLP(MetaModule):
    """Expert MLP Layer"""

    def __init__(self, 
                 layer_idx, 
                 config:ModelConfig, 
                 enable_recompute, 
                 mlp_recompute:MLPRecomputeConfig,
                 strategy:StrategyConfig, 
                 system:SystemConfig, 
                 router_preparation=None,
                 specific_name='') -> None:
        super().__init__(strategy, system, specific_name)
        self.layer_idx = layer_idx
        self.config = config
        self.strategy = strategy
        self.system = system
        self.enable_recompute = enable_recompute  # for old version 
        self.expert_num = self.config.expert_num
        self.topk = self.config.topk
        self.local_expert_num = self.config.expert_num // self.strategy.ep_size
        ffn_hidden_size = (self.config.moe_ffn_hidden_size if self.config.moe_ffn_hidden_size is not None 
                        else self.config.intermediate_size)
        intermediate_size = (
            2 * ffn_hidden_size
            if self.config.use_swiglu
            else ffn_hidden_size
        )
        self.mlp_recompute = mlp_recompute
        megatron_moe = mlp_recompute.megatron_moe
        megatron_moe_act = mlp_recompute.megatron_moe_act and not megatron_moe
        self.shared_expert = None
        if getattr(self.config, "moe_shared_expert_intermediate_size", None) is not None:
            shared_expert_recompute = deepcopy(mlp_recompute)
            shared_expert_recompute.megatron_layernorm = False
            self.shared_expert = MLP(
                    layer_idx=f"{layer_idx}-shareExpert",
                    config=self.config,
                    enable_recompute=enable_recompute, # for old version 
                    mlp_recompute_conf=shared_expert_recompute,
                    strategy=strategy,
                    system=system,
                    intermediate_size=self.config.moe_shared_expert_intermediate_size
                )

        GroupLinearCol_ = QuantizedGroupLinearCol if self.strategy.fp8 else GroupLinearCol
        GroupLinearRow_ = QuantizedGroupLinearRow if self.strategy.fp8 else GroupLinearRow
        
        self.router = Router(
                layer_idx=layer_idx,
                hidden_size=self.config.hidden_size,
                expert_num=self.config.expert_num,
                topk=self.topk,
                moe_dispatcher_policy=self.strategy.moe_dispatcher_policy,
                has_cached_inputs=mlp_recompute.megatron_layernorm,
                enable_recompute=mlp_recompute.router_recompute or mlp_recompute.megatron_layernorm or megatron_moe,
                is_last_recompute=mlp_recompute.megatron_layernorm,
                use_variance_tail_model=mlp_recompute.megatron_layernorm,
                strategy=strategy,
                system=system,
                specific_name='MoERouter',
            )
        # Optional model-specific implementation stages for the router path.
        #
        # The implementation has a real dependency edge at the metadata
        # exchange: local histogram/prefix work produces the metadata payload,
        # ``Permutation`` exchanges that payload, and only then can the
        # inverse/gate and route-index stages consume the received metadata.
        # Keep the two partitions as separate children so the DES preserves
        # that edge in its event order. ``router_preparation`` remains an
        # accepted single-module input for callers that do not need the split.
        if isinstance(router_preparation, (tuple, list)):
            if len(router_preparation) != 2:
                raise ValueError(
                    "router_preparation tuple must contain (pre_metadata, post_metadata)"
                )
            self.router_preparation_pre = router_preparation[0]
            self.router_preparation_post = router_preparation[1]
        else:
            self.router_preparation_pre = router_preparation
            self.router_preparation_post = None
        # Backward-compatible alias for code that introspects the old field.
        self.router_preparation = self.router_preparation_pre
        permutation_kwargs = dict(
                layer_idx=layer_idx,
                expert_num=self.expert_num,
                local_expert_num=self.local_expert_num,
                topk=self.topk,
                moe_pad_expert_input_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                capacity=self.config.capacity,
                moe_dispatcher_policy=self.strategy.moe_dispatcher_policy,
                has_cached_inputs=False,
                enable_recompute=mlp_recompute.permutation_recompute or megatron_moe,
                strategy=strategy,
                system=system,
            )
        # Keep the metadata exchange and activation dispatch as separate
        # children. This lets the router post-metadata stages sit between the
        # fixed metadata all-to-all and the variable activation all-to-all-v.
        self.permutation_pre = Permutation(
            **permutation_kwargs, stage_partition='pre_metadata')
        self.permutation_post = Permutation(
            **permutation_kwargs, stage_partition='post_metadata')
        # Backward-compatible alias for callers that inspect ``permutation``.
        self.permutation = self.permutation_pre
        self.group_linear1 = GroupLinearCol_(
                layer_idx=layer_idx,
                input_size=self.config.hidden_size,
                output_size=intermediate_size,
                local_expert_num=self.local_expert_num,
                use_bias=False,
                has_cached_inputs=False,
                enable_recompute=mlp_recompute.linear_recompute or megatron_moe,
                mode=self.config.group_linear_mode,
                strategy=strategy,
                system=system,
                specific_name='MoEGroupGemmGateUp',
            )
        if self.strategy.fp8:
            # fp8
            if self.strategy.cache_groupgemm_col_fp8_inputs:
                self.group_linear1.linear.offload_inputs = self.strategy.offload_groupgemm_col_inputs
            else:
                self.group_linear1.quantizer.offload_inputs = self.strategy.offload_groupgemm_col_inputs
        else:
            # bf16
            self.group_linear1.offload_inputs = self.strategy.offload_groupgemm_col_inputs
        
        if self.config.use_swiglu:
            self.expert_activation_layer = Swiglu(
                    is_fused=self.strategy.use_fused_swiglu,
                    has_cached_inputs=False,
                    enable_recompute=mlp_recompute.linear_recompute or megatron_moe or megatron_moe_act,
                    strategy=strategy,
                    system=system,
                    is_weighted_silu= self.strategy.dispatch_probs
                )
        else:
            self.expert_activation_layer =Gelu(
                    has_cached_inputs=False,
                    enable_recompute=mlp_recompute.linear_recompute or megatron_moe or megatron_moe_act,
                    strategy=strategy,
                    system=system,
                )
        self.group_linear2 = GroupLinearRow_(
                layer_idx=layer_idx,
                input_size=ffn_hidden_size,
                output_size=self.config.hidden_size,
                local_expert_num=self.local_expert_num,
                has_cached_inputs=megatron_moe_act,
                enable_recompute=mlp_recompute.linear_recompute or megatron_moe or megatron_moe_act,
                is_last_recompute = True,
                use_variance_tail_model=megatron_moe_act,
                mode=self.config.group_linear_mode,
                use_bias=False,
                strategy=strategy,
                system=system,
                specific_name='MoEGroupGemmDown',
            )
        self.unpermutation = UnPermutation(
                layer_idx=layer_idx,
                expert_num=self.expert_num,
                local_expert_num=self.local_expert_num,
                topk=self.topk,
                moe_dispatcher_policy=self.strategy.moe_dispatcher_policy,
                has_cached_inputs=False,
                enable_recompute=mlp_recompute.permutation_recompute or megatron_moe,
                strategy=strategy,
                system=system,
            )
        # NOTE: MoE dispatch/combine quantization cost (MOE_DISPATCH / MOE_COMBINE)
        # is NOT created as standalone modules here to avoid double-counting
        # communication (Permutation + UnPermutation already model all2all costs).
        #
        # Design decision (see design_plan_remaining_ops_costmodel.md §6.0):
        #   Phase 3 方案 A — the quant FLOPS/memory-access formulas from
        #   MoEDispatchModule / MoECombineModule (moe_comm_module.py) should be
        #   extracted and added directly into Permutation.prefill() and
        #   UnPermutation.prefill(). The standalone classes exist as formula
        #   reference; they are not wired as children here.

        if (
            self.strategy.recompute_granularity == "selective_recompute"
            and mlp_recompute.megatron_layernorm
        ):
            self.router.is_breakpoints = True

        if self.unpermutation.enable_recompute and self.strategy.recompute_granularity == "selective_recompute":
            self.unpermutation.is_breakpoints = True

        full_moe_checkpoint = megatron_moe or (
            mlp_recompute.router_recompute and
            mlp_recompute.permutation_recompute and
            mlp_recompute.linear_recompute and
            (self.shared_expert.recompute_granularity == "full" if self.shared_expert else True)
        )
        if not full_moe_checkpoint:
            self.recompute_granularity = "submodule"
    
    def preprocess(self, input_info:InputOutputInfo):
        self.unpermutation.set_ori_shape(input_info.tensors[0].shape.copy())

    def forward(self, input_info:InputOutputInfo, path_debug_context:PathDebugContext):
        self.preprocess(input_info) 
        if self.shared_expert:
            shared_out = self.shared_expert(input_info, path_debug_context)
        probs = self.router(input_info, path_debug_context) # add router scores
        if self.router_preparation_pre is not None:
            self.router_preparation_pre(
                Input(tensors=[input_info.tensors[0], probs.tensors[0]]),
                path_debug_context)

        if self.strategy.dispatch_probs:
            self.permutation_pre(
                Input(tensors=[input_info.tensors[0], probs.tensors[0]]),
                path_debug_context)
            if self.router_preparation_post is not None:
                self.router_preparation_post(
                    Input(tensors=[input_info.tensors[0], probs.tensors[0]]),
                    path_debug_context)
            # The pre-partition's materialized buffer is an internal 2-D
            # route layout.  The post-partition models the remaining route
            # stages and returns the expert-shaped logical tensor, so keep the
            # public shape carrier at the original [B, S, H] boundary.
            permute_hidden_states = self.permutation_post(
                Input(tensors=[input_info.tensors[0], probs.tensors[0]]),
                path_debug_context)
            grou1_out = self.group_linear1(permute_hidden_states, path_debug_context)
            act_out = self.expert_activation_layer(Input(tensors=[grou1_out.tensors[0], probs.tensors[0]]), path_debug_context)
            group2_out = self.group_linear2(act_out, path_debug_context)
            out = self.unpermutation(group2_out, path_debug_context)

        else:
            self.permutation_pre(
                Input(tensors=[input_info.tensors[0], probs.tensors[0]]),
                path_debug_context)
            if self.router_preparation_post is not None:
                self.router_preparation_post(
                    Input(tensors=[input_info.tensors[0], probs.tensors[0]]),
                    path_debug_context)
            permute_hidden_states = self.permutation_post(
                Input(tensors=[input_info.tensors[0], probs.tensors[0]]),
                path_debug_context)
            grou1_out = self.group_linear1(permute_hidden_states, path_debug_context)
            act_out = self.expert_activation_layer(grou1_out, path_debug_context)
            group2_out = self.group_linear2(act_out, path_debug_context)
            out = self.unpermutation(Input(tensors=[group2_out.tensors[0], probs.tensors[0]]), path_debug_context)
           
        # FIXME(sherry):  add mul, routed_expert hidden_states * router scores
        if self.shared_expert:
            # return out + shared_out
            return AddFunction.apply(parent_model=self,
                                     enable_recompute=self.recompute_granularity == 'full_block',
                                     tensor_size1=out,
                                     tensor_size2=shared_out,
                                     path_debug_context=path_debug_context,
                                     name='SharedExpertAddFunction')
        return out 
       
    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        for layer in self.children_ordered_module:
            self.layers.append(layer)
            layer.prefill(args, self.call_stk, com_buff)
#endregion 
