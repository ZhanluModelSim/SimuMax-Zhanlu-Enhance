"""SWA (Sliding Window Attention) Roofline Cost Model.

Based on op_define.md MojoPagedPrefillSWAOp / MojoPagedDecodeSWAOp.
"""
from simumax.core.tensor import TensorSize
from simumax.core.base_struct import MetaModule, InputOutputInfo, PathDebugContext, ActivationInfo, AtomModel
from simumax.core.config import StrategyConfig, SystemConfig
from simumax.core.utils import format_model_info_microbatch_tag, get_rank_group


def swa_causal_mask_area(q_len: int, kv_len: int = 0, window: int = 1028) -> float:
    """Compute causal mask area for SWA attention.

    Source: op_define.md L477-485

    Args:
        q_len: query sequence length
        kv_len: existing KV cache length (0 for prefill/training)
        window: sliding window size (default 1028)

    Returns:
        Effective causal mask area (replaces S×S in standard attention FLOPS)
    """
    limit_i = window - kv_len - 1
    if limit_i < 0:
        return q_len * window
    if limit_i >= q_len - 1:
        return q_len * kv_len + q_len * (q_len + 1) // 2
    arithmetic = (limit_i + 1) * (kv_len + 1 + window) // 2
    constant = (q_len - (limit_i + 1)) * window
    return arithmetic + constant


class SWACoreAttention(MetaModule):
    """Sliding Window Attention — Roofline Cost Model.

    Corresponds to MojoPagedPrefillSWAOp in op_define.md (L548-575).
    Uses windowed causal mask area instead of full S×S attention,
    reducing compute complexity from O(S²) to O(S×W) for sequences longer than the window.
    """

    def __init__(
        self,
        head_size: int,
        head_num: int,
        kv_head_num: int,
        use_math_sdp: bool,
        use_flash_sdp: bool,
        has_cached_inputs: bool,
        enable_recompute: bool,
        strategy: StrategyConfig,
        system: SystemConfig,
        window_size: int = 1028,
        specific_name: str = 'SWAAttention',
        is_last_recompute: bool = False,
        use_variance_tail_model: bool = False,
    ) -> None:
        super().__init__(strategy, system, specific_name)
        self.use_math_sdp = use_math_sdp
        self.use_flash_sdp = use_flash_sdp
        self.window_size = window_size
        self.attention_sparse_ratio = 0.0  # SWA sparsity is inherent in windowed mask
        if self.strategy.tp_size > 1:
            assert head_num % self.strategy.tp_size == 0
            assert kv_head_num % self.strategy.tp_size == 0
            head_num = head_num / self.strategy.tp_size
            kv_head_num = kv_head_num / self.strategy.tp_size
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.head_size = head_size
        self.v_head_dim = head_size
        self.has_cached_inputs = has_cached_inputs
        self.enable_recompute = enable_recompute
        self.is_last_recompute = is_last_recompute
        self.use_variance_tail_model = self.use_variance_tail_model or use_variance_tail_model
        if self.is_last_recompute and self.enable_recompute:
            self.set_variance_node(True)
        # Flag: when True, CP A2A is handled externally (by HybridAttentionBlock),
        # so skip _append_cp_a2a_layers in prefill.
        self._skip_cp_a2a = False

    # ───  prefill  ───
    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        model_info = f"{format_model_info_microbatch_tag(args)}-name:{self.__class__.__name__}"
        rank_info = get_rank_group(args.rank, self.strategy)
        if not self._skip_cp_a2a:
            self._append_cp_a2a_layers(args, model_info, rank_info, com_buff=com_buff)
        self.layers.append(AtomModel(
            fwd_cost=self._cost_info.fwd_compute_time,
            bwd_cost=self._cost_info.bwd_grad_act_time + self._cost_info.bwd_grad_w_time,
            specific_name='SWAAttentionScore',
        ))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)

    # ───  properties  ───
    @property
    def micro_hidden_state_size(self):
        assert self.input_info is not None, "Please set input info"
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.input_info.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    @property
    def micro_output_grad_size(self):
        batch_size = self.output_info_.tensors[0].size(0)
        seq_len = self.output_info_.tensors[0].size(1)
        hidden_size = self.output_info_.tensors[0].size(2)
        return batch_size * seq_len * hidden_size

    def create_output_info(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        hidden_size = self.head_num * self.head_size
        output_info = InputOutputInfo(
            tensors=[TensorSize(shape=(batch_size, seq_len, hidden_size))]
        )
        return output_info

    def _pre_op(self):
        self._act_info.checkpoint_mem = self.micro_hidden_state_size * self.element_size

    # ───  shapes desc for efficiency lookup  ───
    def get_input_shapes_desc(self, stage):
        hidden_states = self.input_info.tensors[0]
        batch, seq_len = hidden_states.shape[:2]
        if self.strategy.cp_size > 1:
            seq_len = seq_len * self.strategy.cp_size
            head_num = self.head_num // self.strategy.cp_size
            kv_head_num = self.kv_head_num // self.strategy.cp_size
        else:
            head_num = self.head_num
            kv_head_num = self.kv_head_num
        qkv_contiguous = False if 's5000' in self.system.sys_name else True
        shape_str = (
            f'batch={int(batch)}, seq_len={int(seq_len)}, '
            f'head_num={int(head_num)}, kv_head_num={int(kv_head_num)}, '
            f'qk_head_dim={int(self.head_size)}, v_head_dim={int(self.v_head_dim)}, '
            f'window_size={int(self.window_size)}, '
            f'qkv_contiguous={qkv_contiguous}'
        )
        return shape_str

    # ───  Method 1: FLOPS  ───
    def _comp_leaf_flops_info(self):
        seq_len = self.input_info.tensors[0].size(1)
        if self.strategy.cp_size > 1:
            if self.strategy.cp_comm_type == "a2a":
                assert self.head_num % self.strategy.cp_size == 0, (
                    f"head_num {self.head_num} must be divisible by cp_size {self.strategy.cp_size}"
                )
                seq_len = seq_len * self.strategy.cp_size
                head_num = self.head_num // self.strategy.cp_size
            else:
                raise NotImplementedError(
                    f"SWA cp_comm_type {self.strategy.cp_comm_type} not implemented yet."
                )
        else:
            head_num = self.head_num
        batch_size = self.input_info.tensors[0].size(0)

        # SWA core: use swa_causal_mask_area instead of seq_len*seq_len
        # (training: KV_LEN=0)
        causal_mask_area = swa_causal_mask_area(int(seq_len), 0, self.window_size)
        base_flops = (
            2 * batch_size * head_num * self.head_size * causal_mask_area
        )
        # No sparse_ratio multiplication — window already encodes sparsity

        self._compute_info.fwd_flops = 2 * base_flops      # QK^T + PV
        self._compute_info.recompute_flops = (
            self._compute_info.fwd_flops if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_flops = 4 * base_flops
        if self.use_flash_sdp:
            self._compute_info.bwd_grad_act_flops += base_flops
        self._compute_info.bwd_grad_w_flops = 0

    # ───  Method 2: memory access  ───
    def _comp_leaf_mem_accessed_info(self):
        """Matches CoreAttention.flash_sdp memory access pattern.
        Note: k_size = v_size = q_size (all use head_num, not kv_head_num),
        consistent with CoreAttention's estimation convention.
        """
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        q_size = batch_size * self.head_num * seq_len * self.head_size
        k_size = q_size   # consistent with CoreAttention
        v_size = q_size   # consistent with CoreAttention
        output_grad_size = batch_size * seq_len * self.head_num * self.head_size
        lse_size = batch_size * self.head_num * seq_len

        if self.use_flash_sdp:
            self._compute_info.fwd_accessed_mem = (
                q_size + k_size + v_size + output_grad_size + lse_size
            ) * self.element_size
            self._compute_info.bwd_grad_act_accessed_mem = (
                2 * q_size + 2 * k_size + 2 * v_size
                + output_grad_size + lse_size
            ) * self.element_size
            self._compute_info.bwd_grad_w_accessed_mem = 0
            self._compute_info.recompute_accessed_mem = (
                self._compute_info.fwd_accessed_mem
                if self.enable_recompute else 0
            )
            return

        # math_sdp fallback
        softmax_size = batch_size * self.head_num * seq_len * seq_len
        self._compute_info.fwd_accessed_mem += (
            q_size + k_size + softmax_size
        ) * self.element_size
        self._compute_info.fwd_accessed_mem += 2 * softmax_size * self.element_size
        self._compute_info.fwd_accessed_mem += (
            softmax_size + v_size + output_grad_size
        ) * self.element_size
        self._compute_info.recompute_accessed_mem = (
            self._compute_info.fwd_accessed_mem if self.enable_recompute else 0
        )
        self._compute_info.bwd_grad_act_accessed_mem += (
            2 * (softmax_size + v_size + output_grad_size) * self.element_size
        )
        self._compute_info.bwd_grad_act_accessed_mem += (
            2 * softmax_size * self.element_size
        )
        self._compute_info.bwd_grad_act_accessed_mem += (
            2 * (q_size + k_size + softmax_size) * self.element_size
        )
        self._compute_info.bwd_grad_w_accessed_mem = 0

    # ───  Method 3: activation memory  ───
    def _comp_leaf_act_info_impl(self):
        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)

        q_size = batch_size * self.head_num * seq_len * self.head_size
        k_size = q_size
        v_size = q_size
        lse_size = batch_size * self.head_num * seq_len
        output_grad_size = batch_size * seq_len * self.head_size * self.head_num

        qkv_mem = (q_size + k_size + v_size) * self.element_size

        if self.use_flash_sdp:
            lse_mem = lse_size * self.element_size
            out_mem = output_grad_size * self.element_size
            self._act_info.activation_mem_cache = qkv_mem + lse_mem
            if self.has_cached_inputs:
                self._act_info.activation_mem_cache -= qkv_mem
            self._act_info.fwd_peak_mem_no_cache = qkv_mem + lse_mem + out_mem
            self._act_info.bwd_peak_mem_no_cache = (
                2 * q_size + 2 * k_size + 2 * v_size
                + lse_size + output_grad_size
            ) * self.element_size
        else:
            # math_sdp fallback: explicit softmax buffer needed
            softmax_size = batch_size * self.head_num * seq_len * seq_len
            softmax_mem = softmax_size * self.element_size
            out_mem = output_grad_size * self.element_size
            self._act_info.activation_mem_cache = qkv_mem + softmax_mem
            self._act_info.fwd_peak_mem_no_cache = (
                qkv_mem + softmax_mem + out_mem
            )
            self._act_info.bwd_peak_mem_no_cache = (
                qkv_mem + 2 * softmax_mem + out_mem
            )

    # ───  Method 4: model params  ───
    def _comp_leaf_model_info_impl(self):
        # Attention has no trainable weights of its own
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0

    # ───  Method 6: efficiency lookup op_name ───
    def _comp_cost_info(self):
        """SWA reuses sdp_fwd/sdp_bwd efficiency entries (same compute pattern)."""
        self._comp_cost_info_impl(
            fwd_op="sdp_fwd",
            bwd_grad_act_op="sdp_bwd",
            bwd_grad_w_op="sdp_bwd",
            enable_recompute=self.enable_recompute,
        )

    def extra_repr(self) -> str:
        return (
            f"head_size={self.head_size},"
            f"head_num={self.head_num},"
            f"kv_head_num={self.kv_head_num},"
            f"window_size={self.window_size},"
            f"use_flash_sdp={self.use_flash_sdp},"
            f"enable_recompute={self.enable_recompute}"
        )

    # ───  CP A2A helpers (shared with CoreAttention pattern)  ───
    def _get_cp_a2a_stage_specs(self):
        if not (self.strategy.cp_size > 1 and self.strategy.cp_comm_type == "a2a"):
            return None

        batch_size = self.input_info.tensors[0].size(0)
        seq_len = self.input_info.tensors[0].size(1)
        q_mem = batch_size * self.head_num * seq_len * self.head_size * self.element_size
        k_mem = batch_size * self.kv_head_num * seq_len * self.head_size * self.element_size
        v_mem = batch_size * self.kv_head_num * seq_len * self.v_head_dim * self.element_size
        o_do_mem = (
            batch_size * self.head_num * seq_len * self.v_head_dim * self.element_size
        )
        bwd_pre = [("Attention_BWD_CP2_DOUT", o_do_mem)]
        if not self.strategy.te_cp_a2a_saves_pre_posta2a_output:
            bwd_pre.insert(0, ("Attention_BWD_CP2_OUT", o_do_mem))
        return {
            "fwd_pre": [
                ("Attention_FWD_CP1_Q", q_mem),
                ("Attention_FWD_CP1_K", k_mem),
                ("Attention_FWD_CP1_V", v_mem),
            ],
            "fwd_post": [("Attention_FWD_CP2", o_do_mem)],
            "bwd_pre": bwd_pre,
            "bwd_post": [
                ("Attention_BWD_CP1_DQ", q_mem),
                ("Attention_BWD_CP1_DK", k_mem),
                ("Attention_BWD_CP1_DV", v_mem),
            ],
        }

    def _append_cp_a2a_layers(self, args, model_info, rank_info, com_buff=None):
        from simumax.core.base_struct import all2all_fwd, all2all_bwd

        stage_specs = self._get_cp_a2a_stage_specs()
        if stage_specs is None:
            return

        state = args.thread_state
        cp_rank = rank_info.get(
            "cp_rank", (args.rank // self.strategy.tp_size) % self.strategy.cp_size
        )
        cp_group_id = rank_info.get(
            "cp_group_id",
            f"tp:{rank_info['tp_rank']}-pp:{rank_info['pp_rank']}-dp:{rank_info['dp_rank']}",
        )

        def append_comm(comm_cls, stage_name, comm_size):
            cost = self.system.compute_net_op_time(
                "all2all",
                comm_size,
                comm_num=self.strategy.cp_size,
                net=self.strategy.cp_net,
                comm_stage=stage_name,
                strategy=self.strategy,
                group_kind="cp",
            )
            fwd_cost = cost if comm_cls is all2all_fwd else 0
            bwd_cost = cost if comm_cls is all2all_bwd else 0
            self.layers.append(
                comm_cls(
                    f"{state.comm_order}-{model_info}-cp_group:{cp_group_id}-stage:{stage_name}",
                    cp_rank,
                    self.strategy.cp_size,
                    com_buff=com_buff,
                    fwd_cost=fwd_cost,
                    bwd_cost=bwd_cost,
                    global_rank=args.rank,
                )
            )
            state.comm_order += 1

        for stage_name, comm_size in stage_specs["fwd_pre"]:
            append_comm(all2all_fwd, stage_name, comm_size)
        for stage_name, comm_size in reversed(stage_specs["bwd_post"]):
            append_comm(all2all_bwd, stage_name, comm_size)
        for stage_name, comm_size in stage_specs["fwd_post"]:
            append_comm(all2all_fwd, stage_name, comm_size)
        for stage_name, comm_size in reversed(stage_specs["bwd_pre"]):
            append_comm(all2all_bwd, stage_name, comm_size)

    # ───  CP network info  ───
    def _comp_leaf_intra_net_info(self):
        if self.strategy.cp_size > 1:
            batch_size = self.input_info.tensors[0].size(0)
            seq_len = self.input_info.tensors[0].size(1)
            q_size = batch_size * self.head_num * seq_len * self.head_size
            k_size = batch_size * self.kv_head_num * seq_len * self.head_size
            v_size = batch_size * self.kv_head_num * seq_len * self.v_head_dim
            qkv_mem = (q_size + k_size + v_size) * self.element_size
            kv_mem = (k_size + v_size) * self.element_size

            if self.strategy.cp_comm_type == "a2a":
                stage_specs = self._get_cp_a2a_stage_specs()
                for bucket, stage_group in (
                    ("fwd_net_time", stage_specs["fwd_pre"] + stage_specs["fwd_post"]),
                    ("bwd_grad_act_net_time", stage_specs["bwd_post"] + stage_specs["bwd_pre"]),
                ):
                    for stage_name, comm_size in stage_group:
                        setattr(
                            self._cost_info,
                            bucket,
                            getattr(self._cost_info, bucket)
                            + self.system.compute_net_op_time(
                                "all2all",
                                comm_size,
                                comm_num=self.strategy.cp_size,
                                net=self.strategy.cp_net,
                                comm_stage=stage_name,
                                strategy=self.strategy,
                                group_kind="cp",
                            ),
                        )
            elif self.strategy.cp_comm_type == "all_gather":
                fwd_comm_size = kv_mem * self.strategy.cp_size * self.dtype_to_element_size[self.strategy.dtype]
                self._cost_info.fwd_net_time += self.system.compute_net_op_time(
                    "all_gather", fwd_comm_size,
                    comm_num=self.strategy.cp_size, net=self.strategy.cp_net,
                    comm_stage="Attention_FWD_CP",
                    strategy=self.strategy, group_kind="cp",
                )
                self._cost_info.bwd_net_time += self.system.compute_net_op_time(
                    "all_gather", fwd_comm_size,
                    comm_num=self.strategy.cp_size, net=self.strategy.cp_net,
                    comm_stage="Attention_BWD_CP1",
                    strategy=self.strategy, group_kind="cp",
                )
                self._cost_info.bwd_net_time += self.system.compute_net_op_time(
                    "reduce_scatter", fwd_comm_size,
                    comm_num=self.strategy.cp_size, net=self.strategy.cp_net,
                    comm_stage="Attention_BWD_CP2",
                    strategy=self.strategy, group_kind="cp",
                )
