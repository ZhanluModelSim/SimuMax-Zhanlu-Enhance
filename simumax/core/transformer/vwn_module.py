"""VWN (Variable Window Network) Roofline Cost Model.

Based on op_define.md MojoVWNInOp (L119-169) and MojoVWNOutOp (L941-964).

VWN is a gating / stream-mixing network that replaces the standard RMSNorm
in selected transformer layers. It consists of two operators:

  VWNInOp  — input side: RMSNorm + gating GEMM + tanh + stream mixing
  VWNOutOp — output side: B-gate mixing + residual add

Parameters (from op_define L126-131):
  C      = hidden size
  VWN_N  = residual streams count
  VWN_M  = block output streams count
  V      = C / VWN_M   (per-stream hidden dim)
  A      = VWN_N + VWN_M
  G      = VWN_N + 2 * VWN_M
"""

from simumax.core.base_struct import MetaModule, InputOutputInfo
from simumax.core.tensor import TensorSize
from simumax.core.config import StrategyConfig, SystemConfig


class VWNInModule(MetaModule):
    """VWN Input Gating Operator.

    Corresponds to MojoVWNInOp in op_define.md (L119-169).

    Fuses RMSNorm + gating GEMM + tanh + stream mixing into a single
    roofline-modeled operator.

    Input:
      x[T, VWN_N, V]  —  V = C / VWN_M
    Outputs:
      block_input[T, VWN_M, V]  — feeds into main transformer block
      residual[T, VWN_N, V]     — residual bypass for VWNOut
      b_gate[T, VWN_N, VWN_M]   — gate values for VWNOut
    """

    def __init__(
        self,
        hidden_size: int,
        vwn_n: int,
        vwn_m: int,
        strategy: StrategyConfig,
        system: SystemConfig,
        specific_name: str = 'VWNIn',
    ) -> None:
        super().__init__(strategy, system, specific_name)
        self.hidden_size = hidden_size   # C
        self.vwn_n = vwn_n               # residual streams
        self.vwn_m = vwn_m               # block output streams
        assert hidden_size % vwn_m == 0, (
            f"hidden_size({hidden_size}) must be divisible by vwn_m({vwn_m})"
        )
        self.V = hidden_size // vwn_m     # per-stream dim
        self.A = vwn_n + vwn_m           # A gate dimension
        self.G = vwn_n + 2 * vwn_m       # G gate dimension

    @property
    def _token_count(self) -> int:
        """Total token count T from input shape."""
        t = self.input_info.tensors[0]
        # Input: [B, S, H] → T = B × S
        # Input: [T, VWN_N, V] → T = T (already flat)
        if len(t.shape) == 3:
            # Check if input is already flat [T, N, V] or batched [B, S, H]
            # By convention, [T, N, V] has size(0) >> size(1)
            if t.size(0) > t.size(1):
                return int(t.size(0))
            return int(t.size(0)) * int(t.size(1))
        return int(t.size(0))

    def create_output_info(self):
        # VWNIn produces three outputs; the primary output is block_input
        # with shape [T, VWN_M, V]
        T = self._token_count
        return InputOutputInfo(
            tensors=[
                TensorSize(shape=(T, self.vwn_m, self.V))
            ]
        )

    def get_input_shapes_desc(self, stage: str) -> str:
        T = self._token_count
        return (
            f'T={int(T)}, C={int(self.hidden_size)}, '
            f'VWN_N={int(self.vwn_n)}, VWN_M={int(self.vwn_m)}, '
            f'V={int(self.V)}, A={int(self.A)}, G={int(self.G)}'
        )

    # ───  Method 1: FLOPS (source: op_define L154-156)  ───
    def _comp_leaf_flops_info(self):
        T = self._token_count

        # 1) gating GEMM: x[T, VWN_N, V] × w[V, G] → [T, VWN_N, G]
        #    (op_define L154)
        gating_gemm_flops = 2 * T * self.vwn_n * self.V * self.G

        # 2) stream mix (A-gate mix):
        #    y = einsum("tan,tnv->tav", a_gate^T, x_norm)
        #    a_gate: [T, VWN_N, A], x_norm: [T, VWN_N, V] → [T, A, V]
        #    (op_define L155)
        mix_tensor_flops = 2 * T * self.A * self.vwn_n * self.V

        # 3) vector FLOPs (RMSNorm + tanh)
        #    (op_define L156)
        vector_flops = 5 * T * self.vwn_n * self.V + 3 * T * self.vwn_n * self.G

        self._compute_info.fwd_flops = gating_gemm_flops + mix_tensor_flops + vector_flops
        # backward: ~2× forward for gradient computation,
        # plus weight gradient of the gating GEMM
        self._compute_info.bwd_grad_act_flops = 2 * (mix_tensor_flops + vector_flops)
        self._compute_info.bwd_grad_w_flops = gating_gemm_flops
        self._compute_info.recompute_flops = 0  # no separate recompute for norm-like ops

    # ───  Method 2: memory access  ───
    def _comp_leaf_mem_accessed_info(self):
        T = self._token_count
        bf16_sz = self.element_size              # 2: x, norm_w, gate_w, gate_b,
        fp32_sz = self.dtype_to_element_size["fp32"]  # 4: residual only

        # Reads (all bf16):
        #   - x[T, VWN_N, V] bf16
        #   - norm_weight[V] bf16
        #   - gating_weight[V, G] bf16
        #   - gating_bias[G] bf16
        read_act = T * self.vwn_n * self.V * bf16_sz
        read_w   = self.V * bf16_sz
        read_gw  = self.V * self.G * bf16_sz
        read_gb  = self.G * bf16_sz
        read_total = read_act + read_w + read_gw + read_gb

        # Writes:
        #   - block_input[T, VWN_M, V] bf16
        #   - residual[T, VWN_N, V]     fp32  ← 残差用 fp32
        #   - b_gate[T, VWN_N, VWN_M]   bf16
        write_block = T * self.vwn_m * self.V * bf16_sz
        write_resid = T * self.vwn_n * self.V * fp32_sz
        write_gate  = T * self.vwn_n * self.vwn_m * bf16_sz
        write_total = write_block + write_resid + write_gate

        self._compute_info.fwd_accessed_mem = read_total + write_total
        self._compute_info.bwd_grad_act_accessed_mem = 2 * read_total + write_total
        self._compute_info.bwd_grad_w_accessed_mem = read_gw
        self._compute_info.recompute_accessed_mem = 0

    # ───  Method 3: activation memory  ───
    def _comp_leaf_act_info_impl(self):
        T = self._token_count
        bf16_sz = self.element_size              # 2
        fp32_sz = self.dtype_to_element_size["fp32"]  # 4: residual only

        # Cache for backward:
        #   - x_norm[T, VWN_N, V] bf16
        #   - gate[T, VWN_N, G]    bf16
        cache_mem = (
            T * self.vwn_n * self.V  * bf16_sz   # x_norm
            + T * self.vwn_n * self.G * bf16_sz   # gate
        )

        # Forward peak (all intermediates alive):
        fwd_peak = (
            T * self.vwn_n * self.V  * bf16_sz   # x (input)
            + T * self.vwn_n * self.V  * bf16_sz   # x_norm
            + T * self.vwn_n * self.G  * bf16_sz   # gate_logits
            + T * self.vwn_n * self.G  * bf16_sz   # gate
            + T * self.A        * self.V  * bf16_sz   # y (mix output)
            + T * self.vwn_m    * self.V  * bf16_sz   # block_input
            + T * self.vwn_n    * self.V  * fp32_sz   # residual (fp32)
            + T * self.vwn_n * self.vwn_m * bf16_sz   # b_gate
        )

        # Backward peak:
        bwd_peak = (
            T * self.vwn_n * self.V  * bf16_sz   # re-read x
            + T * self.vwn_n * self.V  * bf16_sz   # dx
            + T * self.vwn_n * self.V  * bf16_sz   # x_norm cached
            + T * self.vwn_n * self.G  * bf16_sz   # gate cached
            + T * self.A        * self.V  * bf16_sz   # dy
            + T * self.vwn_m    * self.V  * bf16_sz   # d(block_input)
            + T * self.vwn_n    * self.V  * fp32_sz   # d(residual) (fp32)
        )

        self._act_info.activation_mem_cache = cache_mem
        self._act_info.fwd_peak_mem_no_cache = max(fwd_peak, cache_mem)
        self._act_info.bwd_peak_mem_no_cache = max(bwd_peak - cache_mem, 0)

    # ───  Method 4: model params  ───
    def _comp_leaf_model_info_impl(self):
        element_size = self.dtype_to_element_size[self.strategy.dtype]
        # norm_weight[V] + gating_weight[V, G] + gating_bias[G]
        weight_bytes = (self.V + self.V * self.G + self.G) * element_size
        grad_bytes = weight_bytes
        # Adam optimizer: fp32 master weight + m + v = 3×fp32×numel
        self._model_info.dense_weight_bytes = weight_bytes
        self._model_info.dense_grad_bytes = grad_bytes
        num_params = self.V + self.V * self.G + self.G
        self._model_info.dense_state_bytes = (
            3 * self.dtype_to_element_size["fp32"] * num_params
        )
        self._model_info.moe_weight_bytes = 0
        self._model_info.moe_grad_bytes = 0
        self._model_info.moe_state_bytes = 0

    # ───  Method 5: intra-net communication  ───
    def _comp_leaf_intra_net_info(self):
        # VWNIn has no TP/SP/EP communication
        pass

    # ───  Method 6: efficiency lookup  ───
    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="vwn_in",
            bwd_grad_act_op="vwn_in",
            bwd_grad_w_op="vwn_in",
            enable_recompute=False,
        )

    # ───  prefill (DES simulation)  ───
    def prefill(self, args, call_stk='', com_buff=None):
        from simumax.core.base_struct import AtomModel
        from simumax.core.utils import format_model_info_microbatch_tag
        self.call_stk = call_stk + self.call_stk
        self.layers.append(AtomModel(
            fwd_cost=self._cost_info.fwd_compute_time,
            bwd_cost=(
                self._cost_info.bwd_grad_act_time
                + self._cost_info.bwd_grad_w_time
            ),
            specific_name=self.specific_name,
        ))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)


class VWNOutModule(MetaModule):
    """VWN Output Mixing Operator.

    Corresponds to MojoVWNOutOp in op_define.md (L941-964).

    Fuses B-gate mixing + residual add into a single roofline-modeled operator.
    Has no trainable parameters.

    Inputs:
      block_out[T, VWN_M, V]   — output from main transformer block
      b_gate[T, VWN_N, VWN_M]  — gate values from VWNIn
      residual[T, VWN_N, V]    — residual from VWNIn
    Output:
      y[T, VWN_N, V]           — gated output
    """

    def __init__(
        self,
        hidden_size: int,
        vwn_n: int,
        vwn_m: int,
        strategy: StrategyConfig,
        system: SystemConfig,
        specific_name: str = 'VWNOut',
    ) -> None:
        super().__init__(strategy, system, specific_name)
        self.hidden_size = hidden_size
        self.vwn_n = vwn_n
        self.vwn_m = vwn_m
        assert hidden_size % vwn_m == 0, (
            f"hidden_size({hidden_size}) must be divisible by vwn_m({vwn_m})"
        )
        self.V = hidden_size // vwn_m

    @property
    def _token_count(self) -> int:
        t = self.input_info.tensors[0]
        if len(t.shape) == 3:
            # [T, VWN_M, V] → T, or [B, S, C] → B×S
            if t.size(0) > t.size(1):
                return int(t.size(0))
            return int(t.size(0)) * int(t.size(1))
        return int(t.size(0))

    def create_output_info(self):
        T = self._token_count
        return InputOutputInfo(
            tensors=[
                TensorSize(shape=(T, self.vwn_n, self.V))
            ]
        )

    def get_input_shapes_desc(self, stage: str) -> str:
        T = self._token_count
        return (
            f'T={int(T)}, C={int(self.hidden_size)}, '
            f'VWN_N={int(self.vwn_n)}, VWN_M={int(self.vwn_m)}, V={int(self.V)}'
        )

    # ───  Method 1: FLOPS (source: op_define L958-959)  ───
    def _comp_leaf_flops_info(self):
        T = self._token_count

        # 1) B-gate mix: b_gate[T, VWN_N, VWN_M] × block_out[T, VWN_M, V]
        #    → mixed[T, VWN_N, V]   (op_define L958)
        mix_tensor_flops = 2 * T * self.vwn_n * self.vwn_m * self.V

        # 2) residual add: mixed[T, VWN_N, V] + residual[T, VWN_N, V]
        #    (op_define L959)
        residual_add_flops = T * self.vwn_n * self.V

        self._compute_info.fwd_flops = mix_tensor_flops + residual_add_flops
        self._compute_info.bwd_grad_act_flops = 2 * (mix_tensor_flops + residual_add_flops)
        self._compute_info.bwd_grad_w_flops = 0   # no learnable params
        self._compute_info.recompute_flops = 0

    # ───  Method 2: memory access  ───
    def _comp_leaf_mem_accessed_info(self):
        T = self._token_count
        bf16_sz = self.element_size              # 2: block_out, b_gate, y
        fp32_sz = self.dtype_to_element_size["fp32"]  # 4: residual only

        # Reads:
        #   - block_out[T, VWN_M, V]  bf16
        #   - b_gate[T, VWN_N, VWN_M] bf16
        #   - residual[T, VWN_N, V]   fp32  ← 残差用 fp32
        read_block = T * self.vwn_m * self.V * bf16_sz
        read_gate  = T * self.vwn_n * self.vwn_m * bf16_sz
        read_resid = T * self.vwn_n * self.V * fp32_sz
        read_total = read_block + read_gate + read_resid

        # Write:
        #   - y[T, VWN_N, V] bf16
        write_output = T * self.vwn_n * self.V * bf16_sz

        self._compute_info.fwd_accessed_mem = read_total + write_output
        self._compute_info.bwd_grad_act_accessed_mem = 2 * read_total + write_output
        self._compute_info.bwd_grad_w_accessed_mem = 0
        self._compute_info.recompute_accessed_mem = 0

    # ───  Method 3: activation memory  ───
    def _comp_leaf_act_info_impl(self):
        T = self._token_count
        bf16_sz = self.element_size              # 2
        fp32_sz = self.dtype_to_element_size["fp32"]  # 4: residual only

        # Cache for backward:
        #   - b_gate[T, VWN_N, VWN_M] bf16
        #   - block_out[T, VWN_M, V]  bf16
        cache_mem = (
            T * self.vwn_n * self.vwn_m * bf16_sz   # b_gate
            + T * self.vwn_m * self.V  * bf16_sz     # block_out
        )

        # Forward peak:
        fwd_peak = (
            T * self.vwn_m * self.V  * bf16_sz       # block_out
            + T * self.vwn_n * self.vwn_m * bf16_sz   # b_gate
            + T * self.vwn_n * self.V  * fp32_sz       # residual (fp32)
            + 2 * T * self.vwn_n * self.V * bf16_sz   # mixed (temp) + y
        )

        # Backward peak:
        bwd_peak = (
            cache_mem                                  # cached b_gate + block_out
            + T * self.vwn_n * self.V  * bf16_sz       # dy
            + T * self.vwn_n * self.V  * fp32_sz       # d(residual) (fp32)
            + T * self.vwn_n * self.vwn_m * bf16_sz    # d(b_gate)
        )

        self._act_info.activation_mem_cache = cache_mem
        self._act_info.fwd_peak_mem_no_cache = max(fwd_peak, cache_mem)
        self._act_info.bwd_peak_mem_no_cache = max(bwd_peak - cache_mem, 0)

    # ───  Method 4: model params ───
    def _comp_leaf_model_info_impl(self):
        # VWNOut has no trainable parameters
        self._model_info.dense_weight_bytes = 0
        self._model_info.dense_grad_bytes = 0
        self._model_info.dense_state_bytes = 0
        self._model_info.moe_weight_bytes = 0
        self._model_info.moe_grad_bytes = 0
        self._model_info.moe_state_bytes = 0

    # ───  Method 5: intra-net communication  ───
    def _comp_leaf_intra_net_info(self):
        # VWNOut has no TP/SP/EP communication
        pass

    # ───  Method 6: efficiency lookup  ───
    def _comp_cost_info(self):
        self._comp_cost_info_impl(
            fwd_op="vwn_out",
            bwd_grad_act_op="vwn_out",
            bwd_grad_w_op="vwn_out",
            enable_recompute=False,
        )

    # ───  prefill (DES simulation)  ───
    def prefill(self, args, call_stk='', com_buff=None):
        from simumax.core.base_struct import AtomModel
        from simumax.core.utils import format_model_info_microbatch_tag
        self.call_stk = call_stk + self.call_stk
        self.layers.append(AtomModel(
            fwd_cost=self._cost_info.fwd_compute_time,
            bwd_cost=(
                self._cost_info.bwd_grad_act_time
                + self._cost_info.bwd_grad_w_time
            ),
            specific_name=self.specific_name,
        ))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)
