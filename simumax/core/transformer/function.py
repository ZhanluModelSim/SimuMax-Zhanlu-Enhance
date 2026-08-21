from typing import List
from simumax.core.tensor import TensorSize
from simumax.core.base_struct import (
    AtomModel,
    MetaModule,
    InputOutputInfo,
    PathDebugContext,
)
# from simumax.core.transformer.dense_module import Qunatizer


def _tensor_bytes(tensor):
    return tensor.get_memory_size()


class _LayoutCostMixin:
    """Forward-derived cost path for materialized layout operators.

    Layout operations have no meaningful model FLOPs, but they launch kernels
    and transfer input/output tensors. Concrete modules provide byte formulas;
    SystemConfig supplies the HBM and launch parameters. No measured duration
    or utilization enters this path.
    """

    fwd_op = "layout"
    bwd_op = "layout"

    def _comp_leaf_flops_info(self):
        self._compute_info.fwd_flops = 0
        self._compute_info.recompute_flops = 0
        self._compute_info.bwd_grad_act_flops = 0
        self._compute_info.bwd_grad_w_flops = 0

    def _comp_leaf_intra_net_info(self):
        pass

    def _comp_cost_info(self):
        _, path_key = self.get_cost_keys()
        if path_key in (None, "self"):
            path_key = getattr(self, "current_full_module_path", None) or path_key

        def layout_time(op_name, stage, accessed_mem):
            if not accessed_mem:
                return 0.0
            input_bytes, output_bytes = self._layout_io_bytes(stage)
            return self.system.compute_layout_time(
                op_name,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                stage=stage,
                path_key=path_key,
                shape_desc=self.get_input_shapes_desc(stage),
            )

        self._cost_info.fwd_compute_time = layout_time(
            self.fwd_op, "fwd", self._compute_info.fwd_accessed_mem)
        self._cost_info.bwd_grad_act_time = layout_time(
            self.bwd_op, "bwd_grad_act",
            self._compute_info.bwd_grad_act_accessed_mem)
        self._cost_info.bwd_grad_w_time = 0
        self._cost_info.recompute_compute_time = (
            self._cost_info.fwd_compute_time if self.enable_recompute else 0)

    def _layout_io_bytes(self, stage):
        if stage == "fwd":
            return self._compute_info.fwd_accessed_mem, 0
        return self._compute_info.bwd_grad_act_accessed_mem, 0

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = call_stk + self.call_stk
        self.layers.append(AtomModel(
            fwd_cost=self._cost_info.fwd_compute_time,
            bwd_cost=(self._cost_info.bwd_grad_act_time
                      + self._cost_info.bwd_grad_w_time),
            specific_name=self.name or self.__class__.__name__,
        ))
        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)

    def get_input_shapes_desc(self, stage):
        del stage
        shapes = ["x".join(str(dim) for dim in tensor.shape)
                  for tensor in self.input_info.tensors]
        return f"inputs={';'.join(shapes)}"

class Function:
    @staticmethod
    def apply(cls, *args, **kwargs):
        raise NotImplementedError


class ConcatModule(_LayoutCostMixin, MetaModule):
    fwd_op = "layout_concat"
    bwd_op = "layout_concat_bwd"

    def __init__(self, dim: int = -1, enable_recompute: bool = False,
                 strategy=None, system=None, name=None):
        # Keep the semantic boundary in both the scope and the leaf event.
        # Previously the AtomModel was named ``ConcatD`` while the enclosing
        # scope remained ``ConcatModule`` because MetaModule received an empty
        # specific_name.  That made a materialized concat look like two
        # unrelated identities to trace consumers.  The name is supplied by
        # the model graph, never from a measured duration.
        semantic_name = name if name else 'ConcatD'
        super().__init__(strategy, system, semantic_name)
        self.dim = dim
        self.enable_recompute = enable_recompute
        self.is_leaf_module = True
        # The implementation materializes a Cat/ConcatD layout kernel.  Keep
        # the semantic event name stable across model callers; this is only a
        # kernel identity, not a measured duration or calibration parameter.
        self.name = semantic_name

    def create_output_info(self):
        # return TensorSize or InputOutputInfo
        tensor_sizes = self.input_info.tensors
        if len(tensor_sizes) == 0:
            return InputOutputInfo([])
        concat_size = sum([t[self.dim] for t in tensor_sizes])
        shape = list(tensor_sizes[0].shape)
        shape[self.dim] = concat_size
        return TensorSize(shape=tuple(shape), dtype=tensor_sizes[0].dtype)

    def _comp_leaf_mem_accessed_info(self):
        # Materialized concat: read every input and write the output. Backward
        # performs the inverse materialized split on the output gradient.
        input_bytes = sum(
            _tensor_bytes(tensor) for tensor in self.input_info.tensors)
        output_bytes = _tensor_bytes(self.output_info_)
        traffic = input_bytes + output_bytes
        self._compute_info.fwd_accessed_mem = traffic
        self._compute_info.bwd_grad_act_accessed_mem = traffic
        self._compute_info.bwd_grad_w_accessed_mem = 0
        self._compute_info.recompute_accessed_mem = (
            traffic if self.enable_recompute else 0)

    def _layout_io_bytes(self, stage):
        input_bytes = sum(
            _tensor_bytes(tensor) for tensor in self.input_info.tensors)
        output_bytes = _tensor_bytes(self.output_info_)
        if stage == "fwd":
            return input_bytes, output_bytes
        return output_bytes, input_bytes

    def extra_repr(self) -> str:
        repr_info = f"concat_dim={self.dim}, enable_recompute={self.enable_recompute}"
        return repr_info

class SplitModule(_LayoutCostMixin, MetaModule):
    fwd_op = "layout_split"
    bwd_op = "layout_split_bwd"

    def __init__(self, split_size_or_sections: List[int], split_dim,
                 enable_recompute: bool = False, strategy=None, system=None,
                 name=None):
        # As with ConcatModule, expose the graph-provided semantic identity as
        # the scope name as well as the leaf event name.  This keeps QKV_Split
        # and other materialized splits independently addressable without
        # changing their cost formulas.
        semantic_name = name if name else 'SplitModule'
        super().__init__(strategy, system, semantic_name)
        self.split_size_or_sections = split_size_or_sections
        self.split_dim = split_dim
        self.enable_recompute = enable_recompute
        self.name = semantic_name

    def create_output_info(self):
        tensor_size = (self.input_info.tensors[0]
                       if isinstance(self.input_info, InputOutputInfo)
                       else self.input_info)
        split_dim = self.split_dim
        split_size_or_sections = self.split_size_or_sections
        if isinstance(split_size_or_sections, int):
            assert tensor_size[split_dim] % split_size_or_sections == 0, (
                f"tensor_size[dim]={tensor_size[split_dim]} "
                f"split_size_or_sections={split_size_or_sections}")
            print(
                "split_size_or_sections is int, tensor_size[dim] is "
                f"{tensor_size[split_dim]}, split_size_or_sections is "
                f"{split_size_or_sections}")
            split_size_or_sections = [
                tensor_size[split_dim] // split_size_or_sections
            ] * split_size_or_sections

        assert tensor_size[split_dim] == sum(split_size_or_sections), (
            f"tensor_size[dim]={tensor_size[split_dim]} "
            f"sum(split_size_or_sections)={sum(split_size_or_sections)}, "
            f"tensor_size={tensor_size.shape}, split_dim={split_dim}")
        outputs = []
        for size in split_size_or_sections:
            shape = list(tensor_size.shape)
            shape[split_dim] = size
            outputs.append(TensorSize(
                shape=tuple(shape), dtype=tensor_size.dtype))
        output_info = InputOutputInfo(tensors=outputs)
        return output_info

    def _comp_leaf_mem_accessed_info(self):
        # Materialized split: read the input and write every output slice.
        # Backward materializes the inverse concatenation.
        input_bytes = _tensor_bytes(self.input_info.tensors[0])
        output_bytes = sum(
            _tensor_bytes(tensor) for tensor in self.output_info_.tensors)
        traffic = input_bytes + output_bytes
        self._compute_info.fwd_accessed_mem = traffic
        self._compute_info.bwd_grad_act_accessed_mem = traffic
        self._compute_info.bwd_grad_w_accessed_mem = 0
        self._compute_info.recompute_accessed_mem = (
            traffic if self.enable_recompute else 0)

    def _layout_io_bytes(self, stage):
        input_bytes = _tensor_bytes(self.input_info.tensors[0])
        output_bytes = sum(
            _tensor_bytes(tensor) for tensor in self.output_info_.tensors)
        if stage == "fwd":
            return input_bytes, output_bytes
        return output_bytes, input_bytes

    def extra_repr(self) -> str:
        repr_info = f"split_dim={self.split_dim}, enable_recompute={self.enable_recompute}"
        return repr_info

class AddModule(_LayoutCostMixin, MetaModule):
    fwd_op = "layout_add"
    bwd_op = "layout_add_bwd"

    def __init__(self, enable_recompute: bool = False, strategy=None,
                 system=None, name=None):
        super().__init__(strategy, system)
        self.enable_recompute = enable_recompute
        self.name = name

    def create_output_info(self):
        # recover the original input
        assert self.output_info_ is None
        source = self.input_info.tensors[0]
        output_info = InputOutputInfo(tensors=[TensorSize(
            shape=tuple(source.shape), dtype=source.dtype)])
        return output_info

    def _comp_leaf_mem_accessed_info(self):
        output_bytes = _tensor_bytes(self.output_info_.tensors[0])
        # Forward reads two operands and writes one result. Backward routes one
        # output gradient to two input gradients: one read plus two writes.
        traffic = 3 * output_bytes
        self._compute_info.fwd_accessed_mem = traffic
        self._compute_info.bwd_grad_act_accessed_mem = traffic
        self._compute_info.bwd_grad_w_accessed_mem = 0
        self._compute_info.recompute_accessed_mem = (
            traffic if self.enable_recompute else 0)

    def _layout_io_bytes(self, stage):
        output_bytes = _tensor_bytes(self.output_info_.tensors[0])
        # Fwd: two operand reads + one result write. Bwd: one gradient read +
        # two materialized input-gradient writes.
        if stage == "fwd":
            return 2 * output_bytes, output_bytes
        return output_bytes, 2 * output_bytes

    def extra_repr(self) -> str:
        repr_info = f"enable_recompute={self.enable_recompute}"
        return repr_info

class UnsqueezeModule(MetaModule):
    def __init__(self, unsqueeze_dim:int, enable_recompute:bool = False, strategy=None, system=None, name=None):
        super().__init__(strategy, system)
        self.unsqueeze_dim = unsqueeze_dim
        self.enable_recompute = enable_recompute
        self.name =  name if name else 'UnsqueezeModule'
    
    def create_output_info(self):
        inputs = self.input_info.tensors[0] if isinstance(self.input_info, InputOutputInfo) else self.input_info
        outputs = inputs.new()
        outputs.squeeze(self.unsqueeze_dim)
        return InputOutputInfo(tensors=outputs)

class ConcatFunction(Function):
    @staticmethod
    def apply(parent_model:MetaModule, enable_recompute:bool, tensor_sizes: List[TensorSize], dim:int = -1, path_debug_context: PathDebugContext = None, name = None):
        # model.output_size = TensorSize.concat(tensor_sizes, dim)
        concat_module = ConcatModule(dim, enable_recompute, parent_model.strategy, parent_model.system, name)
        concat_module.parent_module = parent_model  # Bind parent module 

        input_info = InputOutputInfo(tensor_sizes)
        out = concat_module(input_info, path_debug_context = path_debug_context) # Reuse the __call__ method of MetaModule, call related functions for statistics, and register the concat_module into parent_module
        return out

class SplitFunction(Function):
    @staticmethod
    def apply(parent_model:MetaModule, enable_recompute:bool, tensor_size:TensorSize, split_size_or_sections:int, split_dim:int = -1, path_debug_context: PathDebugContext = None, name = None):
        # model.output_size = TensorSize.split(tensor_size, split_size_or_sections, dim)    
        split_module = SplitModule(split_size_or_sections, split_dim,  enable_recompute, parent_model.strategy, parent_model.system, name)
        split_module.parent_module = parent_model  # Bind parent module 

        input_info = InputOutputInfo([tensor_size]) if isinstance(tensor_size, TensorSize) else tensor_size
        out = split_module(input_info, path_debug_context = path_debug_context) # Reuse the __call__ method of MetaModule, call related functions for statistics, and register the split_module into parent_module

        return out
    
class AddFunction(Function):
    @staticmethod
    def apply(parent_model:MetaModule, enable_recompute:bool, tensor_size1:TensorSize, tensor_size2:TensorSize, path_debug_context: PathDebugContext = None, name = None):
        # model.output_size = TensorSize.split(tensor_size, split_size_or_sections, dim)    
        add_module = AddModule(enable_recompute, parent_model.strategy, parent_model.system, name)
        add_module.parent_module = parent_model  # Bind parent module 

        if isinstance(tensor_size1, InputOutputInfo):
            tensor_size1 = tensor_size1.tensors[0]
        if isinstance(tensor_size2, InputOutputInfo):
            tensor_size2 = tensor_size2.tensors[0]
        input_info = InputOutputInfo([tensor_size1, tensor_size2])
        out = add_module(input_info, path_debug_context = path_debug_context) # Reuse the __call__ method of MetaModule, call related functions for statistics, and register the split_module into parent_module
        return out
