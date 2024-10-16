# Third Party
from transformers.activations import ACT2FN
import torch
import torch.nn.functional as F
from torch.distributed._tensor import DTensor

try:
    from khd.kernels.scattermoe.triton_implementation.ops import (
        scattered_experts, padded_block_indices
    )
except ImportError:
    pass
 

def resolve_dtensor(weight):
    if isinstance(weight, DTensor):
        return weight.to_local()
    return weight

class ScatteredExperts(torch.nn.Module):
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        num_experts: int,
        fan_out: int,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(num_experts, in_features, out_features, dtype=torch.bfloat16),
            requires_grad=True,
        )
        self.fan_out = fan_out
        self.grouped_in = grouped_in
        self.grouped_out = grouped_out

    def forward(
        self, x, bin_ids, indices, padded_block_idxs, 
        expert_offsets, gates=None,
    ):
        weight = resolve_dtensor(self.weight)
        return scattered_experts(
            x,
            weight,
            self.fan_out,
            bin_ids, # sorted_expert_idxs,
            indices, # sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates=gates, # we dont have router weights
            grouped_in=self.grouped_in,
            grouped_out=self.grouped_out,
        )

# slightly rewritten from KHD
class ScatterMoE(torch.nn.Module):

    def __init__(
        self, 
        hidden_size: int,
        hidden_act: str,
        intermediate_size: int,
        num_experts: int,
        has_bias: bool = False,
        top_k: int = 2,
        all_to_all: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: str = torch.device('cpu'),
        # **kwargs,
    ):
        assert has_bias == False, "dunno how to handle bias"

        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.hidden_act = hidden_act
        self.activation = ACT2FN[hidden_act]
        self.top_k = top_k
        self.all_to_all = all_to_all

        self.router = torch.nn.Linear(
            in_features=hidden_size,
            out_features=num_experts,
            bias=False,
            dtype=dtype,
            device=device,
        )
        # self._kwargs = kwargs
        # - col are those whose 
        # - keep em empty
        # self.col_linears: Dict[str, ScatteredExperts] = torch.nn.ModuleDict()
        # self.row_linears = torch.nn.ModuleDict()
        # NOTE: in the future we handle this by passing into 
        # this class a spec on how many to create
        self.w1 = ScatteredExperts(
                in_features=self.hidden_size,
                out_features=self.intermediate_size,
                num_experts=self.num_experts,
                fan_out=self.top_k if not self.all_to_all else 1,
                grouped_out=True,
        )
        self.w2 = ScatteredExperts(
                in_features=self.hidden_size,
                out_features=self.intermediate_size,
                num_experts=self.num_experts,
                fan_out=self.top_k if not self.all_to_all else 1,
                grouped_out=True,
        )
        self.w3 = ScatteredExperts(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            num_experts=self.num_experts,
            fan_out=self.top_k if not self.all_to_all else 1,
            grouped_in=True,
            # grouped_out=False if self.args.moe_expert_model_parallelism else True
        )

    # def add_expert(self, key, 
    # dolomite, MoE_Torch
    def _compute_routing_weights(self, hidden_states):

        # router_logits: (batch * sequence_length, n_experts)
        weight = resolve_dtensor(self.router.weight)
        bias = self.router.bias
        if bias: bias = resolve_dtensor(bias)
        router_logits = F.linear(hidden_states, weight, bias)
        # router_logits = self.router(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        return router_logits, routing_weights, selected_experts

    def _maybe_gather(self, selected_experts):
        # can replace with megablocks version of _indices_and_bins
        # this is if there is no 
        if not self.all_to_all:
            sorted_expert_idxs, sorted_scattered_idxs = torch.sort(selected_experts.flatten())
            return sorted_expert_idxs, sorted_scattered_idxs
        
        return NotImplementedError

    def _maybe_scatter(self):
        pass

    def forward(self, hidden_states):

        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # _, batch_index, batch_gates, expert_size, router_logits = self.router(layer_input)
        router_logits, routing_weights, selected_experts = self._compute_routing_weights(hidden_states)

        sorted_expert_idxs, sorted_scattered_idxs = self._maybe_gather(selected_experts)

        with torch.no_grad():
            padded_block_idxs, expert_offsets = padded_block_indices(sorted_expert_idxs, self.num_experts)
        
        out = self.w1(
            hidden_states,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets
        )
        out = self.activation(out)
        # NOTE: in the future we can check this as a condition
        if self.w3:
            out *= self.w3(
                hidden_states,
                sorted_expert_idxs, sorted_scattered_idxs,
                padded_block_idxs, expert_offsets
            ) 

        hidden_states = self.w2(
            out,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            gates=routing_weights # Have to disable this later
        ) 

        self._maybe_scatter() # placeholder, noop

        # col_outs = ()
        # for tag, scattered in self.col_linears.items():
        #     out = scattered(
        #         hidden_states,
        #         sorted_expert_idxs, sorted_scattered_idxs,
        #         padded_block_idxs, expert_offsets
        #     )
        #     col_outs = col_outs + (tag, out)

        # col_outs = self._activation(col_outs)

        # router
        # expert_weights = routing_weights.flatten()
        # top_experts = selected_experts.flatten()

        hidden_states = hidden_states.view(original_shape)
        return hidden_states, router_logits
