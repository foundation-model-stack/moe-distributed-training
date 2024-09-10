# utilities to update megablocks to register the MLP_v2 that 
# handles gate, up, down projections

from megablocks.layers.dmlp_registry import _REGISTRY


# from megablocks.layers import mlp
from .sparse_mlp2 import SparseMLPv2
from .peft_utils import ParallelDroplessMLP as LoRAParallelDroplessMLP
from megablocks.layers.moe import ParallelMLP
from megablocks.layers.dmoe import ParallelDroplessMLP
from megablocks.layers.mlp import resolve_dtensor
from megablocks.layers.router import LearnedRouter, _uniform_expert_assignment
from megablocks.layers import mpu
import torch
import torch.nn.functional as F

# if is_scattermoe_available():
import scattermoe
from scattermoe.parallel_experts import parallel_linear
from scattermoe_utils.parallel_linear_lora import parallel_linear_lora
from functools import partial

# from peft.tuners.lora import LoraLayer
# def is_lora_linear(mod: torch.nn.Module):
#     return isinstance(mod, LoraLayer)

def get_mlp_weights(mod: torch.nn.Module, name: str = "w1"):

    weight = getattr(mod.mlp, name)
    weight = weight.view(-1, mod.ffn_hidden_size, mod.hidden_size)
    weight = weight.to_local() # because its sharded
    if name in {"w1", "w3"}:
        weight = weight.permute(0, 2, 1)
    return (weight,)

def get_mlp_weights_lora(mod: torch.nn.Module, name: str = "w1"):
    # - inside, mod.mlp will resolve to mod.base_layer.mlp
    W, = get_mlp_weights(mod, name)

    (
        A, B, r, lora_alp, 
        hidden_size, ffn_hidden_size
    ) = mod._lora_pointers[name]

    # Assume LORA has no bias (should be correct because)
    # these should be updated with no bias in LoraLayer.update_layer
    # A = getattr(mod.lora_A, name).weight
    A = A.view(-1, hidden_size, r)
    # B = getattr(mod.lora_B, name).weight
    B = B.view(-1, ffn_hidden_size, r).permute(0, 2, 1)

    if name  == "w2":
        A = A.permute(0, 2, 1)
        B = B.permute(0, 2, 1)
        temp = A # swap
        A = B
        B = temp

    # r = mod.r[name]
    # lora_alp = mod.lora_alpha[name]

    return (W, A, B, r, lora_alp)

def update_mlp_registry():
    # patch the registry to point to our v2
    _REGISTRY['mlp']['sparse'] = SparseMLPv2

    def forward(self, x, scores, expert_weights, top_experts):
        in_shape = x.size()

        # Compute the experts.
        x, _ = self.forward_fn(x, expert_weights, top_experts)

        x = x.view(in_shape)
        if self.bias is not None:
            if self.args.return_bias:
                return x, self.bias
            return x + self.bias

        # in this case we should be returning the router
        # logits out of the MoeE forward. However, since
        # the way the code is written now, it si difficult 
        # to extract these logits out, so at the moment,
        # we return None as the placeholder.
        return x, None

    # patch the forward function
    ParallelMLP.forward = forward

    # in the case of scattermoe, replace ParallelDroplessMLP.permute_and_compute
    # instead to this version for our v2
    def scattermoe_permute_and_compute(
        self,
        x,
        tokens_per_expert,
        indices,
        bin_ids,
        expert_weights,
        bins,
        expert_capactiy,
        top_k,
    ):
        with torch.no_grad():
            # num_experts if from ParallelMLP
            padded_block_idxs, expert_offsets = scattermoe.kernels.ops.padded_block_indices(
                bin_ids, mpu.experts_per_rank(self.args)
            )

        if hasattr(self, "_lora_pointers"):
            scattered_experts = parallel_linear_lora
            get_weights = get_mlp_weights_lora
        else:
            scattered_experts = parallel_linear
            get_weights = get_mlp_weights

        hidden_states = scattered_experts(
            x,
            # self.mlp.w1.view(
            #     -1, self.ffn_hidden_size, self.hidden_size
            # ).to_local().permute(0, 2, 1),
            *get_weights(self, "w1"),
            1,
            bin_ids, # sorted_expert_idxs,
            indices, # sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates=None, # we dont have router weights
            grouped_in=False,
            grouped_out=True,
        )
        hidden_states2 = scattered_experts(
            x,
            # self.mlp.w3.view(
            #     -1, self.ffn_hidden_size, self.hidden_size
            # ).to_local().permute(0, 2, 1),
            *get_weights(self, "w3"),
            1,
            bin_ids, # sorted_expert_idxs,
            indices, # sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates=None, # we dont have router weights
            grouped_in=False,
            grouped_out=True,
        )
        hidden_states = F.silu(hidden_states) * hidden_states2
        return scattered_experts(
            hidden_states,
            # self.mlp.w2.view(
            #     -1, self.ffn_hidden_size, self.hidden_size
            # ).to_local(),
            *get_weights(self, "w2"),
            1,
            bin_ids, # sorted_expert_idxs,
            indices, # sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates=None, # we dont have router weights
            grouped_in=True,
            grouped_out=False,
        )

    def permute_and_compute(
        self,
        x,
        tokens_per_expert,
        indices,
        bin_ids,
        expert_weights,
        bins,
        expert_capactiy,
        top_k,
    ):

        if self.args.mlp_impl == 'sparse':
            _func = self.sparse_permute_and_compute
        elif self.args.mlp_impl == 'scattermoe':
            # see above
            _func = partial(scattermoe_permute_and_compute, self)
        else:
            _func = self.grouped_permute_and_compute

        return _func(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            expert_capactiy,
            top_k,
        )

    # reg SparseMLPv2 in place for the init functions
    _REGISTRY['mlp']['scattermoe'] = SparseMLPv2
    
    ParallelDroplessMLP.permute_and_compute = permute_and_compute
    # LoRAParallelDroplessMLP.permute_and_compute = permute_and_compute

    def forward_router(self, x):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        _weight = resolve_dtensor(self.layer.weight)
        _bias = None if self.layer.bias is None else resolve_dtensor(self.layer.bias)
        # pylint: disable=not-callable
        scores = F.linear(x.view(-1, x.shape[-1]), _weight, _bias).softmax(dim=-1)
        expert_weights, expert_indices = self._top_k(scores)
        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights,
                p=self.args.moe_normalize_expert_weights,
                dim=-1,
                keepdim=True,
            )

        expert_indices = (
            _uniform_expert_assignment(
                expert_indices,
                self.args.moe_num_experts,
            )
            if self.args.uniform_expert_assignment
            else expert_indices
        )
        return scores, expert_weights, expert_indices

    # replace the forward function in the router
    # - same as above
    LearnedRouter.forward = forward_router
