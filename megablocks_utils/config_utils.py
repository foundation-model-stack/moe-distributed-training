# utilities to update megablocks to register the MLP_v2 that 
# handles gate, up, down projections

from megablocks.layers.dmlp_registry import _REGISTRY


# from megablocks.layers import mlp
from .sparse_mlp2 import SparseMLPv2
from .peft_utils import ParallelDroplessMLP as LoRAParallelDroplessMLP
import megablocks.ops as ops
from megablocks.layers.moe import ParallelMLP
from megablocks.layers.dmoe import ParallelDroplessMLP
from megablocks.layers.mlp import resolve_dtensor
from megablocks.layers.router import LearnedRouter, _uniform_expert_assignment
from megablocks.layers import mpu
import torch
import torch.nn.functional as F

try:
    import scattermoe
    from scattermoe.parallel_experts import parallel_linear
except ImportError:
    pass

from scattermoe_utils.parallel_linear_lora import parallel_linear_lora
from functools import partial

def get_mlp_weights(mod: torch.nn.Module, name: str = "w1"):

    weight = getattr(mod.mlp, name)
    weight = weight.view(-1, mod.ffn_hidden_size, mod.hidden_size)
    weight = weight.to_local() # because its sharded
    if name in {"w1", "w3"}:
        weight = weight.permute(0, 2, 1)
    return (weight,)

def get_mlp_weights_lora(mod: torch.nn.Module, name: str = "w1"):
    W, = get_mlp_weights(mod, name)

    (
        A, B, r, lora_alp, 
        hidden_size, ffn_hidden_size
    ) = mod._lora_pointers[name]

    # Assume LORA has no bias (should be correct because)
    # these should be updated with no bias in LoraLayer.update_layer
    # A = getattr(mod.lora_A, name).weight
    A = A.view(-1, hidden_size, r)
    B = B.view(-1, ffn_hidden_size, r).permute(0, 2, 1)

    if name  == "w2":
        A = A.permute(0, 2, 1)
        B = B.permute(0, 2, 1)
        temp = A # swap
        A = B
        B = temp

    A = resolve_dtensor(A)
    B = resolve_dtensor(B)

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

        torch.distributed.breakpoint(0)
        hidden_states = scattered_experts(
            x,
            *get_weights(self, "w1"),
            1 if self.args.moe_expert_model_parallelism else top_k,
            bin_ids, # sorted_expert_idxs,
            indices, # sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates=None, # we dont have router weights
            grouped_in=False,
            grouped_out=True,
        )
        ds = 1
        torch.distributed.breakpoint(0)
        hidden_states2 = scattered_experts(
            x,
            *get_weights(self, "w3"),
            1 if self.args.moe_expert_model_parallelism else top_k,
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

    # this is the permute and compute for the ParallelMLP version
    # - we need to overide it with the above permute and compute
    # - the "dmoe" version of parallel and compute has the 
    #   binned gather and scatter done outside
    # - this needs to reduce the expert weights
    def permute_and_compute_parallel_mlp(
        self,
        x,
        tokens_per_expert,  # unused
        indices,
        bin_ids,  # unused
        expert_weights,
        bins,
        expert_capacity,
        top_k,
    ):
        # Route the tokens for MoE computation.
        # x = x.view(-1, x.shape[-1])
        # x = ops.binned_gather(x, indices, bins, expert_capacity, top_k)

        # Perform the expert computation. Note that we don't
        # use biases for these linear operations.
        # x = self.mlp(x)

        # need to handle the weights
        # in_shape = None
        # torch.distributed.breakpoint()
        # this is done in the MoE.permute_and_compute
        if len(x.shape) == 3:
            # in_shape = x.size()
            x = x.view(-1, x.shape[-1])


        print ("rank-bef-perm-compu", torch.distributed.get_rank())
        x = permute_and_compute(
            self, x, tokens_per_expert, indices,
            bin_ids, expert_weights, bins, expert_capacity, top_k
        )
        # following
        # https://github.com/IBM/dolomite-engine/blob/b3ad5703ae32060f07339b664d3caca3e2c5e70a/dolomite_engine/hf_models/models/moe_dolomite/moe/base.py

        print ("rank-bef", torch.distributed.get_rank())
        from time import sleep
        sleep(100000)
        y = ops.scatter(x, indices, bin_ids, expert_weights, bins, top_k)
        print ("rank-af", torch.distributed.get_rank())
        return y

        # x *= expert_weights.unsqueeze()
        # if in_shape is not None:
        #     x = x.view(in_shape)
        x = x.view(expert_weights.shape)

        # Un-route the data for the MoE output.
        return ops.binned_scatter(x, indices, expert_weights, bins, top_k)

    ParallelMLP.permute_and_compute = permute_and_compute_parallel_mlp
    # ParallelMLP.permute_and_compute = permute_and_compute

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
