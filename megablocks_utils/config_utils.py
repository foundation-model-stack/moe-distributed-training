# utilities to update megablocks to register the MLP_v2 that 
# handles gate, up, down projections

from megablocks.layers.dmlp_registry import _REGISTRY


# from megablocks.layers import mlp
from .sparse_mlp2 import SparseMLPv2
from megablocks.layers.moe import ParallelMLP
from megablocks.layers.mlp import resolve_dtensor
from megablocks.layers.router import LearnedRouter, _uniform_expert_assignment
import torch
import torch.nn.functional as F

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
    