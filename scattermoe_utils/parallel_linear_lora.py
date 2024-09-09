import torch
import torch.nn as nn
from .kernels import ops
import scattermoe.kernels as orig_kernels

# lora_scaling is lora_alpha / lora_r
class ParallelLinearLora(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, expert_weights, 
        expert_lora_A, expert_lora_B, 
        lora_r, lora_alp, k,
        sorted_expert_idxs, sorted_scattered_idxs,
        padded_block_idxs, expert_offsets,
        gates=None, grouped_in=False, grouped_out=False,
    ):

        output = ops.scatter2scatter_lora(
            X=x, W=expert_weights, A=expert_lora_A, B=expert_lora_B,
            lora_alp=lora_alp,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            k=k, x_grouped=grouped_in, y_grouped=grouped_out
        )
        if gates is not None:
            output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
            output = torch.bmm(
                gates[:, None, :],
                output_expanded
            ).squeeze(1)
        else:
            output_expanded = None

        ctx.save_for_backward(
            x, expert_weights,
            expert_lora_A, expert_lora_B,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            gates,
            output_expanded
        )
        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k
        ctx.lora_r = lora_r
        ctx.lora_alp = lora_alp
        return output

    @staticmethod
    def backward(ctx, grad_out):
        (
            x, expert_weights,
            expert_lora_A, expert_lora_B, 
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            gates, output_expanded
        ) = ctx.saved_tensors
        k = ctx.k
        lora_r = ctx.lora_r
        lora_alp = ctx.lora_alp
        grouped_in = ctx.grouped_in
        grouped_out = ctx.grouped_out
        if gates is not None:
            # calculate gates gradient
            d_gates = torch.bmm(output_expanded, grad_out[:, :, None]).squeeze(-1)
            gates_flat = gates.flatten()
            gate_fan = gates.size(1)
            grouped_grad_out = output_expanded.flatten(0, 1) # reuse expanded buffer later
        else:
            d_gates = None
            gates_flat = None
            gate_fan = 1
            grouped_grad_out = None

        # group_bwd_AB only operates on grouped grad_out and x.
        if grouped_out:
            grouped_grad_out = grad_out
        else:
            # - if grad_out is not grouped, need to do it here
            grouped_grad_out = orig_kernels.ops.group(
                grad_out, sorted_scattered_idxs,
                fan_out=gate_fan, coeff=gates_flat,
                out=grouped_grad_out
            )

        if grouped_in:
            grouped_x = x
            d_expanded_input = None
        else:
            # - if x is not grouped, need to do it here
            grouped_x = orig_kernels.ops.group(x, sorted_scattered_idxs, fan_out=k)
            d_expanded_input = grouped_x

        # 1. compute weight gradients on the grouped grad_out and x
        d_weights_A, d_weights_B = ops.group_bwd_AB(
            DY=grouped_grad_out, X=grouped_x,
            A=expert_lora_A, B=expert_lora_B,
            expert_offsets=expert_offsets,
            E=expert_weights.size(0),
            scaling=(lora_alp / lora_r),
        )

        # NOTE: this maybe can be fused
        # 2. compute the input gradients.
        # - the operation Y = X * (W + A * B * scaling) is learn
        # - thus the input gradients DX should be simply
        #   DX = DY * (W + A * B * scaling)^T
        #      = DY * W^T + DY * B^T * A^T * scaling
        d_expanded_input = ops.scatter2scatter_lora(
            X=grouped_grad_out, x_grouped=True, # DY
            W=expert_weights.permute(0, 2, 1), # W^T
            A=expert_lora_B.permute(0, 2, 1), # B^T
            B=expert_lora_A.permute(0, 2, 1), # A^T
            lora_alp=lora_alp,
            padded_block_idxs=padded_block_idxs,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            k=1, # process it as no-fan out and handle below
            y_grouped=grouped_in,
            out=d_expanded_input # Reuse grouped_x buffer
        )

        # handle the fan out
        if k == 1:
            d_input = d_expanded_input
        else:
            d_input = d_expanded_input.view(x.size(0), k, d_expanded_input.size(-1)).sum(-2)

        return (
            # x, expert_weights,
            d_input, None, 
            # expert_lora_A, expert_lora_B, 
            d_weights_A, d_weights_B, None,
            # lora_r, lora_alp, k
            None, None, None,
            # sorted_expert_idxs, sorted_scattered_idxs,
            None, None,
            # padded_block_idxs, expert_offsets,
            None, None,
            # gates
            d_gates, None, None
        )

def parallel_linear_lora(
    inputs, 
    expert_weights, expert_lora_A, expert_lora_B, 
    lora_r, lora_alp, k,
    sorted_expert_idxs, sorted_scattered_idxs,
    padded_block_idxs, expert_offsets,
    gates=None, grouped_in=False, grouped_out=False
):
    results = ParallelLinearLora.apply(
        inputs, expert_weights, 
        expert_lora_A, expert_lora_B, 
        lora_r, lora_alp, k,
        sorted_expert_idxs, sorted_scattered_idxs,
        padded_block_idxs, expert_offsets, gates,
        grouped_in, grouped_out
    )
    return results
