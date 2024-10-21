import torch
import megablocks.ops as ops
from megablocks.layers.all_to_all import all_to_all

# from megablocks
def no_indices_just_bins(top_expert, num_experts):
    # Sort the expert ids to produce the scatter/gather
    # indices for the permutation.

    # Histogram the expert ids to identify the number of
    # tokens routed to each expert.
    #
    # TODO(tgale): Does the sorted data produce a more favorable
    # data distribution for histogram? Or is the op parallelism
    # worth more?
    tokens_per_expert = ops.histogram(top_expert, num_experts)

    # Calculate the bin bounds for the sorted tokens.
    bins = ops.inclusive_cumsum(tokens_per_expert, 0)
    bins = bins.view(1) if not len(bins.size()) else bins
    return bins, tokens_per_expert

# modified from megablocks 
# - original credit to trevor-gale
def all_to_all_gather_inputs(
    x, 
    top_experts,
    bin_ids, indices, 
    expert_parallel_group, 
    top_k, 
    experts_per_rank, 
):
    # Compute the mapping of local tokens to experts.
    # expert_weights = expert_weights.flatten()
    top_experts = top_experts.flatten()
    world_size = expert_parallel_group.size()
    with torch.no_grad():
        bins, tokens_per_expert = no_indices_just_bins(
            top_experts, experts_per_rank * world_size
        )

        # Pass token count information to the device on which the
        # target expert resides.
        parallel_tokens_per_expert = torch.empty_like(tokens_per_expert,)
        tpe_handle = torch.distributed.all_to_all_single(
            parallel_tokens_per_expert,
            tokens_per_expert,
            group=expert_parallel_group,
            async_op=True,
        )

    # Permute locally and without any padding so that tokens for each
    # parallel device are stored contiguously.
    #
    # This view updates the shape of the tensor from [sl, bs, hs] to
    # [sl * bs, hs] prior to the permutation.
    # x = x.view(-1, x.shape[-1]) # NOTE: not needed
    x = ops.gather(x, indices, bin_ids, bins, top_k)

    # Compute the number of tokens that will be received from each
    # device and permute the input data across the devices.
    with torch.no_grad():
        tpe_handle.wait()

        # Reshape to [world_size, num_experts_per_rank].
        tokens_per_expert = (tokens_per_expert.view(world_size, experts_per_rank))
        parallel_tokens_per_expert = (parallel_tokens_per_expert.view(world_size, experts_per_rank))

        # TODO(tgale): It might be faster to do this on the GPU and
        # then communicate the results back to the host.
        send_counts = tokens_per_expert.cpu().sum(dim=-1)
        parallel_tokens_per_expert_cpu = parallel_tokens_per_expert.cpu()
        recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1)

        # Convert the send/recv counts to lists.
        send_counts = send_counts.tolist()
        recv_counts = recv_counts.tolist()
        tokens_received = sum(recv_counts)

    # Start the cross-device permutation asynchronously so we can
    # overlap communication with computation.
    parallel_x, parallel_x_handle = all_to_all(
        x,
        recv_counts,
        send_counts,
        expert_parallel_group,
        async_op=True,
    )

    with torch.no_grad():
        # After we do the cross-device permutation we have the tokens on the
        # correct device but not yet grouped by expert because we received
        # tokens from each device as contiguous chunks. To group the tokens
        # for expert computation we'll do one more local permutation. The
        # rest of this torch.no_grad() scope sets up the indices and bins
        # for this permutation.
        replicate_bins = ops.inclusive_cumsum(
            parallel_tokens_per_expert.flatten(),
            0,
        )
        replicate_bins = (replicate_bins.view(1) if not len(replicate_bins.size()) else replicate_bins)

        # Construct the expert indices for the permuted tokens.
        parallel_top_expert = torch.remainder(
            torch.arange(
                experts_per_rank * world_size,
                dtype=torch.int32,
                device=indices.device,
            ),
            experts_per_rank
        )
        parallel_top_expert = ops.replicate(
            parallel_top_expert.unsqueeze(dim=0),
            replicate_bins,
            tokens_received,
        ).flatten()

        # TODO(tgale): The sort_end_bit here can be reduced.
        # NOTE: replace this first
        # parallel_bin_ids, parallel_indices = ops.sort(
        #     parallel_top_expert,
        #     self.sort_end_bit,
        # )
        parallel_bin_ids, parallel_indices = torch.sort(parallel_top_expert)

        # Calculate the bins boundaries from the token counts.
        parallel_tokens_per_expert = parallel_tokens_per_expert.sum(
            dim=0,
            dtype=torch.int,
        )
        parallel_bins = ops.inclusive_cumsum(parallel_tokens_per_expert, 0)
        parallel_bins = (parallel_bins.view(1) if not len(parallel_bins.size()) else parallel_bins)

        # # If expert_capacity is set to zero, set the number of tokens
        # # per expert to the maximum we need to avoid dropping tokens.
        # tokens, hs = x.size()
        # expert_capacity = self.expert_capacity(tokens)
        # if expert_capacity == 0:
        #     expert_capacity = torch.max(parallel_tokens_per_expert).item()
    
    parallel_x_handle.wait()

    return (
        parallel_x, parallel_bin_ids, parallel_indices, 
        send_counts, recv_counts, # for all to all
        bins # local
    )

def scatter_with_routing_weights(
    x,
    expert_weights,
    send_counts, recv_counts,
    bins, original_shape,
    bin_ids, indices, 
    expert_parallel_group, 
    top_k,
):

    # Un-permute the tokens across the devices.
    x, _ = all_to_all(
        x,
        send_counts,
        recv_counts,
        expert_parallel_group,
    )


    # TODO: maybe we can remove this op
    # x = ops.sum(x, dim=0)

    # Un-permute locally to setup for the next series of operations.
    x = ops.scatter(
        x, indices, bin_ids, expert_weights, 
        bins, top_k
    )

    return x.view(original_shape)
