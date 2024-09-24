import torch
import triton
import triton.language as tl
from torch.nn import functional as F

BLOCK_M = 128

def _scatter2scatter_configs():
    return [
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ]

@triton.autotune(configs=_scatter2scatter_configs(), key=['M', 'N', 'K'], )
@triton.heuristics({
    "NO_K_MASK": lambda args: (args['K'] % args['BLOCK_K']) == 0,
    "NO_N_MASK": lambda args: (args['N'] % args['BLOCK_N']) == 0,
})
@triton.jit
def _scatter2scatter_lora(
    X_ptr, stride_xm, stride_xk,
    W_ptr, stride_we, stride_wk, stride_wn,
    A_ptr, stride_ae, stride_ak, stride_ar,
    B_ptr, stride_be, stride_br, stride_bn,
    Y_ptr, stride_ym, stride_yn,
    grouped_idx_ptr, expert_idxs_ptr, block_start_idx_ptr,
    FAN_OUT: tl.constexpr,
    M, K: tl.constexpr, N: tl.constexpr, E: tl.constexpr,
    R: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    OUT_M,
    scaling,
    allow_tf32: tl.constexpr,
    x_grouped: tl.constexpr, y_grouped: tl.constexpr,
    NO_K_MASK: tl.constexpr, NO_N_MASK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # The grid is assumed to be num_padded_blocks * N_BLOCK_COUNT. The input tokens are
    # on the M dimension, that is blocked. The first task is to identify which expert
    # is being worked on by the kernel instance.
    # - block_start_idx_ptr contains offsets that allow the processing to occur in M Blocks
    # - one padded block could contain multiple experts, in which case block_start_idx_ptr
    #   will index multiple times into a single BLOCK_M.
    # - e.g., block_start_idx_ptr = [0, 128, 256, 300, 428]
    #   * there are 300 E_idx = 0 tokens, this requires 3 * 128 blocks to process, at 
    #     offsets 0, 128, 256.
    #   * the remaining are E_idx = 1 tokens, where the first 128 block starts at 300,
    #     then goes on to 428, etc.
    # - use start index to instantiate the M_block
    # - M_block indices the X's worked on by this kernel instance
    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    PB_idx = pid // N_BLOCK_COUNT # padded block index
    N_block_id = pid % N_BLOCK_COUNT
    M_range = tl.arange(0, BLOCK_M)
    block_start_idx = tl.load(block_start_idx_ptr + PB_idx) # M block starts from here
    M_block = tl.max_contiguous(block_start_idx + M_range, BLOCK_M)

    # Assumption: expert_idxs_ptr is a sorted list of expert ids.
    # - load expert_idxs_ptr into E_idxs
    # - construct the E_mask so we operate only on expert for this kernel instance (e.g., E_idx)
    # - in cases where the M_block may overlap multiple experts, then tl.min(E_idxs) or 
    #   E_idxs[0] (if it is sorted) can be used to infer the expert being worked on
    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_block < (FAN_OUT * M), other=E)
    E_idx = tl.min(E_idxs) # if we do this we do not need to index the tensor
    E_mask = E_idxs == E_idx

    # Assumption: grouped_idx_ptr puts X in the order as expected by expert_idxs_ptr.
    # - same length as expert_idxs_ptr

    # depending on grouped settings, set M_in_idx (input) and M_out_idx (output) appropriately
    # - if already grouped, then M_idx is not required and use M_block
    if x_grouped:
        M_in_idx = M_block
    else:
        M_idx = tl.load(grouped_idx_ptr + M_block, mask=E_mask, other=0)
        M_in_idx = M_idx // FAN_OUT

    if y_grouped:
        M_out_idx = M_block
    else:
        M_idx = tl.load(grouped_idx_ptr + M_block, mask=E_mask, other=0)
        M_out_idx = M_idx

    # - K_block for input dimension
    K_block = tl.arange(0, BLOCK_K)

    # - N_block for output dimension
    N_block = N_block_id * BLOCK_N  + tl.arange(0, BLOCK_N)
    N_mask = N_block < N

    # - R range for lora dimension
    R_range = tl.arange(0, R)

    # X: dimensions M, K
    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk

    # W: dimensions E, K,         N
    # A: dimensions E, K, lora_r
    # B: dimensions E,    lora_r, N
    W_blk_ptrs = W_ptr + E_idx * stride_we + K_block[:, None] * stride_wk + N_block[None, :] * stride_wn
    A_blk_ptrs = A_ptr + E_idx * stride_ae + K_block[:, None] * stride_ak                                + R_range[None, :] * stride_ar
    B_blk_ptrs = B_ptr + E_idx * stride_be                                + N_block[None, :] * stride_bn + R_range[:, None] * stride_br

    # b can be loaded outside because it has no dependence on input dimension K
    b = tl.load(B_blk_ptrs, mask=N_mask[None, :])

    # accumulate loop over input dimension, for iters number of times
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    iters = tl.cdiv(K, BLOCK_K)
    for K_block_id in range(0, iters):

        # - load x, w, a quantities depending on NO_K_MASK or NO_N_MASK
        if NO_K_MASK:
            # - if K mask not required
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            a = tl.load(A_blk_ptrs)

            if NO_N_MASK or K_block_id < (iters - 1):
                # - if N mask also not reqiured
                w = tl.load(W_blk_ptrs)
            else:
                w = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            # - construct K mask (NO_N_MASK has no effect here)
            K_mask = (K_block_id * BLOCK_K + K_block) < K
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])
            a = tl.load(A_blk_ptrs, mask=K_mask[:, None])

        # Y = X * (W + A*B*scaling)
        #   = X * W + X * A * B * scaling
        # - acummulate base layer
        acc += tl.dot(x, w, allow_tf32=allow_tf32, out_dtype=ACC_TYPE)

        # - accumulate adapter
        # - interim = X * A * scaling
        # - interm wil be of dimensions M_block by lora_r
        interim = tl.dot(x, a)
        interim *= scaling
        acc += tl.dot(interim.to(b.dtype), b, allow_tf32=allow_tf32, out_dtype=ACC_TYPE)

        # move pointers in K
        # NOTE: b has no dependence on K, so it doesnt need to move
        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        A_blk_ptrs += BLOCK_K * stride_ak

    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=E_mask[:, None] & N_mask[None, :])

# This is the lora enabled version of scatter2scatter, where alongisde the weights
# W, we take in the adapters A and B.
# - in lora adaption the combined weights are W + A*B*scaling
# - scaling is typically lora_alp / lora_r
def scatter2scatter_lora(
    X, W, A, B, lora_alp,
    sorted_expert_idxs, sorted_scattered_idxs, k,
    padded_block_idxs, x_grouped=False, y_grouped=False,
    out=None
):
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * k

    assert W.size(1) == A.size(1), "A has incorrect input size."
    assert W.size(2) == B.size(2), "B has incorrect output size."
    assert A.size(2) == B.size(1), "A and B have inconsistent inner dims."

    # Pre-kernel setup
    x_dim = X.size(-1)
    y_dim = W.size(-1)
    L_scattered = sorted_expert_idxs.size(0)
    if out is None:
        O = torch.empty((L_scattered, y_dim), device=X.device, dtype=X.dtype)
    else:
        assert out.size(0) == L_scattered and out.size(1) == y_dim
        O = out

    def grid(META):
        grid_num = (
            padded_block_idxs.size(0) *
            triton.cdiv(META['N'], META['BLOCK_N']),
        )
        return grid_num
    with torch.cuda.device(X.device):
        _scatter2scatter_lora[grid](
            # X_ptr, stride_xm, stride_xk,
            X, X.stride(0), X.stride(1),
            # W_ptr, stride_we, stride_wk, stride_wn,
            W, W.stride(0), W.stride(1), W.stride(2),
            # A_ptr, stride_ae, stride_ak, stride_ar,
            A, A.stride(0), A.stride(1), A.stride(2),
            # B_ptr, stride_be, stride_br, stride_bn,
            B, B.stride(0), B.stride(1), B.stride(2),
            # Y_ptr, stride_ym, stride_yn,
            O, O.stride(0), O.stride(1),
            grouped_idx_ptr=sorted_scattered_idxs,
            expert_idxs_ptr=sorted_expert_idxs,
            block_start_idx_ptr=padded_block_idxs,
            FAN_OUT=k,
            M=X.size(0),
            K=X.size(1),
            N=O.size(1), E=W.size(0),
            R=A.size(2),
            BLOCK_M=BLOCK_M,
            ACC_TYPE=tl.float32,
            OUT_M=O.size(0),
            scaling=(lora_alp / A.size(2)),
            allow_tf32=True,
            x_grouped=x_grouped, y_grouped=y_grouped,
        )
        return O


def _config_XtY():
    return [
        triton.Config(
            {
                'BLOCK_N': 128, 'BLOCK_K': 128, 'BLOCK_M': 32, 
                'GROUP_K': 4, 'CHUNK_M': 128,
            }, 
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {
                'BLOCK_N': 128, 'BLOCK_K': 128, 'BLOCK_M': 128, 
                'GROUP_K': 4, 'CHUNK_M': 128,
            }, 
            num_stages=1, num_warps=4
        ),
    ]
    # return [
    #     triton.Config(
    #         {
    #             'BLOCK_N': 16, 'BLOCK_K': 16, 'BLOCK_M': 16, 
    #             'GROUP_K': 2, 'CHUNK_M': 2,
    #         }, 
    #         num_stages=1, num_warps=1
    #     ),
    # ]

# this is version2 that chunks the sequence (M) dimension into groups.
# - the M dimension will get chunked to scale as M increases.
# - each chunk will take CHUNK_M number of BLOCK_M.
# - each accumulate will be performed in G groups. Each chunk gets assigned to one of the groups from 0..G-1.
#   They compute in parallel.
# - If two chunks get assigned to one of the groups, then they compute in series. They is a locking mechanism. Each
#   group in G consumes addiitional memory due to an extra buffer for the accum. The buffers are reduced at the 
#   end of the compute.
# - A single chunk may process more than one expert if the chunk covers sequences that involve
#   multiple experts. This may more so happen when BLOCK_M * CHUNK_M is large
# - Similarly, the (K, N) tiles are independent of each other, but we also hash them in buckets. This is controlled
#   by P. Unlike G, having a large P incurs minimum additional memory (only E * G lock entries).
def group_bwd_W_v2(DY, X, expert_offsets, E, G=1, P=1):

    # - gridded by (K, N) tile and a BLOCK_M * CHUNK_M chunk in the inner dimension
    def grid(META):
        grid = (
            triton.cdiv(META['K'], META['BLOCK_K']),
            triton.cdiv(META['N'], META['BLOCK_N']),
            triton.cdiv(META['M'], META['BLOCK_M'] * META['CHUNK_M']),
        )
        return grid

    M, K, N = X.size(0), X.size(-1), DY.size(-1)

    # - need to ensure that stuff is properly zeroed since
    #   there is no zeroing inside the kernel.
    # - allow G groups for each of the E experts
    DW = torch.zeros((E * G, K, N), device=DY.device, dtype=DY.dtype)

    # Locks
    # - we require P series of locks, one each for the tiles (K,N)
    # - then of the P's will have E * G locks, for the experts X CHUNK_M
    Lock = torch.zeros(P * E * G, dtype=torch.int32, device=DY.device)
    
    with torch.cuda.device(DY.device):
        _groupXtY_v2[grid](
            # DY_ptr, stride_dym, stride_dyk,
            DY, DY.stride(0), DY.stride(1),
            # X_ptr, stride_xm, stride_xn,
            X, X.stride(0), X.stride(1),
            # DW_ptr, stride_dweg, stride_dwk, stride_dwn,
            DW, DW.stride(0), DW.stride(1), DW.stride(2),
            # expert_offsets_ptr,
            expert_offsets, Lock,
            # K: tl.constexpr, N: tl.constexpr,
            M=M, N=N, K=K, E=E,
            G=G, P=P,
            # ACC_TYPE: tl.constexpr,
            ACC_TYPE=tl.float32,
            allow_tf32=True
        )
        # - need to investigate if this needs to be replaced with a kernel or not
        return DW.view(E, -1, K, N).sum(1)  # reduce over groups


# use locking across programs
# - https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
@triton.autotune(configs=_config_XtY(), key=['N', 'K'], )
@triton.heuristics({
    "NO_K_MASK": lambda args: (args['K'] % args['BLOCK_K']) == 0,
    "NO_N_MASK": lambda args: (args['N'] % args['BLOCK_N']) == 0,
})
@triton.jit
def _groupXtY_v2(
    DY_ptr, stride_dym, stride_dyk,
    X_ptr, stride_xm, stride_xn,
    DW_ptr, stride_dweg, stride_dwk, stride_dwn,
    expert_offsets_ptr,
    Lock, 
    M, K: tl.constexpr, N: tl.constexpr, E: tl.constexpr, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr, # for swizzling
    CHUNK_M: tl.constexpr, # number of BLOCK_M's used to form a chunk
    G: tl.constexpr, # number of chunks computed in parallel
    P: tl.constexpr, # number of tiles computed in parallel
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
    NO_K_MASK: tl.constexpr, NO_N_MASK: tl.constexpr,
):

    # first dimension is the tile index into K (input dim)
    # second dimension is the tile index into N (output dim)
    # third dimension is the group index on M (inputs)
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    pid2 = tl.program_id(axis=2)
    num0 = tl.num_programs(0)
    num1 = tl.num_programs(1)

    # - swizzle it up to improve cache usage
    pid0, pid1 = tl.swizzle2d(pid0, pid1, num0, num1, GROUP_K)

    # - rename the pids into something more readable
    K_block_id = pid0
    N_block_id = pid1
    M_chunk_id = pid2

    # find the starting point of the M chunk
    start_idx = M_chunk_id * BLOCK_M * CHUNK_M

    # - find the expert that corresponds involved in the sequence starting
    #   at start_idx
    # - if offset == M_start => ends at offset-1, so exclude
    E_idx = 0
    end_idx = tl.load(expert_offsets_ptr).to(tl.int32)
    while E_idx < E and end_idx <= start_idx:
        E_idx += 1
        expert_offsets_ptr += 1

        if E_idx < E:
            # - guard against illegal access
            end_idx = tl.load(expert_offsets_ptr).to(tl.int32)

    # - each program will stop at the end of the M chunk
    # - this may cover multiple experts depending on values in expert_offsets_ptr
    M_block_end = (M_chunk_id + 1) * BLOCK_M * CHUNK_M

    # - K and N tile ranges for the current program
    K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    K_mask = K_block < K
    K_block = tl.max_contiguous(tl.multiple_of(K_block % K, BLOCK_K), BLOCK_K)

    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)

    # - use primes to distribute to bin the tile
    # - TODO: replace with larger primes
    tile_bin = (K_block_id * 11 + N_block_id * 13) % P

    # - get the E * G locks corresponding to the tile bin
    Lock += tile_bin * E * G

    # - get the group id for the chunk (within the expert E_idx)
    grp_id = M_chunk_id % G

    # - get the lock 
    Lock += E_idx * G + grp_id

    # - the invariant for the inner loop iterations in the chunk that may take place 
    #   across muliple experts (start_idx < M_block_end)
    # - process tiles (K, N) tiles of size BLOCK_M. Process a total of CHUNK_M such tiles
    # - the outer iteration handles expert E_idx by:
    #   * shifting the pointer start_idx and end_idx
    # - the inner iteration goes from start_idx to end_idx or M_block_end, whichever is closer
    #   * if end_idx < M_block_end, then there are leftover sequences in this CHUNK_M corresponding
    #     to the next expert. Which then iterates to the outer loop to be handled.
    # - at th end of the inner iteration, a Lock is grabbed for the (E, G) to update the DW buffer
    while E_idx < E and start_idx < M_block_end:

        # - guarded against illegal access by E_idx < E above
        end_idx = tl.load(expert_offsets_ptr).to(tl.int32)

        # - M tiling
        # - M_block set to start from start_idx
        M_block = tl.max_contiguous(start_idx + tl.arange(0, BLOCK_M), BLOCK_M)

        xt_blk_ptrs = X_ptr + K_block[:, None] * stride_xn + M_block[None, :] * stride_xm
        dy_blk_ptrs = DY_ptr + M_block[:, None] * stride_dym + N_block[None, :] * stride_dyk

        # - for output for expert E_idx
        DW_blk_ptrs = DW_ptr + (E_idx * G + grp_id) * stride_dweg + K_block[:, None] * stride_dwk + N_block[None, :] * stride_dwn

        # - process for a current expert that runs between 
        #   start_idx and min(end_idx, M_block_end)
        # - if the expert runs over M_block_end, then it will be processed
        #   by the next program
        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=ACC_TYPE)

        # - this is the end_idx that takes into account the M_block_end
        end_idx2 = tl.minimum(end_idx, M_block_end)
        iters = tl.cdiv(end_idx2 - start_idx, BLOCK_M)
        for i in range(0, iters):
            M_mask = (i * BLOCK_M + M_block) < end_idx2
            if NO_K_MASK:
                xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :])
            else:
                xt = tl.load(xt_blk_ptrs, mask=K_mask[:, None] & M_mask[None, :])
            if NO_N_MASK:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None])
            else:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None] & N_mask[None, :])

            xt_blk_ptrs += BLOCK_M * stride_xm
            dy_blk_ptrs += BLOCK_M * stride_dym
            acc += tl.dot(xt, dy, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)

        # accum is done over BLOCK_M * CHUNK_M samples
        # - grab a lock to try to update the DW buffer
        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass

        # Lock aquired. 
        # - load the partial compute in the DW buffer
        # - add the partial compute to acc
        # partial = tl.load(DW_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])
        # acc += partial

        # - store the accum back to the DW buffer
        acc = acc.to(DW_blk_ptrs.dtype.element_ty)
        tl.store(DW_blk_ptrs, acc, mask=K_mask[:, None] & N_mask[None, :])

        # - release Lock
        tl.atomic_xchg(Lock, 0)

        # Set end_idx to start_idx as we prepare for the outer iterate
        # - if start_idx < M_block_end, then there are some more sequences to cover for
        #   E_idx + 1
        # - if start_idx >= M_block_end, then the outer iterate will break, in which case the 
        #   value of E_idx, Lock, and offsets are irrelevant
        start_idx = end_idx

        # increment E_idx and Lock
        # - each expert has G locks, so add G
        E_idx += 1

        # - may go out of bounds, but we will not load if if E_idx < E check fails above
        Lock += G
        expert_offsets_ptr += 1


def group_bwd_AB(DY, X, A, B, scaling, expert_offsets, E):

    assert A.size(2) == B.size(1), "A and B have inconsistent inner dims."

    DA = torch.zeros((E, X.size(-1), A.size(2)), device=DY.device, dtype=DY.dtype)
    DB = torch.zeros((E, B.size(1), DY.size(-1)), device=DY.device, dtype=DY.dtype)
    def grid(META):
        grid = (
            E * triton.cdiv(META['K'], META['BLOCK_K']),
            triton.cdiv(META['N'], META['BLOCK_N']),
        )
        return grid
    
    with torch.cuda.device(DY.device):
        _groupXtY_lora[grid](
            # DY_ptr, stride_dym, stride_dyn,
            DY, DY.stride(0), DY.stride(1),
            # X_ptr, stride_xm, stride_xk,
            X, X.stride(0), X.stride(1),
            # DA_ptr, stride_dae, stride_dak, stride_dar
            DA, DA.stride(0), DA.stride(1), DA.stride(2),
            # DB_ptr, stride_dbe, stride_dbr, stride_dbn,
            DB, DB.stride(0), DB.stride(1), DB.stride(2),
            # A_ptr, stride_ae, stride_ak, stride_ar,
            A, A.stride(0), A.stride(1), A.stride(2),
            # B_ptr, stride_be, stride_br, stride_bn,
            B, B.stride(0), B.stride(1), B.stride(2),
            # expert_offsets_ptr,
            expert_offsets,
            # K: tl.constexpr, N: tl.constexpr,
            M=DY.size(0), N=DY.size(-1), K=X.size(-1), R=A.size(2),
            scaling=scaling,
            ACC_TYPE=tl.float32,
            allow_tf32=True
        )
        return DA, DB

@triton.autotune(configs=_config_XtY(), key=['M', 'N', 'K'], )
@triton.heuristics({
    "NO_K_MASK": lambda args: (args['K'] % args['BLOCK_K']) == 0,
    "NO_N_MASK": lambda args: (args['N'] % args['BLOCK_N']) == 0,
})
@triton.jit
def _groupXtY_lora(
    DY_ptr, stride_dym, stride_dyn,
    X_ptr, stride_xm, stride_xk,
    DA_ptr, stride_dae, stride_dak, stride_dar,
    DB_ptr, stride_dbe, stride_dbr, stride_dbn,
    A_ptr, stride_ae, stride_ak, stride_ar,  # transposed
    B_ptr, stride_be, stride_br, stride_bn,  # transposed
    expert_offsets_ptr,
    M, K: tl.constexpr, N: tl.constexpr, R: tl.constexpr,
    scaling,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
    NO_K_MASK: tl.constexpr, NO_N_MASK: tl.constexpr
):

    # this function is required for computing the weight gradients as per the
    # lora updates:
    # Y = X * (W + A*B*scaling)
    #   = X * W + X * A * B * scaling

    # Consider a function f over domain R^n, i.e., f(Y)
    # - Let DY the be gradients flowing backwards, i.e., DY = df/dY
    # - Let DY be of dimensions M, N, where N is from the domain of f and M is 
    #   sequence dimension.
    # - then the gradients DA for adapter A will be DA = X^t * DY * B^T * scaling
    #   and the gradients DB for adapter B will be DB = A^t * X^t * DY * scaling

    # The 2D grid is assumed to be:
    # (num_experts * K_BLOCK_COUNT, N_BLOCK_COUNT)
    # - the input dimension is K
    # - the output dimension is N
    # - X and DY are assumed to be grouped.
    # - expert_offsets_ptr are the offsets for accessing the expert groups.

    # handling pid0 and pid1
    # - memory swizzling for minimizing shared memory conflicts
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    num0 = tl.num_programs(0)
    num1 = tl.num_programs(1)
    pid1, pid0 = tl.swizzle2d(pid1, pid0, num1, num0, 128)

    # - get E_idx, K_block_id, N_block_id
    K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    E_idx = pid0 // K_BLOCK_COUNT
    K_block_id = pid0 % K_BLOCK_COUNT
    N_block_id = pid1

    # - get the offset and ending of the expert group
    # - use E_idx that points ot the expert, 
    # - start_idx is the offset. end_idx indicates where the group ends.
    if E_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(expert_offsets_ptr + E_idx - 1).to(tl.int32)
    end_idx = tl.load(expert_offsets_ptr + E_idx).to(tl.int32)

    # - get the K_block for the input dimension
    K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    K_mask = K_block < K
    K_block = tl.max_contiguous(tl.multiple_of(K_block % K, BLOCK_K), BLOCK_K)

    # - get the N_block for the output dimension
    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)

    # - R range for lora dimension
    R_range = tl.arange(0, R)

    # - M block for indices (sequence) dimension
    M_block = tl.max_contiguous(start_idx + tl.arange(0, BLOCK_M), BLOCK_M)

    # At: dimensions E, lora_r, K (transposed)
    # Bt: dimensions E, N, lora_r (transposed)
    At_blk_ptrs = A_ptr + E_idx * stride_ae + K_block[None, :] * stride_ak + R_range[:, None] * stride_ar
    Bt_blk_ptrs = B_ptr + E_idx * stride_be + N_block[:, None] * stride_bn + R_range[None, :] * stride_br

    # - iterate over the (grouped) expert indices (sequence)
    # - check if end_idx and start_idx are valid
    if end_idx > start_idx:

        # - get the at and bt (transposed) weights
        # - do not depend on indices dimension so can be loaded outside of loop
        if NO_K_MASK:
            at = tl.load(At_blk_ptrs)
        else:
            at = tl.load(At_blk_ptrs, mask=K_mask[None, :])

        if NO_N_MASK:
            bt = tl.load(Bt_blk_ptrs)
        else:
            bt = tl.load(Bt_blk_ptrs, mask=N_mask[:, None])

        # - prepare for iteration
        # - xt (transposed) created from (un-transposed) X_ptr
        xt_blk_ptrs = X_ptr + M_block[None, :] * stride_xm + K_block[:, None] * stride_xk
        dy_blk_ptrs = DY_ptr + M_block[:, None] * stride_dym + N_block[None, :] * stride_dyn

        # - for accumulation
        acc_A = tl.zeros((BLOCK_K, R), dtype=ACC_TYPE)
        acc_B = tl.zeros((R, BLOCK_N), dtype=ACC_TYPE)
        iters = tl.cdiv(end_idx - start_idx, BLOCK_M)

        # - iterate
        for i in range(0, iters):

            # - get the (sequence) M mask 
            M_mask = (i * BLOCK_M + M_block) < end_idx

            # - load xt and dy
            if NO_K_MASK:
                xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :])
            else:
                xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :] & K_mask[:, None])
            if NO_N_MASK:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None])
            else:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None] & N_mask[None, :])

            # compute DA = X^t * DY * B^T * scaling
            interm = tl.dot(dy, bt)
            interm *= scaling
            acc_A += tl.dot(xt, interm.to(xt.dtype), out_dtype=ACC_TYPE, allow_tf32=allow_tf32)

            # compute DB = A^t * X^t * DY * scaling
            interm = tl.dot(at, xt)
            interm *= scaling
            acc_B += tl.dot(interm.to(dy.dtype), dy, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)

            # - move pointers
            xt_blk_ptrs += BLOCK_M * stride_xm
            dy_blk_ptrs += BLOCK_M * stride_dym

        # - store output for DA and DB
        DA_blk_ptrs = DA_ptr + E_idx * stride_dae + K_block[:, None] * stride_dak + R_range[None, :] * stride_dar
        acc_A = acc_A.to(DA_blk_ptrs.dtype.element_ty)
        tl.store(DA_blk_ptrs, acc_A, mask=K_mask[:, None])

        DB_blk_ptrs = DB_ptr + E_idx * stride_dbe + N_block[None, :] * stride_dbn + R_range[:, None] * stride_dbr
        acc_B = acc_B.to(DB_blk_ptrs.dtype.element_ty)
        tl.store(DB_blk_ptrs, acc_B, mask=N_mask[None, :])

