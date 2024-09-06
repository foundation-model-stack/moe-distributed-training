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
    A_ptr, stride_ae, stride_ak, 
    B_ptr, stride_be, stride_bn, 
    Y_ptr, stride_ym, stride_yn,
    grouped_idx_ptr, expert_idxs_ptr, block_start_idx_ptr,
    FAN_OUT: tl.constexpr,
    M, K: tl.constexpr, N: tl.constexpr, E: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    OUT_M,
    scaling,
    allow_tf32: tl.constexpr,
    x_grouped: tl.constexpr, y_grouped: tl.constexpr,
    NO_K_MASK: tl.constexpr, NO_N_MASK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    M_block_id = pid // N_BLOCK_COUNT
    N_block_id = pid % N_BLOCK_COUNT
    M_range = tl.arange(0, BLOCK_M)
    block_start_idx = tl.load(block_start_idx_ptr + M_block_id)
    # M_block = tl.max_contiguous((block_start_idx + M_range) % OUT_M, BLOCK_M)
    M_block = tl.max_contiguous(block_start_idx + M_range, BLOCK_M)
    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_block < (FAN_OUT * M), other=E)
    E_idx = tl.min(E_idxs)
    E_mask = E_idxs == E_idx
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=E_mask, other=0)
    if x_grouped:
        M_in_idx = M_block
    else:
        M_in_idx = M_idx // FAN_OUT

    if y_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx

    K_block = tl.arange(0, BLOCK_K)

    N_block = N_block_id * BLOCK_N  + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    # N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)
    # N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)

    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
    W_blk_ptrs = W_ptr + K_block[:, None] * stride_wk + N_block[None, :] * stride_wn + E_idx * stride_we
    A_blk_ptrs = A_ptr + K_block[:, None] * stride_ak + E_idx * stride_ae
    B_blk_ptrs = B_ptr + N_block[None, :] * stride_bn + E_idx * stride_be

    # b can be loaded outside because it has no dependence on K
    b = tl.load(B_blk_ptrs, mask=N_mask[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    iters = tl.cdiv(K, BLOCK_K)
    for K_block_id in range(0, iters):
        if NO_K_MASK:
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            if NO_N_MASK or K_block_id < (iters - 1):
                w = tl.load(W_blk_ptrs)
            else:
                w = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = (K_block_id * BLOCK_K + K_block) < K
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])
            a = tl.load(A_blk_ptrs, mask=K_mask[:, None])

        # accumulate
        # - acummulate the base layer
        acc += tl.dot(x, w, allow_tf32=allow_tf32, out_dtype=ACC_TYPE)

        # - accumulate the adapter
        interim = tl.dot(x, a, allow_tf32=allow_tf32, out_dtype=ACC_TYPE)
        interim *= scaling
        acc += tl.dot(interim, b, allow_tf32=allow_tf32, out_dtype=ACC_TYPE)

        # move pointers in K
        # NOTE: b has no dependence on K, so it doesnt need to move
        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        A_blk_ptrs += BLOCK_K * stride_ak

    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=E_mask[:, None] & N_mask[None, :])

def scatter2scatter_lora(
    X, W, A, B, scaling,
    sorted_expert_idxs, sorted_scattered_idxs, k,
    padded_block_idxs, x_grouped=False, y_grouped=False,
    out=None
):
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * k

    ## TODO: do size checks on A, B

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
            # A_ptr, stride_ae, stride_ak, 
            A, A.stride(0), A.stride(1), 
            # B_ptr, stride_be, stride_bn, 
            B, B.stride(0), B.stride(1), 
            # Y_ptr, stride_ym, stride_yn,
            O, O.stride(0), O.stride(1),
            grouped_idx_ptr=sorted_scattered_idxs,
            expert_idxs_ptr=sorted_expert_idxs,
            block_start_idx_ptr=padded_block_idxs,
            FAN_OUT=k,
            M=X.size(0),
            K=X.size(1),
            N=O.size(1), E=W.size(0),
            BLOCK_M=BLOCK_M,
            ACC_TYPE=tl.float32,
            OUT_M=O.size(0),
            alpha=scaling,
            allow_tf32=True,
            x_grouped=x_grouped, y_grouped=y_grouped,
        )
        return O


def _config_XtY():
    return [
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'BLOCK_M': 32}, num_stages=4, num_warps=4),
    ]

def group_bwd_AB(DY, X, A, B, expert_offsets, E, lora_alp, lora_r):
    DA = torch.zeros((E, X.size(-1), lora_r), device=DY.device, dtype=DY.dtype)
    DB = torch.zeros((E, lora_r, DY.size(-1)), device=DY.device, dtype=DY.dtype)
    def grid(META):
        grid = (
            E * triton.cdiv(META['K'], META['BLOCK_K']),
            triton.cdiv(META['N'], META['BLOCK_N']),
        )
        return grid
    
    with torch.cuda.device(DY.device):
        _groupXtY_lora[grid](
            # DY_ptr, stride_dym, stride_dyk,
            DY, DY.stride(0), DY.stride(1),
            # X_ptr, stride_xm, stride_xn,
            X, X.stride(0), X.stride(1),
            # DA_ptr, stride_dae, stride_dak,
            DA, DA.stride(0), DA.stride(1),
            # DB_ptr, stride_dbe, stride_dbn,
            DB, DB.stride(0), DB.stride(1), 
            # A_ptr, stride_ae, stride_ak, 
            A.permute(0,2,1), A.stride(0), A.stride(2), 
            # B_ptr, stride_be, stride_bn, 
            B.permute(0,2,1), B.stride(0), B.stride(2), 
            # expert_offsets_ptr,
            expert_offsets,
            # K: tl.constexpr, N: tl.constexpr,
            M=DY.size(0), N=DY.size(-1), K=X.size(-1),
            lora_r=lora_r, lora_alp=lora_alp,
            # ACC_TYPE: tl.constexpr,
            ACC_TYPE=tl.float32,
            allow_tf32=True
        )
        return DA.permute(0, 2, 1), DB.permute(0,2,1)

@triton.autotune(configs=_config_XtY(), key=['M', 'N', 'K'], )
@triton.heuristics({
    "NO_K_MASK": lambda args: (args['K'] % args['BLOCK_K']) == 0,
    "NO_N_MASK": lambda args: (args['N'] % args['BLOCK_N']) == 0,
})
@triton.jit
def _groupXtY_lora(
    DY_ptr, stride_dym, stride_dyk,
    X_ptr, stride_xm, stride_xn,
    DA_ptr, stride_dae, stride_dak, 
    DB_ptr, stride_dbe, stride_dbn,
    At_ptr, stride_ae, stride_ak,  # transposed
    Bt_ptr, stride_be, stride_bn,  # transposed
    expert_offsets_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    lora_r, lora_alp,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
    NO_K_MASK: tl.constexpr, NO_N_MASK: tl.constexpr
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    num0 = tl.num_programs(0)
    num1 = tl.num_programs(1)
    pid1, pid0 = tl.swizzle2d(pid1, pid0, num1, num0, 128)

    K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    E_idx = pid0 // K_BLOCK_COUNT
    K_block_id = pid0 % K_BLOCK_COUNT
    N_block_id = pid1

    if E_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(expert_offsets_ptr + E_idx - 1).to(tl.int32)
    end_idx = tl.load(expert_offsets_ptr + E_idx).to(tl.int32)

    K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    K_mask = K_block < K
    K_block = tl.max_contiguous(tl.multiple_of(K_block % K, BLOCK_K), BLOCK_K)

    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)

    # the K and B_blocks are transposed
    At_blk_ptrs = At_ptr + K_block[None, :] * stride_ak + E_idx * stride_ae
    Bt_blk_ptrs = Bt_ptr + N_block[:, None] * stride_bn + E_idx * stride_be

    if NO_K_MASK:
        a = tl.load(At_blk_ptrs)
        bt = tl.load(Bt_blk_ptrs)
    else:
        # the masks are transposed
        a = tl.load(At_blk_ptrs, mask=K_mask[None, :])
        bt = tl.load(Bt_blk_ptrs, mask=N_mask[:, None])

    scaling = lora_alp / lora_r
    if end_idx > start_idx:
        M_block = tl.max_contiguous(start_idx + tl.arange(0, BLOCK_M), BLOCK_M)

        M_idxs = M_block
        xt_blk_ptrs = X_ptr + K_block[:, None] * stride_xn + M_idxs[None, :] * stride_xm
        dy_blk_ptrs = DY_ptr + M_idxs[:, None] * stride_dym + N_block[None, :] * stride_dyk

        acc_A = tl.zeros((BLOCK_K, lora_r), dtype=ACC_TYPE)
        acc_B = tl.zeros((lora_r, BLOCK_N), dtype=ACC_TYPE)
        iters = tl.cdiv(end_idx - start_idx, BLOCK_M)
        for i in range(0, iters):
            M_mask = (i * BLOCK_M + M_block) < end_idx
            if NO_K_MASK:
                xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :])
            else:
                xt = tl.load(xt_blk_ptrs, mask=K_mask[:, None] & M_mask[None, :])
            if NO_N_MASK:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None])
            else:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None] & N_mask[None, :])
            # acc += tl.dot(xt, dy, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)
            xt_blk_ptrs += BLOCK_M * stride_xm
            dy_blk_ptrs += BLOCK_M * stride_dym

            interm = tl.dot(dy, bt, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)
            interm *= scaling
            acc_A += tl.dot(xt, interm, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)

            interm = tl.dot(at, x, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)
            interm *= scaling
            acc_B += tl.dot(interm, dy, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)


        # this one is not transposed
        DA_blk_ptrs = DA_ptr + E_idx * stride_dae + K_block[:, None] * stride_dak 
        acc_A = acc_A.to(DA_blk_ptrs.dtype.element_ty)
        tl.store(DA_blk_ptrs, acc_A, mask=K_mask[:, None])

        DB_blk_ptrs = DB_ptr + E_idx * stride_dbe + N_block[None, :] * stride_dbn
        acc_B = acc_B.to(DB_blk_ptrs.dtype.element_ty)
        tl.store(DB_blk_ptrs, acc_B, mask=N_mask[None, :])

