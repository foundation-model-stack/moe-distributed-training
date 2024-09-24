
import torch
import triton
import triton.language as tl

def repro(DY, X, E, G=1, P=1):

    # - gridded by (K, N) tile and a BLOCK_M * CHUNK_M chunk in the inner dimension
    def grid(META):
        grid = (
            triton.cdiv(META['K'], META['BLOCK_K']),
            triton.cdiv(META['N'], META['BLOCK_N']),
            triton.cdiv(META['M'], META['BLOCK_M'] * META['CHUNK_M']),
        )
        return grid

    # - need to ensure that stuff is properly zeroed since
    #   there is no zeroing inside the kernel.
    # - allow G groups for each of the E experts
    DW = torch.zeros((E * G, X.size(-1), DY.size(-1)), device=DY.device, dtype=DY.dtype)

    with torch.cuda.device(DY.device):
        _repro[grid](
            DW, DW.stride(0), DW.stride(1), DW.stride(2),
            M=DY.size(0),
            N=DY.size(-1), K=X.size(-1), E=E,
            G=G, 
            ACC_TYPE=tl.float32,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
            GROUP_K=4, CHUNK_M=128,
        )
        # - need to investigate if this needs to be replaced with a kernel or not
        return DW  # reduce over groups


@triton.jit
def _repro(
    DW_ptr, stride_dweg, stride_dwk, stride_dwn,
    M,
    K: tl.constexpr, N: tl.constexpr, E: tl.constexpr, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr, # for swizzling
    CHUNK_M: tl.constexpr, # number of BLOCK_M's used to form a chunk
    G: tl.constexpr, # number of chunks computed in parallel
    ACC_TYPE: tl.constexpr,
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

    # - K and N tile ranges for the current program
    K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    K_mask = K_block < K
    K_block = tl.max_contiguous(tl.multiple_of(K_block % K, BLOCK_K), BLOCK_K)

    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)

    while E_idx < E:

        # - for output for expert E_idx
        DW_blk_ptrs = DW_ptr + (E_idx * G) * stride_dweg + K_block[:, None] * stride_dwk + N_block[None, :] * stride_dwn

        # - process for a current expert that runs between 
        #   start_idx and min(end_idx, M_block_end)
        # - if the expert runs over M_block_end, then it will be processed
        #   by the next program
        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=ACC_TYPE)

        # - store the accum back to the DW buffer
        acc = acc.to(DW_blk_ptrs.dtype.element_ty)
        tl.store(DW_blk_ptrs, acc, mask=K_mask[:, None] & N_mask[None, :])

        E_idx += 1

if __name__ == '__main__':


    # small example

    torch.manual_seed(0)

    K, N = (4096, 14336)
    M = 100000
    E = 4

    X = torch.rand((M, K)).to('cuda')
    DY = torch.randn((M, N)).to('cuda')

    repro(
        DY,
        X,
        E=4,
        G=16,
    )
        