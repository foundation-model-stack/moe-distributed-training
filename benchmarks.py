import sys, os
from typing import List

import numpy as np
import torch
import time
from tqdm import trange

# for testing kernels
# copied from
# https://github.com/IST-DASLab/marlin/blob/master/bench.py
def benchmark(f, warmup=1, iter=10):
    for i in trange(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return res


from scattermoe_utils.kernels.ops import scatter2scatter_lora, group_bwd_AB
from scattermoe_utils.kernels.ops import group_bwd_W_v2

LOADED_KERNEL_HYPERDRIVE = False
if os.environ.get('LOAD_KERNEL_HYPERDRIVE', 'true') == "true":
    try:
        # this is from Mayank's repo
        from khd.kernels.scattermoe.triton_implementation.ops import scatter2scatter, group_bwd_W, padded_block_indices
        LOADED_KERNEL_HYPERDRIVE = True
        print('scatter kernels loaded kernel-hyperdrive')
    except ImportError as e:
        pass

if not LOADED_KERNEL_HYPERDRIVE:
    from scattermoe.kernels.ops import scatter2scatter, group_bwd_W, padded_block_indices

import torch
import inspect

def benchmark_group(
    DY, X, A, B, expert_offsets, fn_args={}, **kwargs
):

    if A is None or B is None:
        if 'P' in fn_args:
            return benchmark(
                lambda: group_bwd_W_v2(
                    DY, X, expert_offsets, E=len(expert_offsets), 
                    G=fn_args['G'], P=fn_args['P']
                ),
                **kwargs
            )

        if 'DW' in inspect.signature(group_bwd_W).parameters:
            DW = torch.zeros(
                (len(expert_offsets), X.size(-1), DY.size(-1)),
                device=DY.device, dtype=DY.dtype
            )
            return benchmark(
                lambda: group_bwd_W(DY, X, expert_offsets, DW=DW, E=len(expert_offsets)),
                **kwargs
            )
        else:
            return benchmark(
                lambda: group_bwd_W(DY, X, expert_offsets, E=len(expert_offsets)),
                **kwargs
            )

    return benchmark(
        lambda: group_bwd_AB(DY, X, A, B, 1., expert_offsets, E=len(expert_offsets)),
        **kwargs
    )

def get_histograms(
    rng: np.random.Generator,
    num_experts: int,
    space: int = 100000,
    sampling: str = 'order-stat',
    **kwargs,
):
    P = rng.random((space, num_experts))
    # P[:,0] *= impt_samp_scalar
    P = np.sort(-P, -1)
    P = P / P.sum(-1)[:, None]

    ent = lambda p: - sum( p * np.log(p) / np.log(2))
    entropy = np.apply_along_axis(ent, 1, P)
    idx = np.argsort(entropy)

    # 
    if sampling == 'order-stat':
        grid = kwargs.get('grid', 10)

        idx = idx[::int(space/grid)]
        return (
            entropy[idx], P[idx]
        )
    elif sampling == 'bisect':

        from bisect import bisect
        delta = kwargs.get('delta', .2)

        entropy = entropy[idx]

        d, end = 0, np.log(num_experts) / np.log(2)
        choices = []
        while True:
            found = bisect(entropy, d)
            if found == space:
                choices.append(space-1)
                break

            choices.append(found)
            d += delta

        choices = np.unique(choices)
        return (
            entropy[choices], P[idx[choices]]
        )

def run(
    sizes_inputs: List[int] = [10000, 100000],
    num_experts: int = [2, 4],
    num_lora: List[int] = [None],
    delta_p: float = .2,
    sampling_p: str = 'bisect',
    num_warmup: int = 100,
    num_trials: int = 200,
    num_repeats: int = 3,
):

    rng = np.random.default_rng(1)
    KN_VEC = [(4096, 14336)]

    print ("sizes_inputs", sizes_inputs)
    print ("num_experts", num_experts)
    print ("num_lora", num_lora)

    from itertools import product
    for rep in range(num_repeats):
        for M, (K,N), E, R in product(
            sizes_inputs, 
            KN_VEC, 
            num_experts, 
            num_lora,
        ):
            X = torch.rand((M, K)).to('cuda')
            # W = torch.rand((E, K, K)).to('cuda')

            entropy, P = get_histograms(
                rng,
                sampling=sampling_p,
                num_experts=E,
                delta=delta_p,
            )

            for p_idx, (v, p) in enumerate(zip(entropy, P)):

                O = rng.choice(range(0,E), p=p, size=M)
                bin_ids, indices = torch.sort(torch.tensor(O).to('cuda'))
                # print (bin_ids)

                _fn_args = {}
                if isinstance(R, int):
                    A = torch.randn((E, K, R)).to('cuda') / K
                    B = torch.randn((E, R, N)).to('cuda') / N
                elif isinstance(R, dict):
                    A, B = None, None 
                    _fn_args = R
                else:
                    A, B = None, None 
                DY = torch.randn((M, N)).to('cuda')

                padded_block_idxs, expert_offsets = padded_block_indices(bin_ids, 2)

                tot = benchmark_group(
                    DY, X, A, B, expert_offsets,
                    warmup=num_warmup,
                    iter=num_trials,
                    fn_args=_fn_args,
                )
                tot *= num_trials

                print({
                    'M': M, 'K': K, 'N': N, 'E': E, 
                    'R': R if R is not None else "no-lora", 
                    'repeat': rep + 1,
                    'idx': p_idx + 1,
                    # 'time': tot,
                    'throughput': M * round(num_trials / tot, 2),
                    'entropy': float(v),
                    'p': [
                        - float(round(np.log(x) / np.log(2), 2))
                        for x in p
                    ],
                    # 'kernel' : kernel
                })

if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity

    import fire
    fire.Fire(run)
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:

    # prof.export_chrome_trace("trace.json")

    # import numpy as np
    # from tqdm import trange
    # rng = np.random.default_rng(1)


    # KN_VEC = [(4096, 14336)]

    # M_VEC = [
    #     # 1000, 
    #     10000, 100000
    # ]
    # E_VEC = [2, 4]
    # R_VEC = [16, 32]
    # impt_samp_scalar = 1
    # wramup_trials = 100
    # trials = 200
    # repeats = 3
    # sampling = 'bisect'
    # # kernel = 'lora' 
    # kernel = 'orig'
    # delta = .2

    # hack grials 
    # M_VEC = [100000]
    # E_VEC = [4]
    # impt_samp_scalar = 1
    # wramup_trials = 100
    # trials = 200
    # repeats = 3
    # sampling = 'bisect'
    # kernel = 'orig'
    # # kernel = 'lora'
    # R_VEC = [16]

    # wramup_trials = 10
    # trials = 10
    # repeats = 1
    # delta = 1000



    # M_VEC = [10000]
    # # E_VEC = [4]
    # E_VEC = [4]
    # R_VEC = [16]
    # impt_samp_scalar = 1
    # wramup_trials = 100
    # trials = 100
    # repeats = 1

