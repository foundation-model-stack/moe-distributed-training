
import torch
from transformers import TrainingArguments, PretrainedConfig
from typing import Union, Dict, List, Type
from torch.distributed._tensor import Placement, Replicate, Shard, distribute_tensor
from torch.distributed._tensor.device_mesh import init_device_mesh
import os
from tqdm import tqdm

from safetensors import safe_open
import json, re
from collections import defaultdict

from accelerate import init_empty_weights

from contextlib import ExitStack

FILE_SAFETENSOR_INDEX = 'model.safetensors.index.json'
KEY_DATA_PARALLEL = 'data_parallel'
KEY_EXPERT_PARALLEL = 'expert_parallel'
DIM_EXPERT = 0

KEY_ROUTER = 'router.layer.weight'
KEY_EXPERTS = 'experts.mlp'

def get_moe_kwargs(
    config: PretrainedConfig,
    has_bias: bool = False, # if the MOE has bias
    fp16: bool = False,
    bf16: bool = False,
    mlp_impl: str = 'sparse',
    use_tensor_parallelism: bool = False,
):
    return {
        "hidden_size": config.hidden_size,
        "ffn_hidden_size": config.intermediate_size,
        "moe_num_experts": config.num_local_experts,
        "moe_top_k": config.num_experts_per_tok,
        "moe_expert_model_parallelism": not use_tensor_parallelism,
        "memory_optimized_mlp": False,
        "bias": has_bias,
        "moe_normalize_expert_weights": True,
        "fp16": fp16,
        "bf16": bf16,
        "mlp_impl": mlp_impl,
    }

# trick to get the resolved cache file to acccess the safetensor
# NOTE: this does not work if _dict_from_json_file, like GGUF files
def get_resolved_checkpoint_location(model_name_or_path: str):

    result = None
    _old_func = PretrainedConfig._dict_from_json_file
    def _dict_from_json_file(resolved_config_file):
        nonlocal result
        result = resolved_config_file
        return _old_func(resolved_config_file)

    # make a hook and restrive
    PretrainedConfig._dict_from_json_file = _dict_from_json_file
    PretrainedConfig.from_pretrained(model_name_or_path)
    PretrainedConfig._dict_from_json_file = _old_func
    return os.path.dirname(result)

# see https://github.com/mosaicml/llm-foundry/blob/main/tests/models/layers/test_dmoe.py
# for a basic example

# this one is called for one layer
# e.g., 'model.layers.0, block_sparse_moe
def get_router_experts_sharded_safetensor(
    weight_map: Dict,
    prefix: str, # e.g., 'model.layers.0,
    instance_name: str, # e.g., block_sparse_moe
    router_name: str = 'gate',
    expert_name: str = 'experts'
):
    # insert in order
    def _insert(L: List, i: int, v):
        n = len(L)
        if i < n:
            L[i] = v
            return 

        n = i - n + 1
        while n > 0:
            L.append(None)
            n -= 1
        L[i] = v

    # state dict -> weights
    # 'router.layer.weight': [(k, file),...]
    # `experts.mlp.w1`: [...]
    _map = defaultdict(list)
    prefix = f"{prefix}.{instance_name}."
    for k, stfile in weight_map.items():
        if not k.startswith(prefix):
            continue

        # e.g. after replacement we get
        # - gate.weight
        # - experts.0.w1.weight
        rel_k = k.replace(prefix, "")
        m = re.match(
            f'({router_name}|{expert_name})\.?(\d+)?\.?(\w+)?\.weight',
            rel_k
        )
        if m is None:
            raise ValueError(
                f"Unable to handle key '{k}' with provided router_name "
                f"'{router_name}' or expert_name '{expert_name}'"
            )
        if m.group(1) == router_name:
            _map[KEY_ROUTER].append((k, stfile))
        elif m.group(1) == expert_name:
            index = int(m.group(2))
            mod = m.group(3)
            # expert_map[stfile].append((mod, index, k))
            _insert(_map[f'{KEY_EXPERTS}.{mod}'], index, (k, stfile))

    if len(_map) == 0:
        raise ValueError(
            f"Could not get safetensor map for '{prefix}' and '{instance_name}'"
        )

    return _map

# for megablocks.SparseMLPv2
# assign dmoe with mlp_v2
# settings is: 
# experts.mlp.w1: [(k, file)]
def assign_mlp_v2_weights(
    dmoe: torch.nn.Module,
    directory: str,
    settings: Dict,
    device_mesh, 
    placements,
    parallize_tensor: bool = False,
):
    # typically they all should be same file
    with ExitStack() as stack:
        files = {}
        for _, vs in settings.items():
            for _, fi in vs:
                if fi not in files:
                    files[fi] = stack.enter_context(
                        safe_open(os.path.join(directory, fi), framework='pt', device='cpu')
                    )
            
        # go by one weight
        for weight_name, vs in settings.items():
            data = []
            for k, fi in vs:
                T = files[fi].get_tensor(k)
                if 'experts' in k:
                    if T.shape[1] > T.shape[0]:
                        T = T.t()
                data.append(T)

            # get the module we want to shard
            name = weight_name.split(".")
            path, name = ".".join(name[:-1]), name[-1]
            mod = dmoe.get_submodule(path)

            # get 
            mod_dtype = getattr(mod, name).dtype
            requires_grad = getattr(mod, name).requires_grad

            # concat on dim 0 and distribute
            num_experts = len(data)
            param = torch.concat(data, dim=DIM_EXPERT).to(mod_dtype)
            _placements = placements
            if KEY_ROUTER in weight_name:
                # - the router needs to be replicated
                _placements = [Replicate() for _ in range(len(placements))]
            elif parallize_tensor:
                # in the TP case, we need to inteleave DIM_EXPERT
                # - we assume the ex
                # e.g., ep_size = 4
                assert len(device_mesh.shape) > 1, "TP cannot work with 1D mesh"
                ep_size = device_mesh.shape[-1] # assume its the last dim

                # - this is the number of features per expert
                dim1_part = param.shape[0] // num_experts
                dim2 = param.shape[1]
                # - create a strided index, e.g. 8 experts, ep_size=4
                #   then I = [0, 4, 8, ..., 28, 1, 5, ...]
                I = torch.arange(num_experts * ep_size).view(
                    -1, ep_size
                ).permute(1, 0).reshape(-1)
                # - cut the features per expert and interleave
                param = param.view(
                    num_experts * ep_size, # e.g., groups of size 8, each of size 4
                    dim1_part // ep_size,  # e.g., expert features cut split by 4
                    dim2
                )[I].reshape(-1, dim2)

            param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, _placements),
                requires_grad=requires_grad,
            )
            mod.register_parameter(name, param)

def shard_moe(
    model: torch.nn.Module,
    moe_cls: Union[str, Type],
    checkpoint_name_or_path: str,
    rank: int, 
    world_size: int,
    ep_size: int,
    moe_kwargs: Dict,
    device_type: str = 'cuda',
    key_dp: str = KEY_DATA_PARALLEL,
    key_ep: str = KEY_EXPERT_PARALLEL,
    parallize_tensor: bool = False,
):
    # guarded import
    from megablocks.layers import moe, dmoe, arguments, mpu
    from megablocks.layers import dmlp_registry

    assert ep_size > 1, "this function is used for sharding moe" 

    # this function will shard the MOE on this rank
    device = torch.device(f'cuda:{rank}')
    dp_size = world_size // ep_size

    if dp_size == 1:

        assert not parallize_tensor, \
            "we do not support parallize_tensor in the 1D mesh case."

        # in this case we will have a 1D mesh and collapse the 
        # expert parallel with data_parallel

        device_mesh = init_device_mesh(
            device_type,
            (ep_size,),
            mesh_dim_names=(key_dp,),
        )
        key_ep = key_dp
        placements: List[Placement] = [Shard(DIM_EXPERT)]
    else:
        # in this case it will be a 2D mesh
        device_mesh = init_device_mesh(
            device_type,
            (dp_size, ep_size),
            mesh_dim_names=(key_dp, key_ep),
        )
        placements: List[Placement] = [Replicate(), Shard(DIM_EXPERT)]

    mp_dmoe_args = arguments.Arguments(
        **moe_kwargs, device=device,
        expert_parallel_group=device_mesh[key_ep].get_group(0)
    )

    assert mp_dmoe_args.moe_num_experts % world_size == 0, \
        "number of moe experts not divisible by world_size"

    # for all the MoE related params, e.g., gate, experts
    # get a dictc
    # parent_mod: (child_instance_name, [list of fqdn keys])
    found = {}
    for name, mod in model.named_modules():
        name = name.split('.')
        parent, child = ".".join(name[:-1]), name[-1]
        if isinstance(mod, moe_cls):
            found[parent] = (
                child,
                [ # all params, including childs'
                    f'{parent}.{child}.{n}'
                    for n, _ in mod.named_parameters()
                ]
            )

    # NOTE: for now we only support sharded safetensors
    # - most MOE models should be used using this checkpoint format
    try:
        loc = get_resolved_checkpoint_location(checkpoint_name_or_path)
        with open(os.path.join(loc, FILE_SAFETENSOR_INDEX)) as f:
            index = json.load(f)

        # e.g., prefix: 'model.layers.0',
        #       module_name: 'block_sparse_moe'
        for prefix, (module_name, relevant_keys) in tqdm(
            found.items(), 
            disable=torch.distributed.get_rank() > 0,
            desc=(
                'Sharding MoE (TP)' if parallize_tensor
                else 'Sharding MoE (EP)'
            )
        ):
            settings = get_router_experts_sharded_safetensor(
                index['weight_map'], prefix, module_name,
            )
            with init_empty_weights():
                if not parallize_tensor:
                    mp_dmoe = dmoe.dMoE(mp_dmoe_args) # drop in replacement for now
                else:
                    # in the TP case, we wil use moe.MoE instead
                    # - we will update moe_kwargs to set moe_expert_model_parallelism=False
                    #   this is so that forward_once will be called instead of parallel_forward_once
                    # - we need to accurately set ffn_hidden_size after the TP sharding by ep_size
                    mp_dmoe = moe.MoE(mp_dmoe_args)
                    exp = mp_dmoe.experts
                    exp.hidden_size = mp_dmoe_args.hidden_size
                    exp.ffn_hidden_size = mpu.features_per_rank(mp_dmoe_args) // ep_size
                    exp.mlp = dmlp_registry.get(mp_dmoe_args)

            assign_mlp_v2_weights(
                mp_dmoe, loc, settings, 
                device_mesh, placements, parallize_tensor
            )
            parent = model.get_submodule(prefix)
            setattr(parent, module_name, mp_dmoe)

    except ValueError as e:
        raise ValueError(
            f"Unable to load checkpoint_path '{checkpoint_name_or_path}'. "
            "Currently only support safetensor checkpoints. "
            f": {e}"
        )


    return device_mesh
