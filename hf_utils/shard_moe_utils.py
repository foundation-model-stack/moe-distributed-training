# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from collections import defaultdict
from contextlib import ExitStack
from copy import copy
from typing import Dict, List, Tuple, Type, Union
import json
import os
import re
import warnings

# Third Party
from accelerate import init_empty_weights
from safetensors import safe_open
from torch.distributed._tensor import Placement, Replicate, Shard, distribute_tensor, Partial
from torch.distributed._tensor.device_mesh import DeviceMesh, init_device_mesh
from tqdm import tqdm
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
import torch

FILE_SAFETENSOR_INDEX = "model.safetensors.index.json"
KEY_REPLICATE = "data_parallel"
KEY_EXPERT_PARALLEL = "expert_parallel"
DIM_EXPERT = 0

# class ScatterMoe(torch.nn.Module):
# 
#     def __init__(
#         self, 
#         hidden_size: int,
#         intermediate_size: int,
#         num_experts: int,
#         **kwargs,
#     ):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.num_experts = num_experts
#         self._kwargs = kwargs
# 
#     def _init_parameters(self):
#         raise NotImplementedError

# def extract_experts_module_list(
#     expert_module: torch.nn.ModuleList, # the expert module
#     ff_dim: int,
#     rank: int,
#     world_size: int,
#     device_mesh: DeviceMesh,
#     ep_degree: int = 1,
# ):
# 
#     # gather up all the shard
#     state_dict = defaultdict(list)
#     _ep_mesh = device_mesh[KEY_EXPERT_PARALLEL]
#     _num_processes = _ep_mesh.size()
#     _lr = _ep_mesh.get_local_rank()
#     for i in range(_lr * ep_degree, (_lr + 1) * ep_degree):
#         for param_name, param in expert_module[i].named_parameters():
#             # state_dict[param_name].append(
#             #     param if param.shape[0] > param.shape[1] else param.T
#             # )
#             assert len(param.shape) == 2, "unsupported shape"
#             if param.shape[0] != ff_dim:
#                 param = param.T
#             param.unsqueeze(0)
# 
#     state_dict = {k: torch.concat(ps) for k, ps in state_dict.items()}
# 
     
from .shard_moe_utils_legacy import (
    get_resolved_checkpoint_location,
    get_checkpoint_meta_from_sharded_safetensor
)
from .scattermoe import ScatterMoE

def shard_moe(
    model: torch.nn.Module,
    moe_cls: Union[str, Type],
    checkpoint_name_or_path: str,
    rank: int,
    world_size: int,
    device_type: str = "cuda",
    key_rep: str = KEY_REPLICATE,
    key_ep: str = KEY_EXPERT_PARALLEL,
    router_name: str = "gate",
    expert_name: str = "experts", # can be regex with |
    ep_degree: int = 1,
    mixed_precision: bool = False,
):

    assert ep_degree > 1, "expert_parallel dimension must be set larger than 1"
    assert (
        world_size % ep_degree == 0
    ), f"world_size ({world_size}) not divisible by ep_size ({ep_degree})."

    # this function will shard the MOE on this rank
    device = torch.device(f"cuda:{rank}")

    # in this case it will distribute experts on a different
    # mesh dimension than dp.
    # - this will achieve the effect that the expert sharding can be
    #   hierachical (e.g., can be over a slower network plane since
    #   the communication overhead is less
    rep_size = world_size // ep_degree
    device_mesh = init_device_mesh(
        device_type,
        (rep_size, ep_degree),
        mesh_dim_names=(key_rep, key_ep),
    )
    # - experts will replicate over the first dimension
    placements: List[Placement] = [Replicate(), Shard(DIM_EXPERT)]

    # mp_dmoe_args = arguments.Arguments(
    #     **moe_kwargs,
    #     device=device,
    #     expert_parallel_group=device_mesh[key_ep].get_group(0),
    # )

    moe_num_experts: int = model.config.num_local_experts

    assert moe_num_experts % ep_degree == 0, (
        f"number of moe experts ({moe_num_experts}) "
        f"not divisible by ep_size ({ep_degree})."
    )

    num_experts_per_device = moe_num_experts // ep_degree

    # for all the MoE related params, e.g., gate, experts
    # get a dictionary
    # parent_mod: (child_instance_name, [list of fqdn keys])
    found = {}
    for name, mod in model.named_modules():
        name = name.split(".")
        parent, child = ".".join(name[:-1]), name[-1]

        # check the module depending if moe_cls is a str or class
        if (
            mod.__class__.__name__ == moe_cls
            if isinstance(moe_cls, str)
            else isinstance(mod, moe_cls)
        ):
            fqdn_keys = [  # all params, including childs'
                f"{parent}.{child}.{n}" for n, _ in mod.named_parameters()
            ]

            # check if there are any biases in any of the experts
            # if there are biases
            # Assumption: assume that if one expert has bias,then the others
            # will have it to
            has_bias = any(expert_name in k and k.endswith("bias") for k in fqdn_keys)

            found[parent] = (child, fqdn_keys, has_bias)

    moe_module_names = set()

    # NOTE: for now we only support sharded safetensors
    # - most MOE models should be used using this checkpoint format
    try:
        loc = get_resolved_checkpoint_location(checkpoint_name_or_path)
        with open(os.path.join(loc, FILE_SAFETENSOR_INDEX), encoding="utf-8") as f:
            index = json.load(f)

        # e.g., prefix: 'model.layers.0',
        #       module_name: 'block_sparse_moe'
        for prefix, (module_name, _, has_bias) in tqdm(
            found.items(), disable=(rank > 0), desc="Sharding MoE"
        ):
            checkpoint_metadata = get_checkpoint_meta_from_sharded_safetensor(
                index["weight_map"], prefix, module_name, router_name, expert_name
            )

            # _args = copy(mp_dmoe_args)
            # _args.bias = has_bias

            # - will replace the MoE module with the megablocks sharded dMoE
            # - very hard to do patching, settle for module swap
            #   for now
            # - assumption, router will just use a nn.Linear with topk
            with init_empty_weights():
                mp_dmoe = ScatterMoE(
                    hidden_size=model.config.hidden_size,
                    hidden_act=model.config.hidden_act,
                    intermediate_size=model.config.intermediate_size,
                    num_experts=num_experts_per_device,
                    has_bias=has_bias
                    dtype=model.dtype,
                    device=device,
                )  # 

            load_sharded_experts_onto_device(
                mp_dmoe,
                loc,
                checkpoint_metadata,
                device_mesh,
                placements,
                expert_name,
                mixed_precision,
            )
            parent = model.get_submodule(prefix)
            setattr(parent, module_name, mp_dmoe)

            # - keep track of the name for returning
            moe_module_names.add(module_name)

    except ValueError as e:
        raise ValueError(
            f"Unable to load checkpoint_path '{checkpoint_name_or_path}'. "
            "Currently only support non-GGUF safetensor checkpoints. "
        ) from e

    return device_mesh[key_dp], moe_module_names
