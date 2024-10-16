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
from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed._tensor.placement_types import _Partial
from torch.distributed._tensor.device_mesh import DeviceMesh, init_device_mesh
from tqdm import tqdm
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
import torch

FILE_SAFETENSOR_INDEX = "model.safetensors.index.json"
KEY_REPLICATE = "data_parallel"
KEY_EXPERT_PARALLEL = "expert_parallel"
DIM_EXPERT = 0

KEY_SCATTERMOE_ROUTER = 'router'
     
from .shard_moe_utils_legacy import (
    get_checkpoint_meta_from_sharded_safetensor
)
from .scattermoe import ScatterMoE


KEY_SCATTERMOE_ROUTER = 'router.weight'

from megablocks_utils.shard_moe_utils import get_resolved_checkpoint_location

# this function will load the sharded experts onto the device.
# - this assumes that the "dmoe" module is the megablocks.layers.dmoe.dMoE distributed
#   implementation of the mixture of experts.
def load_sharded_experts_onto_device(
    dmoe: torch.nn.Module,
    directory: str,
    checkpoint_metadata: Dict[str, List[Tuple]],
    device_mesh: DeviceMesh,
    # placements: Placement,
    # expert_name: str = "experts",  # e.g., named "experts" within block_sparse_moe
    mixed_precision: bool = False,
):

    rep_size = device_mesh[KEY_REPLICATE].size()
    ep_mesh = device_mesh[KEY_EXPERT_PARALLEL]
    ep_process_index = ep_mesh.get_local_rank()

    # typically they all should be same file, but to play safe, load the checkpoint file onto
    # cpu first since we may not need all weights in that file.
    with ExitStack() as stack:
        files = {}
        for _, vs in checkpoint_metadata.items():
            for _, fi in vs:
                if fi not in files:
                    files[fi] = stack.enter_context(
                        safe_open(
                            os.path.join(directory, fi), framework="pt", device="cpu"
                        )
                    )

        # go by one weight at a time.
        # - weight_name: points to megablocks.dmoe
        upcasted = set()
        for weight_name, vs in checkpoint_metadata.items():

            if KEY_SCATTERMOE_ROUTER in weight_name:
                k, fi = vs[0] # only one item
                param = files[fi].get_tensor(k)
                _placements = [Replicate(), Replicate()]

            elif len(vs) == 1:
                k, fi = vs[0] # only one item
                # if its a non-router weight and its non-sharded
                param = files[fi].get_tensor(k)
                # NOTE: check for num_experts
                assert len(param.shape) == 3, "wrong shape for non-sharded expert"
                _placements = [Replicate(), Shard(0)]
            else:
                # handle sharding if the checkpoint shards experts
                # - 
                data = []
                N = dmoe.num_experts
                for k, fi in [
                    vs[i] for i in range(
                        ep_process_index*N,
                        (ep_process_index+1)*N
                    )
                ]:
                    T = files[fi].get_tensor(k)
                    assert len(T.shape) == 2, "wrong shape"
                    if k.endswith('weight'):
                        T = T.T # then its from a linear

                    T = T.unsqueeze(0)
                    data.append(T)

                param = torch.concat(data, dim=DIM_EXPERT)
                _placements = None
                if rep_size == 1:
                    param = param.to('cuda')
                else:
                    param = DTensor.from_local(
                        param, device_mesh=device_mesh, 
                        placements=[Replicate(), Shard(0)]
                    )

            # get the module we want to shard
            name = weight_name.split(".")
            path, name = ".".join(name[:-1]), name[-1]
            mod = dmoe.get_submodule(path)

            # if mixed_precision and KEY_DMOE_ROUTER not in weight_name:
            if mixed_precision:
                mod_dtype = torch.float32
                upcasted.add(weight_name)
            else:
                mod_dtype = getattr(mod, name).dtype

            requires_grad = getattr(mod, name).requires_grad

            # the megablocks dmoe experts the expert features to be on DIM_EXPERT.
            # - concat on dim 0 and distribute
            # - cast to the correct dtype for the module
            # - if mixed precision is enabled, then sharded params are cased
            param = param.to(mod_dtype)

            # if KEY_SCATTERMOE_ROUTER in weight_name:
            #     # - the router needs to be replicated
            #     _placements = [Replicate() for _ in range(len(placements))]

            param = torch.nn.Parameter(
                (
                    param if _placements is None else
                    distribute_tensor(param, device_mesh, _placements)
                ),
                requires_grad=requires_grad,
            )

            # register the sharded parameter onto the megablocks.dmoe
            mod.register_parameter(name, param)

    if mixed_precision:
        upcasted = ", ".join(sorted(upcasted))
        warnings.warn(f"Mixed precision turned on, upcasted MoE parameters: {upcasted}")

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
    # placements: List[Placement] = [Replicate(), Shard(DIM_EXPERT)]

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

            # - will replace the MoE module with the megablocks sharded dMoE
            # - very hard to do patching, settle for module swap
            #   for now
            # - assumption, router will just use a nn.Linear with topk
            with init_empty_weights():
                moe = ScatterMoE(
                    hidden_size=model.config.hidden_size,
                    hidden_act=model.config.hidden_act,
                    intermediate_size=model.config.intermediate_size,
                    num_experts=num_experts_per_device,
                    all_to_all=True,
                    has_bias=has_bias,
                    dtype=model.dtype,
                    device=device,
                    expert_parallel_group=device_mesh[key_ep].get_group(0)
                )  # 

            load_sharded_experts_onto_device(
                moe,
                loc,
                checkpoint_metadata,
                device_mesh,
                mixed_precision,
            )
            parent = model.get_submodule(prefix)
            setattr(parent, module_name, moe)

            # - keep track of the name for returning
            moe_module_names.add(module_name)

    except ValueError as e:
        raise ValueError(
            f"Unable to load checkpoint_path '{checkpoint_name_or_path}'. "
            "Currently only support non-GGUF safetensor checkpoints. "
        ) from e

    return 
    # return device_mesh[key_dp], moe_module_names
