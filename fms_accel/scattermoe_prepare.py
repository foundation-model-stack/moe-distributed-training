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
from collections import OrderedDict
from typing import Type, Union
import json
import os
from contextlib import nullcontext

# Third Party
from accelerate import init_empty_weights
from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed._tensor.device_mesh import DeviceMesh, init_device_mesh
from tqdm import tqdm
import torch
from transformers.modeling_utils import is_fsdp_enabled, is_local_dist_rank_0
from transformers import PretrainedConfig

from .scattermoe import ScatterMoE
from .scattermoe_constants import (
    FILE_SAFETENSOR_INDEX,
    KEY_REPLICATE, 
    KEY_EXPERT_PARALLEL,
    KEY_SCATTERMOE_ROUTER,
    SCATTERMOE_CONVERSION_SPEC
)

from .scattermoe_state_dict import (
    convert_state_dict,
    get_state_dict_from_checkpoint_metadata,
    get_checkpoint_meta_from_sharded_safetensor,
)

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


# this function will load the sharded experts onto the device.
# - this assumes that the "dmoe" module is the megablocks.layers.dmoe.dMoE distributed
#   implementation of the mixture of experts.
def load_experts_onto_device(
    module: torch.nn.Module,
    state_dict: OrderedDict,
    device_mesh: DeviceMesh,
    num_experts_per_device: int, 
):

    # required replication placements
    reps = [Replicate() for _ in range(device_mesh.ndim -1)]

    for weight_name, param in state_dict.items():

        if KEY_SCATTERMOE_ROUTER in weight_name:

            param = distribute_tensor(
                param, device_mesh, reps + [Replicate()]
            )
        elif param.shape[0] > num_experts_per_device:
            param = distribute_tensor(
                param, device_mesh, 
                reps + [Shard(0)]
            )
        else:
            # NOTE: somehow this takes alot of memory, 
            # - so if not rep skip for now
            # - however, this means that there will be a mixture of 
            #   dtensors and regular tensors in the grad norm calc
            if device_mesh is not None:
                if device_mesh.ndim == 1:
                    param = param.to(device_mesh.device_type)
                else:
                    param = DTensor.from_local(
                        param, device_mesh=device_mesh, 
                        placements= reps + [Shard(0)]
                    )

        # get the module we want to shard
        name = weight_name.split(".")
        path, name = ".".join(name[:-1]), name[-1]
        mod = module.get_submodule(path)
        requires_grad = getattr(mod, name).requires_grad

        param = torch.nn.Parameter(
            param, requires_grad=requires_grad,
        )

        # register the sharded parameter onto the megablocks.dmoe
        mod.register_parameter(name, param)


def prepare_scattemoe(
    model: torch.nn.Module,
    moe_cls: Union[str, Type],
    checkpoint_name_or_path: str = None,
    rank: int = None,
    world_size: int = None,
    ep_degree: int = 1,
    key_rep: str = KEY_REPLICATE,
    key_ep: str = KEY_EXPERT_PARALLEL,
    device_type: str = 'cuda',
    mixed_precision: bool = False,
):
    assert world_size % ep_degree == 0, (
        f"world size ({world_size}) "
        f"not divisible by ep_size ({ep_degree})."
    )

    moe_num_experts: int = model.config.num_local_experts
    num_experts_per_device = moe_num_experts // ep_degree
    assert (
        moe_num_experts % ep_degree == 0
    ), f"moe num experts ({moe_num_experts}) not divisible by ep_shard_factor ({ep_degree})."

    # current rank of the device
    device = torch.device(f"{device_type}:{rank}")

    # infer the router_name and expert_name
    found = False
    for archs, (
        router_name, expert_name, expert_mlp_spec, sharded_expert_ckpt
    ) in SCATTERMOE_CONVERSION_SPEC.items():
        archs = archs.split(',')
        if any(x in archs for x in model.config.architectures):
            found = True
            break
    assert found, "cannot configure scatter moe for this model"
    expert_name = expert_name.split('|')

    rep_size = world_size // ep_degree
    if ep_degree == 1 and rep_size == 1:
        # in this case no need for sharding
        device_mesh = None
    elif rep_size == 1:
        # in this case a 1D device mesh suffices
        device_mesh = init_device_mesh(
            device_type,
            (ep_degree, ),
            mesh_dim_names=(key_ep, ),
        )
    else:
        # in this case it will distribute experts on a different dim
        # - this will achieve the effect that the expert sharding can be
        #   hierachical (e.g., can be over a slower network plane since
        #   the communication overhead is less
        device_mesh = init_device_mesh(
            device_type,
            (rep_size, ep_degree),
            mesh_dim_names=(key_rep, key_ep),
        )

    # - compute the shard indices for current expert, if sharding is 
    #   indeed taking place
    expert_shards = None
    if device_mesh is not None:
        _index = device_mesh[KEY_EXPERT_PARALLEL].get_local_rank()
        expert_shards = list(range(
            _index * num_experts_per_device, 
            (_index+1) * num_experts_per_device
        ))

    # dtype
    dtype = model.dtype if not mixed_precision else torch.bfloat16

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
            has_bias = any(expert_name[0] in k and k.endswith("bias") for k in fqdn_keys)

            found[parent] = (child, fqdn_keys, has_bias)

    assert len(found) > 0, "cannot find scattermoe modules to replace"

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
            found.items(), disable=(rank > 0), desc="Converting ScatterMoE layers"
        ):
            checkpoint_metadata = get_checkpoint_meta_from_sharded_safetensor(
                index["weight_map"], prefix, module_name, 
                router_name, '|'.join(expert_name)
            )

            # the parent module
            parent = model.get_submodule(prefix)

            # - handle state dict loading
            # - NOTE: convert_state_dict does not have logic to concat sharded
            #   experts so cannot handle this case
            if (
                ep_degree == 1 and (
                    not is_fsdp_enabled() or is_local_dist_rank_0()
                ) and
                not sharded_expert_ckpt # cannot be a sharded checkpoint
            ):
                # - if there is no sharding, and model is not loaded on the
                #   meta device, we can simply convert the state dict
                sd = convert_state_dict(
                    prefix + '.' + module_name + '.',
                    checkpoint_metadata,
                    getattr(parent, module_name).state_dict(),
                    model.config.num_local_experts,
                    model.config.intermediate_size,
                    dtype,
                )
            else:
                # if there is sharding, then we want the model to be loaded 
                # on meta in general, since the actual model may be alot smaller
                sd = get_state_dict_from_checkpoint_metadata(
                    loc,
                    checkpoint_metadata,
                    num_experts_per_device,
                    model.config.intermediate_size,
                    expert_shards,
                    dtype
                )

            if device_mesh is None:
                _init_scattermoe_context = nullcontext
            else:
                # in this case we need to distribute parameters, so just initialize
                # the scattermoe module swap with empty weights,
                # since they are going to replaced.
                _init_scattermoe_context = init_empty_weights

            # - conver to a scatter moe
            # - very hard to do patching, settle for module swap
            with _init_scattermoe_context():
                moe = ScatterMoE(
                    hidden_size=model.config.hidden_size,
                    hidden_act=model.config.hidden_act,
                    intermediate_size=model.config.intermediate_size,
                    num_experts=num_experts_per_device,
                    has_bias=has_bias,
                    mlp_arch=expert_mlp_spec,
                    top_k=model.config.num_experts_per_tok,
                    dtype=model.dtype,
                    device=device,
                    device_mesh=device_mesh,
                    key_ep=key_ep,
                )  # 

            if device_mesh is None:
                # - if not on meta, just load the state dict
                # - and then put on the device
                moe.load_state_dict(sd)
                moe = moe.to(device)
            else:
                # - otherwise, we need to distribtue and will 
                #   replace the parameters
                load_experts_onto_device(
                    moe,
                    sd, 
                    device_mesh,
                    num_experts_per_device
                )

            # module swap
            setattr(parent, module_name, moe)

            # - keep track of the name for returning
            moe_module_names.add(module_name)

    except ValueError as e:
        raise ValueError(
            f"Unable to load checkpoint_path '{checkpoint_name_or_path}'. "
            "Currently only support non-GGUF safetensor checkpoints. "
        ) from e