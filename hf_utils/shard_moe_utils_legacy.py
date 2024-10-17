
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
from typing import Dict, List, Tuple, Type, Union
import re

# This function creates a dictionary of keys and paths into the the sharded
# safetensors checkpoint file, that are relevant to the "prefix" and "instance_name"
# being pased in.
# - the keys point to modules found in megablocks.layers.dmoe.dMoE, the distributed
#   expert module provided by megablocks.
# - the values are tuples pointing to the keys within the checkpoint file.
#
# Example: if prefix="module.layers.0" and instance_name="block_sparse_moe", then a dictionary
# of the following will be returned:
# {
#   'w1.weight': [
#      (
#        'model.layers.0.block_sparse_moe.experts.0.w1.weight',
#        'model-00001-of-00019.safetensors'
#      ),
#      (
#         'model.layers.0.block_sparse_moe.experts.1.w1.weight',
#         'model-00001-of-00019.safetensors'
#      ),
#      ...
#    ]
#    'w2.weight': [...],
#    'w3.weight': [...],
#    'router.weight': [
#       (
#          'model.layers.0.block_sparse_moe.gate.weight',
#          'model-00001-of-00019.safetensors'
#       )
#     ]
# }
# 
# or the non-sharded case (and possibly fused case)
# {
#   'w1.weight': [
#      (
#        'model.layers.0.block_sparse_moe.input_linear.layer.weight',
#        'model-00001-of-00001.safetensors'
#      ),
#    ],
#    ...
#   'w3.weight': [
#      (
#        'model.layers.0.block_sparse_moe.input_linear.layer.weight',
#        'model-00001-of-00001.safetensors'
#      ),
#    ]
# }


def get_checkpoint_meta_from_sharded_safetensor(
    weight_map: Dict,
    prefix: str,  # e.g., 'model.layers.0,
    instance_name: str,  # e.g., block_sparse_moe
    router_name: str = "gate",  # e.g., named "gate" within block_sparse_moe
    expert_name: str = "experts",  # e.g., named "experts" within block_sparse_moe
    expert_map: Dict = None, # map -> [w1,w2,w3]
) -> Dict[str, List[Tuple]]:
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

    # if expert_name = input_linear|output_linear|input_linear
    # - in this case will map 
    # - input_linear: [w1, w3], output_linear: {w2}
    # - will assume the latter has double the size and can
    #   be split.
    if expert_map is None:
        if '|' in expert_name:
            expert_map = {}
            _names = expert_name.split('|')
            assert len(_names) in {2,3}, "expert name map has to be length 2/3"
            
            for i, n in enumerate(_names):
                if n not in expert_map:
                    expert_map[n] = []
                expert_map[n].append(f'w{i+1}')
        else:
            expert_map = {x: [x] for x in ['w1', 'w2', 'w3']}

    # state dict -> weights
    # 'router.weight': [(k, file),...]
    # `w1.weight`: [...]
    _map = defaultdict(list)
    prefix = f"{prefix}.{instance_name}."
    for k, stfile in weight_map.items():
        if not k.startswith(prefix):
            continue

        # e.g. after replacement we get
        # - gate.weight
        # - experts.0.w1.weight
        rel_k = k.replace(prefix, "")
        # pylint: disable=anomalous-backslash-in-string
        m = re.match(f"({router_name}|{expert_name})\.?(\d+)?\.?(\w+)?\.weight", rel_k)
        if m is None:
            raise ValueError(
                f"Unable to handle key '{k}' with provided router_name "
                f"'{router_name}' or expert_name '{expert_name}'"
            )
        if m.group(1) == router_name:
            _map["router.weight"].append((k, stfile))
        elif m.group(1) in expert_name:
            index = m.group(2)
            index = 0 if index is None else int(index)
            mod = None
            for mod in expert_map.get(m.group(1), expert_map.get(m.group(3))):
                _insert(_map[f"{mod}.weight"], index, (k, stfile))

            assert mod is not None, f"cannot map \'{rel_k}\'"
        # else:
        #     # might need a another clause if it not sharded type
        #     # - or maybe not, just let it return 
        #     # "w1.weight: [(k,fi), ...]"
        #     pass

    if len(_map) == 0:
        raise ValueError(
            f"Could not get safetensor map for '{prefix}' and '{instance_name}'"
        )

    return _map

from collections import OrderedDict

def convert_state_dict(
    prefix: str,
    metadata: Dict,
    state_dict: Dict,
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
):
    target = OrderedDict()

    for scatter_key, vs in metadata.items():
        for state_key, _ in vs:
            state_key = state_key.replace(prefix, "")
            param = state_dict[state_key]
            
            if (
                scatter_key.startswith('w1') or 
                scatter_key.startswith('w2') or
                scatter_key.startswith('w3')
            ):
                if len(param.shape) == 2:
                    param = param.view(num_experts, -1, param.shape[-1])

                if scatter_key.startswith('w1') or scatter_key.startswith('w3'):
                    if param.shape[-2] == (2 * intermediate_size):
                        # cut it 
                        if scatter_key.startswith('w1'):
                            param = param[..., :intermediate_size, :]
                        else:
                            param = param[..., intermediate_size:, :]

                    # asumme these are linears
                    # assert param.shape[-2] == intermediate_size, "wrong intermediate size"
                    # assert param.shape[-1] == hidden_size, "wrong hidden size"

                # have to transpose for weights since scattermoe accepts the differen
                # order
                param = param.permute(0, 2, 1)

            target[scatter_key] = param

    return target