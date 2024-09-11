
import torch
from peft.tuners.lora import LoraLayer
from torch.distributed._tensor import Placement, Replicate, Shard, distribute_tensor
from megablocks.layers import mpu
from typing import Union, Any

# for injection
ARTIFACTS = {}

# this is the lora version
# class ParallelDroplessMLP(ParallelMLP, LoraLayer):
class ParallelDroplessMLP(torch.nn.Module, LoraLayer):

    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        # mb_args: Arguments,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        # super().__init__(mb_args)
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        # - base_layer will be ParallelMLP
        #  
        num_experts = mpu.experts_per_rank(self.args)

        # we need to it cover the base layer
        # - this will have A to be size (k, hd)
        # - this will have B to be size (ffn, k)
        self.in_features = base_layer.hidden_size * num_experts
        self.out_features = base_layer.ffn_hidden_size * num_experts

        # dummy not really used
        self._active_adapter = adapter_name

        device_mesh = ARTIFACTS.get('device_mesh')
        if device_mesh is not None:
            # for A and B, of size ff_dim x hidd
            placements = (
                [Shard(0)] + 
                [Replicate() for _ in range(device_mesh.ndim-1)]
            )

        base_layer._lora_pointers = {}

        # assume that all the parameters of ParallelMLP.mlp 
        # are weights
        # - should be single layer
        for name, _ in base_layer.mlp.named_parameters():
            self.update_layer(
                name,
                r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=False,
                use_dora=False,
            )

            # need to adjust the A
            # - ()
            A = getattr(self.lora_A, name)
            if device_mesh is None:
                A.weight.data = A.weight.data.T # HACK
            else:
                weight = A.weight.data.T

                # must cast here otherwise, the casting on the 
                # Dtensor will not affect the local
                if self.args.bf16 or self.args.fp16:
                    weight = weight.to(
                        torch.bfloat16 if self.args.bf16
                        else torch.float16
                    )

                # - NOTE: if we do it this way, all reps
                # will begin with the same initialization of A
                # - it is ok for random, but not ok for other
                # - types of intialization
                delattr(A, 'weight')
                setattr(
                    A, 'weight', 
                    torch.nn.Parameter(
                        distribute_tensor(
                            weight.repeat(num_experts, 1), 
                            device_mesh,
                            placements
                        ),
                        requires_grad=True
                    )
                )

            B = getattr(self.lora_B, name)
            if device_mesh is not None:
                weight = B.weight.data

                # cast (see above)
                if self.args.bf16 or self.args.fp16:
                    weight = weight.to(
                        torch.bfloat16 if self.args.bf16
                        else torch.float16
                    )

                delattr(B, 'weight')
                setattr(
                    B, 'weight', 
                    torch.nn.Parameter(
                        distribute_tensor(
                            weight.repeat(num_experts, 1), 
                            device_mesh,
                            placements
                        ),
                        requires_grad=True
                    )
                )

            r = self.r[name]
            alpha = self.lora_alpha[name]

            # put pointers in the base layer 
            base_layer._lora_pointers[
                name
            ] = (
                A.weight, B.weight, r, alpha,
                base_layer.hidden_size, base_layer.ffn_hidden_size,
            )
    
    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_layer, name)

    def forward(self, *args, **kwargs):
        return self.base_layer(*args, **kwargs)
