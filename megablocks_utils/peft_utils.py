
import torch
from peft.tuners.lora import LoraLayer
# from megablocks.layers.moe import ParallelMLP
# from megablocks.layers.dmoe import ParallelDroplessMLP
# from megablocks.layers.arguments import Arguments
from megablocks.layers import mpu
from typing import Union, Any

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
            A.weight.data = A.weight.data.T # HACK

            B = getattr(self.lora_B, name)

            # we need to upcast the adapter weights because
            # it is otherwise problematic for the kernel
            # A.weight.data = A.weight.data.to(torch.float32)
            # B.weight.data = B.weight.data.to(torch.float32)

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
