
from peft import PeftConfig
from peft import get_peft_model
from peft.utils import (
    prepare_model_for_kbit_training,
)
import torch

from trl.trainer.utils import peft_module_casting_to_bf16

import inspect
from typing import Dict

def prepare_peft(
    model: torch.nn.Module, 
    peft_config: PeftConfig,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    gradient_checkpointing_kwargs: Dict = {},
    autocast_adapter_dtype: bool = True,
):

    _support_gc_kwargs = len(
        gradient_checkpointing_kwargs
    ) > 0 and "gradient_checkpointing_kwargs" in list(
        inspect.signature(prepare_model_for_kbit_training).parameters
    )
    is_sharded_qlora = False
    # Below is to support QLoRA + FSDP / DS-Zero3 - one should never call
    # peft_module_casting_to_bf16 or prepare_model_for_kbit_training when doing
    # QLoRA + FSDP / DS-Zero3
    if getattr(model, "is_loaded_in_4bit", False):
        for _, param in model.named_parameters():
            if param.__class__.__name__ == "Params4bit":
                is_sharded_qlora = param.data.device.type == "cpu"
                break
    if getattr(model, "is_loaded_in_8bit", False) or (
        getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora
    ):
        prepare_model_kwargs = {
            "use_gradient_checkpointing": gradient_checkpointing,
        }

        if _support_gc_kwargs:
            prepare_model_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

        model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

        # if args is not None:
        #     args = dataclasses.replace(args, gradient_checkpointing=False)
    elif gradient_checkpointing and (
        "use_reentrant" not in gradient_checkpointing_kwargs
        or gradient_checkpointing_kwargs["use_reentrant"]
    ):
        # For backward compatibility with older versions of transformers
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # if (
    #     "autocast_adapter_dtype" in list(inspect.signature(get_peft_model).parameters)
    #     and getattr(model, "is_loaded_in_4bit", False)
    #     and is_sharded_qlora
    # ):
    #     model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
    # else:
    #     model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)

    model = get_peft_model(model, peft_config, autocast_adapter_dtype=autocast_adapter_dtype)

    if (
        bf16
        and getattr(model, "is_loaded_in_4bit", False)
        and not is_sharded_qlora
    ):
        peft_module_casting_to_bf16(model)

    # add some more extra logic to cast to bf16
    if bf16:
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.data = p.data.to(torch.bfloat16)

    return model