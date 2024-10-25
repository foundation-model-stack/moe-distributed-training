
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
from fms_accel import prepare_scattemoe
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from peft import LoraConfig
from typing import List

# Demo for running databricks megablocks on mixtral using accelerate + FSDP1
# - this uses HF Trainer's integration of FSDP1
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

def main(
    model_name_or_path=MODEL_NAME,
    moe_module_name="MixtralSparseMoeBlock",
    max_seq_length=4096,
    load_model_dtype='bfloat16', # FSDP shared params will take 
    attn_implementation='sdpa',
    use_scattermoe: bool = False,
    ep_degree: int = None,
    truncate_model_for_debug: bool = False,
    lora_r: int = 0,
    lora_alpha: float = None,
):

    parser = HfArgumentParser(
        dataclass_types=TrainingArguments
    )
    training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=getattr(torch, load_model_dtype), ## UPDATED
        attn_implementation=attn_implementation, ## UPDATED
    )

    if truncate_model_for_debug:
        # will just change to two layers for a quick run
        model.model.layers = model.model.layers[:2]

    # we set the max sequence length here
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, model_max_length=max_seq_length,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # use the alpaca dataset
    dataset = load_dataset('tatsu-lab/alpaca', split='train')

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
            output_texts.append(text)
        return output_texts

    # taken from https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/sft_trainer.py
    response_template_ids = tokenizer.encode(
        " ### Answer:", add_special_tokens=False
    )[2:]

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer, return_tensors='pt'
    )

    peft_config = None
    if lora_r > 0 and lora_alpha > 0:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", 
            target_modules='all-linear', # keep it simple for now
            r=lora_r,
            lora_alpha=lora_alpha,
            bias='none', # scattermoe only support no bias
        )

    if use_scattermoe:
        # scattermoe to happen before the building the trainer
        # either at loading or augmentation

        prepare_scattemoe(
            model, 
            moe_module_name, 
            checkpoint_name_or_path=model_name_or_path,
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
            ep_degree=ep_degree,
            lora_config=peft_config
        )


    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        max_seq_length=max_seq_length,
        data_collator=data_collator,
        peft_config=peft_config,
    )

    if use_scattermoe:

        # for newer torch that enables foreach for Dtensors we need to remove it
        from torch.optim.optimizer import _foreach_supported_types

        i = 0
        while i < len(_foreach_supported_types):
            x = _foreach_supported_types[i]
            if x.__name__ == 'DTensor':
                _foreach_supported_types.pop(i)
            else:
                i += 1 

        trainer.accelerator.state.fsdp_plugin.ignored_modules = [
            layer.block_sparse_moe for layer in model.model.layers
        ]

        # NOTE: in the future the quant case because prepare_model_for_kbit_training
        # is called, this will reset the grads of the ScatterMoe
        # so we will neeed to set them back to True
        # - but for regular LoRA this is not needed, because
        #   _mark_only_adapters_as_trainable will check for prefix

    trainer.train()

if __name__ == '__main__':
    import fire
    fire.Fire(main)