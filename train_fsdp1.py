
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
from hf_utils.shard_moe_utils import prepare_scattemoe
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from megablocks_utils.config_utils import update_mlp_registry

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

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        max_seq_length=max_seq_length,
        data_collator=data_collator,
    )

    if use_scattermoe:

        prepare_scattemoe(
            model, 
            moe_module_name, 
            checkpoint_name_or_path=model_name_or_path,
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
            # ep_degree=torch.distributed.get_world_size(),
            ep_degree=ep_degree,
        )

        trainer.accelerator.state.fsdp_plugin.ignored_modules = [
            layer.block_sparse_moe for layer in model.model.layers
        ]

    trainer.train()

if __name__ == '__main__':
    import fire
    fire.Fire(main)