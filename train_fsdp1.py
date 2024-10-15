
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
# from megablocks_utils.shard_moe_utils import shard_moe, get_moe_kwargs
from hf_utils.shard_moe_utils import shard_moe
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from megablocks_utils.config_utils import update_mlp_registry

# Demo for running databricks megablocks on mixtral using accelerate + FSDP1
# - this uses HF Trainer's integration of FSDP1
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

update_mlp_registry()

def main(
    max_seq_length=4096,
    load_model_dtype='bfloat16', # FSDP shared params will take 
    attn_implementation='sdpa',
    use_megablocks_sharding: bool = False,
    truncate_model_for_debug: bool = False,
):

    parser = HfArgumentParser(
        dataclass_types=TrainingArguments
    )
    training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=getattr(torch, load_model_dtype), ## UPDATED
        attn_implementation=attn_implementation, ## UPDATED
    )

    if truncate_model_for_debug:
        # will just change to two layers for a quick run
        model.model.layers = model.model.layers[:2]

    # we set the max sequence length here
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, model_max_length=max_seq_length,
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

    if use_megablocks_sharding:

        dp_mesh = shard_moe(
            model, 
            MixtralSparseMoeBlock, 
            checkpoint_name_or_path=MODEL_NAME,
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
            ep_degree=torch.distributed.get_world_size(),
            # moe_kwargs=get_moe_kwargs(
            #     model.config, 
            #     has_bias=False,
            #     fp16=training_args.fp16,
            #     bf16=training_args.bf16,
            # ),
        )

        trainer.accelerator.state.fsdp_plugin.ignored_modules = [
            layer.block_sparse_moe for layer in model.model.layers
        ]

    trainer.train()

if __name__ == '__main__':
    import fire
    fire.Fire(main)