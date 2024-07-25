
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from transformers import TrainingArguments
from transformers.optimization import get_scheduler
from trl import  DataCollatorForCompletionOnlyLM
from accelerate import Accelerator
import os
import torch
from torch.utils.data import DataLoader
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.checkpoint import checkpoint

from tqdm import tqdm, trange
from torch.optim import AdamW
from megablocks_utils.shard_moe_utils import shard_moe, get_moe_kwargs
from megablocks_utils.config_utils import update_mlp_registry

# Demo for running databricks megablocks on mixtral using FSDP2
# - for this we only use accelerate, we cannot use HF Trainer
#   because FSDP2 is not properly integrated in
# - FSDP2 has some problems with the accelerate launcher, so in this case 
#   we need to use the torchrun launcher
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# call this to patch the megablocks 
update_mlp_registry()

# https://github.com/pytorch/pytorch/issues/114299
# - issue on agwu on how FSDP2 should be per-parameter sharding

# https://github.com/huggingface/accelerate/issues/2873
# - HF issue asking when we should integrate FSDP2, with questions on stability

# https://pytorch.org/blog/training-moes/
# - blog on databricks + pytorch collab on MoE train

def main(
    max_seq_length: str =4096,
    load_model_dtype: str='bfloat16', # FSDP shared params will take 
    attn_implementation: str ='sdpa',
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    use_megablocks_sharding: bool = False,
    lr_scheduler_type: str = 'linear',
    num_epochs: int = 1,
    num_warmup_steps: int = 0,
    # max_grad_norm: float = 1.0,
    learning_rate: float = 1e-5,
):

    # parser = HfArgumentParser(
    #     dataclass_types=TrainingArguments
    # )
    # training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=getattr(torch, load_model_dtype), 
        attn_implementation=attn_implementation, 
        low_cpu_mem_usage=True, # set this manually to also support torchrun
    )

    # HACK
    # model.model.layers = model.model.layers[:2]

    # we set the max sequence length here
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, model_max_length=max_seq_length,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # tokenize the dataset here, because we are not using SFTTrainer anymore
    def convert_data(example):
        input_ids = tokenizer(example['output'])['input_ids']
        return {'input_ids': input_ids, 'labels': input_ids}

    dataset = load_dataset('json', data_files='data.json').map(
        convert_data
    ).remove_columns('output')

    # taken from https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/sft_trainer.py
    response_template_ids = tokenizer.encode(
        " ### Response", add_special_tokens=False
    )[2:]

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer, return_tensors='pt'
    )
    dataloader = DataLoader(
        dataset['train'], 
        batch_size=per_device_train_batch_size, 
        collate_fn=data_collator
    )

    # some stuff that we need to get before
    # torch.dist is initialialzed
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    if use_megablocks_sharding:

        dp_mesh = shard_moe(
            model, 
            MixtralSparseMoeBlock, 
            checkpoint_name_or_path=MODEL_NAME,
            rank=rank,
            world_size=world_size,
            ep_size=world_size,
            moe_kwargs=get_moe_kwargs(
                model.config, 
                has_bias=False,
                fp16=(load_model_dtype == 'float16'),
                bf16=(load_model_dtype == 'bfloat16'),
            ),
        )
        from torch.distributed._composable.contract import REGISTRY_KEY
        for layer in model.model.layers:
            setattr(
                layer.block_sparse_moe, REGISTRY_KEY, {'replicate'}
            )
    else:
        dp_mesh = init_device_mesh(
            "cuda", 
            (world_size,), mesh_dim_names=('dp', )
        )['dp']

    mp_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, load_model_dtype), 
        reduce_dtype=getattr(torch, load_model_dtype), 
    )

    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}


    # apply sharding on the model
    # - this is a drop-in replacement for accelerate.prepare_model
    # - we cannot use accelerate's prepare model because there is no 
    #   FSDP2 integration yet
    def prepare_model(m: torch.nn.Module):
        layers = m.model.layers
        n_layers = len(layers)
        for layer_id, transformer_block in enumerate(tqdm(layers)):
            # As an optimization, do not reshard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = layer_id < (n_layers - 1)

            # activation checkpoint (by default)
            if use_megablocks_sharding:
                # somehow need this for MoE
                _checkpoint_args = {
                    'checkpoint_impl': CheckpointImpl.REENTRANT,
                    'use_reentrant': True,
                }
            else:
                _checkpoint_args = {
                    'checkpoint_impl': CheckpointImpl.NO_REENTRANT,
                    'use_reentrant': False,
                }

            transformer_block = ptd_checkpoint_wrapper(
                transformer_block,
                checkpoint_fn=checkpoint,
                preserve_rng_state=False,
                **_checkpoint_args
            )

            # perform the sharding
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

            # replace the layer
            layers[layer_id] = transformer_block

        fully_shard(
            m, **fsdp_config, reshard_after_forward=True
        )
        return m

    # prepare the model without accelerate
    model = prepare_model(model)

    # create the accelerator, optimizer and prepare for distributed
    # training
    from accelerate.utils import GradientAccumulationPlugin

    accelerator = Accelerator(
        # gradient_accumulation_steps=gradient_accumulation_steps
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=gradient_accumulation_steps,
            sync_each_batch=True, # to 
        )
    )

    # - create optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # - create scheduler
    scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_epochs * len(dataloader)
    )

    # - prepare for distributed training
    dataloader, optimizer, scheduler = accelerator.prepare(
        dataloader, optimizer, scheduler
    )

    # train loop
    len_dataloader = len(dataloader)
    num_update_steps_per_epoch = len_dataloader // gradient_accumulation_steps
    total_train_steps = num_update_steps_per_epoch * num_epochs
    step = -1
    tr_loss = 0.
    with trange(
        total_train_steps,
        disable=torch.distributed.get_rank() > 0
    ) as pbar:
        for epoch in range(num_epochs):

            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)

            for batch in dataloader:

                step += 1

                inputs, targets = batch['input_ids'], batch['labels']
                inputs = inputs.to('cuda')
                targets = targets.to('cuda')

                with accelerator.accumulate(model):
                    outputs = model(inputs, labels=targets, use_cache=False)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                tr_loss += loss.item() / gradient_accumulation_steps

                # needs to be fixed, this doesnt work with FSDP2
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_value_(model.parameters(), max_grad_norm)

                if (
                    step > 0 and
                    step % gradient_accumulation_steps == 0
                ):
                    last_lr = scheduler.get_last_lr()[0]
                    print ({
                        'epoch': round(
                            step / num_update_steps_per_epoch /
                            gradient_accumulation_steps, 4
                        ),
                        'loss': tr_loss,
                        'learning_rate': last_lr,
                    }, flush=True)

                    tr_loss = 0.

                    pbar.update(1)

if __name__ == '__main__':
    import fire
    fire.Fire(main)