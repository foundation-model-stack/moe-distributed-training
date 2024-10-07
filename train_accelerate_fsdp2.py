
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import get_scheduler
from trl import  DataCollatorForCompletionOnlyLM
from accelerate import Accelerator
import os
import torch
import json
from torch.utils.data import DataLoader
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.checkpoint import checkpoint

from peft import LoraConfig, TaskType
from utils import prepare_peft

from tqdm.auto import tqdm, trange
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
    use_scattermoe: bool = False,
    use_tp_sharding: bool = False,
    lr_scheduler_type: str = 'linear',
    num_epochs: int = 1,
    num_warmup_steps: int = 0,
    max_grad_norm: float = 1.0,
    learning_rate: float = 1e-5,
    truncate_model_for_debug: bool = False,
    expert_degree: int = None,
    use_lora: str = "none", # attn-only #all
):

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=getattr(torch, load_model_dtype), 
        attn_implementation=attn_implementation, 
        low_cpu_mem_usage=True, # set this manually to also support torchrun
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

    # tokenize the dataset here, because we are not using SFTTrainer anymore
    # https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/README.md#pre-process-the-jsonjsonl-dataset
    # - follow these instructions to prepare `data.json` that contains a single 
    #   key "output"
    def convert_data(example):
        input_ids = tokenizer(example['output'])['input_ids']
        return {'input_ids': input_ids, 'labels': input_ids}

    dataset = load_dataset('json', data_files='data.json').map(
        convert_data
    ).remove_columns('output')

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

    if use_scattermoe and not (use_megablocks_sharding or use_tp_sharding):
        raise ValueError(
            "use_scatter_moe==True only works if performing "
            "megablocks sharding or tp sharding."
        )

    if use_megablocks_sharding or use_tp_sharding:

        if expert_degree is None:
            expert_degree = world_size

        if use_scattermoe:
            try:
                import scattermoe
            except ImportError:
                raise ValueError(
                    "use_scattermoe=True but scattermoe not installed. "
                    "pip install -r requirements-scattermoe.txt"
                )

        assert use_megablocks_sharding + use_tp_sharding == 1, \
            "must choose between megabocks or tp_sharding"

        # if use_megablocks_sharding:
        device_mesh = shard_moe(
            model, 
            MixtralSparseMoeBlock, 
            checkpoint_name_or_path=MODEL_NAME,
            rank=rank,
            world_size=world_size,
            ep_size=expert_degree,
            moe_kwargs=get_moe_kwargs(
                model.config, 
                has_bias=False,
                fp16=(load_model_dtype == 'float16'),
                bf16=(load_model_dtype == 'bfloat16'),
                mlp_impl=(
                    "sparse" if not use_scattermoe else
                    "scattermoe"
                ),
                use_tensor_parallelism=use_tp_sharding,
            ),
            parallize_tensor=use_tp_sharding
        )
        # elif use_tp_sharding:

        # NOTE: this is a hack to hve the FSDP fully_shard ignore the MoE 
        # module, whilst sharding the attention module.
        # - setting this composable contract "replicate", it will cause it
        #   to be ignored.
        from torch.distributed._composable.contract import REGISTRY_KEY
        for layer in model.model.layers:
            setattr(
                layer.block_sparse_moe, REGISTRY_KEY, {'replicate'}
            )
    elif use_tp_sharding:
        assert expert_degree is not None, "for tp sharding please expert degree < world_size"

        
    else:
        device_mesh = init_device_mesh(
            "cuda", 
            (world_size,), mesh_dim_names=('dp', )
        )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, load_model_dtype), 
        reduce_dtype=getattr(torch, load_model_dtype), 
    )

    fsdp_config = {"mesh": device_mesh, "mp_policy": mp_policy}


    # apply sharding on the model
    # - this is a drop-in replacement for accelerate.prepare_model
    # - we cannot use accelerate's prepare model because there is no 
    #   FSDP2 integration yet
    def prepare_model(m: torch.nn.Module):
        if use_lora == "none":
            layers = m.model.layers
        else:
            layers = m.get_base_model().model.layers

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

    # no stringent value checks, please set carefully
    if use_lora != "none":

        # target modules set if use_lora == "all" or "attn-only"
        if use_lora in {'all', 'attn-only'}:
            tm = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            tm = []

        # if we are not doing megablocks sharding
        if (
            not use_megablocks_sharding and  
            use_lora in {"all", "mlp-only"} # if need adapters
        ):
            # add these extra here 
            tm += ["w1", "w2", "w3"]


        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=16,
            lora_dropout=0.0, # for now do not support dropout
            bias="none",
            target_modules=tm,
        )

        if (
            (use_megablocks_sharding or use_tp_sharding)
            and use_lora in {'all', 'mlp-only'}
        ):
            assert use_scattermoe, "lora adapters cannot be used on MLP without scattermoe"

            from megablocks.layers.moe import ParallelMLP
            from megablocks.layers.dmoe import ParallelDroplessMLP
            from megablocks_utils.peft_utils import ParallelDroplessMLP as LoRAParallelDroplessMLP, ARTIFACTS
            from megablocks_utils.peft_utils import ParallelMLP as LoRADroplessMLP

            # inject this so we can replicate
            ARTIFACTS['device_mesh'] = device_mesh

            # inject a custom module for MLP since SparseMLP is not 
            # a supported class
            peft_config._register_custom_module({
                ParallelDroplessMLP: LoRAParallelDroplessMLP,
                ParallelMLP: LoRADroplessMLP,
            })
            
            peft_config.target_modules.add("experts")

        # NOTE: since this is done before prepare model
        # - since we REGISTRY_KEY='replicate" for the moe above, we ignore any
        #   FSDP2 sharding for the moe
        # - for the case use_megablocks_sharding this is ok, since the adapters for each
        #   shard is seperate
        # - however for use_tp_sharding the adapter gradients will not be 
        #   reduced, and it is currently not accurate. TODO: add some ocde
        #   to reduce the adapter weights for this case
        model = prepare_peft(
            model, peft_config, 
            gradient_checkpointing=True,
            bf16=(load_model_dtype=='bfloat16'),
            autocast_adapter_dtype=False, # do not upcast because FSDP cannot handle mixed types
        )

        # FIXME: its abit problemantic to use different adapter
        # names, so we do this manually
        if use_scattermoe and use_lora in {'all', 'mlp-only'}:
            # - after the PEFT prepare we have to force the parameters
            # back to require_grad = True
            # - lora has no bias
           for name, p in model.named_parameters():
               if "experts" in name and 'lora_' in name:
                   p.requires_grad = True

    # prepare the model without accelerate
    model = prepare_model(model)

    # - create optimizer (after sharding)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # - create scheduler
    scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(dataloader) // gradient_accumulation_steps * num_epochs,
    )

    # create the accelerator, optimizer and prepare for distributed
    # training
    from accelerate.utils import GradientAccumulationPlugin

    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=gradient_accumulation_steps,
            sync_each_batch=True, # to save memory
        )
    )

    # - prepare for distributed training 
    dataloader, optimizer, scheduler = accelerator.prepare(
        dataloader, optimizer, scheduler
    )

    # after prepare compute these (note the dataloader length will change here)
    num_update_steps_per_epoch = len(dataloader) // gradient_accumulation_steps
    total_train_steps = num_update_steps_per_epoch * num_epochs

    # memory tracker
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # train loop
    step = -1
    tr_loss = 0.
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    with trange(
        total_train_steps,
        disable=torch.distributed.get_rank() > 0
    ) as pbar:

        start_time.record()
        for epoch in range(num_epochs):

            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)

            for batch in dataloader:

                # if torch.distributed.get_rank() == 0:
                #     torch.save(batch, 'batch')
                # torch.distributed.breakpoint()
                batch = torch.load('batch.pt')

                step += 1

                inputs, targets = batch['input_ids'], batch['labels']
                inputs = inputs.to('cuda')
                targets = targets.to('cuda')

                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    outputs = model(inputs, labels=targets, use_cache=False)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()

                tr_loss += loss.item() / gradient_accumulation_steps

                if accelerator.sync_gradients:
                    try:
                        _grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                        # if its a DTensor we need to do this
                        _grad_norm = _grad_norm.to_local()
                        _grad_norm = _grad_norm.item()
                    except:
                        _grad_norm = 0.


                if (
                    step > 0 and
                    step % gradient_accumulation_steps == 0
                ):
                    gpu_mem_used_now = torch.cuda.memory_allocated()
                    gpu_mem_used_peak = torch.cuda.max_memory_allocated()
                    last_lr = scheduler.get_last_lr()[0]
                    metrics = {
                        'epoch': round(
                            step / num_update_steps_per_epoch /
                            gradient_accumulation_steps, 2
                        ),
                        'loss': tr_loss,
                        'learning_rate': last_lr,
                        "gpu_mem_used_now": gpu_mem_used_now,
                        "gpu_mem_used_peak": gpu_mem_used_peak,
                        'grad_norm': _grad_norm,
                    }

                    # reset
                    tr_loss = 0.

                    # report
                    pbar.update(1)
                    pbar.write(json.dumps(metrics))

        end_time.record()
        torch.cuda.synchronize()
        elapsed_time_s = start_time.elapsed_time(end_time) / 1000  # secs
        pbar.write(json.dumps({
            'train_runtime': elapsed_time_s,
            'train_steps_per_second': total_train_steps / elapsed_time_s
        }))

if __name__ == '__main__':
    import fire
    fire.Fire(main)