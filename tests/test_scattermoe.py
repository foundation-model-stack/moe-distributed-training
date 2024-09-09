import torch
import pytest
from scattermoe.kernels.ops import padded_block_indices
from scattermoe_utils.parallel_linear_lora import parallel_linear_lora

# ---------------------- HELPERS -----------------------

# build 
def build_expert_model_for_test(
    num_experts: int = 2, 
    module_tag: str = "lin",
    input_dim: int = 128,
    output_dim: int = 16,
    bias: bool = False,
):

    # this is a expert model that takes in scattered inputs
    # and ouputs grouped
    class ExpertModel(torch.nn.Module):
        
        def __init__(self):
            super().__init__()
            
            self.num_experts = num_experts
            self.module_tag = module_tag
            self.experts = torch.nn.ModuleDict({
                f'{module_tag}_{i}': torch.nn.Linear(
                    input_dim, output_dim, bias=bias,
                ) for i in range(num_experts)
            })
                
        def forward(
            self, X, sorted_expert_idxs, sorted_scattered_idxs
        ):
            # this forward takes in scattered inputs
            # - build the output group-by-group
            output = []
            for i in range(num_experts):

                expert = getattr(self.experts, f'{module_tag}_{i}')

                # get the inputs for current expert
                d = X[sorted_scattered_idxs[sorted_expert_idxs == i]]

                # get expert output
                output.append(expert(d))

            # concatenate outputs
            return torch.cat(output)
            
        def prepare_inputs_for_generation(self):
            pass

    return ExpertModel()

def build_lora_model_for_test(
    r: int, lora_alpha: float, lora_dropout: float=0., lora_bias: str = "none",
    **kwargs,
):
    from peft import get_peft_model, LoraConfig, TaskType

    model = build_expert_model_for_test(**kwargs)

    # build a causallm
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules = [
            f'{model.module_tag}_{i}' for i in 
            range(model.num_experts)
        ], 
    )

    return get_peft_model(model, peft_config).get_base_model()

def build_inputs_for_test(
    num_tokens: int = 10, input_dim: int = 128,
):
    # for now:
    assert num_tokens == 10, "only implemented for 10 tokens now"
    X = torch.randn((num_tokens, input_dim))
    sorted_expert_idxs = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    sorted_scattered_idxs = torch.tensor([4, 6, 8, 9, 0, 1, 2, 3, 5, 7])

    return X, sorted_expert_idxs, sorted_scattered_idxs

def get_params_for_parallel_forward_lora(
    model: torch.nn.Module, 
):
    W, A, B = [], [], []

    tag = model.module_tag
    for i in range(model.num_experts):
        expert = getattr(model.experts, f'{tag}_{i}')
        W.append(expert.base_layer.weight.T.unsqueeze(0))
        A.append(expert.lora_A.default.weight.T.unsqueeze(0))
        B.append(expert.lora_B.default.weight.T.unsqueeze(0))

    W = torch.concat(W)
    A = torch.concat(A)
    B = torch.concat(B)
    A.retain_grad()
    B.retain_grad()
    return (W, A, B)

# ---------------------- TESTS -----------------------

# NOTE: currently now cannot handle
# - lora dropout
# - lora bias
# - lora_r < 16
MODEL_LORA = {
    'small-lora-no-dropout': {
        "r": 16, "lora_alpha": 32, "lora_dropout": 0., "lora_bias": "none",
        "input_dim": 128, "output_dim": 16, "bias": False,
        "num_experts": 2,
    }
}

@pytest.fixture()
def models(seed: int = 42):
    torch.manual_seed(seed)

    yield {
        tag : (
            build_lora_model_for_test(**kwargs).to('cuda'), 
            kwargs
        )
        for tag, kwargs in MODEL_LORA.items()
    }

@pytest.fixture()
def inputs(seed: int = 42):
    INPUTS = [(10, 128)]

    torch.manual_seed(seed)

    _inputs = {}
    for num_tokens, input_dim in INPUTS:
        X, sei, ssi = build_inputs_for_test(num_tokens, input_dim)
        _inputs[(num_tokens, input_dim)] = (
            X.to('cuda'), sei.to('cuda'), ssi.to('cuda')
        )

    yield _inputs

@pytest.mark.parametrize(
    "model_tag,num_tokens", [('small-lora-no-dropout', 10)]
)
def test_scattermoe_with_lora_adapters(
    model_tag, num_tokens, models, inputs
):

    model, model_kwargs = models[model_tag]
    X, sorted_expert_idxs, sorted_scattered_idxs = inputs[
        (num_tokens, model_kwargs["input_dim"])
    ]

    padded_block_idxs, expert_offsets = padded_block_indices(
        sorted_expert_idxs, model_kwargs['num_experts']
    )

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    out = model(X, sorted_expert_idxs, sorted_scattered_idxs)
    loss = out.norm()
    loss.backward()

    if 'lora' in model_tag:

        tag = model.module_tag
        refs = []
        for i in range(model.num_experts):
            expert = getattr(model.experts, f'{tag}_{i}')
            refs.append(
                expert.lora_B.default.weight.grad.T.detach().cpu().clone()
            )

        W, A, B = get_params_for_parallel_forward_lora(model)
        A.grad = None
        B.grad = None

        out2 = parallel_linear_lora(
            X, W, A, B, 
            model_kwargs['r'], model_kwargs['lora_alpha'], 1,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_in=False, grouped_out=True
        )
        loss2 = out2.norm()
        loss2.backward()
        assert torch.allclose(out, out2, atol=1e-2), \
            "outputs differ"

        tag = model.module_tag
        for i in range(model.num_experts):
            expert = getattr(model.experts, f'{tag}_{i}')
            assert torch.allclose(refs[i], B.grad[i].detach().cpu(), atol=1e-2), \
                f"B gradients differ at expert {i}"
    else:
        raise NotImplementedError

