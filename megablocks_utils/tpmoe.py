from  megablocks.layers.moe import MoE, ParallelMLP
from megablocks.layers import common, dmlp_registry, moe, mpu
from megablocks.layers.arguments import Arguments

class TpMLP(moe.ParallelMLP):
    def __init__(self, args: Arguments):
        super().__init__(args)
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = mpu.features_per_rank(args)

        # REMOVE THIS?
        assert args.mlp_impl == 'scattermoe', \
            "only scattermoe version of TP available"

    # this will implement a sharded TP MLP
    def forward_once(self, x, expert_weights, top_experts):
        pass


# tensor parallel version
class TpMoE(MoE):
    
    # - MoE._init_experts_mlp initializes experts to 
    #  the correct type of MLP
    # 
    # MoE.router(x) is called and returns
    # - scores, expert_weights, expert_indices
    # - these are passed to MoE.experts
    # ParallelMLP.forward will call ParallelMLP.forward_once

    def _init_experts_mlp(self, args: Arguments):
        return TpMLP(args)
