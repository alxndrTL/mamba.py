from dataclasses import dataclass, fields, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import MambaConfig, MambaBlock, RMSNorm

# todo : une fois la numerical output achieved, on simplifiera à mort (quitte à changer un peu la structure aussi)
# todo : shapes des entrées et des sorties à préciser partout!!

@dataclass
class JambaConfig(MambaConfig):
    
    mlp_size: int = 14336

    num_attention_heads: int = 32

    num_experts_per_tok: int = 2

    def __post_init__(self):
        super().__post_init__()

    """
    def to_mamba_config(self) -> MambaConfig:
        mamba_config_fields = {field.name for field in fields(MambaConfig)}
        filtered_dict = {k: v for k, v in asdict(self).items() if k in mamba_config_fields}
        return MambaConfig(**filtered_dict)
    """

class JambaSdpaAttention(nn.Module):
    def __init__(self, config: JambaConfig): #, layer_idx: Optional[int] = None): # todo : caching
        super().__init__()

        self.config = config
        #self.layer_idx = layer_idx

        self.hidden_size = config.d_model
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

class JambaMambaDecoderLayer(nn.Module):
    def __init__(self, config: JambaConfig, num_experts: int):
        super().__init__()

        self.config = config

        self.mamba = MambaBlock(config=config) #, layer_idx=layer_idx) TODO: caching

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = JambaSparseMoeBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, hidden_states, output_attentions = False, output_router_logits = False, use_cache = False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.mamba(hidden_states)
        bs, seqlen, _ = hidden_states.shape
        self_attn_weights = torch.empty(bs, self.config.num_attention_heads, seqlen, seqlen, device="meta")

        # residual connection after mamba
        hidden_states = residual + hidden_states

        # Experts
        residual = hidden_states
        hidden_states = self.pre_moe_layernorm(hidden_states)
        hidden_states, router_logits = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

class JambaSparseMoeBlock(nn.Module):
    def __init__(self, config: JambaConfig, num_experts: int, num_experts_per_tok: int):
        super().__init__()

        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        if num_experts > 1:
            self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        else:
            self.router = None

        self.experts = nn.ModuleList([JambaMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states):
        # hidden_states : (B, L, D)
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # no routing
        if self.num_experts == 1:
            final_hidden_states = self.experts[0](hidden_states)
            router_logits = torch.ones(
                (batch_size * sequence_length, 1),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                requires_grad=hidden_states.requires_grad,
            )
            return final_hidden_states, router_logits

        # routing
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

class JambaMLP(nn.Module):
    def __init__(self, config: JambaConfig):
        super().__init__()

        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size

        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, x):
        # x : (B, L, D)

        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
