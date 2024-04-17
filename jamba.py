from dataclasses import dataclass
import json
from typing import Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import MambaConfig, MambaBlock, RMSNorm

# todo : inférence !! avec caching évidemment !!
# todo : calcul du loss (avec le load_balancing)

# todo : sur le github, mettre un schéma de la structure de Jamba, et les parametres qui décident de quoi

@dataclass
class JambaLMConfig:
    
    d_model: int
    n_layers: int
    
    mlp_size: int
    
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5

    # mamba
    d_state: int = 16 # N in paper
    expand_factor: int = 2 # N in paper
    d_conv: int = 4
    dt_rank: Union[int, str] = 'auto'

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False
    pscan: bool = True # use parallel scan mode or sequential mode when training

    # attention
    num_attention_heads: int = 32
    num_key_value_heads: int = 8 # GQA
    attention_dropout: float = 0.

    # MoE
    num_experts: int = 16
    num_experts_per_tok: int = 2

    # structure
    attn_layer_offset: int = 4
    attn_layer_period: int = 8
    expert_layer_offset: int = 1
    expert_layer_period: int = 2

    # language modeling
    vocab_size: int = 65536
    pad_token_id: int = 0
    tie_lm_weights: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        self.mamba_config = MambaConfig(self.d_model, 0, self.dt_rank, self.d_state, self.expand_factor,
                                        self.d_conv, self.dt_min, self.dt_max, self.dt_init, self.dt_scale, self.rms_norm_eps,
                                        self.bias, self.conv_bias, self.inner_layernorms, self.pscan)

def from_pretrained(name: str):
    """
    Returns a model loaded with pretrained weights pulled from HuggingFace.
    The model has to follow the same structure as the original Jamba model on HF (ai21labs/Jamba-v0.1).
    You can easily adapt this function.

    Args:
        name: for example:
            * 'TechxGenus/Mini-Jamba'
            * 'ai21labs/Jamba-v0.1'

    Returns:
        model: a Jamba model configured with the proper parameters and initialized with the proper weights
    """

    from transformers import AutoModelForCausalLM

    model_hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, use_mamba_kernels=False, device_map="auto", trust_remote_code=True)
        
    # copy config data
    config = JambaLMConfig(vocab_size=model_hf.config.vocab_size, d_model=model_hf.config.hidden_size, n_layers=model_hf.config.num_hidden_layers, 
                                rms_norm_eps=model_hf.config.rms_norm_eps, mlp_size=model_hf.config.intermediate_size, inner_layernorms=model_hf.config.mamba_inner_layernorms,
                                expand_factor=model_hf.config.mamba_expand, dt_rank=model_hf.config.mamba_dt_rank, d_state=model_hf.config.mamba_d_state,
                                d_conv=model_hf.config.mamba_d_conv, conv_bias=model_hf.config.mamba_conv_bias, initializer_range=model_hf.config.initializer_range,
                                num_experts=model_hf.config.num_experts, num_experts_per_tok=model_hf.config.num_experts_per_tok, 
                                attn_layer_offset=model_hf.config.attn_layer_offset, attn_layer_period=model_hf.config.attn_layer_period, 
                                expert_layer_offset=model_hf.config.expert_layer_offset, expert_layer_period=model_hf.config.expert_layer_period,
                                num_key_value_heads=model_hf.config.num_key_value_heads, num_attention_heads=model_hf.config.num_attention_heads,
                                pad_token_id=model_hf.config.pad_token_id, bias=model_hf.config.mamba_proj_bias, attention_dropout=model_hf.config.attention_dropout,
                                tie_lm_weights=model_hf.config.tie_word_embeddings)

    model = JambaLM(config)

    # copy weights
    for name, param in model_hf.named_parameters():
        name = name.replace("model.", "jamba.")
        
        if "embed_tokens" in name:
            name = "embedding.weight"
        
        if "final_layernorm" in name:
            name = name.replace("jamba.", "")

        counterpart_param = model.get_parameter(name)
        if counterpart_param is not None:
            counterpart_param.data.copy_(param.data)

    del model_hf

    return model

class JambaLM(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.jamba = Jamba(config)
        self.final_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if self.config.tie_lm_weights:
            self.lm_head.weight = self.embedding.weight 

        self.apply(self._init_weights)

    def forward(self, tokens, caches = None):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)

        x, caches = (self.jamba(x), None) if caches is None else self.jamba.step(x, caches)
        x = self.final_layernorm(x)

        logits = self.lm_head(x)

        return logits, caches

    # TODO process prompt in parallel, and pass in sequential mode when prompt is finished ?
    def generate(self, tokenizer, prompt: str, num_tokens: int = 50, batch_size: int = 1, sample: bool = True, top_k: int = 40, temperature: float = 1.0):
        self.eval()

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(self.parameters()).device) # (1, num_tokens)
        input_ids = input_ids.repeat(batch_size, 1)

        # caches is a list of cache, one per layer
        # cache is composed of : - if Mamba layer : the hidden state, and the last d_conv-1 inputs
        #                        - if Attention layer : the KV cache
        caches = [self.jamba.layers[i].get_empty_cache(batch_size) for i in range(self.config.n_layers)]

        for i in range(input_ids.size(1) + num_tokens - 1):
            with torch.no_grad():
                # forward the new output, get new cache
                next_token_logits, caches = self(input_ids[:, [i]], caches) # (batch_size, 1, vocab_size), caches
                next_token_logits = next_token_logits.squeeze(1)

            # sample (no sampling when the prompt is being processed)
            if i+1 >= input_ids.size(1):
                probs = F.softmax(next_token_logits / temperature, dim=-1) # (batch_size, vocab_size)

                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k) # (batch_size, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)

                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # (batch_size)
                else:
                    next_token = torch.argmax(probs, dim=-1) # (batch_size)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
                    
        outputs = [tokenizer.decode(output.tolist()) for output in input_ids[:, 1:]]

        self.train()

        if batch_size==1:
            return outputs[0]
        else:
            return outputs
    
    def _init_weights(self, module):
        std = self.config.initializer_range

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class Jamba(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.config = config

        # init each model layer, decide if it's mamba/attention and has experts or not
        decoder_layers = []
        for i in range(config.n_layers):
            is_attn = True if (i - self.config.attn_layer_offset) % self.config.attn_layer_period == 0 else False
            is_expert = True if (i - self.config.expert_layer_offset) % self.config.expert_layer_period == 0 else False

            num_experts = self.config.num_experts if is_expert else 1

            if is_attn:
                decoder_layers.append(AttentionLayer(config, num_experts=num_experts)) #, layer_idx=i))
            else:
                decoder_layers.append(MambaLayer(config, num_experts=num_experts)) #, layer_idx=i))

        self.layers = nn.ModuleList(decoder_layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)

    def forward(self, x):
        # x: (B, L, D)

        # logits: (B, L, D)

        router_logits = []

        for decoder_layer in self.layers:
            layer_output, _ = decoder_layer(x)
            x = layer_output[0]
            router_logits.append(layer_output[1])

        return x, router_logits
    
    def step(self, x, caches):
        # x: (B, L, D)

        # logits: (B, L, D)
        # caches

        for i, decoder_layer in enumerate(self.layers):
            layer_output, caches[i] = decoder_layer(x, caches[i])
            x = layer_output[0]

        return x, caches

class AttentionLayer(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int): #, layer_idx: int): # todo : caching
        super().__init__()

        self.self_attn = AttentionSDPA(config)

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, cache = None):
        # x: (B, L, D)

        # outputs: (B, L, D)
        
        # attention
        residual = x
        x = self.input_layernorm(x)
        x, cache = self.self_attn(x, cache)
        x = residual + x

        # FFN
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)
        return outputs, cache

    def get_empty_cache(self, batch_size):
        return (None, None)

class AttentionSDPA(nn.Module):
    def __init__(self, config: JambaLMConfig): #, layer_idx: Optional[int] = None): # todo : caching
        super().__init__()

        self.config = config
        #self.layer_idx = layer_idx

        self.hidden_size = config.d_model
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, cache = None): # todo : caching
        # x: (B, L, D)

        # attn_output: (B, L, D)

        # todo : rename (virer le "states" et juste mettre queries, keys, values...)
        # todo : rename aussi le bsz et le q_len

        bsz, q_len, _ = x.size()

        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if cache is not None:
            past_key_states, past_value_states = cache
            
            # first in the sequence
            if past_key_states is not None:
                key_states = torch.cat([past_key_states, key_states], dim=2)
                value_states = torch.cat([past_value_states, value_states], dim=2)
            
            cache = (key_states, value_states)

        #print(f"key states to be used : {key_states}")
        #print(f"value states to be used : {value_states}")

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        #print("----")
        #print(query_states.shape)
        #print(key_states.shape)
        #print(value_states.shape)
        #print("-----")

        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states,
                                                                       dropout_p=self.attention_dropout if self.training else 0.0,
                                                                       is_causal=(cache is None))
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, cache

class MambaLayer(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int):
        super().__init__()

        self.config = config

        self.mamba = MambaBlock(config=config.mamba_config) #, layer_idx=layer_idx) TODO: caching

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, cache = None):
        # x: (B, L, D)

        # outputs: (B, L, D)

        # mamba
        residual = x
        x = self.input_layernorm(x)
        if cache is None:
            x = self.mamba(x)
        else:
            x, cache = self.mamba.step(x.squeeze(1), cache)
            x = x.unsqueeze(1)
        x = residual + x

        # FFN
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)

        return outputs, cache
    
    def get_empty_cache(self, batch_size):
        return (None, torch.zeros(batch_size, self.config.d_inner, self.config.d_conv-1))

class SparseMoEBlock(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int, num_experts_per_tok: int):
        super().__init__()

        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        if num_experts > 1:
            self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        else:
            self.router = None

        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

    def forward(self, x):
        # x: (B, L, D)

        # final_hidden_states: (B, L, D)
        # router_logits: (B*L, n_experts)

        #todo : pq B*L ? ça parait bizarre
        
        batch_size, sequence_length, hidden_dim = x.shape

        # no routing
        if self.num_experts == 1:
            final_hidden_states = self.experts[0](x)
            router_logits = torch.ones(
                (batch_size * sequence_length, 1),
                device=x.device,
                dtype=x.dtype,
                requires_grad=x.requires_grad,
            )
            return final_hidden_states, router_logits

        # routing
        x = x.view(-1, hidden_dim) # (B*L, D)

        router_logits = self.router(x) # (B*L, n_experts)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device)

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
            current_state = x[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, router_logits

class MLP(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size

        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    
def load_balancing_loss(router_logits, num_experts, num_experts_per_tok):
    # router_logits: list of router_logit, one per layer, each (B*D, n_experts)

    # moe_aux_loss : scalar

    router_logits = torch.cat(router_logits, dim=0)

    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    moe_aux_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return moe_aux_loss * num_experts

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
