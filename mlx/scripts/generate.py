import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings('ignore')

import argparse

import mlx.core as mx

import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoTokenizer

from mamba_lm_mlx import MambaLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_name', type=str, default='state-spaces/mamba-130m')
    parser.add_argument('--model_dir', type=str, default=None, help='local model to load. overwrites hf_model_name')
    parser.add_argument('--prompt', type=str, default='Mamba is a type of')
    parser.add_argument('--n_tokens', type=int, default=50, help='number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=None, help='top_k sampling : only sample from the top k most probable tokens')

    args = parser.parse_args()

    if args.model_dir is not None:
        raise NotImplementedError
    else:
        model = MambaLM.from_pretrained(args.hf_model_name)
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

    #print(model.config)
    
    mx.set_default_device(mx.gpu)

    for token in model.generate(tokenizer, args.prompt, n_tokens_to_gen=args.n_tokens, temperature=args.temperature, top_k=args.top_k):
        print(token, end='', flush=True)
