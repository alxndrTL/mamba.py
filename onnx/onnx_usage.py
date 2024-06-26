import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

import onnxruntime as ort

num_tokens = 50
temperature = 1.0
top_k = 40
sample = True

provider = ['CPUExecutionProvider', ]
model = ort.InferenceSession('mamba-370m.onnx', providers=provider)


# See https://huggingface.co/state-spaces/mamba-2.8b/discussions/3
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')


inputs = input(">>> ")
input_ids = tokenizer(inputs, return_tensors='pt').input_ids
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def init_zeros(shape: list):
    return to_numpy(torch.zeros(shape))

hs = init_zeros(model.get_inputs()[1].shape)
inputs = init_zeros(model.get_inputs()[2].shape)

for i in range(input_ids.size(1) + num_tokens - 1):
    with torch.no_grad():
        # print(input_ids[:, i].shape)
        ort_input = {model.get_inputs()[0].name: to_numpy(input_ids[:, i]), model.get_inputs()[1].name: hs, model.get_inputs()[2].name: inputs}
        # return a list of outputs
        run_result = model.run(None, ort_input)
        # print(len(run_result))
        next_token = torch.from_numpy(run_result[0])
        hs = run_result[1]
        inputs = run_result[2]
    
    if i+1 >= input_ids.size(1):
        probs = F.softmax(next_token / temperature, dim=-1) # (batch_size, vocab_size)

        if top_k is not None:
            values, _ = torch.topk(probs, k=top_k) # (batch_size, k) ordered from lowest to biggest
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)

        if sample:
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # (batch_size)
        else:
            next_token = torch.argmax(probs, dim=-1) # (batch_size)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

outputs = [tokenizer.decode(output.tolist()) for output in input_ids]


print(outputs[0])