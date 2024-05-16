import torch
from mamba_lm_onnx import from_pretrained

model = from_pretrained('state-spaces/mamba-370m')
model.eval()

# Note that this may not work
# so you may need to adjust the input shape and type
torch.onnx.export(model,
                  torch.zeros(1, dtype=torch.int64),
                  'mamba-370m.onnx',
                  input_names=['input'],
                  output_names=['output'],
                  opset_version=17,
                  )
