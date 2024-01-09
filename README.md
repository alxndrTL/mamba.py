# mamba.py 🐍 : a simple parallel scan implementation
Mamba in PyTorch with a simple parallel scan implementation

-graphique speed comparaison

-pq ce repo ?

-utilisation simple (def et forward)

## Usage

The most basic usage is to use the ```Mamba``` object, which implements 
No embedding, no head : input is ```(B, L, D)``` and output is ```(B, L, D)```.

```
import torch
from mamba import Mamba, MambaConfig

config = MambaConfig(d_model=16, n_layers=2)
model = Mamba(config)

B, L, D = 2, 64, 16
x = torch.randn(B, L, D)
y = model(x)

assert y.shape == x.shape
```

The class ```MambaLM``` builds on this previous ... and can be used as follows :

```

```

-todo
