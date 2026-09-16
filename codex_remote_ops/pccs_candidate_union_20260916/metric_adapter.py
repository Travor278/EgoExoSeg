import torch
from torch import nn
from torch.nn import functional as F
class Adapter(nn.Module):
    """Rank8 shared residual projection on frozen O-MaMa embeddings."""
    def __init__(self):
        super().__init__();self.down=nn.Linear(768,8,bias=False);self.up=nn.Linear(8,768,bias=False);nn.init.normal_(self.down.weight,std=.02);nn.init.zeros_(self.up.weight)
    def forward(self,x):return F.normalize(x+self.up(torch.tanh(self.down(x))),dim=-1)
