import dataclasses
import os
import torch
import torch.nn as nn

@dataclasses.dataclass
class NN_config:

    xp_name: str
    device: torch.device
    criterion: nn.MSELoss
    
    padding_idx: int
    block_size:int
    vocab_size: int

    embed_size: int
    hidden_factor: int 
    num_layers: int
    num_heads:int 
    dropout:float

    lr: float
    betas:  list
    weight_decay: float

    warmup_epochs: int
        
    data_path: str
    batch_size: int
    max_btch: int
    epoch: int

    mode: str
    bins: int
    bound: int

    shift: int
