"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier

modefied by BWi 
"""

import math
from matplotlib.pyplot import xkcd

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.embed_size % config.num_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.embed_size, config.embed_size)
        self.query = nn.Linear(config.embed_size, config.embed_size)
        self.value = nn.Linear(config.embed_size, config.embed_size)
        # regularization
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        # output projection
        self.proj = nn.Linear(config.embed_size, config.embed_size)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.num_heads

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_size)
        self.ln2 = nn.LayerNorm(config.embed_size)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_size, config.hidden_factor * config.embed_size),
            nn.GELU(),
            nn.Linear(config.hidden_factor * config.embed_size, config.embed_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.xT = config.xT
        self.regression = config.mode == 'reg'
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + config.xT, config.embed_size))
        self.drop = nn.Dropout(config.dropout)
        # xT linear
        self.xT_lin = nn.Linear(2, config.embed_size)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.embed_size)
        self.decoder = nn.Linear(config.embed_size, config.embed_size)
        
        if config.mode == 'reg':
            self.head = nn.Linear(config.embed_size, 1)
        
        self.block_size = config.block_size
        self.apply(self._init_weights)

        #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        #self.head.bias.data.fill_(0) # this is only for current data

    def configure_optimizers(self, config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=config.lr, betas=config.betas)
        return optimizer

    def forward(self, idx, xT, targets=None):
        b, t = idx.size()
        #assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        xT = torch.unsqueeze(xT,1)
        xT_proj = self.xT_lin(xT)

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        
        #concat xT to token_embeddings
        position_embeddings = self.pos_emb[:, :t + 1, :] # each position maps to a (learnable) vector
        
        if self.xT > 0:
            position_embeddings = self.pos_emb[:, :t + 1, :] # each position maps to a (learnable) vector
            token_embeddings = torch.cat([token_embeddings, xT_proj], dim=1)
            x = self.drop(token_embeddings + position_embeddings)
        else:
            position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
            xT_embedding = torch.ones(b, token_embeddings.shape[1], token_embeddings.shape[2]).to('cuda') * xT_proj
            x = self.drop(token_embeddings + position_embeddings + xT_embedding)
        
        x = self.blocks(x)
        x = self.ln_f(x)
        x = torch.max(x, dim=1)[0]
        x = self.decoder(x)
        x = F.relu(x)

        if self.config.mode == 'reg':
            logits = self.head(x)

        logits = logits.squeeze()
        return logits