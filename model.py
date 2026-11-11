"""The transformer, and a hand-run forward pass that exposes residuals and attention and allows patching."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from process import VOCAB


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o, self.heads = nn.Linear(d, 3 * d), nn.Linear(d, d), heads
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        sh = lambda t: t.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """Pre-LN causal transformer with learned absolute positions. Input vocabulary has one extra id, BOS."""

    def __init__(self, T, d, layers, heads):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB + 1, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.lnf, self.unembed = nn.LayerNorm(d), nn.Linear(d, VOCAB)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for b in self.blocks:
            h = b(h)
        return self.unembed(self.lnf(h))


@torch.no_grad()
def trace(model, x, patch=None, want_attn=True):
    """Forward pass by hand. Returns logits, residuals [embedding, after block 1, ..., after block L],
    and attention weights per block [B, heads, T, T].

    patch = {l: (mask [B, T] bool, values [B, T, d])} overwrites the residual after block l
    (l = 0 is the embedding) at the masked positions before the pass continues."""
    T = x.shape[1]
    patch = patch or {}

    def apply(l, h):
        if l in patch:
            mask, values = patch[l]
            h = torch.where(mask[..., None], values, h)
        return h

    h = apply(0, model.tok(x) + model.pos(torch.arange(T)))
    resid, attn = [h], []
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    for l, blk in enumerate(model.blocks, 1):
        B, _, D = h.shape
        q, k, v = blk.qkv(blk.ln1(h)).split(D, dim=-1)
        sh = lambda t: t.view(B, T, blk.heads, D // blk.heads).transpose(1, 2)
        a = (sh(q) @ sh(k).transpose(-1, -2) / (D // blk.heads) ** 0.5).masked_fill(future, -1e9).softmax(-1)
        h = h + blk.o((a @ sh(v)).transpose(1, 2).reshape(B, T, D))
        h = apply(l, h + blk.mlp(blk.ln2(h)))
        resid.append(h)
        if want_attn:
            attn.append(a)
    return model.unembed(model.lnf(h)), resid, attn


def load(run, args):
    model = GPT(args["T"], args["d"], args["layers"], args["heads"])
    model.load_state_dict(torch.load(run + ".pt", map_location="cpu"))
    return model.eval()
