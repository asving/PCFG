"""The transformer, and a hand-run forward pass that exposes residuals and attention and allows patching."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o, self.heads = nn.Linear(d, 3 * d), nn.Linear(d, d), heads
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def qkv_heads(self, x, rot=None):
        """Queries, keys and values per head [B, heads, T, d / heads]; queries and keys rotated if rot is given."""
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        sh = lambda t: t.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        if rot is not None:
            q, k = rotate(q, rot), rotate(k, rot)
        return q, k, v

    def forward(self, x, rot=None):
        B, T, D = x.shape
        q, k, v = self.qkv_heads(x, rot)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


ROT_PERIODS = (2, 3, 5, 7, 11, 16, 32, 64)                  # in levels of depth; one pair of dimensions each


def rotate(t, rot):
    """Rotate the first len(ROT_PERIODS) pairs of dimensions of t [B, heads, T, hd] by the angles in rot."""
    cos, sin = rot                                          # [B, 1, T, M]
    M = cos.shape[-1]
    a, b = t[..., 0:2 * M:2], t[..., 1:2 * M:2]
    turned = torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)
    return torch.cat([turned, t[..., 2 * M:]], -1)


class GPT(nn.Module):
    """Pre-LN causal transformer with learned absolute positions. Input vocabulary has one extra id, BOS.

    Three optional ways of handing the network part of the bracket structure, each switched on by passing
    (first open id, first close id). In every case the structure is computed here, not by the network.

    depth_in   the embedding gains depth / 4 times one learned vector, where depth counts the opens minus
               the closes read so far (including the position's own token).
    tag_in     the embedding gains a learned vector indexed by the position of the open that opened the
               position's bracket (0 for the root): the bracket tag of algorithm.py, step 2.
    depth_rot  in every block, the first pairs of query and key dimensions of every head are rotated by
               depth times a fixed frequency (periods ROT_PERIODS), as rotary embeddings do with position.
               A score between two positions then depends on their depths only through the difference."""

    def __init__(self, vocab, T, d, layers, heads, depth_in=None, tag_in=None, depth_rot=None):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab + 1, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.lnf, self.unembed = nn.LayerNorm(d), nn.Linear(d, vocab)
        self.vocab, self.depth_in, self.tag_in, self.depth_rot = vocab, depth_in, tag_in, depth_rot
        if depth_in is not None:
            self.dep = nn.Linear(1, d, bias=False)
        if tag_in is not None:
            self.tag = nn.Embedding(T, d)

    def depth(self, x, ids):
        o, c = ids
        is_open = (x >= o) & (x < c)
        return is_open.float().cumsum(1) - ((x >= c) & (x < self.vocab)).float().cumsum(1), is_open

    def embed(self, x):
        J = torch.arange(x.shape[1], device=x.device)
        h = self.tok(x) + self.pos(J)
        if self.depth_in is not None:
            h = h + self.dep(self.depth(x, self.depth_in)[0][..., None] / 4)
        if self.tag_in is not None:                         # the most recent open at my depth; BOS opens the root
            depth, is_open = self.depth(x, self.tag_in)
            cand = (depth[:, :, None] == depth[:, None, :]) & (is_open | (J == 0))[:, None, :] & (J[None, :] <= J[:, None])
            own = torch.where(cand, J, torch.full_like(J, -1)).max(-1).values
            h = h + self.tag(own.clamp(min=0))
        return h

    def rotation(self, x):
        if self.depth_rot is None:
            return None
        ang = self.depth(x, self.depth_rot)[0][:, None, :, None] * (2 * torch.pi / torch.tensor(ROT_PERIODS, device=x.device, dtype=torch.float32))
        return ang.cos(), ang.sin()

    def forward(self, x):
        h, rot = self.embed(x), self.rotation(x)
        for b in self.blocks:
            h = b(h, rot)
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

    h, rot = apply(0, model.embed(x)), model.rotation(x)
    resid, attn = [h], []
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    for l, blk in enumerate(model.blocks, 1):
        B, _, D = h.shape
        q, k, v = blk.qkv_heads(h, rot)
        a = (q @ k.transpose(-1, -2) / (D // blk.heads) ** 0.5).masked_fill(future, -1e9).softmax(-1)
        h = h + blk.o((a @ v).transpose(1, 2).reshape(B, T, D))
        h = apply(l, h + blk.mlp(blk.ln2(h)))
        resid.append(h)
        if want_attn:
            attn.append(a)
    return model.unembed(model.lnf(h)), resid, attn


def load(run, args, proc):
    ids = (proc.OPEN0, proc.CLOSE0)
    model = GPT(proc.VOCAB, args["T"], args["d"], args["layers"], args["heads"],
                depth_in=ids if args.get("depth_input") else None, tag_in=ids if args.get("tag_input") else None,
                depth_rot=ids if args.get("depth_rotary") else None)
    model.load_state_dict(torch.load(run + ".pt", map_location="cpu"))
    return model.eval()
