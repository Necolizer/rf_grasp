import torch
import torch.nn.functional as F
from torch import nn, einsum

from typing import List, Optional, Callable, Tuple
from beartype import beartype

from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce

import timm
from timm.models.efficientnet import _cfg

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

# sinusoidal positions

def posemb_sincos_1d(seq, dim, temperature = 10000, device = None, dtype = torch.float32):
    n = torch.arange(seq, device = device)
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim = 1)
    return pos_emb.type(dtype)

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, cond_fn = None):
        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)

# attention

class TransformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        norm_context = False,
        dropout = 0.1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_bias = None,
        attn_mask = None,
        cond_fn: Optional[Callable] = None
    ):
        b = x.shape[0]

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layer-norm
            x = cond_fn(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        q = q * self.scale

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

@beartype
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TransformerAttention(dim = dim, heads =  heads, dropout = attn_dropout),
                FeedForward(dim = dim, dropout = ff_dropout)
            ]))

    def forward(
        self,
        x,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask = None
    ):
        if not exists(cond_fns):
            cond_fns = (None,) * (len(self.layers) * 2)

        cond_fns = iter(cond_fns)

        for attn, ff in self.layers:
             x = attn(x, attn_mask = attn_mask, cond_fn = next(cond_fns)) + x
             x = ff(x, cond_fn = next(cond_fns)) + x
        return x

# token learner module

class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        num_output_tokens = 8,
        num_layers = 2
    ):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups = num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups = num_output_tokens),
        )

    def forward(self, x):
        x, ps = pack_one(x, '* c h w')
        x = repeat(x, 'b c h w -> b (g c) h w', g = self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, 'b g h w -> b 1 g h w')
        x = rearrange(x, 'b (g c) h w -> b c g h w', g = self.num_output_tokens)

        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        x = unpack_one(x, ps, '* c n')
        return x

# @beartype
# class RT1_Mod(nn.Module):
#     def __init__(
#         self,
#         *,
#         depth = 6,
#         heads = 8,
#         dim_head = 64,
#         token_learner_ff_mult = 2,
#         token_learner_num_layers = 2,
#         token_learner_num_output_tokens = 8,
#     ):
#         super().__init__()

#         vit_embed_dim = 768

#         config = _cfg(url='', file='/home/amax/Project/Pretrained/vit_base_patch16_224.augreg2_in21k_ft_in1k/pytorch_model.bin')
#         self.vit = timm.create_model(
#             "vit_base_patch16_224.augreg2_in21k_ft_in1k",
#             pretrained=True,
#             num_classes=0,
#             pretrained_cfg=config
#         )

#         self.token_learner = TokenLearner(
#             dim = vit_embed_dim,
#             ff_mult = token_learner_ff_mult,
#             num_output_tokens = token_learner_num_output_tokens,
#             num_layers = token_learner_num_layers
#         )

#         self.num_learned_tokens = token_learner_num_output_tokens

#         self.transformer_depth = depth

#         self.transformer = Transformer(
#             dim = vit_embed_dim,
#             dim_head = dim_head,
#             heads = heads,
#             depth = depth
#         )
#         self.to_logits = nn.Sequential(
#             LayerNorm(vit_embed_dim),
#         )

#     def forward(
#         self,
#         video,
#     ):
#         # print(video.shape) #torch.Size([1, 3, 1, 224, 224])

#         frames, device = video.shape[2], video.device

#         video = rearrange(video, 'b c f h w -> b f c h w')
#         images, packed_shape = pack_one(video, '* c h w')
#         # print('images=',images.shape)

#         tokens = self.vit.forward_features(
#             images,
#         )[:,1:,:]

#         tokens = tokens.permute(0, 2, 1)
#         tokens = tokens.view(tokens.shape[0], tokens.shape[1], int(tokens.shape[2] ** 0.5), int(tokens.shape[2] ** 0.5))

#         # print('tokens=',tokens.shape)
#         tokens = unpack_one(tokens, packed_shape, '* c h w')
#         # print('unpack tokens=',tokens.shape)

#         learned_tokens = self.token_learner(tokens)
#         # print('learned_tokens=',learned_tokens.shape)
#         learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')
#         # print('learned_tokens=',learned_tokens.shape)
#         # causal attention mask

#         attn_mask = torch.ones((frames, frames), dtype = torch.bool, device = device).triu(1)
#         attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = self.num_learned_tokens, r2 = self.num_learned_tokens)

#         # sinusoidal positional embedding

#         pos_emb = posemb_sincos_1d(frames, learned_tokens.shape[-1], dtype = learned_tokens.dtype, device = learned_tokens.device)

#         learned_tokens = learned_tokens + repeat(pos_emb, 'n d -> (n r) d', r = self.num_learned_tokens)

#         # attention

#         attended_tokens = self.transformer(learned_tokens, attn_mask = ~attn_mask)
#         # print('attended_token=',attended_tokens.shape)
#         pooled = reduce(attended_tokens, 'b (f n) d -> b f d', 'mean', f = frames)
        
#         # print('ss')
#         # print('pooled=',pooled.shape)
#         # return pooled.squeeze(1)
#         logits = self.to_logits(pooled).squeeze(1)
#         # print(logits.shape) #torch.Size([1, 768])
#         return logits
    

@beartype
class RT1_Mod(nn.Module):
    def __init__(
        self,
        *,
        depth = 6,
        heads = 8,
        dim_head = 64,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 8,
    ):
        super().__init__()

        vit_embed_dim = 768

        config = _cfg(url='', file='/home/amax/Project/Pretrained/vit_base_patch16_224.augreg2_in21k_ft_in1k/pytorch_model.bin')
        self.vit = timm.create_model(
            "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            pretrained=True,
            num_classes=0,
            pretrained_cfg=config
        )

        self.token_learner = TokenLearner(
            dim = vit_embed_dim,
            ff_mult = token_learner_ff_mult,
            num_output_tokens = token_learner_num_output_tokens,
            num_layers = token_learner_num_layers
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer_depth = depth

        self.transformer = Transformer(
            dim = vit_embed_dim,
            dim_head = dim_head,
            heads = heads,
            depth = depth
        )
        self.to_logits = nn.Sequential(
            LayerNorm(vit_embed_dim),
        )

        self.state_linear = nn.Sequential(
            LayerNorm(7),
            nn.Linear(7, 768)
        )

    def forward(
        self,
        video,
        states
    ):
        # print(video.shape) #torch.Size([1, 3, 1, 224, 224])

        frames, device = video.shape[2], video.device

        video = rearrange(video, 'b c f h w -> b f c h w')
        images, packed_shape = pack_one(video, '* c h w')
        # print('images=',images.shape)

        tokens = self.vit.forward_features(
            images,
        )[:,1:,:]

        tokens = tokens.permute(0, 2, 1)
        tokens = tokens.view(tokens.shape[0], tokens.shape[1], int(tokens.shape[2] ** 0.5), int(tokens.shape[2] ** 0.5))

        # print('tokens=',tokens.shape)
        tokens = unpack_one(tokens, packed_shape, '* c h w')
        # print('unpack tokens=',tokens.shape)

        learned_tokens = self.token_learner(tokens)
        # print('learned_tokens=',learned_tokens.shape)
        learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')
        # print('learned_tokens=',learned_tokens.shape) #torch.Size([1, 8, 768])

        state_token = self.state_linear(states).unsqueeze(1)
        learned_tokens = torch.cat((learned_tokens, state_token), dim=1)

        # causal attention mask

        attn_mask = torch.ones((frames, frames), dtype = torch.bool, device = device).triu(1)
        attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = self.num_learned_tokens + 1, r2 = self.num_learned_tokens + 1)

        # sinusoidal positional embedding

        pos_emb = posemb_sincos_1d(frames, learned_tokens.shape[-1], dtype = learned_tokens.dtype, device = learned_tokens.device)

        learned_tokens = learned_tokens + repeat(pos_emb, 'n d -> (n r) d', r = self.num_learned_tokens + 1)

        # attention

        attended_tokens = self.transformer(learned_tokens, attn_mask = ~attn_mask)
        # print('attended_token=',attended_tokens.shape)
        pooled = reduce(attended_tokens, 'b (f n) d -> b f d', 'mean', f = frames)
        
        # print('ss')
        # print('pooled=',pooled.shape)
        # return pooled.squeeze(1)
        logits = self.to_logits(pooled).squeeze(1)
        # print(logits.shape) #torch.Size([1, 768])
        return logits