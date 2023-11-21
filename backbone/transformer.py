import torch
import torch.nn.functional as F
from torch import nn, einsum
from beartype import beartype
import timm
from timm.models.efficientnet import _cfg

@beartype
class RT1_Mod(nn.Module):
    def __init__(
        self,
        *,
        vit: MaxViT,
        depth = 6,
        heads = 8,
        dim_head = 64,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 8,
    ):
        super().__init__()

        config = _cfg(url='', file='vit_base_patch16_224.augreg2_in21k_ft_in1k1.pth')
        self.vit = timm.create_model(
            "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            pretrained=True,
            features_only=True,
            pretrained_cfg=config
        )

        self.token_learner = TokenLearner(
            dim = vit.embed_dim,
            ff_mult = token_learner_ff_mult,
            num_output_tokens = token_learner_num_output_tokens,
            num_layers = token_learner_num_layers
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer_depth = depth

        self.transformer = Transformer(
            dim = vit.embed_dim,
            dim_head = dim_head,
            heads = heads,
            depth = depth
        )
        self.to_logits = nn.Sequential(
            LayerNorm(vit.embed_dim),
        )

    def forward(
        self,
        video,
    ):
        # print(video.shape) #torch.Size([1, 3, 1, 224, 224])

        frames, device = video.shape[2], video.device

        video = rearrange(video, 'b c f h w -> b f c h w')
        images, packed_shape = pack_one(video, '* c h w')
        # print('images=',images.shape)
        tokens = self.vit(
            images,
            texts = texts,
            cond_fns = vit_cond_fns,
            cond_drop_prob = cond_drop_prob,
            return_embeddings = True
        )
        # print('tokens=',tokens.shape)
        tokens = unpack_one(tokens, packed_shape, '* c h w')
        # print('unpack tokens=',tokens.shape)
        learned_tokens = self.token_learner(tokens)
        # print('learned_tokens=',learned_tokens.shape)
        learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')
        # print('learned_tokens=',learned_tokens.shape)
        # causal attention mask

        attn_mask = torch.ones((frames, frames), dtype = torch.bool, device = device).triu(1)
        attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = self.num_learned_tokens, r2 = self.num_learned_tokens)

        # sinusoidal positional embedding

        pos_emb = posemb_sincos_1d(frames, learned_tokens.shape[-1], dtype = learned_tokens.dtype, device = learned_tokens.device)

        learned_tokens = learned_tokens + repeat(pos_emb, 'n d -> (n r) d', r = self.num_learned_tokens)

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