'''
Code borrowed from ISAB repository: https://github.com/lucidrains/isab-pytorch/blob/main/isab_pytorch/isab_pytorch.py 

Modified to allow multi-resolution encoding to be processed using transformers
'''
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers
def exists(val):
    return val is not None

# classes
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        and_self_attend = False
    ):
        ''' 
        dim: dimension of input
        heads: number of heads
        dim_head: dimension of each head
        and_self_attend: if True, then the input is concatenated with the context before attention is applied
            this is basically to allow the input to attend to itself
            If False, then input is used for query, and context is used for key and value pairs
        '''
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.and_self_attend = and_self_attend
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context,
        mask = None
    ):
        h, scale = self.heads, self.scale

        if self.and_self_attend:
            context = torch.cat((x, context), dim = -2)

            if exists(mask):
                mask = F.pad(mask, (x.shape[-2], 0), value = True)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        if exists(mask):
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b 1 1 n')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class ISABBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        num_latents = None,
        latent_self_attend = False
    ):
        '''
        dim: dimension of input
        heads: number of heads
        num_latents: number of latents to learn in the ISAB block (as a compression mechanism)
            this will be specified as a parameter to the module, and will be learned
        latent_self_attend: if True, then the latents are concatenated with the input before attention is applied
        '''
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim)) if exists(num_latents) else None
        self.attn1 = Attention(dim, heads, and_self_attend = latent_self_attend)
        self.attn2 = Attention(dim, heads)

    def forward(self, x, latents = None, mask = None):
        b, *_ = x.shape
        assert exists(latents) ^ exists(self.latents), 'you can only either learn the latents within the module, or pass it in externally'
        latents = latents if exists(latents) else self.latents

        if latents.ndim == 2:
            latents = repeat(latents, 'n d -> b n d', b = b)

        latents = self.attn1(latents, x, mask = mask)
        out     = self.attn2(x, latents)
        return out, latents


class ISAB(nn.Module):
    ''' this is to apply the ISAB block to the multi-resolution encoding '''
    def __init__(self, 
            input_dim, 
            num_latents,
            offsets, 
            resolutions,
            heads = 8,
        ):
        super().__init__()
        self.offsets = offsets.data.cpu()
        self.resolutions = resolutions.data.cpu()
        self.num_levels = num_levels = len(resolutions)
        modules = []
        for lvl in range(num_levels):
            modules.append(ISABBlock(dim = input_dim, heads = heads, num_latents = num_latents))
        self.isab_blocks = nn.ModuleList(modules)
        # apply layernorm after?

    def forward(self, x):
        # x: [B, N, C]
        chunks = []
        for i in range(self.num_levels):
            offset_start, offset_end = self.offsets[i], self.offsets[i+1]
            x_chunk = x[:, offset_start:offset_end, :]
            y_chunk = self.isab_blocks[i](x_chunk)[0]
            chunks.append(y_chunk)
        # append chunks at the end
        return torch.cat(chunks, dim=1)
            

if __name__ == '__main__':
    import gridencoder as ge
    L = 19
    encoder = ge.GridEncoder(desired_resolution=196, gridtype='tiled', align_corners=True, log2_hashmap_size=L).cuda()
    embed = encoder.embeddings[None].contiguous() * 1e3  # [1, N, 2]
    # embed = embed.detach()
    # embed = embed.expand(4, -1, -1).contiguous()
    resolutions = encoder.resolutions
    offsets = encoder.offsets

    # test ISAB
    net = ISAB(2, 16**2, offsets, resolutions, heads=1).cuda()
    from time import time
    a = time()
    out = net(embed)
    b = time()
    ((out**2).sum()).backward()
    c = time()
    print(b-a, c-b)
    input()