import torch
import torch.nn as nn
import module
from module import default
from functools import partial

# Input：a batch of noisy images of shape ( batch_size, num_channels, h, w )
# and a batch of steps of shape ( batch_size, 1 )
# output: a tensor of shape ( batch_size, num_channels, h, w )


class Unet(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3,
                 with_time_emb=True, resnet_block_groups=8, use_convnext=True, convnext_mult=2):
        super(Unet, self).__init__()
        self.channels = channels

        init_dim = default(init_dim, dim//3*2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim*m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(module.ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(module.ResnetBlock, groups=resnet_block_groups)

        if with_time_emb:
            time_dim = dim*4
            self.time_mlp = nn.Sequential(
                module.SinusoidalPositionalEmbedding(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                    module.Residual(module.PreNorm(dim_out, module.LinearAttention(dim_out))),
                    module.downsample(dim_out) if not is_last else nn.Identity(),
                ])
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = module.Residual(module.PreNorm(mid_dim, module.Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out*2, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    module.Residual(module.PreNorm(dim_in, module.LinearAttention(dim_in))),
                    module.upsample(dim_in) if not is_last else nn.Identity(),
                ])
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(block_klass(dim, dim),
                                        nn.Conv2d(dim, out_dim, 1))

    def forward(self, x, time):
        x = self.init_conv(x)
        t = self.time_mlp(time) if module.exists(self.time_mlp) else None
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
