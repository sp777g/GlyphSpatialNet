import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import einsum, nn
from einops import rearrange, reduce


# helpers functions
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# small helper modules
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, spatial_emb_dim=None, style_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear((int(time_emb_dim) if exists(time_emb_dim) else 0) + \
                      (int(spatial_emb_dim) if exists(spatial_emb_dim) else 0) + \
                      (int(style_emb_dim) if exists(style_emb_dim) else 0), dim_out * 2)
        ) if exists(time_emb_dim) or exists(spatial_emb_dim) or exists(style_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, spatial_emb=None, style_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(spatial_emb) or exists(style_emb)):
            cond_emb = tuple(filter(exists, (time_emb, spatial_emb, style_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class ConditionAdapter(nn.Module):
    def __init__(self, in_channels, in_size, out_channels, out_size):
        """
        条件特征适配器
        Args:
            in_channels: 输入条件特征通道数
            out_channels: 目标输出通道数
            in_size: 输入条件特征空间尺寸 (正方形边长)
            out_size: 目标输出空间尺寸 (正方形边长)
        """
        super().__init__()

        # 0. Normalization
        self.pre_norm = LayerNorm(in_channels)

        # 1. 通道数适配器 (1x1卷积)
        self.channel_adapter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        # 2. 空间尺寸适配器
        if in_size > out_size:  # 需要下采样
            # 计算下采样倍数 (必须是整数)
            downsample_ratio = in_size // out_size
            assert in_size % out_size == 0, "输入尺寸必须是输出尺寸的整数倍"
            self.spatial_adapter = nn.AvgPool2d(
                kernel_size=downsample_ratio,
                stride=downsample_ratio
            )
        elif in_size < out_size:  # 需要上采样
            # 计算上采样倍数 (必须是整数)
            upsample_ratio = out_size // in_size
            assert out_size % in_size == 0, "输出尺寸必须是输入尺寸的整数倍"
            self.spatial_adapter = nn.Upsample(
                scale_factor=upsample_ratio,
                mode='bilinear',
                align_corners=False
            )
        else:  # 尺寸相同
            self.spatial_adapter = nn.Identity()

    def forward(self, x):
        """
        输入: [bs, in_channels, in_size, in_size]
        输出: [bs, out_channels, out_size, out_size]
        """
        x = self.pre_norm(x)
        x = self.channel_adapter(x)
        x = self.spatial_adapter(x)
        return x


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=None, dim_head=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class CrossLinearAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, heads=None, dim_head=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.pre_norm = LayerNorm(q_dim)

        self.to_q = nn.Conv2d(q_dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(kv_dim, hidden_dim * 2, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, q_dim, 1),
            LayerNorm(q_dim)
        )

    def forward(self, x, context):
        b, c, h, w = x.shape

        r = x.clone()

        x = self.pre_norm(x)
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), (q, k, v))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)

        return self.to_out(out) + r


class Attention(nn.Module):
    def __init__(self, dim, heads=None, dim_head=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, heads=None, dim_head=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.pre_norm = LayerNorm(q_dim)

        self.to_q = nn.Conv2d(q_dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(kv_dim, hidden_dim * 2, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, q_dim, 1)

    def forward(self, x, context):
        b, c, h, w = x.shape

        r = x.clone()

        x = self.pre_norm(x)
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), (q, k, v))

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out) + r


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            dim_multi,
            img_channels,
            resnet_block_groups,
            self_condition=False
    ):
        super().__init__()

        # determine dimensions
        self.channels = img_channels
        self.self_condition = self_condition

        # x_t, x_in, x_self_cond
        input_channels = img_channels * (3 if self_condition else 2)

        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_multi)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.mid_p = nn.ModuleList([])
        self.mid_s = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConditionAdapter(in_channels=256, in_size=16, out_channels=dim_in, out_size=(64 // (2 ** ind))),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                CrossLinearAttention(q_dim=dim_in, kv_dim=dim_in, heads=4, dim_head=(dim_in // 4)),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                CrossLinearAttention(q_dim=dim_in, kv_dim=dim_in, heads=4, dim_head=(dim_in // 4)),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]

        for _ in range(3):
            self.mid_p.append(nn.ModuleList([
                ConditionAdapter(in_channels=256, in_size=16, out_channels=mid_dim,
                                 out_size=(64 // (2 ** (num_resolutions - 1)))),
                block_klass(mid_dim, mid_dim, time_emb_dim=time_dim),
                CrossAttention(q_dim=mid_dim, kv_dim=mid_dim, heads=4, dim_head=(mid_dim // 4))
            ]))

        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(dim=mid_dim, heads=4, dim_head=(mid_dim // 4))))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for _ in range(3):
            self.mid_s.append(nn.ModuleList([
                ConditionAdapter(in_channels=256, in_size=16, out_channels=mid_dim,
                                 out_size=(64 // (2 ** (num_resolutions - 1)))),
                block_klass(mid_dim + mid_dim, mid_dim, time_emb_dim=time_dim),
                CrossAttention(q_dim=mid_dim, kv_dim=mid_dim, heads=4, dim_head=(mid_dim // 4))
            ]))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ConditionAdapter(in_channels=256, in_size=16, out_channels=dim_out,
                                 out_size=(64 // (2 ** (len(in_out) - 1 - ind)))),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                CrossLinearAttention(q_dim=dim_out, kv_dim=dim_out, heads=4, dim_head=(dim_out // 4)),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                CrossLinearAttention(q_dim=dim_out, kv_dim=dim_out, heads=4, dim_head=(dim_out // 4)),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        output_channels = img_channels
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_cond_ada = ConditionAdapter(in_channels=256, in_size=16, out_channels=dim, out_size=64)
        self.final_attn = CrossLinearAttention(q_dim=dim, kv_dim=dim, heads=4, dim_head=(dim // 4))
        self.final_conv = nn.Conv2d(dim, output_channels, 1)

    def forward(self, x, time, style_cond, x_self_cond=None):
        if self.self_condition:
            bs, _, h, w = x.shape
            channels = self.img_channels
            x_self_cond = default(x_self_cond, lambda: torch.zeros(bs, channels, h, w, device=x.device))
            x = torch.cat([x, x_self_cond], dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for cond_ada, block1, attn1, block2, attn2, downsample in self.downs:
            layer_cond = cond_ada(style_cond)

            x = block1(x, t)
            x = attn1(x, layer_cond)
            h.append(x)

            x = block2(x, t)
            x = attn2(x, layer_cond)
            h.append(x)

            x = downsample(x)

        for cond_ada, block, attn in self.mid_p:
            layer_cond = cond_ada(style_cond)

            x = block(x, t)
            x = attn(x, layer_cond)
            h.append(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for cond_ada, block, attn in self.mid_s:
            layer_cond = cond_ada(style_cond)

            x = torch.cat((x, h.pop()), dim=1)
            x = block(x, t)
            x = attn(x, layer_cond)

        for cond_ada, block1, attn1, block2, attn2, upsample in self.ups:
            layer_cond = cond_ada(style_cond)

            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = attn1(x, layer_cond)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn2(x, layer_cond)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        final_cond = self.final_cond_ada(style_cond)
        x = self.final_attn(x, final_cond)
        x = self.final_conv(x)

        return x


def linear_saturation(x, range_val=0.2, slope=0.05):
    """
    分段线性函数（张量版本），在[-range_val, range_val]区间内斜率为1，区间外斜率为slope。
    保持函数连续，分界点不可导，支持自动微分。

    参数:
        x (torch.Tensor): 输入张量
        range_val (float): 范围阈值（正值）
        slope (float): 区间外的斜率

    返回:
        torch.Tensor: 输出张量
    """
    # 创建掩码用于区分不同区域
    in_range_mask = (x >= -range_val) & (x <= range_val)
    left_mask = x < -range_val
    right_mask = x > range_val

    # 初始化输出张量
    result = torch.zeros_like(x)

    # 区间内 [-range_val, range_val]
    result[in_range_mask] = x[in_range_mask]

    # 区间外左侧 (x < -range_val)
    result[left_mask] = slope * (x[left_mask] + range_val) - range_val

    # 区间外右侧 (x > range_val)
    result[right_mask] = slope * (x[right_mask] - range_val) + range_val

    return result


class SpatialUnet(nn.Module):
    def __init__(
            self,
            dim,
            dim_multi,
            img_channels,
            resnet_block_groups,
            self_condition=False
    ):
        super().__init__()

        # determine dimensions
        self.channels = img_channels
        self.self_condition = self_condition

        # x_t, x_in, x_self_cond
        input_channels = img_channels * (3 if self_condition else 2)

        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_multi)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.mid_p = nn.ModuleList([])
        self.mid_s = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConditionAdapter(in_channels=256, in_size=16, out_channels=dim_in, out_size=(64 // (2 ** ind))),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                CrossLinearAttention(q_dim=dim_in, kv_dim=dim_in, heads=4, dim_head=(dim_in // 4)),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                CrossLinearAttention(q_dim=dim_in, kv_dim=dim_in, heads=4, dim_head=(dim_in // 4)),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                nn.Sequential(
                    nn.Conv2d(dim_in, 1, 1),
                    nn.Flatten(1),
                    nn.LayerNorm((64 // (2 ** ind)) ** 2),
                    nn.Linear((64 // (2 ** ind)) ** 2,
                              ((64 // (2 ** ind)) ** 2) >> 1),
                    nn.GELU(),
                    nn.Linear(((64 // (2 ** ind)) ** 2) >> 1,
                              ((64 // (2 ** ind)) ** 2) >> 1),
                    nn.GELU()
                )
            ]))

        mid_dim = dims[-1]

        for _ in range(3):
            self.mid_p.append(nn.ModuleList([
                ConditionAdapter(in_channels=256, in_size=16, out_channels=mid_dim,
                                 out_size=(64 // (2 ** (num_resolutions - 1)))),
                block_klass(mid_dim, mid_dim, time_emb_dim=time_dim),
                CrossAttention(q_dim=mid_dim, kv_dim=mid_dim, heads=4, dim_head=(mid_dim // 4))
            ]))

        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(dim=mid_dim, heads=4, dim_head=(mid_dim // 4))))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for _ in range(3):
            self.mid_s.append(nn.ModuleList([
                ConditionAdapter(in_channels=256, in_size=16, out_channels=mid_dim,
                                 out_size=(64 // (2 ** (num_resolutions - 1)))),
                block_klass(mid_dim + mid_dim, mid_dim, time_emb_dim=time_dim),
                CrossAttention(q_dim=mid_dim, kv_dim=mid_dim, heads=4, dim_head=(mid_dim // 4))
            ]))

        self.mid_reduce = nn.Sequential(
            nn.Conv2d(mid_dim, 1, 1),
            nn.Flatten(1),
            nn.LayerNorm((64 // (2 ** (num_resolutions - 1))) ** 2),
            nn.Linear((64 // (2 ** (num_resolutions - 1))) ** 2,
                      ((64 // (2 ** (num_resolutions - 1))) ** 2) >> 1),
            nn.GELU(),
            nn.Linear(((64 // (2 ** (num_resolutions - 1))) ** 2) >> 1,
                      ((64 // (2 ** (num_resolutions - 1))) ** 2) >> 1),
            nn.GELU()
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ConditionAdapter(in_channels=256, in_size=16, out_channels=dim_out,
                                 out_size=(64 // (2 ** (len(in_out) - 1 - ind)))),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                CrossLinearAttention(q_dim=dim_out, kv_dim=dim_out, heads=4, dim_head=(dim_out // 4)),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                CrossLinearAttention(q_dim=dim_out, kv_dim=dim_out, heads=4, dim_head=(dim_out // 4)),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                nn.Sequential(
                    nn.Conv2d(dim_out, 1, 1),
                    nn.Flatten(1),
                    nn.LayerNorm((64 // (2 ** (len(in_out) - 1 - ind))) ** 2),
                    nn.Linear((64 // (2 ** (len(in_out) - 1 - ind))) ** 2,
                              ((64 // (2 ** (len(in_out) - 1 - ind))) ** 2) >> 1),
                    nn.GELU(),
                    nn.Linear(((64 // (2 ** (len(in_out) - 1 - ind))) ** 2) >> 1,
                              ((64 // (2 ** (len(in_out) - 1 - ind))) ** 2) >> 1),
                    nn.GELU()
                )
            ]))

        output_channels = img_channels
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_cond_ada = ConditionAdapter(in_channels=256, in_size=16, out_channels=dim, out_size=64)
        self.final_attn = CrossLinearAttention(q_dim=dim, kv_dim=dim, heads=4, dim_head=(dim // 4))
        self.final_conv = nn.Conv2d(dim, output_channels, 1)

        self.final_spatial_mlp = nn.Sequential(
            nn.Linear(5472, 512),
            nn.GELU(),
            nn.Linear(512, 2)
        )

    def forward(self, x, time, style_cond, x_self_cond=None):
        if self.self_condition:
            bs, _, h, w = x.shape
            channels = self.img_channels
            x_self_cond = default(x_self_cond, lambda: torch.zeros(bs, channels, h, w, device=x.device))
            x = torch.cat([x, x_self_cond], dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        sh = []

        for cond_ada, block1, attn1, block2, attn2, downsample, _reduce in self.downs:
            layer_cond = cond_ada(style_cond)

            x = block1(x, t)
            x = attn1(x, layer_cond)
            h.append(x)

            x = block2(x, t)
            x = attn2(x, layer_cond)
            h.append(x)

            sh.append(_reduce(x))
            x = downsample(x)

        for cond_ada, block, attn in self.mid_p:
            layer_cond = cond_ada(style_cond)

            x = block(x, t)
            x = attn(x, layer_cond)
            h.append(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for cond_ada, block, attn in self.mid_s:
            layer_cond = cond_ada(style_cond)

            x = torch.cat((x, h.pop()), dim=1)
            x = block(x, t)
            x = attn(x, layer_cond)

        sh.append(self.mid_reduce(x))

        for cond_ada, block1, attn1, block2, attn2, upsample, _reduce in self.ups:
            layer_cond = cond_ada(style_cond)

            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = attn1(x, layer_cond)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn2(x, layer_cond)

            sh.append(_reduce(x))
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        final_cond = self.final_cond_ada(style_cond)
        x = self.final_attn(x, final_cond)
        x = self.final_conv(x)

        s = torch.cat(sh, dim=-1)
        theta_t = self.final_spatial_mlp(s)
        theta_t = linear_saturation(theta_t)

        return x, theta_t


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=9, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2

        # 创建高斯核
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('kernel', kernel)

    def create_gaussian_kernel(self, size, sigma):
        """纯 PyTorch 实现高斯核创建"""
        # 创建坐标网格
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2

        # 计算高斯函数
        g = coords ** 2
        g = (-g / (2 * sigma ** 2)).exp()

        # 计算二维高斯核
        g = g.unsqueeze(0) * g.unsqueeze(1)
        g = g / g.sum()  # 归一化

        # 扩展为卷积核格式 [out_channels, in_channels, H, W]
        return g.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        # 获取输入通道数
        channels = x.size(1)

        # 扩展核以匹配输入通道数
        kernel = self.kernel.repeat(channels, 1, 1, 1)

        # 应用分组卷积实现通道独立的高斯模糊
        return F.conv2d(
            x,
            kernel,
            padding=self.padding,
            groups=channels
        )


class GradientPropagator(nn.Module):
    def __init__(self, kernel_size=9, sigma=1.0):
        super().__init__()
        self.blur = GaussianBlur(kernel_size, sigma)

    def forward(self, x):
        # 计算模糊版本
        blurred = self.blur(x)

        # 核心梯度传播公式
        return blurred + (x - blurred.detach())


class SpatialInformationAggregationNetwork(nn.Module):
    def __init__(
            self,
            dim,
            dim_multi,
            img_channels,
            resnet_block_groups,
    ):
        super().__init__()

        self.gradient_propagator = GradientPropagator(kernel_size=31, sigma=4.0)  # 大范围传播

    def forward(self, x, time, theta_t):
        identity = torch.eye(2, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        translation = theta_t.unsqueeze(-1)  # [B, 2, 1]
        theta = torch.cat([identity, translation], dim=2)  # [B, 2, 3]

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

        x = self.gradient_propagator(x)

        return x


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = list()
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

    return torch.tensor(betas, dtype=torch.float32)


def get_coefficient(time_steps):
    # Glide cosine schedule, squaredcos_cap_v2
    betas = betas_for_alpha_bar(time_steps)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumsum = 1 - alphas_cumprod ** 0.5
    betas2_cumsum = 1 - alphas_cumprod

    alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
    betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
    alphas = alphas_cumsum - alphas_cumsum_prev
    alphas[0] = 0
    betas2 = betas2_cumsum - betas2_cumsum_prev
    betas2[0] = 0

    betas_cumsum = torch.sqrt(betas2_cumsum)

    return {'alphas_cumsum': alphas_cumsum, 'betas_cumsum': betas_cumsum}


class ResidualDiffusion(nn.Module):
    def __init__(
            self,
            rd_params,
            sia_params,
            time_steps=1000,
    ):
        super().__init__()

        self.residual_net = SpatialUnet(**rd_params)
        self.noise_net = Unet(**rd_params)
        self.sia_net = SpatialInformationAggregationNetwork(**sia_params)

        self.time_steps = time_steps

        coefficient = get_coefficient(time_steps=self.time_steps)
        self.register_buffer('alphas_cumsum', coefficient['alphas_cumsum'])
        self.register_buffer('betas_cumsum', coefficient['betas_cumsum'])

    def forward(self, x_t, t, x_in, style_cond, x_self_cond=None):
        # style_cond.shape = [b, 128, 16, 16]

        model_input = torch.cat((x_t, x_in), dim=1)
        time_alpha = self.alphas_cumsum[t] * self.time_steps
        time_beta = self.betas_cumsum[t] * self.time_steps

        x_0, theta_t = self.residual_net(model_input, time_alpha, style_cond, x_self_cond)
        x_0 = self.sia_net(x_0, time_alpha, theta_t)

        noise = self.noise_net(model_input, time_beta, style_cond, x_self_cond)

        return {'x_0': x_0, 'noise': noise}


class Sampler(object):
    def __init__(
            self,
            device=None,
            time_steps=1000,
            sampling_time_steps=5,
            self_condition=False,
    ):
        self.time_steps = time_steps
        self.sampling_time_steps = sampling_time_steps
        assert self.sampling_time_steps <= self.time_steps

        self.self_condition = self_condition

        self.device = device

        coefficient = get_coefficient(time_steps=self.time_steps)
        self.alphas_cumsum = coefficient['alphas_cumsum'].to(self.device)
        self.betas_cumsum = coefficient['betas_cumsum'].to(self.device)

    def DDIM_sample(self, model, src, style_cond):
        batch_size, *_ = src.shape

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, self.time_steps - 1, steps=self.sampling_time_steps + 1)
        times = list(reversed(times.int().tolist()))

        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        img = src + torch.randn(src.shape, device=self.device)

        x_0 = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            self_cond = x_0 if self.self_condition else None

            with torch.no_grad():
                pred = model(img, time_cond, src, style_cond, self_cond)

            x_0 = pred['x_0']
            noise = pred['noise']
            res = src - x_0

            if time_next < 0:
                img = x_0
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]

            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]

            img = img - (alpha_cumsum - alpha_cumsum_next) * res - (betas_cumsum - betas_cumsum_next) * noise

        return img


########################################################################################################################
#                                                                                                                      #
########################################################################################################################
def StyleDownsample(dim, dim_out=None):
    return nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(dim, dim_out, 3, padding=1)
    )


class StyleEncoder(nn.Module):
    def __init__(
            self,
            dim,
            dim_multi,
            img_channels,
            resnet_block_groups,
    ):
        super().__init__()

        # determine dimensions
        self.channels = img_channels

        # x_t, x_in, x_self_cond
        input_channels = img_channels

        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_multi)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                StyleDownsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)
        self.mid_block3 = block_klass(mid_dim, mid_dim)
        self.mid_block4 = block_klass(mid_dim, mid_dim)

    def forward(self, x):
        x = self.init_conv(x)

        for block1, block2, downsample in self.downs:
            x = block1(x)
            x = block2(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_block2(x)
        x = self.mid_block3(x)
        x = self.mid_block4(x)

        return x


########################################################################################################################
#                                                                                                                      #
########################################################################################################################

class DownSampleEncoder(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size

    def forward(self, x):
        x = F.interpolate(x, size=[self.img_size, self.img_size], mode='bilinear', align_corners=True, antialias=True)
        return x.detach()


class UpSampleDecoder(nn.Module):
    def __init__(
            self,
            dim,
            dim_multi,
            img_channels,
            resnet_block_groups,
            img_size
    ):
        super().__init__()

        # determine dimensions
        self.channels = img_channels
        self.img_size = img_size

        # x_t, x_in, x_self_cond
        input_channels = img_channels

        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_multi)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConditionAdapter(in_channels=256, in_size=16, out_channels=dim_in, out_size=(128 // (2 ** ind))),
                block_klass(dim_in, dim_in),
                CrossLinearAttention(q_dim=dim_in, kv_dim=dim_in, heads=4, dim_head=(dim_in // 4)),
                block_klass(dim_in, dim_in),
                CrossLinearAttention(q_dim=dim_in, kv_dim=dim_in, heads=4, dim_head=(dim_in // 4)),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))

        mid_dim = dims[-1]

        self.mid_cond_ada = ConditionAdapter(in_channels=256, in_size=16, out_channels=mid_dim,
                                             out_size=(128 // (2 ** (num_resolutions - 1))))
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_style_attn1 = CrossAttention(q_dim=mid_dim, kv_dim=mid_dim, heads=4, dim_head=(mid_dim // 4))
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(dim=mid_dim, heads=4, dim_head=(mid_dim // 4))))
        self.mid_block2 = block_klass(mid_dim, mid_dim)
        self.mid_style_attn2 = CrossAttention(q_dim=mid_dim, kv_dim=mid_dim, heads=4, dim_head=(mid_dim // 4))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ConditionAdapter(in_channels=256, in_size=16, out_channels=dim_out,
                                 out_size=(128 // (2 ** (len(in_out) - 1 - ind)))),
                block_klass(dim_out + dim_in, dim_out),
                CrossLinearAttention(q_dim=dim_out, kv_dim=dim_out, heads=4, dim_head=(dim_out // 4)),
                block_klass(dim_out + dim_in, dim_out),
                CrossLinearAttention(q_dim=dim_out, kv_dim=dim_out, heads=4, dim_head=(dim_out // 4)),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))

        output_channels = img_channels

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_cond_ada = ConditionAdapter(in_channels=256, in_size=16, out_channels=dim, out_size=128)
        self.final_attn = CrossLinearAttention(q_dim=dim, kv_dim=dim, heads=4, dim_head=(dim // 4))
        self.final_conv = nn.Conv2d(dim, output_channels, 1)

    def forward(self, x, style_cond):
        x = F.interpolate(x, size=[self.img_size, self.img_size], mode='bilinear', align_corners=True, antialias=True)
        rx = x.clone()

        x = self.init_conv(x)
        r = x.clone()

        h = []

        for cond_ada, block1, attn1, block2, attn2, downsample in self.downs:
            layer_cond = cond_ada(style_cond)

            x = block1(x)
            x = attn1(x, layer_cond)
            h.append(x)

            x = block2(x)
            x = attn2(x, layer_cond)
            h.append(x)

            x = downsample(x)

        mid_layer_cond = self.mid_cond_ada(style_cond)
        x = self.mid_block1(x)
        x = self.mid_style_attn1(x, mid_layer_cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        x = self.mid_style_attn2(x, mid_layer_cond)

        for cond_ada, block1, attn1, block2, attn2, upsample in self.ups:
            layer_cond = cond_ada(style_cond)

            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)
            x = attn1(x, layer_cond)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x)
            x = attn2(x, layer_cond)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x)
        final_cond = self.final_cond_ada(style_cond)
        x = self.final_attn(x, final_cond)
        x = self.final_conv(x)

        return x + rx
