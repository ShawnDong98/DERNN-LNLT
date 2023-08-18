import math
import warnings

import numbers

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

from einops import rearrange
from torch import einsum

from box import Box
from fvcore.nn import FlopCountAnalysis

from csi.data import shift_batch, shift_back_batch, gen_meas_torch_batch

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



class LocalMSA(nn.Module):
    """
    The Local MSA partitions the input into non-overlapping windows of size M × M, treating each pixel within the window as a token, and computes self-attention within the window.
    """
    def __init__(self, 
                 dim, 
                 num_heads, 
                 window_size, 
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5


        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=False)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.pos_emb = nn.Parameter(torch.Tensor(1, num_heads, window_size[0]*window_size[1], window_size[0]*window_size[1]))
        trunc_normal_(self.pos_emb)


    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b c (h b0) (w b1) -> (b h w) (b0 b1) c',
                                              b0=self.window_size[0], b1=self.window_size[1]), (q, k, v))
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = rearrange(out, '(b h w) (b0 b1) c -> b c (h b0) (w b1)', h=h // self.window_size[0], w=w // self.window_size[1],
                            b0=self.window_size[0])
        out = self.project_out(out)
        
        return out
    


class NonLocalMSA(nn.Module):
    """
    The Non-Local MSA divides the input into N × N non-overlapping windows, treating each window as a token, and computes self-attention across the windows.
    """
    def __init__(self, 
                 dim, 
                 num_heads, 
                 window_num 
    ):
        super().__init__()
        self.dim = dim
        self.window_num = window_num
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5


        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)


        self.pos_emb = nn.Parameter(torch.Tensor(1, num_heads, window_num[0]*window_num[1], window_num[0]*window_num[1]))
        trunc_normal_(self.pos_emb)


    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b c (h b0) (w b1)-> b (h w) (b0 b1 c)',
                                              h=self.window_num[0], w=self.window_num[1]), (q, k, v))
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        head_dim = ((h // self.window_num[0]) * (w // self.window_num[1]) * c) / self.num_heads 
        scale = head_dim ** -0.5

        q *= scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
       
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, 'b (h w) (b0 b1 c) -> (b h w) (b0 b1) c', h=self.window_num[0], b0=h // self.window_num[0], b1=w // self.window_num[1])

        out = rearrange(out, '(b h w) (b0 b1) c -> b c (h b0) (w b1)', h=self.window_num[0], w= self.window_num[1],
                            b0=h//self.window_num[0])
        out = self.project_out(out)
        
        
        return out
    

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b, h, w, c]
        return out: [b, h, w, c]
        """
        out = self.net(x)
        return out
    

## Gated-Dconv Feed-Forward Network (GDFN)
class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, 
                 dim, 
                 ffn_expansion_factor = 2.66
    ):
        super(Gated_Dconv_FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=True)

        self.act_fn = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c, h, w]
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x
    

def FFN_FN(
    cfg,
    ffn_name,
    dim
):
    if ffn_name == "Gated_Dconv_FeedForward":
        return Gated_Dconv_FeedForward(
                dim, 
                ffn_expansion_factor=cfg.MODEL.DENOISER.DERNN_LNLT.FFN_EXPAND, 
            )
    elif ffn_name == "FeedForward":
        return FeedForward(dim = dim)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn, layernorm_type='WithBias'):
        super().__init__()
        self.fn = fn
        self.layernorm_type = layernorm_type
        if layernorm_type == 'BiasFree' or layernorm_type == 'WithBias':
            self.norm = LayerNorm(dim, layernorm_type)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        if self.layernorm_type == 'BiasFree' or self.layernorm_type == 'WithBias':
            x = self.norm(x)
        else:
            h, w = x.shape[-2:]
            x = to_4d(self.norm(to_3d(x)), h, w)
        return self.fn(x, *args, **kwargs)
    


class LocalNonLocalBlock(nn.Module):
    """
    The Local and Non-Local Transformer Block (LNLB) is the most important component. Each LNLB consists of three layer-normalizations (LNs), a Local MSA, a Non-Local MSA, and a GDFN (Zamir et al. 2022).
    """
    def __init__(self, 
                 cfg,
                 dim, 
                 num_heads,
                 window_size:tuple,
                 window_num:tuple,
                 layernorm_type,
                 num_blocks,
                 ):
        super().__init__()
        self.cfg = cfg
        self.window_size = window_size
        self.window_num = window_num

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, LocalMSA(
                        dim = dim, 
                        window_size = window_size,
                        num_heads = num_heads,
                    ),
                    layernorm_type = layernorm_type) if self.cfg.MODEL.DENOISER.DERNN_LNLT.LOCAL else nn.Identity(),
                PreNorm(dim, NonLocalMSA(
                        dim = dim, 
                        num_heads = num_heads,
                        window_num = window_num,
                    ),
                    layernorm_type = layernorm_type) if self.cfg.MODEL.DENOISER.DERNN_LNLT.NON_LOCAL else nn.Identity(),
                PreNorm(dim, FFN_FN(
                    cfg,
                    ffn_name = cfg.MODEL.DENOISER.DERNN_LNLT.FFN_NAME,
                    dim = dim
                ),
                layernorm_type = layernorm_type)
            ]))


    def forward(self, x):
        for (local_msa, nonlocal_msa, ffn) in self.blocks:
            x = x + local_msa(x) 
            x = x + nonlocal_msa(x) 
            x = x + ffn(x)

        return x
    

class DownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, stride=2, kernel_size=2, padding=0, output_padding=0)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    

class LNLT(nn.Module):
    """
    The Local and Non-Local Transformer (LNLT) adopts a three-level U-shaped structure, and each level consists of multiple basic units called Local and Non-Local Transformer Blocks (LNLBs). Up- and down-sampling modules are positioned between LNLBs.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Conv2d(cfg.MODEL.DENOISER.DERNN_LNLT.IN_DIM, cfg.MODEL.DENOISER.DERNN_LNLT.DIM, kernel_size=3, stride=1, padding=1, bias=False)


        self.Encoder = nn.ModuleList([
            LocalNonLocalBlock(
                cfg = cfg, 
                dim = cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 0, 
                num_heads = 2 ** 0, 
                window_size = cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_SIZE,
                window_num = cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_NUM,
                layernorm_type = cfg.MODEL.DENOISER.DERNN_LNLT.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.DENOISER.DERNN_LNLT.NUM_BLOCKS[0],
            ),
            LocalNonLocalBlock(
                cfg = cfg, 
                dim = cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 1, 
                num_heads = 2 ** 1, 
                window_size = cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_SIZE,
                window_num = cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_NUM,
                layernorm_type = cfg.MODEL.DENOISER.DERNN_LNLT.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.DENOISER.DERNN_LNLT.NUM_BLOCKS[1],
            ),
        ])

        self.BottleNeck = LocalNonLocalBlock(
                cfg = cfg, 
                dim = cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 2, 
                num_heads = 2 ** 2, 
                window_size = cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_SIZE,
                window_num = cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_NUM,
                layernorm_type = cfg.MODEL.DENOISER.DERNN_LNLT.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.DENOISER.DERNN_LNLT.NUM_BLOCKS[2],
            )

        self.Decoder = nn.ModuleList([
            LocalNonLocalBlock(
                cfg = cfg, 
                dim = cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 1, 
                num_heads = 2 ** 1, 
                window_size = cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_SIZE,
                window_num = cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_NUM,
                layernorm_type = cfg.MODEL.DENOISER.DERNN_LNLT.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.DENOISER.DERNN_LNLT.NUM_BLOCKS[3],
            ),
            LocalNonLocalBlock(
                cfg = cfg, 
                dim = cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 0, 
                num_heads = 2 ** 0, 
                window_size = cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_SIZE,
                window_num = cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_NUM,
                layernorm_type = cfg.MODEL.DENOISER.DERNN_LNLT.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.DENOISER.DERNN_LNLT.NUM_BLOCKS[4],
            )
        ])

        self.Downs = nn.ModuleList([
            DownSample(cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 0),
            DownSample(cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 1)
        ])

        self.Ups = nn.ModuleList([
            UpSample(cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 2),
            UpSample(cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 1)
        ])

        self.fusions = nn.ModuleList([
            nn.Conv2d(
                in_channels = cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 2,
                out_channels = cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 1,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False
            ),
            nn.Conv2d(
                in_channels = cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 1,
                out_channels = cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 0,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False
            )
        ])

        self.mapping = nn.Conv2d(cfg.MODEL.DENOISER.DERNN_LNLT.DIM, cfg.MODEL.DENOISER.DERNN_LNLT.OUT_DIM, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')


        x1 = self.embedding(x)
        res1 = self.Encoder[0](x1)

        x2 = self.Downs[0](res1)
        res2 = self.Encoder[1](x2)

        x4 = self.Downs[1](res2)
        res4 = self.BottleNeck(x4)

        dec_res2 = self.Ups[0](res4) # dim * 2 ** 2 -> dim * 2 ** 1
        dec_res2 = torch.cat([dec_res2, res2], dim=1) # dim * 2 ** 2
        dec_res2 = self.fusions[0](dec_res2) # dim * 2 ** 2 -> dim * 2 ** 1
        dec_res2 = self.Decoder[0](dec_res2)

        dec_res1 = self.Ups[1](dec_res2) # dim * 2 ** 1 -> dim * 2 ** 0
        dec_res1 = torch.cat([dec_res1, res1], dim=1) # dim * 2 ** 1 
        dec_res1 = self.fusions[1](dec_res1) # dim * 2 ** 1 -> dim * 2 ** 0        
        dec_res1 = self.Decoder[1](dec_res1)

        if self.cfg.MODEL.DENOISER.DERNN_LNLT.WITH_NOISE_LEVEL:
            out = self.mapping(dec_res1) + x[:, 1:, :, :]
        else:
            out = self.mapping(dec_res1) + x
            

        return out[:, :, :h_inp, :w_inp]
    

def PWDWPWConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 1, 1, 0, bias=True),
        nn.GELU(),
        nn.Conv2d(64, 64, 3, 1, 1, bias=True, groups=64),
        nn.GELU(),
        nn.Conv2d(64, out_channels, 1, 1, 0, bias=False)
    )

def A(x, Phi):
    B, nC, H, W = x.shape
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y

def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


def shift_3d(inputs, step=2):
    [B, C, H, W] = inputs.shape
    temp = torch.zeros((B, C, H, W+(C-1)*step)).to(inputs.device)
    temp[:, :, :, :W] = inputs
    for i in range(C):
        temp[:,i,:,:] = torch.roll(temp[:,i,:,:], shifts=step*i, dims=2)
    return temp

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs


class DegradationEstimation(nn.Module):
    """
    The Degradation Estimation Network (DEN) is proposed to estimate degradation-related parameters from the input of the current recurrent step and with reference to the sensing matrix.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.DL = nn.Sequential(
            PWDWPWConv(self.cfg.DATASETS.WAVE_LENS*2, self.cfg.DATASETS.WAVE_LENS*2),
            PWDWPWConv(self.cfg.DATASETS.WAVE_LENS*2, self.cfg.DATASETS.WAVE_LENS),
        )
        self.down_sample = nn.Conv2d(self.cfg.DATASETS.WAVE_LENS, self.cfg.DATASETS.WAVE_LENS*2, 3, 2, 1, bias=True) # (B, 64, H, W) -> (B, 64, H//2, W//2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(self.cfg.DATASETS.WAVE_LENS*2, self.cfg.DATASETS.WAVE_LENS*2, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cfg.DATASETS.WAVE_LENS*2, self.cfg.DATASETS.WAVE_LENS*2, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cfg.DATASETS.WAVE_LENS*2, 2, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)


    def forward(self, y, phi):

        inp = torch.cat([phi, y], dim=1)
        phi_r = self.DL(inp)

        phi = phi + phi_r

        x = self.down_sample(self.relu(phi_r))
        x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6
        mu = x[:, 0, :, :]
        noise_level = x[:, 1, :, :]

        return phi, mu, noise_level[:, None, :, :]
    

class DERNN_LNLT(nn.Module):
    """
    The DERNN-LNLT unfolds the HQS algorithm within the MAP framework and transfors the DUN into an RNN by sharing parameters across stages.  
    
    Then, the DERNN-LNLT integrate the Degradation Estimation Network into the RNN, which estimates the degradation matrix for the data subproblem and the noise level for the prior subproblem by residual learning with reference to the sensing matrix.

    Subsequently, the Local and Non-Local Transformer (LNLT) utilizes the Local and Non-Local Multi-head Self-Attention (MSA) to effectively exploit both local and non-local HSIs priors. 
    
    Finally, incorporating the LNLT into the DERNN as the denoiser for the prior subproblem leads to the proposed DERNN-LNLT.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.fusion = nn.Conv2d(cfg.DATASETS.WAVE_LENS*2, cfg.DATASETS.WAVE_LENS, 1, padding=0, bias=True)

        self.DP = nn.ModuleList([
           DegradationEstimation(cfg) for _ in range(cfg.MODEL.DENOISER.DERNN_LNLT.STAGE)
        ]) if not cfg.MODEL.DENOISER.DERNN_LNLT.SHARE_PARAMS else DegradationEstimation(cfg)
        self.PP = nn.ModuleList([
            LNLT(cfg) for _ in range(cfg.MODEL.DENOISER.DERNN_LNLT.STAGE)
        ]) if not cfg.MODEL.DENOISER.DERNN_LNLT.SHARE_PARAMS else LNLT(cfg)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initial(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: temp: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        nC = self.cfg.DATASETS.WAVE_LENS
        step = self.cfg.DATASETS.STEP
        bs, nC, row, col = Phi.shape
        y_shift = torch.zeros(bs, nC, row, col).to(y.device).float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fusion(torch.cat([y_shift, Phi], dim=1))
        return z

    def prepare_input(self, data):
        hsi = data['hsi']
        mask = data['mask']

        YH = gen_meas_torch_batch(hsi, mask, step=self.cfg.DATASETS.STEP, wave_len=self.cfg.DATASETS.WAVE_LENS, mask_type=self.cfg.DATASETS.MASK_TYPE, with_noise=self.cfg.DATASETS.TRAIN.WITH_NOISE)

        data['Y'] = YH['Y']
        data['H'] = YH['H']

        return data
    

    def forward_train(self, data):
        y = data['Y']
        phi = data['mask']
        x0 = data['H']

        z = self.initial(y, phi)


        B, C, H, W = phi.shape
        B, C, H_, W_ = x0.shape        

        for i in range(self.cfg.MODEL.DENOISER.DERNN_LNLT.STAGE):
            Phi, mu, noise_level = self.DP[i](z, phi) if not self.cfg.MODEL.DENOISER.DERNN_LNLT.SHARE_PARAMS else self.DP(z, phi)

            if not self.cfg.MODEL.DENOISER.DERNN_LNLT.WITH_DL:
                Phi = phi
            if not self.cfg.MODEL.DENOISER.DERNN_LNLT.WITH_MU:
                mu = torch.FloatTensor([1e-6]).to(y.device)

            Phi_s = torch.sum(Phi**2,1)
            Phi_s[Phi_s==0] = 1
            Phi_z = A(z, Phi)
            x = z + At(torch.div(y-Phi_z,mu+Phi_s), Phi)
            x = shift_back_3d(x)[:, :, :, :W_]
            noise_level_repeat = noise_level.repeat(1,1,x.shape[2], x.shape[3])
            if not self.cfg.MODEL.DENOISER.DERNN_LNLT.WITH_NOISE_LEVEL:
                z = self.PP[i](x) if not self.cfg.MODEL.DENOISER.DERNN_LNLT.SHARE_PARAMS else self.PP(x)
            else:
                z = self.PP[i](torch.cat([noise_level_repeat, x],dim=1)) if not self.cfg.MODEL.DENOISER.DERNN_LNLT.SHARE_PARAMS else self.PP(torch.cat([noise_level_repeat, x],dim=1))
            z = shift_3d(z)


        z = shift_back_3d(z)[:, :, :, :W_]

        out = z[:, :, :, :W_]

        return out
    
    def forward_test(self, data):
        y = data['Y']
        phi = data['mask']
        x0 = data['H']

        z = self.initial(y, phi)


        B, C, H, W = phi.shape
        B, C, H_, W_ = x0.shape        

        for i in range(self.cfg.MODEL.DENOISER.DERNN_LNLT.STAGE):
            Phi, mu, noise_level = self.DP[i](z, phi) if not self.cfg.MODEL.DENOISER.DERNN_LNLT.SHARE_PARAMS else self.DP(z, phi)

            if not self.cfg.MODEL.DENOISER.DERNN_LNLT.WITH_DL:
                Phi = phi
            if not self.cfg.MODEL.DENOISER.DERNN_LNLT.WITH_MU:
                mu = torch.FloatTensor([1e-6]).to(y.device)

            Phi_s = torch.sum(Phi**2,1)
            Phi_s[Phi_s==0] = 1
            Phi_z = A(z, Phi)
            x = z + At(torch.div(y-Phi_z,mu+Phi_s), Phi)
            x = shift_back_3d(x)[:, :, :, :W_]
            noise_level_repeat = noise_level.repeat(1,1,x.shape[2], x.shape[3])
            if not self.cfg.MODEL.DENOISER.DERNN_LNLT.WITH_NOISE_LEVEL:
                z = self.PP[i](x) if not self.cfg.MODEL.DENOISER.DERNN_LNLT.SHARE_PARAMS else self.PP(x)
            else:
                z = self.PP[i](torch.cat([noise_level_repeat, x],dim=1)) if not self.cfg.MODEL.DENOISER.DERNN_LNLT.SHARE_PARAMS else self.PP(torch.cat([noise_level_repeat, x],dim=1))
            z = shift_3d(z)


        z = shift_back_3d(z)[:, :, :, :W_]

        out = z[:, :, :, :W_]

        return out
    
    def forward(self, data):
        if self.training:
            data = self.prepare_input(data)
            x = self.forward_train(data)

        else:
            x = self.forward_test(data)

        return x
    

