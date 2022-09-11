import torch
# from base import BaseVAE
from abc import abstractmethod
import torch.utils.checkpoint as checkpoint
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
# from diff_aug import DiffAugment
# from torch import tensor as Tensor
import math

Tensor = TypeVar('torch.tensor')


class configError(Exception):
    """
        封装错误信息
    """

    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print(H // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # print(x.shape)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class Downsample(nn.Sequential):
    """Downsample module.

    Args:

    """

    def __init__(self, in_channels, h_dim):
        m = []
        m.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim,
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU())
        )
        super(Downsample, self).__init__(*m)


class UPsample(nn.Sequential):
    """Downsample module.

    Args:

    """

    def __init__(self, in_channels, h_dim, upscale = 2):
        m = []
        m.append(
            nn.Sequential(
                nn.Conv2d(in_channels, h_dim * upscale * upscale, kernel_size=3, stride=1, padding=1),
                nn.ELU(True),
                nn.Conv2d(h_dim * upscale * upscale, h_dim * upscale * upscale, kernel_size=3, stride=1, padding=1),
                nn.ELU(True),
                nn.PixelShuffle(upscale_factor=upscale)
                # nn.UpsamplingNearest2d(scale_factor=2)
            ))
        super(UPsample, self).__init__(*m)

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size, img_size=256, patch_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=Downsample, upsample=UPsample, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim,
                                norm_layer=nn.LayerNorm)
        self.input_resolution = (self.embed.patches_resolution[0], self.embed.patches_resolution[1])
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.patchdrop = nn.Dropout(p=drop)
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=self.input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        self.unembed = PatchUnEmbed(embed_dim=dim)
        self.downsample = downsample(in_channels=dim, h_dim=2 * dim)
        self.upsample = upsample(in_channels=dim, h_dim=dim // 2)

    def forward(self, x):
        # print(self.input_resolution)
        x_size = (x.shape[2], x.shape[3])
        x = self.embed(x)
        x = self.patchdrop(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        x_mid = self.unembed(x, x_size)
        x_up = self.upsample(x_mid)
        x_down = self.downsample(x_mid)
        return {'up':x_up, 'mid':x_mid, 'down':x_down}

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class MSST(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 num_out_ch: int,
                 hidden_dims: List = None,
                 anneal_steps: int = 200,
                 window_size: int = 8,
                 img_size=256,
                 alpha: float = 1.,
                 beta: float = 6.,
                 gamma: float = 1.,
                 resi_connection = '3conv',
                 **kwargs) -> None:
        super(MSST, self).__init__()

        self.anneal_steps = anneal_steps
        self.img_size = img_size
        self.window_size = window_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.modules_top = []
        self.modules_mid = []
        self.modules_down = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        self.convfirst = nn.Conv2d(in_channels, hidden_dims[0], 3, 1, 1)

        # Build Encoder
        self.top_e0 = BasicLayer(img_size=self.img_size, dim=hidden_dims[0], depth=2, num_heads=8, window_size=self.window_size)
        self.top_e1 = BasicLayer(img_size=self.img_size, dim=hidden_dims[0], depth=2, num_heads=8, window_size=self.window_size)
        self.top_e2 = BasicLayer(img_size=self.img_size, dim=hidden_dims[0], depth=2, num_heads=8, window_size=self.window_size)

        self.mid_e0 = BasicLayer(img_size=self.img_size, dim=hidden_dims[1], depth=2, num_heads=8,
                            window_size=self.window_size)
        self.mid_e1 = BasicLayer(img_size=self.img_size, dim=hidden_dims[1], depth=2, num_heads=8,
                            window_size=self.window_size)

        self.down_e0 = BasicLayer(img_size=self.img_size, dim=hidden_dims[2], depth=2, num_heads=8,
                            window_size=self.window_size)

        # self.encoder = nn.Sequential(*modules)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(hidden_dims[0], hidden_dims[0], 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(hidden_dims[0], hidden_dims[0] // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(hidden_dims[0] // 4, hidden_dims[0] // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(hidden_dims[0] // 4, hidden_dims[0], 3, 1, 1))
        # Build Decoder
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(hidden_dims[0], hidden_dims[0]//2, 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True))
        self.upsample = UPsample(hidden_dims[0]//2, hidden_dims[0]//2, upscale=1)
        self.conv_last = nn.Conv2d(hidden_dims[0]//2, num_out_ch, 3, 1, 1)

    def forward(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        # x_size = (x.shape[2], x.shape[3])
        x = self.convfirst(x)
        out_top0 = self.top_e0(x)
        out_mid0 = self.mid_e0(out_top0['down'])
        out_down0 = self.down_e0(out_mid0['down'])

        out_top1 = self.top_e1(out_top0['mid'] + out_mid0['up'])
        out_mid1 = self.mid_e1(out_mid0['mid'] + out_down0['up'] + out_top1['down'])

        out_top2 = self.top_e2(out_top1['mid'] + out_mid1['up'])

        x = self.conv_after_body(out_top2['mid'])
        # print(x.shape)
        x = self.conv_before_upsample(x)
        # print(x.shape)
        x = self.upsample(x)
        # print(x.shape)
        x = self.conv_last(x)
        return F.sigmoid(x)

    # def loss_function(self,
    #                   args,
    #                   **kwargs) -> dict:
    #     """
    #     Computes the VAE loss function.
    #     KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    #     :param args:
    #     :param kwargs:
    #     :return:
    #     """
    #
    #     recons = args[0]
    #     input = args[1]
    #     mu = args[2]
    #     log_var = args[3]
    #     z = args[4]
    #
    #     weight = 0.005  # kwargs['M_N']  # Account for the minibatch samples from the dataset
    #
    #     recons_loss = F.mse_loss(recons, input)
    #     # recons_loss = F.l1_loss(recons, input)
    #
    #     log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim=1)
    #
    #     zeros = torch.zeros_like(z)
    #     log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)
    #
    #     batch_size, latent_dim = z.shape
    #     mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, latent_dim),
    #                                             mu.view(1, batch_size, latent_dim),
    #                                             log_var.view(1, batch_size, latent_dim))
    #
    #     # Reference
    #     # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
    #     dataset_size = (1 / weight) * batch_size  # dataset size
    #     strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
    #     importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1)).to(input.device)
    #     importance_weights.view(-1)[::batch_size] = 1 / dataset_size
    #     importance_weights.view(-1)[1::batch_size] = strat_weight
    #     importance_weights[batch_size - 2, 0] = strat_weight
    #     log_importance_weights = importance_weights.log()
    #
    #     mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)
    #
    #     log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
    #     log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)
    #
    #     mi_loss = (log_q_zx - log_q_z).mean()
    #     tc_loss = (log_q_z - log_prod_q_z).mean()
    #     kld_loss = (log_prod_q_z - log_p_z).mean()
    #
    #     # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    #
    #     if self.training:
    #         self.num_iter += 1
    #         anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
    #     else:
    #         anneal_rate = 1.
    #
    #     loss = recons_loss / batch_size + \
    #            self.alpha * mi_loss + \
    #            weight * (self.beta * tc_loss +
    #                      anneal_rate * self.gamma * kld_loss)
    #     # loss = recons_loss + weight * anneal_rate * self.gamma * abs(kld_loss)
    #
    #     return {'loss': loss,
    #             'Reconstruction_Loss': recons_loss,
    #             'KLD': kld_loss,
    #             'TC_Loss': tc_loss,
    #             'MI_Loss': mi_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples



if __name__ == '__main__':
    upscale = 4
    window_size = 8
    # height = (512 // upscale // window_size + 1) * window_size
    # width = (512 // upscale // window_size + 1) * window_size
    height = 256
    width = 256
    # print(height)
    model = MSST(img_size=256, window_size=window_size, in_channels=3, num_out_ch=1, hidden_dims = [32, 64, 128])
    print(model)
    # print(height, width, model.flops() / 1e9)
    input = torch.randn((1, 3, height, width))
    x = model(input)
    print(x.shape)