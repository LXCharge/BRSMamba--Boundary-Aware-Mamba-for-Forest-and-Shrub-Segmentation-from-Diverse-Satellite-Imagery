import math
import os
from matplotlib import pyplot as plt
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from torchvision import transforms

try:
    from backbone.MambaVision import MambaVision
except ImportError:
    from .backbone.MambaVision import MambaVision
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import (
    causal_conv1d_fn,
    mamba_chunk_scan_combined,
)

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class GaussianFilter(nn.Module):
    def __init__(self, in_channels, kernel_size=3, sigma=None, **kwargs):
        super(GaussianFilter, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        if sigma is None:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        self.sigma = sigma
        self.in_channels = in_channels
        self.kernel = (
            self.create_gaussian_kernel(kernel_size, sigma)
            .view(1, 1, self.kernel_size, self.kernel_size)
            .repeat(self.in_channels, 1, 1, 1)
        )

    def _init_weights(self, m):
        kernel = self.create_gaussian_kernel(self.kernel_size, self.sigma)
        self.gauss_conv.weight.data = kernel.view(
            1, 1, self.kernel_size, self.kernel_size
        ).repeat(self.in_channels, 1, 1, 1)
        self.gauss_conv.weight.requires_grad = False

    def create_gaussian_kernel(self, kernel_size, sigma):
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[x, y] = (1 / (2 * 3.14159 * sigma**2)) * math.exp(
                    -((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2)
                )
        # print(kernel / torch.sum(kernel))
        return kernel / torch.sum(kernel)

    def forward(self, x):
        # kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size).to(x.device)
        return F.conv2d(
            x,
            self.kernel.to(x.device),
            padding=(self.kernel_size - 1) // 2,
            groups=self.in_channels,
        )
        # return self.gauss_conv(x)


class Laplacian(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = self.create_Laplacian_kernel().repeat(self.out_channels, 1, 1, 1)
        # self.kernel.weight.requires_grad = False

    def create_Laplacian_kernel(self):
        return (
            torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            # torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            .float().reshape(1, 1, 3, 3)
        )

    def forward(self, x):
        return F.conv2d(x, self.kernel.to(x.device), padding=1, groups=self.in_channels)


class Sobel(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_x = self.create_sobel_kernel_x().repeat(self.out_channels, 1, 1, 1)
        self.kernel_y = self.create_sobel_kernel_y().repeat(self.out_channels, 1, 1, 1)

    def create_sobel_kernel_x(self):
        return (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            .float()
            .reshape(1, 1, 3, 3)
        )

    def create_sobel_kernel_y(self):
        return (
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            .float()
            .reshape(1, 1, 3, 3)
        )

    def forward(self, x):
        x_x = F.conv2d(
            x, self.kernel_x.to(x.device), padding=1, groups=self.in_channels
        )
        x_y = F.conv2d(
            x, self.kernel_y.to(x.device), padding=1, groups=self.in_channels
        )
        grad_mag = torch.sqrt(torch.pow(x_x, 2) + torch.pow(x_y, 2))
        grad_dir = torch.atan2(x_y, x_x)
        return grad_mag, grad_dir


class LAS(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        act_layer=nn.ELU,
        norm_layer=nn.BatchNorm2d,
        drop_rate=0.1,
        eps=1e-5,
        no_LoG=False,
        no_Conv=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert not (no_LoG and no_Conv)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act_layer()
        # self.get_gray_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, 3, bias=False),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     act_layer(),
        # )
        self.no_LoG = no_LoG
        self.no_Conv = no_Conv
        log_edge_channels, conv_edge_channels = 0, 0
        if not no_LoG:
            self.to_gray_conv = nn.Sequential(
                nn.Conv2d(in_channels, 1, 1, bias=False),
            )
            # self.gaussian = GaussianFilter(1, kernel_size=3)
            self.gaussianLaplacian = nn.Sequential(
                GaussianFilter(1, kernel_size=7),
                Laplacian(1, 1),
                norm_layer(1),
            )
            log_edge_channels = 1
        if not no_Conv:
            self.edge_conv = nn.Sequential(
                nn.Conv2d(1, 3, 3, 1, 1, bias=False),
                nn.Conv2d(3, 3, 5, 1, 2, bias=False),
                norm_layer(3),
            )
            conv_edge_channels = 3
        self.conv_down = nn.Sequential(
            # nn.Conv2d(in_channels, hidden_dim, 3, 2, 1, bias=False),
            # norm_layer(hidden_dim),
            # act_layer(negative_slope=0.01),
            nn.Conv2d(
                log_edge_channels + conv_edge_channels,
                out_channels,
                3,
                2,
                1,
                bias=False,
            ),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity(),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def gaussian_forward(self, x):
        return self.gaussian(x)

    def forward(self, x):
        B, C, H, W = x.shape
        # x = F.interpolate(x, size=(H // 2, W // 2), mode="bilinear")
        # gray_kernel = self.get_gray_conv(x)
        # gray_x = torch.sum(x * gray_kernel, dim=1, keepdim=True)
        if not self.no_LoG:
            gray_x = self.to_gray_conv(x)
            log_edge_x = self.gaussianLaplacian(gray_x)
            edge_x = log_edge_x
        # median_x = torch.median(edge_x)
        # edge_x = torch.sigmoid((edge_x - median_x) * 10)
        if not self.no_Conv:
            gray_x = transforms.Grayscale()(x)
            conv_edge_x = self.edge_conv(gray_x)
            edge_x = conv_edge_x
        if not self.no_LoG and not self.no_Conv:
            edge_x = torch.cat([conv_edge_x, log_edge_x], dim=1)
        edge_x = self.act(edge_x)
        x = F.interpolate(edge_x, size=(H // 2, W // 2), mode="bilinear")
        # x = self.res_conv_norm_act(x)
        x = self.conv_down(x)
        return x


# if __name__ == "__main__":
#     model = LAS(3, 1)
#     x = torch.rand(1, 3, 224, 224)
#     y = model(x)
#     print(y.shape)


class SSM(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        in_dt=False,
        dt_dim=-1,
        activation="silu",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.in_dt = in_dt
        self.dt_dim = dt_dim
        self.activation = activation

        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner, bias=bias, **factory_kwargs
        )
        self.x_proj = nn.Linear(
            self.d_inner // 2,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs
        )
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        if self.in_dt:
            assert self.dt_dim > 0
            self.dt_in_conv = nn.Sequential(
                nn.Conv2d(
                    self.dt_dim,
                    self.d_inner // 2,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    **factory_kwargs,
                ),
                nn.ELU(),
            )
            self.dt_conv1d = nn.Conv1d(
                in_channels=self.d_inner // 2,
                out_channels=self.d_inner // 2,
                bias=conv_bias // 2,
                kernel_size=d_conv,
                groups=self.d_inner // 2,
                **factory_kwargs,
            )

    def forward(self, hidden_states, in_dt=None):
        """
        hidden_states: (B, C, H, W)
        Returns: same shape as hidden_states
        """
        B, C, H, W = hidden_states.shape
        hidden_states = PatchEmbed.SerpentineEmbedding(hidden_states)
        hidden_states = rearrange(hidden_states, "b c h w -> b (h w) c")
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        if causal_conv1d_fn is not None and self.activation in ["silu", "swish"]:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d_x.weight, "d 1 w -> d w"),
                bias=self.conv1d_x.bias,
                activation=self.activation,
            )
            z = causal_conv1d_fn(
                x=z,
                weight=rearrange(self.conv1d_z.weight, "d 1 w -> d w"),
                bias=self.conv1d_z.bias,
                activation=self.activation,
            )
        else:
            x = F.silu(
                F.conv1d(
                    input=x,
                    weight=self.conv1d_x.weight,
                    bias=self.conv1d_x.bias,
                    padding="same",
                    groups=self.d_inner // 2,
                )
            )
            z = F.silu(
                F.conv1d(
                    input=z,
                    weight=self.conv1d_z.weight,
                    bias=self.conv1d_z.bias,
                    padding="same",
                    groups=self.d_inner // 2,
                )
            )
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        if self.in_dt:
            assert in_dt is not None
            in_dt = self.dt_in_conv(in_dt)
            in_dt = rearrange(in_dt, "b c h w -> b (h w) c")
            if causal_conv1d_fn is not None and self.activation in ["silu", "swish"]:
                in_dt = causal_conv1d_fn(
                    x=in_dt.transpose(1, 2),
                    weight=rearrange(self.dt_conv1d.weight, "d 1 w -> d w"),
                    bias=self.dt_conv1d.bias,
                    activation=self.activation,
                )
            else:
                in_dt = F.silu(
                    F.conv1d(
                        input=in_dt.transpose(1, 2),
                        weight=self.dt_conv1d.weight,
                        bias=self.dt_conv1d.bias,
                        padding="same",
                        groups=self.nheads,
                    )
                )
            dt = dt * in_dt
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        out = PatchEmbed.SerpentineEmbedding(out)
        return out


class PatchEmbed(nn.Module):
    def __init__():
        super().__init__()

    def forward(self, x):
        return x

    @staticmethod
    def SerpentineEmbedding(x: torch.Tensor):
        # flip_x = torch.flip(flip_x, [3])
        temp_x = x.clone()
        temp_x[:, :, 1::2, :] = torch.flip(temp_x[:, :, 1::2, :], [3])
        # x[:, :, 1::2, :] = torch.flip(x[:, :, 1::2, :], [3])
        return temp_x


class NCSSD(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=64,
        chunk_size: int = 32,
        d_conv=3,
        expand=2,
        headdim=64,
        D_has_hdim=False,
        norm_before_gate=False,
        d_ssm=None,
        ngroups=1,
        bias=False,
        conv_bias=True,
        rmsnorm=True,
        device=None,
        dtype=None,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        A_init_range=(1, 16),
        activation="silu",
        linear_attn_duality=True,
        d_mlp=False,
        bidirection=False,
        ssd_positve_dA=False,
        in_dt=False,
        dt_dim=-1,
        window_size=-1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.rmsnorm = rmsnorm
        self.dt_limit = dt_limit
        assert activation in [None, "silu", "swish"]
        self.activation = activation
        self.d_inner = self.expand * self.d_model  # dim
        if ngroups <= 0:
            ngroups = self.d_inner // self.headdim
        self.ngroups = ngroups
        self.linear_attn_duality = linear_attn_duality
        self.ssd_positve_dA = ssd_positve_dA
        self.bidirection = bidirection
        self.D_has_hdim = D_has_hdim
        self.norm_before_gate = norm_before_gate
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_ssm = d_ssm if d_ssm is not None else self.d_inner
        self.nheads = self.d_ssm // self.headdim
        self.window_size = window_size
        assert self.d_inner % self.ngroups == 0
        self.d_mlp = self.d_model // 2 if d_mlp else 0
        # Order: [z, x, B, C, dt]
        d_in_proj = (
            2 * self.d_inner
            + 2 * self.ngroups * self.d_state
            + self.nheads
            + 2 * self.d_mlp
        )
        # self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.in_proj = nn.Conv2d(
            self.d_model, d_in_proj, 1, bias=bias, **factory_kwargs
        )

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        # TODO: group conv1d change
        self.conv1d_xBC = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_ssm,
            out_channels=self.d_ssm,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_ssm,
        )

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        self.conv1d_dt = nn.Conv1d(
            in_channels=self.nheads,
            out_channels=self.nheads,
            bias=conv_bias,
            kernel_size=4,
            groups=self.nheads,
            **factory_kwargs,
        )

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # 跳跃连接D
        self.D = nn.Parameter(
            torch.ones(
                self.d_ssm if self.D_has_hdim else self.nheads,
                device=device,
                dtype=dtype,
            )
        )
        self.D._no_weight_decay = True
        self.in_dt = in_dt
        if in_dt:
            assert dt_dim > 0
            # self.in_dt_proj = nn.Sequential(
            #     nn.Conv2d(dt_dim, self.nheads, 1, bias=False, **factory_kwargs),
            #     nn.BatchNorm2d(self.nheads),
            # )
            # self.dt_norm = nn.BatchNorm2d(self.nheads)
            # self.weights = nn.Parameter(
            #     torch.ones(2, dtype=torch.float32), requires_grad=True
            # )
            self.in_dt_out_proj = nn.Sequential(Conv(dt_dim, self.nheads, 3))
            # self.eps = 1e-8
        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )
        # self.out_proj = nn.Linear(
        #     self.d_inner + self.d_mlp, self.d_model, bias=bias, **factory_kwargs
        # )
        self.out_proj = Conv(
            self.d_inner + self.d_mlp,
            self.d_model,
            3,
            1,
            1,
            bias=bias,
        )

    def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
        """
        non-casual attention duality of mamba v2
        x: (B, L, H, D), equivalent to V in attention
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state), equivalent to K in attention
        C: (B, L, d_state), equivalent to Q in attention
        D: (nheads), equivalent to the skip connection
        """

        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]
        V = x.permute(0, 2, 1, 3)  # (B, H, L, D)
        dt = dt.permute(0, 2, 1)  # (B, H, L)

        dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
        if self.ssd_positve_dA:
            dA = -dA

        V_scaled = V * dA

        K = B.view(batch, 1, seqlen, dstate)  # (B, 1, L, D)

        if self.ngroups == 1:
            ## get kv via transpose K and V
            KV = K.transpose(-2, -1) @ V_scaled  # (B, H, dstate, D)
            Q = C.view(batch, 1, seqlen, dstate)  # .repeat(1, head, 1, 1)
            x = Q @ KV  # (B, H, L, D)
            x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
            if getattr(self, "__DEBUG__", False):
                print("dt.shape", dt.shape)
                print("dt.unsqueeze(-1).shape", dt.unsqueeze(-1).shape)
                print("A.shape", A.shape)
                print("A.view(1, -1, 1, 1).shape", A.view(1, -1, 1, 1).shape)
                print("dA.shape", dA.shape)
                print("V_scaled.shape", V_scaled.shape)
                print(
                    f"A: {A.shape}, dA: {dA.shape}, V: {V.shape}, V_scaled: {V_scaled.shape}, K: {K.shape}"
                )
                print(
                    f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}, KV: {KV.shape}, x: {x.shape}"
                )
        else:
            assert head % self.ngroups == 0
            dstate = dstate // self.ngroups
            K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(
                0, 1, 3, 2, 4
            )  # (B, 1, g, L, dstate)
            V_scaled = V_scaled.view(
                batch, head // self.ngroups, self.ngroups, seqlen, dim
            )  # (B, H//g, g, L, D)
            Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(
                0, 1, 3, 2, 4
            )  # (B, 1, g, L, dstate)

            KV = K.transpose(-2, -1) @ V_scaled  # (B, H//g, g, dstate, D)
            x = Q @ KV  # (B, H//g, g, L, D)
            V_skip = (V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(
                batch, head // self.ngroups, self.ngroups, seqlen, dim
            )  # (B, H//g, g, L, D)
            x = x + V_skip  # (B, H//g, g, L, D)
            x = (
                x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim)
            )  # (B, L, H, D)
            x = x.contiguous()

        return x

    def forward(self, u, seq_idx=None, in_dt=None):
        """
        u: (batch, channel, height, width).
        Returns: same shape as u
        """

        batch, channel, H, W = u.shape
        dim = channel
        assert dim * self.expand == self.expand * self.d_model
        dim = self.d_model * self.expand
        # conv_state, ssm_state = None, None
        zxbcdt = self.in_proj(u)  # (B, d_in_proj, H, W)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) # or (d_inner, d_state)
        assert self.nheads % self.ngroups == 0
        d_mlp = (
            zxbcdt.shape[1]
            - 2 * self.d_ssm
            - 2 * self.ngroups * self.d_state
            - self.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                self.d_ssm,
                self.d_ssm + 2 * self.ngroups * self.d_state,
                self.nheads,
            ],
            dim=1,
        )
        if self.in_dt:
            assert in_dt is not None
            in_dt = self.in_dt_out_proj(in_dt)
            dt = dt * in_dt

        # B C H W -> (B H // ws W // ws) (ws ws) C
        # PatchEmbed.SerpentineEmbedding(dt)
        dt = causal_conv1d_fn(
            x=rearrange(dt, "B C H W -> B C (H W)"),
            weight=rearrange(self.conv1d_dt.weight, "d 1 w -> d w"),
            bias=self.dt_bias,
            activation=self.activation,
        )
        dt = rearrange(dt, "B C L-> B L C")
        # dt = F.softplus(rearrange(dt, "B C H W -> B (H W) C") + self.dt_bias)

        # xBC = PatchEmbed.SerpentineEmbedding(xBC)
        xBC = causal_conv1d_fn(
            x=rearrange(xBC, "B C H W -> B C (H W)"),
            weight=rearrange(self.conv1d_xBC.weight, "d 1 w -> d w"),
            bias=self.conv1d_xBC.bias,
            activation=self.activation,
        )
        x, B, C = torch.split(
            xBC,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=1,
        )
        # x = rearrange(x, "B (h c) H W -> B (H W) h c", c=self.headdim)
        # B = rearrange(B, "b C H W -> b (H W) C")
        # C = rearrange(C, "b C H W -> b (H W) C")
        x = rearrange(x, "B (h c) L-> B L h c", c=self.headdim)
        B = rearrange(B, "b C L -> b L C")
        C = rearrange(C, "b C L -> b L C")
        if self.linear_attn_duality:
            y = self.non_casual_linear_attn(
                # rearrange(x, "B (h c) H W -> B (H W) h c", c=self.headdim),
                x,
                dt,
                A,
                # rearrange(B, "b C H W -> b (H W) C"),
                # rearrange(C, "b C H W -> b (H W) C"),
                B,
                C,
                self.D,
                H,
                W,
            )
        elif self.bidirection:
            assert self.ngroups % 2 == 0
            x = x.chunk(2, dim=-2)
            B = (rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
            C = (rearrange(C, "b (g n) l -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
            dt, A, D = dt.chunk(2, dim=-1), A.chunk(2, dim=-1), self.D.chunk(2, dim=-1)
            y_forward = mamba_chunk_scan_combined(
                x[0],
                dt[0],
                A[0],
                B[0],
                C[0],
                chunk_size=self.chunk_size,
                D=D[0],
                z=None,
                seq_idx=seq_idx,
            )
            y_backward = mamba_chunk_scan_combined(
                x[1].flip(1),
                dt[1].flip(1),
                A[1].flip(0),
                B[1].flip(1),
                C[1].flip(1),
                chunk_size=self.chunk_size,
                D=D[1].flip(0),
                z=None,
                seq_idx=seq_idx,
            )
            y = torch.cat([y_forward, y_backward.flip(1)], dim=-2)
        else:
            # dt_temp = dt.permute(0, 2, 1)
            # dA_temp = dt_temp.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(
            #     batch, 1, H * W, 1
            # )

            y = mamba_chunk_scan_combined(
                x,
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=(
                    rearrange(self.D, "(h p) -> h p", p=self.headdim)
                    if self.D_has_hdim
                    else self.D
                ),
                z=None,
                seq_idx=seq_idx,
                # **dt_limit_kwargs,
            )
            # print("dA_temp.shape", dA_temp.shape)
            # print("dA_temp:", dA_temp[0, 0, :, :])
        y = rearrange(y, "b l h p -> b l (h p)")
        # z = PatchEmbed.SerpentineEmbedding(z)
        z = causal_conv1d_fn(
            x=rearrange(z, "B C H W -> B C (H W)"),
            weight=rearrange(self.conv1d_z.weight, "d 1 w -> d w"),
            bias=self.conv1d_z.bias,
            activation=self.activation,
        )
        if self.rmsnorm:
            # y = self.norm(y, rearrange(z, "B C H W -> B (H W) C"))
            y = self.norm(y, rearrange(z, "B C L -> B L C"))
        if d_mlp > 0:
            y = torch.cat(
                [rearrange(F.silu(z0) * x0, "B C H W -> B (H W) C"), y], dim=2
            )
        y = rearrange(y, "B (H W) C -> B C H W", H=H, W=W)
        out = self.out_proj(y)
        # out = PatchEmbed.SerpentineEmbedding(out)
        return out


class ConvBNELU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super(ConvBNELU, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
            nn.ELU(),
        )


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super(ConvBN, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
        )


class Conv(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False
    ):
        super(Conv, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            )
        )


class SeparableConvBNELU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    ):
        super(SeparableConvBNELU, self).__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias=False,
            ),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ELU(),
        )


class SeparableConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    ):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias=False,
            ),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# class DenseAggregation(nn.Module):
#     def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
#         super(DenseAggregation, self).__init__()
#         self.down_proj = nn.Sequential(

#         )
#         self.up_proj = nn.ConvTranspose2d()

#     def forward(self, x):
#         assert x.dim() == 4
#         (res1, res2, res3) = x
#         return out


class MambaBlock(nn.Module):
    def __init__(
        self,
        dim=256,
        window_size=8,
        attn=True,
        num_heads=16,
        qkv_bias=False,
        attn_drop=0.0,
        in_dt=False,
        dt_dim=-1,
        ssm=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.ws = window_size
        self.attn = attn
        self.in_dt = in_dt
        if self.attn:
            self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
            if self.in_dt:
                self.dt_proj = ConvBN(dt_dim, dim, kernel_size=1)
                self.x_norm = nn.BatchNorm2d(dim)
        else:
            if ssm:
                self.mamba = SSM(dim, d_state=dim * 4, dt_dim=dt_dim, in_dt=in_dt)
            else:
                expand = 2
                self.mamba = NCSSD(
                    dim,
                    expand=expand,
                    d_state=dim * 4,
                    headdim=head_dim * expand,
                    # headdim=dim // 2,
                    dt_dim=dt_dim,
                    in_dt=in_dt,
                    d_conv=4,
                    window_size=window_size,
                )
        self.drop = nn.Dropout(attn_drop, inplace=True)
        # self.local1 = ConvBN(dim, dim, kernel_size=3)
        # self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)
        self.pooling_x = nn.AvgPool2d(
            kernel_size=(window_size, 1),
            stride=1,
            padding=(window_size // 2 - 1, 0),
        )
        self.pooling_y = nn.AvgPool2d(
            kernel_size=(1, window_size),
            stride=1,
            padding=(0, window_size // 2 - 1),
        )

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            pad_w = ps - W % ps
            right_pad = pad_w // 2
            left_pad = pad_w - right_pad
            x = F.pad(x, (left_pad, right_pad, 0, 0), mode="reflect")
        if H % ps != 0:
            pad_h = ps - H % ps
            top_pad = pad_h // 2
            bottom_pad = pad_h - top_pad
            x = F.pad(x, (0, 0, top_pad, bottom_pad), mode="reflect")
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode="reflect")
        return x

    def attion_forward(self, x):
        B, C, H, W = x.shape
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv,
            "b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d",
            h=self.num_heads,
            d=C // self.num_heads,
            hh=Hp // self.ws,
            ww=Wp // self.ws,
            qkv=3,
            ws1=self.ws,
            ws2=self.ws,
        )
        # dots = (q @ k.transpose(-2, -1)) * self.scale
        # attn = dots.softmax(dim=-1)
        # attn = attn @ v

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.drop.p,
        )
        attn = rearrange(
            attn,
            "(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)",
            h=self.num_heads,
            d=C // self.num_heads,
            hh=Hp // self.ws,
            ww=Wp // self.ws,
            ws1=self.ws,
            ws2=self.ws,
        )
        attn = attn[:, :, :H, :W]
        attn = self.pooling_x(
            F.pad(attn, pad=(0, 0, 0, 1), mode="reflect")
        ) + self.pooling_y(F.pad(attn, pad=(0, 1, 0, 0), mode="reflect"))
        return attn

    def mamba_forward(self, x, in_dt=None):
        B, C, H, W = x.shape
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        x = x.view(B, C, Hp // self.ws, self.ws, Wp // self.ws, self.ws)
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(-1, C, self.ws, self.ws)
        if self.in_dt:
            in_dt = self.pad(in_dt, self.ws)
            in_dt_C = in_dt.shape[1]
            in_dt = in_dt.view(
                B, in_dt_C, Hp // self.ws, self.ws, Wp // self.ws, self.ws
            )
            in_dt = in_dt.permute(0, 1, 2, 4, 3, 5).reshape(
                -1, in_dt_C, self.ws, self.ws
            )
        mamba_out = self.mamba(x, in_dt=in_dt)
        mamba_out = mamba_out.reshape(
            B, C, Hp // self.ws, Wp // self.ws, self.ws, self.ws
        )
        mamba_out = mamba_out.permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hp, Wp)
        mamba_out = mamba_out[:, :, :H, :W]
        mamba_out = self.pooling_x(
            F.pad(mamba_out, pad=(0, 0, 0, 1), mode="reflect")
        ) + self.pooling_y(F.pad(mamba_out, pad=(0, 1, 0, 0), mode="reflect"))
        return mamba_out

    def forward(self, x, in_dt=None):
        B, C, H, W = x.shape

        # local = self.local2(x) + self.local1(x)
        # mamba = rearrange(mamba, "b (h w) c -> b c h w", h=H, w=W)
        if self.attn:
            if self.in_dt:
                dt = self.dt_proj(in_dt)
                x = self.x_norm(x) - dt
            out = self.attion_forward(x)
            # out = attn + local
        else:
            # mamba_out = self.mamba(x, in_dt=in_dt if in_dt is not None else local)
            out = self.mamba_forward(x, in_dt=in_dt)
            # out = local + mamba_out
        # print(out.shape)
        # out = torch.cat((out, local, mamba), dim=1)
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(
        self,
        dim=256,
        num_heads=16,
        mlp_ratio=1.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        act_layer=nn.ReLU6,
        norm_layer=nn.BatchNorm2d,
        window_size=8,
        ssd_in_dt=False,
        dt_dim=-1,
        attn=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mamba = MambaBlock(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            window_size=window_size,
            in_dt=ssd_in_dt,
            dt_dim=dt_dim,
            attn=attn,
            attn_drop=attn_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        # self.res_conv1 = ConvBN(dim, dim, kernel_size=1)
        # self.res_conv2 = ConvBN(dim, dim, kernel_size=1)

    def forward(self, x, las=None):
        # Res improvement
        if las is not None:
            las = F.interpolate(
                las, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        x = x + self.drop_path(self.mamba(self.norm1(x), in_dt=las))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = self.res_conv1(x) + self.drop_path(self.attn(self.norm1(x)))
        # x = self.res_conv2(x) + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WF(nn.Module):
    def __init__(
        self, in_channels=128, decode_channels=128, res_channels=128, eps=1e-8
    ):
        super(WF, self).__init__()
        self.res_pre_conv = Conv(res_channels, decode_channels, kernel_size=1)
        self.channels_same = in_channels == decode_channels
        # if not self.channels_same:
        self.x_pre_deconv = nn.ConvTranspose2d(
            in_channels, decode_channels, kernel_size=4, stride=2, padding=1
        )
        self.weights = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.eps = eps
        self.post_conv = ConvBNELU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        # if not self.channels_same:
        x = self.x_pre_deconv(x)
        # else:
        #     x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.res_pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class LASMerge(nn.Module):  # unuse
    def __init__(
        self,
        x_channels=128,
        las_channels=128,
        decode_channels=64,
        p=0.1,
        mamba=False,
        num_heads=8,
        window_size=8,
    ):
        super().__init__()
        self.mamba = mamba
        # self.las_pre_conv3 = ConvBN(las_channels, decode_channels, kernel_size=3)
        # self.las_drop = nn.Dropout2d(p, inplace=True)
        self.x_pre_conv = Conv(x_channels, decode_channels, kernel_size=3)
        if mamba:
            self.mamba = MambaBlock(
                dim=decode_channels,
                num_heads=num_heads,
                window_size=window_size,
                in_dt=True,
                dt_dim=las_channels,
                attn=False,
            )
        else:
            self.las_pre_conv1 = Conv(las_channels, decode_channels, kernel_size=1)
            self.weights = nn.Parameter(
                torch.ones(2, dtype=torch.float32), requires_grad=True
            )
            self.eps = 1e-8
            self.post_conv = nn.Sequential(
                ConvBNELU(decode_channels, decode_channels, kernel_size=3),
            )

    def forward(self, x, las=None):
        x = self.x_pre_conv(x)
        if self.mamba is not False:
            x = x + self.mamba(x, in_dt=las)
        else:
            las = self.las_pre_conv1(las)
            weights = nn.ReLU()(self.weights)
            fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
            x = fuse_weights[0] * x + fuse_weights[1] * las
            x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(
        self,
        in_channels=64,
        decode_channels=64,
        res_channels=64,
        las_channels=16,
        in_dt=False,
    ):
        super().__init__()
        self.res_pre_conv = Conv(res_channels, decode_channels, kernel_size=1)
        self.in_dt = in_dt
        if in_dt:
            self.las_pre_conv = Conv(las_channels, decode_channels, kernel_size=1)
            self.weights = nn.Parameter(
                torch.ones(3, dtype=torch.float32), requires_grad=True
            )
        else:
            self.weights = nn.Parameter(
                torch.ones(2, dtype=torch.float32), requires_grad=True
            )
        # else:
        #     self.weights = nn.Parameter(
        #         torch.ones(2, dtype=torch.float32), requires_grad=True
        #     )
        self.channels_same = in_channels == decode_channels
        if not self.channels_same:
            self.x_pre_deconv = nn.ConvTranspose2d(
                in_channels, decode_channels, kernel_size=4, stride=2, padding=1
            )
        self.eps = 1e-8
        self.post_conv = ConvBNELU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(
                decode_channels,
                decode_channels,
                kernel_size=3,
                padding=1,
                groups=decode_channels,
            ),
            nn.Sigmoid(),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(decode_channels, decode_channels // 16, kernel_size=1),
            nn.ELU(),
            Conv(decode_channels // 16, decode_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ELU()

    def mamba_forward(self, x, las):
        B, C, H, W = x.shape
        las = F.interpolate(las, size=(H, W), mode="bilinear", align_corners=False)
        x = self.mamba(x, in_dt=las)
        return x

    def forward(self, x, res, las):
        if not self.channels_same:
            x = self.x_pre_deconv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        if self.in_dt:
            assert las is not None, "las must be provided when in_dt is True"
            x = (
                fuse_weights[0] * self.res_pre_conv(res)
                + fuse_weights[1] * x
                + fuse_weights[2] * self.las_pre_conv(las)
            )
        else:
            x = fuse_weights[0] * self.res_pre_conv(res) + fuse_weights[1] * x
        #     x = fuse_weights[0] * self.res_pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        # x = self.mamba_forward(x, las)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)
        return x


class AuxUpsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=2,
        kernel_size=1,
        stride=1,
        dropout=0.2,
    ):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=((stride - 1) + (kernel_size - 1)) // 2,
            ),
            nn.Dropout2d(p=dropout, inplace=True),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNELU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        return feat


class Decoder(nn.Module):

    def __init__(
        self,
        encoder_channels=[80, 160, 320, 640],
        decode_channels=[32, 64, 128, 128],
        num_heads=[32, 32, 32, 8],
        las_decode_channels=16,
        dropout=0.1,
        window_size=[8, 64, 32, 8],
        num_classes=6,
        ssd_in_dt=False,
        edge=True,
        erh_in_dt=False,
        aux=True,
        no_LoG=False,
        no_Conv=False,
        in_dt_blk=[True, True, True, True],
    ):
        super(Decoder, self).__init__()
        self.ssd_in_dt = ssd_in_dt
        self.edge = edge
        if self.ssd_in_dt:
            assert self.edge == True, "ssd_in_dt must be used with edge"
        self.aux = aux
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels[-1], kernel_size=1)
        self.blacks = nn.ModuleList(  # -3, -2, -1
            [
                Block(
                    dim=decode_channels[i],
                    num_heads=num_heads[i],
                    window_size=window_size[i],
                    ssd_in_dt=ssd_in_dt and in_dt_blk[i],
                    dt_dim=las_decode_channels,
                    # attn=(i == -1),
                    attn=False,
                )
                for i in range(-3, 0, 1)
            ]
        )
        self.upsample = nn.ModuleList(
            [
                FeatureRefinementHead(
                    decode_channels[-3],
                    decode_channels[-4],
                    encoder_channels[-4],
                    las_decode_channels,
                    in_dt=erh_in_dt and in_dt_blk[-4],
                ),
                WF(decode_channels[-2], decode_channels[-3], encoder_channels[-3]),
                WF(decode_channels[-1], decode_channels[-2], encoder_channels[-2]),
            ]
        )
        if self.aux and self.training:
            self.up4 = AuxUpsample(
                decode_channels[-1],
                decode_channels[-4],
                scale_factor=4,
                dropout=dropout,
            )
            self.up3 = AuxUpsample(
                decode_channels[-2],
                decode_channels[-4],
                scale_factor=2,
                dropout=dropout,
            )
            self.up2 = AuxUpsample(
                decode_channels[-3],
                decode_channels[-4],
                scale_factor=1,
                dropout=dropout,
            )
            self.aux_head = AuxHead(decode_channels[-4], num_classes)

        self.segmentation_head = nn.Sequential(
            ConvBNELU(decode_channels[-4], decode_channels[-4]),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels[-4], num_classes, kernel_size=1),
        )
        if self.edge:
            self.las = LAS(3, las_decode_channels, no_LoG=no_LoG, no_Conv=no_Conv)
            # 融合低级边缘与高级语义特征模块
            self.enc4_pre = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                Conv(encoder_channels[-1], decode_channels[-2], kernel_size=3),
                Conv(decode_channels[-2], decode_channels[-2], kernel_size=1),
                nn.BatchNorm2d(decode_channels[-2]),
                nn.ELU(),
            )
            self.las_out = nn.Sequential(
                nn.Conv2d(
                    las_decode_channels + decode_channels[-2],
                    las_decode_channels * 2,
                    kernel_size=3,
                    padding=1,
                ),
                nn.Conv2d(
                    las_decode_channels * 2,
                    las_decode_channels,
                    kernel_size=1,
                ),
                nn.Dropout2d(0.1, inplace=True),
                nn.BatchNorm2d(las_decode_channels),
                nn.ELU(),
            )

        self.init_weight()

    def forward(
        self,
        res1,
        res2,
        res3,
        res4,
        h,
        w,
        las=None,
        image_name=None,
        feature_map_save_path=None,
    ):
        if self.edge:
            las = self.las(las)
            las_H, las_W = las.shape[-2:]
            enc4 = self.enc4_pre(res4)
            enc4 = F.interpolate(
                enc4, size=(las_H, las_W), mode="bilinear", align_corners=False
            )
            las = torch.cat([las, enc4], dim=1)
            las = self.las_out(las)
            # 融合低级边缘与高级语义特征

        # las2 = self.las_pre_conv[0](las1)
        # las3 = self.las_pre_conv[1](las1)
        # las4 = self.las_pre_conv[2](las1)
        if self.aux and self.training:
            x = self.blacks[-1](self.pre_conv(res4), las=las)
            h4 = self.up4(x)
            x = self.upsample[-1](x, res3)
            x = self.blacks[-2](x, las=las)
            h3 = self.up3(x)

            x = self.upsample[-2](x, res2)
            x = self.blacks[-3](x, las=las)
            h2 = self.up2(x)

            x = self.upsample[-3](x, res1, las)
            # x = self.merge_las(x, las=las)
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)

            return x, ah
        else:
            x = self.blacks[-1](self.pre_conv(res4), las=las)
            x = self.upsample[-1](x, res3)
            x = self.blacks[-2](x, las=las)

            x = self.upsample[-2](x, res2)
            x = self.blacks[-3](x, las=las)

            x = self.upsample[-3](x, res1, las)
            # x = self.merge_las(x, las=las)
            if (
                not self.training
                and feature_map_save_path is not None
                and image_name is not None
            ):
                if not os.path.exists(feature_map_save_path):
                    os.makedirs(feature_map_save_path)
                [batch, channel, height, width] = x.shape
                for b in range(batch):
                    sum_x = torch.sum(x, dim=1, keepdim=False)
                    sum_x[b, :, :].cpu().numpy()
                    # plt.figure(figsize=(10, 10))
                    # for c in range(channel):
                    #     ax = plt.subplot(6, 6, c + 1)
                    #     ax.imshow(x[b, c, :, :].cpu().numpy())
                    plt.imsave(
                        os.path.join(
                            feature_map_save_path,
                            image_name[b] + ".png",
                        ),
                        sum_x[b, :, :].cpu().numpy(),
                    )
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class BRSMamba(nn.Module):

    def __init__(
        self,
        backbone_name="mamba_vision_t_1k",
        decode_channels=[32, 32, 32, 64],
        # decode_channels=[64, 64, 64, 64],
        dropout=0.1,
        window_size=[32, 16, 8, 4],
        # window_size=[8, 8, 8, 8],
        num_classes=6,
        num_heads=[16, 32, 16, 8],
        # num_heads=[8, 8, 8, 8],
        las_decode_channels=8,
        in_dt=True,
        erh_in_dt=True,
        edge=True,
        pretrained=True,
        pretrained_ckpt_path="./pretrained_ckpt/mamba_vision_t_1k.pth",
        aux=True,
        no_LoG=False,
        no_Conv=False,
        in_dt_blk=[True, True, True, True],
    ):
        super().__init__()
        print(backbone_name)
        self.in_dt = in_dt
        self.aux = aux
        if backbone_name == "resnet18":
            self.backbone = "resnet18.fb_swsl_ig1b_ft_in1k"
        encoder_channels = [80, 160, 320, 640]
        if backbone_name == "mamba_vision_t_1k":
            self.backbone = MambaVision()
            if pretrained:
                assert os.path.exists(pretrained_ckpt_path)
                ckpt = torch.load(
                    pretrained_ckpt_path, map_location="cpu", weights_only=True
                )
            self.backbone.load_state_dict(ckpt, strict=False)
        else:
            self.backbone = timm.create_model(
                backbone_name,
                features_only=True,
                output_stride=32,
                out_indices=(1, 2, 3, 4),
                pretrained=pretrained,
            )
            encoder_channels = self.backbone.feature_info.channels()
            # print(encoder_channels)
        # if self.in_dt:
        # self.las = LAS(3, las_decode_channels)
        self.decoder = Decoder(
            encoder_channels=encoder_channels,
            decode_channels=decode_channels,
            las_decode_channels=las_decode_channels,
            dropout=dropout,
            window_size=window_size,
            ssd_in_dt=in_dt,
            erh_in_dt=erh_in_dt,
            num_classes=num_classes,
            num_heads=num_heads,
            aux=aux,
            edge=edge,
            no_LoG=no_LoG,
            no_Conv=no_Conv,
            in_dt_blk=in_dt_blk,
        )

    def forward(self, x, image_name=None, feature_map_save_path=None):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # h, w = x.size()[-2:]
        h, w = x.size()[-2:]
        las = x
        [enc1, enc2, enc3, enc4] = self.backbone(x)
        # (enc1, enc2, enc3, enc4) = x
        if self.aux and self.training:
            x, ah = self.decoder(enc1, enc2, enc3, enc4, h, w, las)
            return x, ah
        else:
            x = self.decoder(
                enc1, enc2, enc3, enc4, h, w, las, image_name, feature_map_save_path
            )
            return x


def cat_fps():
    H, W = 512, 512
    input = torch.rand(1, 3, H, W).cuda()
    # net = AtMbNet(backbone_name="resnet50", in_dt=True).to("cuda")
    net = BRSMamba(
        backbone_name="efficientnet_b3",
        in_dt=True,
        #   window_size=[32, 16, 8, 4],
        window_size=[8, 8, 8, 8],
        num_classes=6,
        # num_heads=[16, 32, 16, 8],
        num_heads=[8, 8, 8, 8],
        las_decode_channels=8,
    ).to("cuda")
    net.eval()
    import numpy as np
    import time

    with torch.no_grad():
        t_all = []
        for i in range(100):
            t1 = time.time()
            y = net(input)
            t2 = time.time()
            t_all.append(t2 - t1)
        print("average time:", np.mean(t_all) / 1)
        print("average fps:", 1 / np.mean(t_all))

        print("fastest time:", min(t_all) / 1)
        print("fastest fps:", 1 / min(t_all))

        print("slowest time:", max(t_all) / 1)
        print("slowest fps:", 1 / max(t_all))


def Test():
    model = BRSMamba(
        backbone_name="efficientnet_b3",
        in_dt=True,
        aux=False,
        erh_in_dt=True,
        edge=True,
        in_dt_blk=[False, False, True, True],
    ).to("cuda")
    model.train()
    x = torch.rand(2, 3, 512, 512).to("cuda")
    y = model(x)
    print(y.shape)
    model.eval()
    y = model(x)
    print(y.shape)


def NCSSD_test():
    ncssd = NCSSD(64).to("cuda")
    # ncssd.__DEBUG__ = True
    x = torch.rand(2, 64, 32, 32).to("cuda")
    y = ncssd(x)
    print(y.shape)


if __name__ == "__main__":
    # main()
    Test()
    # cat_fps()
    # NCSSD_test()
# model = GaussianFilter(3)
# model_list = timm.list_models(pretrained=True)
# print(model_list)
