import torch
import torch.nn as nn
import torch.nn.functional as Func

class ConfigMamba:
    DATA_TYPE = "stack"
    WINDOW_SIZE = 100
    N_FEATURES = 52
    D_MODEL = 32
    N_LAYERS = 2
    D_STATE = 16
    DT_RANK = 4
    D_CONV = 4
    EXPAND_FACTOR = 2.0
    N_HEADS = 4
    DROPOUT = 0.5
    N_MOTION_CLASSES = 2
    N_DISTANCE_CLASSES = 4
    INPUT_CONV_KERNEL = 7
    POOL_SIZE = 2

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, dt_rank=None, delta_softplus=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(1, d_model // 16)
        self.delta_softplus = delta_softplus
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, 1)
        self.x_proj_state = nn.Linear(d_model, d_state, bias=False)
        A = torch.arange(1, d_state + 1).float().view(1, d_state)
        A = -torch.exp(A)
        self.A_log = nn.Parameter(torch.log(-A))
        self.out_proj = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        B, L, D = x.shape
        N = self.d_state
        proj = self.x_proj(x)
        dt_rank_part = proj[..., :self.dt_rank]
        B_part = proj[..., self.dt_rank:self.dt_rank + N]
        C_part = proj[..., self.dt_rank + N:]
        dt = self.dt_proj(dt_rank_part)
        if self.delta_softplus:
            dt = Func.softplus(dt)
        A = -torch.exp(self.A_log)
        A_bar = torch.exp(dt * A)
        B_bar = dt * B_part
        x_state = self.x_proj_state(x)
        h = torch.zeros(B, N, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x_state[:, t]
            y_t = self.out_proj(h * C_part[:, t])
            outputs.append(y_t.unsqueeze(1))
        y = torch.cat(outputs, dim=1)
        y = y + self.D * x
        return y

class MambaBranch(nn.Module):
    def __init__(self, d_model, d_state=16, expand_factor=2, conv_kernel=4, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        d_inner = int(d_model * expand_factor)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=conv_kernel, padding=conv_kernel-1, groups=d_inner)
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x_proj = self.in_proj(x)
        x_ssm, gate = x_proj.chunk(2, dim=-1)
        x_ssm = self.conv1d(x_ssm.transpose(1,2)).transpose(1,2)[:, :x.size(1), :]
        x_ssm = Func.silu(x_ssm)
        x_ssm = self.ssm(x_ssm)
        gate = Func.silu(gate)
        x = x_ssm * gate
        x = self.out_proj(x)
        x = self.dropout(x)
        return residual + x

class ConvBranch(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2, groups=d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv(x.transpose(1,2)).transpose(1,2)
        x = self.act(x)
        x = self.dropout(x)
        return residual + x

class HybridMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand_factor=2, conv_kernel=4, dropout=0.1):
        super().__init__()
        self.conv_branch = ConvBranch(d_model, kernel_size=3, dropout=dropout)
        self.mamba_branch = MambaBranch(d_model, d_state, expand_factor, conv_kernel, dropout)
    def forward(self, x):
        x = self.conv_branch(x)
        x = self.mamba_branch(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x_motion, x_dist):
        q = self.norm_q(x_motion)
        kv = self.norm_kv(x_dist)
        attn_out, _ = self.cross_attn(q, kv, kv)
        return x_motion + self.dropout(attn_out)

class InputConvBlock(nn.Module):
    def __init__(self, in_channels, d_model, kernel_size=7, pool_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm = nn.BatchNorm1d(d_model)
        self.pool = nn.MaxPool1d(pool_size)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(Func.relu(self.norm(self.conv(x))))
        x = x.transpose(1, 2)
        return x

class MultiTaskMambaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = config.N_FEATURES * 3
        self.input_conv = InputConvBlock(in_channels, config.D_MODEL, config.INPUT_CONV_KERNEL, config.POOL_SIZE)
        self.pos_embed = nn.Parameter(torch.randn(1, config.WINDOW_SIZE // config.POOL_SIZE, config.D_MODEL) * 0.02)
        self.layers = nn.ModuleList([
            HybridMambaBlock(config.D_MODEL, config.D_STATE, config.EXPAND_FACTOR, config.D_CONV, config.DROPOUT)
            for _ in range(config.N_LAYERS)
        ])
        self.proj_motion = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.proj_distance = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.cross_attn_1 = CrossAttentionBlock(config.D_MODEL, config.N_HEADS, config.DROPOUT)
        self.cross_attn_2 = CrossAttentionBlock(config.D_MODEL, config.N_HEADS, config.DROPOUT)
        self.norm_motion = RMSNorm(config.D_MODEL)
        self.norm_distance = RMSNorm(config.D_MODEL)
        self.motion_head = nn.Sequential(
            nn.Linear(config.D_MODEL, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.N_MOTION_CLASSES)
        )
        self.distance_head = nn.Sequential(
            nn.Linear(config.D_MODEL, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.N_DISTANCE_CLASSES)
        )
    def forward(self, x):
        B, T, F, D = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, T, F * D)
        x = self.input_conv(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        x_motion = self.proj_motion(x)
        x_distance = self.proj_distance(x)
        x_motion = self.cross_attn_1(x_motion, x_distance)
        x_distance = self.cross_attn_2(x_distance, x_motion)
        x_motion = self.norm_motion(x_motion).mean(dim=1)
        x_distance = self.norm_distance(x_distance).mean(dim=1)
        motion_logits = self.motion_head(x_motion)
        distance_logits = self.distance_head(x_distance)
        return motion_logits, distance_logits