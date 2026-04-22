import torch
import torch.nn as nn


class ConfigTransformer:
    N_SENSORS = 3
    TIME_STEPS = 100
    N_SUBCARRIERS = 52
    N_CLASSES = 4
    D_MODEL = 64
    N_HEADS = 4
    NUM_LAYERS_LOCAL = 2
    NUM_LAYERS_GLOBAL = 3
    DIM_FEEDFORWARD = 256
    CNN_CHANNELS_1 = 64
    KERNEL_SIZE_1 = 5
    KERNEL_SIZE_2 = 3
    DROPOUT_TRANSFORMER = 0.3
    DROPOUT_CLASSIFIER = 0.5


def compute_csi_features(x):
    delta1 = x[:, 1:, :] - x[:, :-1, :]
    delta1 = torch.cat([delta1[:, :1, :], delta1], dim=1)
    delta2 = delta1[:, 1:, :] - delta1[:, :-1, :]
    delta2 = torch.cat([delta2[:, :1, :], delta2], dim=1)
    return torch.cat([x, delta1, delta2], dim=2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class WindowEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(config.N_SUBCARRIERS * 3, config.CNN_CHANNELS_1, config.KERNEL_SIZE_1, stride=2, padding=2),
            nn.BatchNorm1d(config.CNN_CHANNELS_1),
            nn.ReLU(),
            nn.Conv1d(config.CNN_CHANNELS_1, config.D_MODEL, config.KERNEL_SIZE_2, padding=1),
            nn.BatchNorm1d(config.D_MODEL),
            nn.ReLU(),
        )
        self.input_norm = nn.LayerNorm(config.D_MODEL)
        self.pos_encoding = PositionalEncoding(config.D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEADS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT_TRANSFORMER,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS_LOCAL)
        self.attn_pool = nn.Linear(config.D_MODEL, 1)

    def forward(self, x):
        x = compute_csi_features(x)
        x = x.transpose(1, 2)
        x = self.cnn(x).transpose(1, 2)
        x = self.pos_encoding(x)
        x = self.input_norm(x)
        x = self.transformer(x)
        weights = torch.softmax(self.attn_pool(x).squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class HierarchicalCSIModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.local_encoder = WindowEncoder(config)
        self.global_pos_encoding = PositionalEncoding(config.D_MODEL)
        global_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEADS,
            batch_first=True,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT_TRANSFORMER,
            activation="gelu",
        )
        self.global_transformer = nn.TransformerEncoder(global_layer, num_layers=config.NUM_LAYERS_GLOBAL)
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT_CLASSIFIER),
            nn.Linear(config.D_MODEL, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, config.N_CLASSES),
        )

    def forward(self, x_seq):
        batch_size, num_sensors, time_len, subs = x_seq.size()
        x = x_seq.view(-1, time_len, subs)
        sensor_embeds = self.local_encoder(x)
        sensor_embeds = sensor_embeds.view(batch_size, num_sensors, -1)
        sensor_embeds = self.global_pos_encoding(sensor_embeds)
        global_feat = self.global_transformer(sensor_embeds)
        final_feat = torch.mean(global_feat, dim=1)
        return self.classifier(final_feat)


class TemporalBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.N_SENSORS * config.N_SUBCARRIERS * 3
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, config.D_MODEL),
            nn.LayerNorm(config.D_MODEL),
            nn.GELU(),
        )
        self.pos_encoding = PositionalEncoding(config.D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEADS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            batch_first=True,
            dropout=config.DROPOUT_TRANSFORMER,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(config.D_MODEL, config.N_CLASSES)

    def forward(self, x):
        x = compute_csi_features(x)
        x = self.proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        return self.classifier(x)


class FusionCSIModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.branch_a = HierarchicalCSIModel(config)
        self.branch_b = TemporalBranch(config)
        self.fusion_weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x_seq):
        logits_a = self.branch_a(x_seq)
        batch_size = x_seq.size(0)
        x_branch_b = x_seq.permute(0, 2, 1, 3).contiguous().view(batch_size, ConfigTransformer.TIME_STEPS, -1)
        logits_b = self.branch_b(x_branch_b)
        alpha = torch.sigmoid(self.fusion_weight)
        return alpha * logits_a + (1 - alpha) * logits_b
