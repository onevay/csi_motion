import torch
import torch.nn as nn
import torch.nn.functional as Func

class ConfigCNN:
    DATA_TYPE = "stack"
    N_SUBCARRIERS = 52
    N_MOTION_CLASSES = 2
    N_DISTANCE_CLASSES = 4
    CONV1_FILTERS = 64
    CONV1_KERNEL = 3
    CONV2_FILTERS = 128
    CONV2_KERNEL = 3
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.35
    BIDIRECTIONAL = True
    USE_ATTENTION = True
    ATTENTION_SIZE = 64

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1)
        )
    def forward(self, lstm_output):
        weights = self.attention(lstm_output).squeeze(-1)
        weights = Func.softmax(weights, dim=1)
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
        return context, weights

class CSI_CNN_LSTM_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_channels = config.N_SUBCARRIERS * 3
        self.conv1 = nn.Conv1d(input_channels, config.CONV1_FILTERS,
                               kernel_size=config.CONV1_KERNEL, padding='same')
        self.bn1 = nn.BatchNorm1d(config.CONV1_FILTERS)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(config.CONV1_FILTERS, config.CONV2_FILTERS,
                               kernel_size=config.CONV2_KERNEL, padding='same')
        self.bn2 = nn.BatchNorm1d(config.CONV2_FILTERS)
        self.pool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(
            input_size=config.CONV2_FILTERS,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL
        )
        lstm_out_size = config.HIDDEN_SIZE * 2 if config.BIDIRECTIONAL else config.HIDDEN_SIZE
        self.use_attention = config.USE_ATTENTION
        if self.use_attention:
            self.attention = TemporalAttention(lstm_out_size, config.ATTENTION_SIZE)
            fc_in = lstm_out_size
        else:
            self.attention = None
            fc_in = lstm_out_size
        self.motion_head = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.N_MOTION_CLASSES)
        )
        self.distance_head = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.N_DISTANCE_CLASSES)
        )
    def forward(self, x):
        B, T, F, D = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B, F * D, T)
        x = self.pool1(Func.relu(self.bn1(self.conv1(x))))
        x = self.pool2(Func.relu(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        if self.use_attention:
            context, attn = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]
            attn = None
        motion_logits = self.motion_head(context)
        distance_logits = self.distance_head(context)
        return motion_logits, distance_logits, attn