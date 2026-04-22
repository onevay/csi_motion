from pathlib import Path
import numpy as np
import torch
from .models.cnn_lstm import CSI_CNN_LSTM_Attention, ConfigCNN
from .models.mamba_s6 import MultiTaskMambaModel, ConfigMamba
from .models.transformer import FusionCSIModel, ConfigTransformer


_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_WEIGHTS_DIR = Path("./weights")
_CNN_MODEL = None
_MAMBA_MODEL = None
_TRANSFORMER_MODEL = None
_TRANSFORMER_MEAN = None
_TRANSFORMER_STD = None


def _load_cnn_model():
    model = CSI_CNN_LSTM_Attention(ConfigCNN).to(_DEVICE)
    model.load_state_dict(torch.load(_WEIGHTS_DIR / "best_cnn_lstm.pth", map_location=_DEVICE))
    model.eval()
    return model


def _load_mamba_model():
    model = MultiTaskMambaModel(ConfigMamba).to(_DEVICE)
    model.load_state_dict(torch.load(_WEIGHTS_DIR / "best_mamba_model.pth", map_location=_DEVICE))
    model.eval()
    return model


def _load_transformer_model():
    model = FusionCSIModel(ConfigTransformer).to(_DEVICE)
    model.load_state_dict(torch.load(_WEIGHTS_DIR / "best_fusion_csi_model.pth", map_location=_DEVICE))
    model.eval()
    return model


def _ensure_models_loaded():
    global _CNN_MODEL, _MAMBA_MODEL, _TRANSFORMER_MODEL, _TRANSFORMER_MEAN, _TRANSFORMER_STD
    if _CNN_MODEL is None:
        _CNN_MODEL = _load_cnn_model()
    if _MAMBA_MODEL is None:
        _MAMBA_MODEL = _load_mamba_model()
    if _TRANSFORMER_MODEL is None:
        _TRANSFORMER_MODEL = _load_transformer_model()
    if _TRANSFORMER_MEAN is None:
        _TRANSFORMER_MEAN = np.load(_WEIGHTS_DIR / "train_mean.npy")
    if _TRANSFORMER_STD is None:
        _TRANSFORMER_STD = np.load(_WEIGHTS_DIR / "train_std.npy")


def _prepare_transformer_input(sample):
    transformer_sample = np.transpose(sample, (2, 0, 1))[None, ...]
    transformer_sample = (transformer_sample - _TRANSFORMER_MEAN) / (_TRANSFORMER_STD + 1e-8)
    return torch.FloatTensor(transformer_sample).to(_DEVICE)


def get_response(sample):
    _ensure_models_loaded()
    common_sample = torch.FloatTensor(sample).unsqueeze(0).to(_DEVICE)
    transformer_sample = _prepare_transformer_input(sample)
    with torch.no_grad():
        m_out_mamba, d_out_mamba = _MAMBA_MODEL(common_sample)
        m_out_cnn, d_out_cnn, _ = _CNN_MODEL(common_sample)
        transformer_logits = _TRANSFORMER_MODEL(transformer_sample)
    motion_pred_cnn = m_out_cnn.argmax(dim=1).item()
    dist_pred_cnn = d_out_cnn.argmax(dim=1).item()
    motion_pred_mamba = m_out_mamba.argmax(dim=1).item()
    dist_pred_mamba = d_out_mamba.argmax(dim=1).item()
    dist_pred_transformer = transformer_logits.argmax(dim=1).item()
    return motion_pred_cnn, dist_pred_cnn, motion_pred_mamba, dist_pred_mamba, dist_pred_transformer