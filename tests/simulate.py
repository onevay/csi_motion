# File: simulate.py
import sys
import time
import random
from pathlib import Path
from app.data_load.parse_data import load_csv_file, parse_csv_line
from app.load_model import get_response
from sklearn.metrics import accuracy_score

def simulate_from_csv(csv_path, delay=0.1, max_samples=None):
    print(f"Loading {csv_path}...")
    data, labels = load_csv_file(str(csv_path))
    if max_samples:
        indices = random.sample(range(len(data)), min(max_samples, len(data)))
    else:
        indices = range(len(data))
    
    y_true_motion = []
    y_pred_cnn_m = []
    y_pred_mamba_m = []
    y_true_dist = []
    y_pred_cnn_d = []
    y_pred_mamba_d = []
    y_pred_transformer_d = []
    
    for idx in indices:
        sample = data[idx]
        true_label = labels[idx]
        true_motion = 1 if true_label > 0 else 0
        
        cnn_m, cnn_d, mamba_m, mamba_d, transformer_d = get_response(sample)
        
        y_true_motion.append(true_motion)
        y_pred_cnn_m.append(cnn_m)
        y_pred_mamba_m.append(mamba_m)
        y_true_dist.append(true_label)
        y_pred_cnn_d.append(cnn_d)
        y_pred_mamba_d.append(mamba_d)
        y_pred_transformer_d.append(transformer_d)
        
        print(f"Sample {idx}: True(m={true_motion},d={true_label}) | "
              f"CNN(m={cnn_m},d={cnn_d}) | Mamba(m={mamba_m},d={mamba_d}) | Transformer(d={transformer_d})")
        time.sleep(delay)
    
    print("\n=== Final Metrics ===")
    print(f"CNN   Motion Acc : {accuracy_score(y_true_motion, y_pred_cnn_m):.4f}")
    print(f"CNN   Distance Acc: {accuracy_score(y_true_dist, y_pred_cnn_d):.4f}")
    print(f"Mamba Motion Acc : {accuracy_score(y_true_motion, y_pred_mamba_m):.4f}")
    print(f"Mamba Distance Acc: {accuracy_score(y_true_dist, y_pred_mamba_d):.4f}")
    print(f"Transformer Distance Acc: {accuracy_score(y_true_dist, y_pred_transformer_d):.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simulate.py <path_to_csv> [max_samples] [delay]")
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else None
    delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    simulate_from_csv(csv_path, delay, max_samples)