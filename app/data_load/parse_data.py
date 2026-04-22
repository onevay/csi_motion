import numpy as np
import re
import pickle
from tqdm import tqdm

USE_NORMALIZATION = False

def load_scalers(path='scalers_person_split.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def parse_csv_line(line):
    line = line.strip()
    if not line:
        return None, None
    match = re.search(r':\[\[(.*?)\]\]:(.*?)$', line)
    if not match:
        return None, None
    content = match.group(1)
    label_str = match.group(2).strip()
    try:
        label = int(float(label_str))
    except ValueError:
        return None, None
    parts = content.split('],[')
    if len(parts) != 3:
        return None, None
    all_numbers = []
    for part in parts:
        part = part.strip('[').strip(']')
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', part)
        if not numbers:
            return None, None
        try:
            values = [float(x) for x in numbers]
        except ValueError:
            return None, None
        all_numbers.extend(values)
    expected_len = 3 * 52 * 100
    if len(all_numbers) < expected_len:
        return None, None
    data = np.array(all_numbers[:expected_len])
    data = data.reshape(3, 100, 52).transpose(1, 2, 0)
    return data, label

def normalize_sample(sample, scalers):
    n_devices = len(scalers)
    n_subcarriers = len(scalers[0])
    sample_norm = np.zeros_like(sample)
    for d in range(n_devices):
        for f in range(n_subcarriers):
            flat = sample[:, f, d].reshape(-1, 1)
            sample_norm[:, f, d] = scalers[d][f].transform(flat).reshape(-1)
    return sample_norm

def load_csv_file(filepath):
    segments = []
    labels = []
    with open(filepath, 'r') as f:
        for line in tqdm(f, desc=f"Loading {filepath}"):
            seg, lbl = parse_csv_line(line)
            if seg is not None:
                segments.append(seg)
                labels.append(lbl)
    return np.array(segments), np.array(labels)