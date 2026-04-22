import argparse
import ast
import os
import shutil
import subprocess
import time
import numpy as np
from app.load_model import get_response


def parse_args():
    parser = argparse.ArgumentParser(description="Watch test folders and run inference")
    parser.add_argument("--watch-dir", required=True, type=str)
    parser.add_argument("--preprocessor", default="./cpp_preprocessor/build/preprocessor", type=str)
    parser.add_argument("--interval", default=0.5, type=float)
    parser.add_argument("--timeout", default=10.0, type=float)
    parser.add_argument("--cleanup", action="store_true", default=True)
    parser.add_argument("--max-retries", default=5, type=int)
    parser.add_argument("--retry-delay", default=0.5, type=float)
    return parser.parse_args()


def is_file_ready(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        size_before = os.path.getsize(filepath)
        time.sleep(0.1)
        size_after = os.path.getsize(filepath)
        return size_before == size_after and size_before > 0
    except OSError:
        return False


def parse_preprocessor_output(raw):
    payload = ast.literal_eval(raw)
    if not isinstance(payload, list) or len(payload) != 3:
        raise ValueError("invalid preprocessor output")
    arrays = []
    for item in payload:
        if not isinstance(item, list):
            raise ValueError("invalid device payload")
        arr = np.asarray(item, dtype=np.float32)
        if arr.size < 5200:
            arr = np.pad(arr, (0, 5200 - arr.size), mode="constant")
        elif arr.size > 5200:
            arr = arr[:5200]
        arrays.append(arr.reshape(100, 52))
    stacked = np.stack(arrays, axis=0)
    return np.transpose(stacked, (1, 2, 0))


def run_models(preprocessor_output):
    sample = parse_preprocessor_output(preprocessor_output)
    cnn_m, cnn_d, mamba_m, mamba_d, transformer_d = get_response(sample)
    print(f"CNN motion={cnn_m} distance={cnn_d}")
    print(f"Mamba motion={mamba_m} distance={mamba_d}")
    print(f"Transformer distance={transformer_d}")


def try_process(folder_path, preprocessor_path, timeout):
    dev_files = [os.path.join(folder_path, "dev1.data"), os.path.join(folder_path, "dev2.data"), os.path.join(folder_path, "dev3.data")]
    if not all(is_file_ready(path) for path in dev_files):
        return False
    try:
        result = subprocess.run([preprocessor_path, *dev_files], capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"[ERR] {folder_path}: {result.stderr.strip()}")
            return False
        payload = result.stdout.strip()
        if not payload:
            return False
        run_models(payload)
        return True
    except Exception as e:
        print(f"[WATCHER ERR] {e}")
        return False


def main():
    args = parse_args()
    processed = set()
    failed = {}
    print(f"Watching {args.watch_dir}")
    while True:
        try:
            with os.scandir(args.watch_dir) as entries:
                candidates = sorted((entry for entry in entries if entry.is_dir() and entry.name.startswith("test_")), key=lambda e: e.name)
            for entry in candidates:
                if entry.name in processed:
                    continue
                attempts = failed.get(entry.name, 0)
                if attempts >= args.max_retries:
                    continue
                ok = try_process(entry.path, args.preprocessor, args.timeout)
                if ok:
                    processed.add(entry.name)
                    failed.pop(entry.name, None)
                    if args.cleanup:
                        try:
                            shutil.rmtree(entry.path)
                        except Exception as e:
                            print(f"[CLEANUP ERR] {entry.path}: {e}")
                else:
                    failed[entry.name] = attempts + 1
                    if failed[entry.name] < args.max_retries:
                        time.sleep(args.retry_delay)
        except Exception as e:
            print(f"[WATCHER CRITICAL ERR] {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()