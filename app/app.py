# File: runner.py
import os
import subprocess
from pathlib import Path
import time
from .data_load.parse_data import load_csv_file
from .load_model import get_response

class Runner:
    def __init__(self, target_dir, output_dir="../data/preprocessed", threshold=100, executable="preprocess.elf"):
        self.target_dir = Path(target_dir)
        self.threshold = threshold
        self.cpp_executable = executable
        self.output_dir = Path(output_dir)
        self.last_count = 0
        self._validate_setup()

    def _validate_setup(self):
        if not self.target_dir.exists():
            raise FileNotFoundError("Target directory not found")
        if not os.access(self.cpp_executable, os.X_OK):
            raise FileNotFoundError("Executable not accessible")

    def count_folders(self):
        try:
            folders = [item for item in self.target_dir.iterdir() if item.is_dir() and not item.name.startswith('.')]
            return len(folders)
        except PermissionError:
            return 0

    def run_cpp_inference(self):
        try:
            result = subprocess.run([self.cpp_executable], capture_output=True, text=True, timeout=5.0)
            if result.returncode != 0:
                print(f"C++ error: {result.stderr}")
                return False
            return True
        except Exception as e:
            print(f"C++ exception: {e}")
            return False

    def monitor(self, interval=0.5):
        print(f"Monitoring {self.target_dir}")
        while True:
            current_count = self.count_folders()
            if self.last_count == 0:
                self.last_count = current_count
            new_folders = current_count - self.last_count
            if new_folders >= self.threshold:
                print(f"Threshold reached ({new_folders} folders). Running C++ preprocessing...")
                if self.run_cpp_inference():
                    csv_path = self.output_dir / "wifi_data_set_after_preprocessing.csv"
                    if csv_path.exists():
                        data, _ = load_csv_file(str(csv_path))
                        if len(data) > 0:
                            sample = data[0]
                            cnn_m, cnn_d, mamba_m, mamba_d, transformer_d = get_response(sample)
                            print("-" * 80)
                            print(f"CNN   : motion={cnn_m}, distance={cnn_d}")
                            print(f"Mamba : motion={mamba_m}, distance={mamba_d}")
                            print(f"Transformer : distance={transformer_d}")
                            print("-" * 80)
                self.last_count = current_count
            else:
                print(f"Waiting: {new_folders}/{self.threshold}")
            time.sleep(interval)