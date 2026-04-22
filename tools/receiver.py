import argparse
import csv
import os
import threading
import time
from datetime import datetime
import serial
import serial.tools.list_ports


HEADER = [
    "logger_timestamp",
    "record_type",
    "packet_seq",
    "source_mac",
    "rssi_dbm",
    "rate_code",
    "sig_mode",
    "mcs_index",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor_dbm",
    "ampdu_count",
    "wifi_channel",
    "secondary_channel",
    "local_timestamp",
    "antenna",
    "signal_length",
    "rx_state",
    "csi_len",
    "first_word",
    "csi_data",
]
device_buffers = [[], [], []]
buffer_lock = threading.Lock()
test_counter = 1
window_size = 0
overlap = 0
output_dir = ""


def parse_args():
    parser = argparse.ArgumentParser(description="CSI Logger for 3 ESP32-S3")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--overlap", type=int, required=True)
    parser.add_argument("--ports", nargs=3, required=True)
    parser.add_argument("--baud", type=int, default=921600)
    return parser.parse_args()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def logger_timestamp():
    return datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f")


def list_available_ports():
    return [p.device for p in serial.tools.list_ports.comports()]


def check_all_connected(required_ports):
    available = list_available_ports()
    print("\nAvailable ports:", available)
    all_ok = True
    for port in required_ports:
        if port not in available:
            print(f"[ERROR] {port} not found")
            all_ok = False
        else:
            print(f"[OK] {port} detected")
    return all_ok


def parse_csi_line(raw_line):
    try:
        parts = raw_line.split(",", 24)
        if len(parts) < 25:
            return None
        return parts
    except Exception:
        return None


def try_save_segment():
    global test_counter
    with buffer_lock:
        if all(len(buf) >= window_size for buf in device_buffers):
            folder_name = f"test_{test_counter}"
            folder_path = os.path.join(output_dir, folder_name)
            ensure_dir(folder_path)
            for idx, buf in enumerate(device_buffers):
                segment = buf[:window_size]
                filename = f"dev{idx + 1}.data"
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(HEADER)
                    writer.writerows(segment)
                device_buffers[idx] = buf[window_size - overlap :]
            test_counter += 1
            print(f"[SYNC] Saved segment {folder_name}")


def serial_listener(device_index, port, baudrate):
    while True:
        try:
            print(f"[dev{device_index + 1}] Opening {port}")
            ser = serial.Serial(port, baudrate, timeout=1)
            print(f"[dev{device_index + 1}] Connected")
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if "CSI_DATA" not in line:
                    continue
                parsed = parse_csi_line(line)
                if parsed is None:
                    continue
                row = [logger_timestamp()] + parsed
                should_save = False
                with buffer_lock:
                    device_buffers[device_index].append(row)
                    if all(len(buf) >= window_size for buf in device_buffers):
                        should_save = True
                if should_save:
                    try_save_segment()
        except Exception as e:
            print(f"[dev{device_index + 1}] Error: {e}. Reconnecting in 2s")
            time.sleep(2)


def save_partial():
    with buffer_lock:
        if any(device_buffers):
            folder_name = f"test_{test_counter}_partial"
            folder_path = os.path.join(output_dir, folder_name)
            ensure_dir(folder_path)
            for idx, buf in enumerate(device_buffers):
                if buf:
                    filename = f"dev{idx + 1}.data"
                    filepath = os.path.join(folder_path, filename)
                    with open(filepath, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(HEADER)
                        writer.writerows(buf)
            print(f"[SYNC] Saved partial segment {folder_name}")


def main():
    global window_size, overlap, output_dir
    args = parse_args()
    if args.overlap >= args.window:
        raise ValueError("Overlap must be smaller than window size")
    window_size = args.window
    overlap = args.overlap
    output_dir = args.output
    ensure_dir(args.output)
    while not check_all_connected(args.ports):
        print("Waiting for all devices")
        time.sleep(3)
    threads = []
    for i, port in enumerate(args.ports):
        t = threading.Thread(target=serial_listener, args=(i, port, args.baud), daemon=True)
        t.start()
        threads.append(t)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        save_partial()


if __name__ == "__main__":
    main()