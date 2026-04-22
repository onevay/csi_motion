import sys
from app.app import Runner
from tests import simulate_from_csv

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [monitor|simulate|receiver|receiver_split|watcher] [args...]")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "monitor":
        target = sys.argv[2] if len(sys.argv) > 2 else "../data/esp"
        threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        exe = sys.argv[4] if len(sys.argv) > 4 else "./preprocess.elf"
        runner = Runner(target_dir=target, threshold=threshold, executable=exe)
        try:
            runner.monitor()
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    elif mode == "simulate":
        csv_path = sys.argv[2]
        max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else None
        delay = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
        simulate_from_csv(csv_path, delay, max_samples)
    elif mode == "receiver":
        from tools import receiver_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        receiver_main()
    elif mode == "watcher":
        from tools import watcher_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        watcher_main()
    elif mode == "receiver_split":
        from tools import receiver_split_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        receiver_split_main()
    else:
        print("Unknown mode")