import os
from datetime import datetime

def get_log_file(prefix: str) -> str:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    return os.path.join(log_dir, f"{prefix}_{date_str}.log")


def log(message: str, prefix: str = "task"):
    log_file = get_log_file(prefix)
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{time_str}] {message}\n")