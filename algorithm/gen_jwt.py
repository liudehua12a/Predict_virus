from __future__ import annotations

import time
from pathlib import Path

import jwt

import os
def generate_qweather_jwt(
    private_key_path: str | Path = "ed25519-private.pem",
    sub: str = "4FKRV33M9W",
    kid: str = "KJ59BN995H",
    ttl_seconds: int = 900,
) -> str:
    """
    生成和风天气 JWT。
    """
    private_key_path = Path(private_key_path)
    # 获取当前脚本 (gen_jwt.py) 所在的绝对路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 拼接出 pem 文件的绝对路径
    private_key_path = os.path.join(BASE_DIR, "ed25519-private.pem")
    with open(private_key_path, "r", encoding="utf-8") as f:
        private_key = f.read()

    now_ts = int(time.time())

    payload = {
        "iat": now_ts - 30,
        "exp": now_ts + ttl_seconds,
        "sub": sub,
    }

    headers = {
        "kid": kid,
    }

    token = jwt.encode(
        payload,
        private_key,
        algorithm="EdDSA",
        headers=headers,
    )
    return token


if __name__ == "__main__":
    token = generate_qweather_jwt()
    print(token)