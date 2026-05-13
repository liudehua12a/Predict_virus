from __future__ import annotations

import time
from pathlib import Path
import sys

import jwt

import os

# 引入打包路径兼容工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pyinstaller_utils as pkgutil


def generate_qweather_jwt(
    private_key_path: str | Path = "ed25519-private.pem",
    sub: str = "4FKRV33M9W",
    kid: str = "KJ59BN995H",
    ttl_seconds: int = 900,
) -> str:
    """
    生成和风天气 JWT。
    """
    # 使用兼容打包后路径的私钥文件
    private_key_path = pkgutil.get_private_key_path()
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