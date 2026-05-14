# -*- coding: utf-8 -*-
"""
PyInstaller 打包路径兼容工具
提供函数以兼容开发环境（普通运行）和打包环境（exe 运行）
"""

import sys
import os
from pathlib import Path


def get_root_path():
    """
    获取项目根目录路径。

    - 打包后（exe）：返回 sys._MEIPASS 临时解压目录
    - 开发时：返回 main.py 所在目录（即项目根目录）
    """
    if getattr(sys, 'frozen', False):
        # 打包后：sys._MEIPASS 是 PyInstaller 解压数据文件的临时目录
        return Path(sys._MEIPASS)
    else:
        # 开发时：main.py 在项目根目录
        return Path(__file__).resolve().parent


def get_data_path(relative_path: str) -> Path:
    """
    获取数据文件的正确路径（兼容打包前后）。

    参数:
        relative_path: 相对于项目根目录的路径，如 'algorithm/data/imgs/background/224.jpg'

    返回:
        Path: 完整的绝对路径
    """
    root = get_root_path()
    return root / relative_path


def get_algorithm_dir() -> Path:
    """获取 algorithm 目录路径"""
    return get_data_path("algorithm")


def get_config_path() -> Path:
    """获取配置文件路径"""
    return get_data_path("algorithm/data/config.ini")


def get_db_path() -> Path:
    """
    获取数据库文件路径。

    - 打包后（exe）：从配置文件读取 db_path，否则使用 exe 同级目录。
      文件不存在时抛出 FileNotFoundError。
    - 开发时：优先从配置文件读取，否则回退到 algorithm/data/nky-CornPre.db。
    """
    config_path = get_config_path()
    db_path = None

    if config_path.exists():
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(str(config_path), encoding="utf-8")
            if config.has_section("database") and config.has_option("database", "db_path"):
                raw_path = config.get("database", "db_path").strip()
                if raw_path:
                    db_path = Path(raw_path)
                    if not db_path.is_absolute():
                        db_path = get_root_path() / db_path
        except Exception:
            pass

    if db_path is None:
        if getattr(sys, 'frozen', False):
            root = get_root_path()
            db_path = root / "nky-CornPre.db"
        else:
            root = get_root_path()
            db_path = root / "algorithm" / "data" / "nky-CornPre.db"

    if not db_path.exists():
        raise FileNotFoundError(
            f"数据库文件不存在: {db_path}\n"
            "请在 config.ini 中指定正确的 db_path，或将 nky-CornPre.db 放在正确位置。"
        )

    return db_path


def get_private_key_path() -> Path:
    """获取和风天气 JWT 私钥路径"""
    return get_data_path("algorithm/ed25519-private.pem")