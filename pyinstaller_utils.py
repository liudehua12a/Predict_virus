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


def get_db_path() -> Path:
    """获取数据库文件路径"""
    return get_data_path("algorithm/data/nky-CornPre.db")


def get_config_path() -> Path:
    """获取配置文件路径"""
    return get_data_path("algorithm/data/config.ini")


def get_private_key_path() -> Path:
    """获取和风天气 JWT 私钥路径"""
    return get_data_path("algorithm/ed25519-private.pem")