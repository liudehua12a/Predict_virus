# -*- coding: utf-8 -*-
"""
PyInstaller Spec File for 玉米病害预测软件
打包命令: pyinstaller 玉米病害预测软件.spec --clean
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all, collect_dynamic_libs
from pathlib import Path

block_cipher = None

# 项目根目录
import os
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(sys.argv[0])))

# ============================================================
# 数据文件打包配置
# 注意: Windows 用 ; 分隔 src 和 dst, Linux/Mac 用 :
# ============================================================

datas_list = [
    # algorithm Python 源码（作为 package）
    (str(PROJECT_ROOT / 'algorithm'), 'algorithm'),
    # 背景图片
    (str(PROJECT_ROOT / 'algorithm/data/imgs/background'), 'algorithm/data/imgs/background'),
    # 天气图标
    (str(PROJECT_ROOT / 'algorithm/data/weather'), 'algorithm/data/weather'),
    # Excel 模板
    (str(PROJECT_ROOT / 'algorithm/data/调查数据表--模板.xlsx'), 'algorithm/data'),
    (str(PROJECT_ROOT / 'algorithm/data/点位批次基础信息表.xlsx'), 'algorithm/data'),
    # 配置文件
    (str(PROJECT_ROOT / 'algorithm/data/config.ini'), 'algorithm/data'),
    # 和风天气 JWT 私钥
    (str(PROJECT_ROOT / 'algorithm/ed25519-private.pem'), 'algorithm'),
    # 模型文件
    (str(PROJECT_ROOT / 'algorithm/models'), 'algorithm/models'),
    # pyinstaller_utils 打包工具（确保能被找到）
    (str(PROJECT_ROOT / 'pyinstaller_utils.py'), '.'),
]

# ============================================================
# 隐藏导入（无法被 PyInstaller 自动检测的模块）
# ============================================================

hiddenimports_list = [
    # PyQt5 核心模块
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.sip',

    # Matplotlib 相关
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_agg',
    'matplotlib.backends._backend_agg',
    'matplotlib.font_manager',
    'matplotlib.ft2font',
    'matplotlib.offsetbox',

    # 数据处理
    'pandas',
    'openpyxl',
    'numpy',
    'PIL',
    'PIL.Image',
    'PIL._Image',
    'xml.etree.ElementTree',
    'xmlrpc.client',

    # 机器学习
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.serialization',
    'sklearn',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors._typedefs',
    'xgboost',
    'xgboost.core',
    'joblib',
    'joblib.externals.cloudpickle',
    'joblib.externals.loky',

    # 加密/认证
    'cryptography',
    'cryptography.x509',
    'jwt',
    'PyJWT',
    'PyJWT.cryptography',

    # API 相关
    'requests',
    'urllib3',
    'certifi',
    'pyparsing',

    # 日期时间
    'datetime',
    'dateutil',
    'dateutil.parser',

    # SQLite
    'sqlite3',
    'sqlite3.dbapi2',

    # 其他工具
    'json',
    'pathlib',
    'importlib',
    'importlib.util',
    'pkgutil',
    'dis',
    'inspect',
    'traceback',
    'typing',

    # scipy（sklearn 依赖）
    'scipy',
    'scipy.spatial',
    'scipy.spatial.distance',

    # threadpoolctl（sklearn 依赖）
    'threadpoolctl',

    # 算法模块中的延迟导入
    'a_config',
    'b_data_cleaning',
    'c_feature_engineering',
    'd_model_training_testing',
    'g_qweather_client',
    'h_qweather_api',
    'i_online_prediction_preparation',
    'j_online_rolling_forecast',
    'k_weather_data_storage',
    'l_history_padding_and_prediction_runner',
    'm_observation_excel_reader',
    'n_online_prediction_service',
    'o_observation_import_service',
    'p_site_batch_excel_reader',
    'q_site_batch_import_service',
    'prediction',
    'gen_jwt',

    # 算法模块完整收集（确保所有内部引用都能找到）
    'algorithm',

    # UI 模块
    'ui_adapter.adapter',
    'ui_adapter.data_management.window',
    'ui_adapter.data_management.base',
    'ui_adapter.data_management.site_module',
    'ui_adapter.data_management.batch_module',
    'ui_adapter.data_management.staleness_module',
    'ui_adapter.data_management.dialogs.site_edit_dialog',
    'ui_adapter.data_management.dialogs.batch_edit_dialog',
    'pyinstaller_utils',
]

# 收集算法模块的所有数据文件和子模块
datas_list += collect_data_files('algorithm')
hiddenimports_list += collect_submodules('algorithm')

# ============================================================
# 构建 Analysis
# ============================================================

a = Analysis(
    ['main.py'],
    pathex=[str(PROJECT_ROOT)],
    binaries=collect_dynamic_libs('torch'),
    datas=datas_list,
    hiddenimports=hiddenimports_list,
    win_private_assemblies=False,
    cipher=block_cipher,
    exclude_binaries=False,
    name='玉米病害预测软件',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulator=False,
    env={
        'KMP_DUPLICATE_LIB_OK': 'TRUE',
        'OMP_NUM_THREADS': '1',
    },
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ============================================================
# 构建 PYZ（压缩所有 Python 字节码）
# ============================================================

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ============================================================
# 构建 EXE（最终的可执行文件）
# ============================================================

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='玉米病害预测软件',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=True,  # GUI 程序，不显示控制台
    disable_windowed_traceback=False,
    argv_emulator=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)