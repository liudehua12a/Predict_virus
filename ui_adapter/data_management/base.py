from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal


class ManagementModule(QWidget):
    """
    所有管理模块的基类/接口。
    每个模块实现 refresh() 方法即可接入 DataManagementWindow。
    """

    MODULE_NAME: str = ""  # 左侧 sidebar 显示名称
    MODULE_ID: str = ""    # 模块唯一标识

    # 跨模块数据变更广播信号，参数为 (module_id, record_id, action)
    data_changed = pyqtSignal(str, int, str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def refresh(self):
        """模块被激活时调用，用于刷新数据"""
        raise NotImplementedError