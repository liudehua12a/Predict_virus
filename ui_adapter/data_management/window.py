from PyQt5.QtWidgets import (
    QDialog, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QStackedWidget, QLabel
)
from PyQt5.QtCore import Qt, pyqtSignal


class DataManagementWindow(QDialog):
    data_updated = pyqtSignal()  # 数据变更时发射此信号
    """
    数据管理弹窗。
    通过 register_module() 注册管理模块，左侧 sidebar 切换，
    右侧 StackedWidget 承载各模块页面。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据管理")
        self.setMinimumSize(1200, 600)
        self._modules = {}  # module_id -> module_instance
        self._module_list = []  # 按注册顺序保存 module_id
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧导航
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(160)
        self.sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(8, 16, 8, 8)
        sidebar_layout.setSpacing(4)

        title = QLabel("数据管理")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1B5E20; padding: 4px;")
        sidebar_layout.addWidget(title)

        self.module_list = QListWidget()
        self.module_list.setObjectName("moduleList")
        self.module_list.currentRowChanged.connect(self._on_module_changed)
        sidebar_layout.addWidget(self.module_list)
        sidebar_layout.addStretch()

        # 右侧内容区
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.content_stack, 1)

        self.setStyleSheet("""
            QDialog {
                background-color: #F4F7F4;
            }
            #sidebar {
                background-color: #FFFFFF;
                border-right: 1px solid #E2E8F0;
            }
            #moduleList {
                border: none;
                background: transparent;
                font-size: 14px;
            }
            #moduleList::item {
                padding: 8px 12px;
                border-radius: 6px;
                color: #334155;
            }
            #moduleList::item:selected {
                background-color: #E8F5E9;
                color: #1B5E20;
                font-weight: bold;
            }
            #moduleList::item:hover {
                background-color: #F1F5F9;
            }
        """)

    def register_module(self, module: QWidget):
        """
        注册一个 ManagementModule 到窗口。
        左侧 sidebar 出现入口，右侧 StackedWidget 加入页面。
        """
        from .base import ManagementModule
        if not isinstance(module, ManagementModule):
            raise TypeError("module must be a ManagementModule")

        mid = module.MODULE_ID
        if mid in self._modules:
            return

        self._modules[mid] = module
        self._module_list.append(mid)
        self.module_list.addItem(module.MODULE_NAME)
        self.content_stack.addWidget(module)

        # 连接模块的 data_changed 信号到窗口的 data_updated 信号
        module.data_changed.connect(self.data_updated.emit)

        if len(self._modules) == 1:
            self.module_list.setCurrentRow(0)

    def _on_module_changed(self, row: int):
        if row < 0:
            return
        self.content_stack.setCurrentIndex(row)
        module_id = self._module_list[row]
        self._modules[module_id].refresh()