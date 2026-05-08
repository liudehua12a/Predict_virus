# 数据管理模块实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 PyQt5 主界面新增"数据管理"按钮，弹出独立窗口，实现站点和批次的完整 CRUD，采用模块化插件架构，支持未来功能扩展。

**Architecture:** 采用模块注册机制，`DataManagementWindow` 作为弹窗容器，左侧 sidebar 导航，右侧 `StackedWidget` 承载各模块页面。每个管理对象（站点、批次）实现为独立的 `ManagementModule` 子类，通过 `register_module()` 注册。

**Tech Stack:** PyQt5 (已有) / SQLite / Python 3

---

## 文件结构

```
ui_adapter/
├── adapter.py                  # 现有代码（不动）
└── data_management/
    ├── __init__.py             # 空文件，package 标识
    ├── base.py                 # ManagementModule 基类
    ├── window.py               # DataManagementWindow + register_module
    ├── site_module.py          # SiteManagementModule
    ├── batch_module.py         # BatchManagementModule
    └── dialogs/
        ├── __init__.py         # 空文件
        ├── site_edit_dialog.py # SiteEditDialog
        └── batch_edit_dialog.py# BatchEditDialog
```

**主应用改动:**
- `main.py`: 新增"数据管理"按钮，绑定打开弹窗

---

## Task 1: 创建目录结构和空 package 文件

**Files:**
- Create: `ui_adapter/data_management/__init__.py`
- Create: `ui_adapter/data_management/dialogs/__init__.py`

- [ ] **Step 1: 创建 data_management 目录和 package 文件**

```bash
mkdir -p /Users/newly/pyproject/预测软件-1.3/ui_adapter/data_management/dialogs
touch /Users/newly/pyproject/预测软件-1.3/ui_adapter/data_management/__init__.py
touch /Users/newly/pyproject/预测软件-1.3/ui_adapter/data_management/dialogs/__init__.py
```

- [ ] **Step 2: 提交**

```bash
git add ui_adapter/data_management/__init__.py ui_adapter/data_management/dialogs/__init__.py
git commit -m "feat(data-management): 创建目录结构和空 package 文件"
```

---

## Task 2: 编写 ManagementModule 基类 (base.py)

**Files:**
- Create: `ui_adapter/data_management/base.py`

```python
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
```

- [ ] **Step 1: 编写 base.py**

```python
# 文件: ui_adapter/data_management/base.py
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
```

- [ ] **Step 2: 提交**

```bash
git add ui_adapter/data_management/base.py
git commit -m "feat(data-management): 添加 ManagementModule 基类"
```

---

## Task 3: 编写 SiteEditDialog (站点编辑弹窗)

**Files:**
- Create: `ui_adapter/data_management/dialogs/site_edit_dialog.py`

```python
# ui_adapter/data_management/dialogs/site_edit_dialog.py
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QDoubleSpinBox, QPushButton, QLabel
)
from PyQt5.QtCore import Qt
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = ROOT_DIR / "algorithm" / "data" / "nky-CornPre.db"


class SiteEditDialog(QDialog):
    def __init__(self, parent=None, site_row: dict = None):
        super().__init__(parent)
        self.site_row = site_row or {}
        self.is_edit = bool(site_row)
        self.setWindowTitle("编辑站点" if self.is_edit else "新增站点")
        self.setMinimumWidth(480)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        form = QFormLayout()
        form.setSpacing(12)

        self.name_edit = QLineEdit()
        self.name_edit.setText(self.site_row.get("site_name", ""))
        self.province_edit = QLineEdit()
        self.province_edit.setText(self.site_row.get("province", ""))
        self.city_edit = QLineEdit()
        self.city_edit.setText(self.site_row.get("city", ""))

        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-90.0, 90.0)
        self.lat_spin.setDecimals(6)
        self.lat_spin.setValue(float(self.site_row.get("lat", 0.0)))

        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180.0, 180.0)
        self.lon_spin.setDecimals(6)
        self.lon_spin.setValue(float(self.site_row.get("lon", 0.0)))

        self.elevation_spin = QDoubleSpinBox()
        self.elevation_spin.setRange(-1000.0, 10000.0)
        self.elevation_spin.setDecimals(2)
        self.elevation_spin.setValue(float(self.site_row.get("elevation", 0.0) or 0.0))

        form.addRow("站点名称 *", self.name_edit)
        form.addRow("省份", self.province_edit)
        form.addRow("城市", self.city_edit)
        form.addRow("纬度 (-90~90)", self.lat_spin)
        form.addRow("经度 (-180~180)", self.lon_spin)
        form.addRow("海拔 (m)", self.elevation_spin)

        layout.addLayout(form)

        # 按钮行
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _on_save(self):
        name = self.name_edit.text().strip()
        if not name:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "校验失败", "站点名称为必填项")
            return

        lat = self.lat_spin.value()
        lon = self.lon_spin.value()

        if not (-90 <= lat <= 90):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "校验失败", "纬度范围 -90~90")
            return
        if not (-180 <= lon <= 180):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "校验失败", "经度范围 -180~180")
            return

        self._save_to_db(name)
        self.accept()

    def _save_to_db(self, name):
        from datetime import datetime
        conn = sqlite3.connect(str(DB_PATH))
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.is_edit:
            sql = """
            UPDATE site_info SET
                site_name = ?,
                province = ?,
                city = ?,
                lat = ?,
                lon = ?,
                elevation = ?,
                updated_at = ?
            WHERE site_id = ?
            """
            conn.execute(sql, (
                name,
                self.province_edit.text().strip() or None,
                self.city_edit.text().strip() or None,
                self.lat_spin.value(),
                self.lon_spin.value(),
                self.elevation_spin.value(),
                now_str,
                self.site_row["site_id"],
            ))
        else:
            sql = """
            INSERT INTO site_info (
                site_name, province, city, lat, lon, elevation,
                is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            """
            conn.execute(sql, (
                name,
                self.province_edit.text().strip() or None,
                self.city_edit.text().strip() or None,
                self.lat_spin.value(),
                self.lon_spin.value(),
                self.elevation_spin.value(),
                now_str,
                now_str,
            ))
        conn.commit()
        conn.close()
```

- [ ] **Step 1: 编写 site_edit_dialog.py**

（代码同上，写入 `ui_adapter/data_management/dialogs/site_edit_dialog.py`）

- [ ] **Step 2: 提交**

```bash
git add ui_adapter/data_management/dialogs/site_edit_dialog.py
git commit -m "feat(data-management): 添加 SiteEditDialog 站点编辑弹窗"
```

---

## Task 4: 编写 BatchEditDialog (批次编辑弹窗)

**Files:**
- Create: `ui_adapter/data_management/dialogs/batch_edit_dialog.py`

```python
# ui_adapter/data_management/dialogs/batch_edit_dialog.py
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QDateEdit, QPushButton, QLabel, QComboBox
)
from PyQt5.QtCore import Qt, QDate
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = ROOT_DIR / "algorithm" / "data" / "nky-CornPre.db"


class BatchEditDialog(QDialog):
    def __init__(self, parent=None, batch_row: dict = None, site_id: int = None, site_name: str = ""):
        super().__init__(parent)
        self.batch_row = batch_row or {}
        self.site_id = site_id or batch_row.get("site_id")
        self.site_name = site_name
        self.is_edit = bool(batch_row)
        self.setWindowTitle("编辑批次" if self.is_edit else "新增批次")
        self.setMinimumWidth(480)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        form = QFormLayout()
        form.setSpacing(12)

        # 所属站点（只读）
        site_label = QLabel(self.site_name + f" (site_id={self.site_id})")
        site_label.setStyleSheet("color: #64748B;")
        form.addRow("所属站点", site_label)

        self.name_edit = QLineEdit()
        self.name_edit.setText(self.batch_row.get("batch_name", ""))
        self.code_edit = QLineEdit()
        self.code_edit.setText(self.batch_row.get("batch_code", ""))
        self.variety_edit = QLineEdit()
        self.variety_edit.setText(self.batch_row.get("crop_variety", ""))

        self.sowing_date_edit = QDateEdit()
        self.sowing_date_edit.setCalendarPopup(True)
        self.sowing_date_edit.setDisplayFormat("yyyy-MM-dd")
        if self.batch_row.get("sowing_date"):
            self.sowing_date_edit.setDate(QDate.fromString(self.batch_row["sowing_date"], "yyyy-MM-dd"))
        else:
            self.sowing_date_edit.setDate(QDate.currentDate())

        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        if self.batch_row.get("survey_start_date"):
            self.start_date_edit.setDate(QDate.fromString(self.batch_row["survey_start_date"], "yyyy-MM-dd"))

        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        if self.batch_row.get("survey_end_date"):
            self.end_date_edit.setDate(QDate.fromString(self.batch_row["survey_end_date"], "yyyy-MM-dd"))

        form.addRow("批次名称 *", self.name_edit)
        form.addRow("批次编码", self.code_edit)
        form.addRow("作物品种", self.variety_edit)
        form.addRow("播种日期", self.sowing_date_edit)
        form.addRow("调查开始", self.start_date_edit)
        form.addRow("调查结束", self.end_date_edit)

        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _on_save(self):
        name = self.name_edit.text().strip()
        if not name:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "校验失败", "批次名称为必填项")
            return
        self._save_to_db(name)
        self.accept()

    def _save_to_db(self, name):
        from datetime import datetime
        conn = sqlite3.connect(str(DB_PATH))
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _dt(edit):
            return edit.date().toString("yyyy-MM-dd")

        if self.is_edit:
            sql = """
            UPDATE survey_batch SET
                batch_name = ?,
                batch_code = ?,
                crop_variety = ?,
                sowing_date = ?,
                survey_start_date = ?,
                survey_end_date = ?,
                updated_at = ?
            WHERE batch_id = ?
            """
            conn.execute(sql, (
                name,
                self.code_edit.text().strip() or None,
                self.variety_edit.text().strip() or None,
                _dt(self.sowing_date_edit),
                _dt(self.start_date_edit) if self.start_date_edit.date().toString("yyyy-MM-dd") else None,
                _dt(self.end_date_edit) if self.end_date_edit.date().toString("yyyy-MM-dd") else None,
                now_str,
                self.batch_row["batch_id"],
            ))
        else:
            sql = """
            INSERT INTO survey_batch (
                site_id, batch_name, batch_code, crop_variety,
                sowing_date, survey_start_date, survey_end_date,
                is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """
            conn.execute(sql, (
                self.site_id,
                name,
                self.code_edit.text().strip() or None,
                self.variety_edit.text().strip() or None,
                _dt(self.sowing_date_edit),
                _dt(self.start_date_edit) if self.start_date_edit.date().toString("yyyy-MM-dd") else None,
                _dt(self.end_date_edit) if self.end_date_edit.date().toString("yyyy-MM-dd") else None,
                now_str,
                now_str,
            ))
        conn.commit()
        conn.close()
```

- [ ] **Step 1: 编写 batch_edit_dialog.py**

（代码同上，写入 `ui_adapter/data_management/dialogs/batch_edit_dialog.py`）

- [ ] **Step 2: 提交**

```bash
git add ui_adapter/data_management/dialogs/batch_edit_dialog.py
git commit -m "feat(data-management): 添加 BatchEditDialog 批次编辑弹窗"
```

---

## Task 5: 编写 DataManagementWindow 主弹窗 + 模块注册机制

**Files:**
- Create: `ui_adapter/data_management/window.py`

核心逻辑：
- `LeftSidebar` (QWidget, 160px宽): 放 `QListWidget` 显示已注册模块名称，信号 `currentRowChanged` 切换右侧 StackedWidget 页面
- `register_module(module)`: 将模块加入 `_modules` 字典，左侧 sidebar 加一项，右侧 StackedWidget 加一个 page，激活第一个注册的模块
- 窗口大小: `1000 x 600`

```python
# ui_adapter/data_management/window.py
from PyQt5.QtWidgets import (
    QDialog, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QStackedWidget, QLabel
)
from PyQt5.QtCore import Qt


class DataManagementWindow(QDialog):
    """
    数据管理弹窗。
    通过 register_module() 注册管理模块，左侧 sidebar 切换，
    右侧 StackedWidget 承载各模块页面。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据管理")
        self.setMinimumSize(1000, 600)
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

        if len(self._modules) == 1:
            self.module_list.setCurrentRow(0)

    def _on_module_changed(self, row: int):
        if row < 0:
            return
        self.content_stack.setCurrentIndex(row)
        module_id = self._module_list[row]
        self._modules[module_id].refresh()
```

- [ ] **Step 1: 编写 window.py**

（代码同上，写入 `ui_adapter/data_management/window.py`）

- [ ] **Step 2: 提交**

```bash
git add ui_adapter/data_management/window.py
git commit -m "feat(data-management): 添加 DataManagementWindow 主弹窗和模块注册机制"
```

---

## Task 6: 编写 SiteManagementModule (站点管理模块)

**Files:**
- Create: `ui_adapter/data_management/site_module.py`

功能：
- 上方案具栏：`QLineEdit` 搜索框（按 site_name 模糊）、`+ 新增站点`按钮
- 下方 `QTableWidget`：列 = site_id(隐藏)、站点名称、省份、城市、纬度、经度、海拔、操作(编辑/删除)
- 搜索实时过滤表格行
- 编辑/删除按钮在操作列，点击弹出 `SiteEditDialog` 或确认删除
- 删除：二次确认 QMessageBox，然后 UPDATE is_active = 0
- `refresh()` 从数据库重新加载所有 is_active=1 的站点

```python
# ui_adapter/data_management/site_module.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QPushButton, QLineEdit, QHeaderView,
    QMessageBox
)
from PyQt5.QtCore import Qt
import sqlite3
from pathlib import Path
from .base import ManagementModule
from .dialogs.site_edit_dialog import SiteEditDialog

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT_DIR / "algorithm" / "data" / "nky-CornPre.db"


class SiteManagementModule(ManagementModule):
    MODULE_NAME = "站点管理"
    MODULE_ID = "site"

    COLUMNS = ["站点名称", "省份", "城市", "纬度", "经度", "海拔", "操作"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # 工具栏
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("🔍 搜索站点名称...")
        self.search_edit.textChanged.connect(self._on_search)
        toolbar.addWidget(self.search_edit)

        add_btn = QPushButton("➕ 新增站点")
        add_btn.setStyleSheet("background-color: #4CAF50; color: white; border: none; padding: 6px 12px; border-radius: 4px;")
        add_btn.clicked.connect(self._on_add)
        toolbar.addWidget(add_btn)
        toolbar.addStretch()

        layout.addLayout(toolbar)

        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #F1F5F9;
                font-weight: bold;
                color: #334155;
            }
        """)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table)

    def refresh(self):
        self._load_data()

    def _load_data(self):
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT site_id, site_name, province, city, lat, lon, elevation
            FROM site_info
            WHERE is_active = 1
            ORDER BY site_id ASC
        """).fetchall()
        conn.close()

        self._all_rows = [dict(r) for r in rows]
        self._apply_filter()

    def _apply_filter(self):
        keyword = self.search_edit.text().strip().lower()
        filtered = [
            r for r in self._all_rows
            if keyword in r["site_name"].lower()
        ]
        self._display_rows(filtered)

    def _display_rows(self, rows):
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(row["site_name"] or ""))
            self.table.setItem(i, 1, QTableWidgetItem(row["province"] or ""))
            self.table.setItem(i, 2, QTableWidgetItem(row["city"] or ""))
            self.table.setItem(i, 3, QTableWidgetItem(str(row["lat"] or "")))
            self.table.setItem(i, 4, QTableWidgetItem(str(row["lon"] or "")))
            self.table.setItem(i, 5, QTableWidgetItem(str(row["elevation"] or "")))

            # 操作按钮
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(2, 2, 2, 2)
            btn_layout.setSpacing(4)

            edit_btn = QPushButton("✏️")
            edit_btn.setFixedWidth(40)
            edit_btn.clicked.connect(lambda _, r=row: self._on_edit(r))

            del_btn = QPushButton("🗑️")
            del_btn.setFixedWidth(40)
            del_btn.clicked.connect(lambda _, r=row: self._on_delete(r))

            btn_layout.addWidget(edit_btn)
            btn_layout.addWidget(del_btn)

            self.table.setCellWidget(i, 6, btn_widget)
            self.table.setRowHeight(i, 36)

    def _on_search(self):
        self._apply_filter()

    def _on_add(self):
        dlg = SiteEditDialog(self)
        if dlg.exec_():
            self.refresh()
            self.data_changed.emit(self.MODULE_ID, -1, "add")

    def _on_edit(self, row):
        dlg = SiteEditDialog(self, site_row=row)
        if dlg.exec_():
            self.refresh()
            self.data_changed.emit(self.MODULE_ID, row["site_id"], "update")

    def _on_delete(self, row):
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除站点「{row['site_name']}」吗？\n删除后将无法在预测中使用，但仍保留历史数据。",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            conn = sqlite3.connect(str(DB_PATH))
            from datetime import datetime
            conn.execute(
                "UPDATE site_info SET is_active = 0, updated_at = ? WHERE site_id = ?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row["site_id"])
            )
            conn.commit()
            conn.close()
            self.refresh()
            self.data_changed.emit(self.MODULE_ID, row["site_id"], "delete")
```

- [ ] **Step 1: 编写 site_module.py**

（代码同上，写入 `ui_adapter/data_management/site_module.py`）

- [ ] **Step 2: 提交**

```bash
git add ui_adapter/data_management/site_module.py
git commit -m "feat(data-management): 添加 SiteManagementModule 站点管理模块"
```

---

## Task 7: 编写 BatchManagementModule (批次管理模块)

**Files:**
- Create: `ui_adapter/data_management/batch_module.py`

功能：
- 左侧 `QListWidget` 显示所有 is_active=1 的站点，点击选中
- 右侧上方 `+ 新增批次` 按钮（顶部 toolbar）
- 右侧下方 `QTableWidget`：列 = batch_id(隐藏)、批次名称、批次编码、作物品种、播种日期、调查开始、调查结束、操作(编辑/删除)
- 选中站点变化时，从数据库加载该 site_id 下的所有 is_active=1 批次
- 站点列表顶部有一个 `+` 按钮，调用 `SiteEditDialog` 新增站点
- 左侧站点选中变化时刷新右侧批次表格
- `refresh()` 重新加载站点列表，若有上次选中的 site_id 则恢复选中

```python
# ui_adapter/data_management/batch_module.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QPushButton, QHeaderView,
    QMessageBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt
import sqlite3
from pathlib import Path
from .base import ManagementModule
from .dialogs.site_edit_dialog import SiteEditDialog
from .dialogs.batch_edit_dialog import BatchEditDialog

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT_DIR / "algorithm" / "data" / "nky-CornPre.db"


class BatchManagementModule(ManagementModule):
    MODULE_NAME = "批次管理"
    MODULE_ID = "batch"

    COLUMNS = ["批次名称", "批次编码", "作物品种", "播种日期", "调查开始", "调查结束", "操作"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_site_id = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # 左侧：站点列表
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        add_site_btn = QPushButton("➕ 新增站点")
        add_site_btn.setStyleSheet("background-color: #4CAF50; color: white; border: none; padding: 6px; border-radius: 4px;")
        add_site_btn.clicked.connect(self._on_add_site)
        left_layout.addWidget(add_site_btn)

        self.site_list = QListWidget()
        self.site_list.setObjectName("siteList")
        self.site_list.currentRowChanged.connect(self._on_site_selected)
        self.site_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                background: white;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #F1F5F9;
            }
            QListWidget::item:selected {
                background-color: #E8F5E9;
                color: #1B5E20;
            }
        """)
        left_layout.addWidget(self.site_list)

        # 右侧：批次表格
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.site_label = QLabel("请选择站点")
        self.site_label.setStyleSheet("font-weight: bold; color: #334155;")
        toolbar.addWidget(self.site_label)
        toolbar.addStretch()

        add_batch_btn = QPushButton("➕ 新增批次")
        add_batch_btn.setStyleSheet("background-color: #4CAF50; color: white; border: none; padding: 6px 12px; border-radius: 4px;")
        add_batch_btn.clicked.connect(self._on_add_batch)
        toolbar.addWidget(add_batch_btn)

        right_layout.addLayout(toolbar)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #F1F5F9;
                font-weight: bold;
                color: #334155;
            }
        """)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        right_layout.addWidget(self.table)

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 3)

    def refresh(self):
        self._load_sites()

    def _load_sites(self):
        self.site_list.clear()
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT site_id, site_name FROM site_info
            WHERE is_active = 1 ORDER BY site_id ASC
        """).fetchall()
        conn.close()

        self._site_rows = [dict(r) for r in rows]
        for row in self._site_rows:
            item = QListWidgetItem(row["site_name"])
            item.setData(Qt.UserRole, row["site_id"])
            self.site_list.addItem(item)

        if self._site_rows:
            restored = False
            if self._current_site_id is not None:
                for i, row in enumerate(self._site_rows):
                    if row["site_id"] == self._current_site_id:
                        self.site_list.setCurrentRow(i)
                        restored = True
                        break
            if not restored:
                self.site_list.setCurrentRow(0)
        else:
            self._current_site_id = None
            self.site_label.setText("无站点，请先新增")

    def _on_site_selected(self, row: int):
        if row < 0 or not self._site_rows:
            self.table.setRowCount(0)
            return
        site = self._site_rows[row]
        self._current_site_id = site["site_id"]
        self.site_label.setText(site["site_name"])
        self._load_batches(site["site_id"])

    def _load_batches(self, site_id: int):
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT batch_id, batch_name, batch_code, crop_variety,
                   sowing_date, survey_start_date, survey_end_date
            FROM survey_batch
            WHERE site_id = ? AND is_active = 1
            ORDER BY batch_id ASC
        """, (site_id,)).fetchall()
        conn.close()

        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(row["batch_name"] or ""))
            self.table.setItem(i, 1, QTableWidgetItem(row["batch_code"] or ""))
            self.table.setItem(i, 2, QTableWidgetItem(row["crop_variety"] or ""))
            self.table.setItem(i, 3, QTableWidgetItem(row["sowing_date"] or ""))
            self.table.setItem(i, 4, QTableWidgetItem(row["survey_start_date"] or ""))
            self.table.setItem(i, 5, QTableWidgetItem(row["survey_end_date"] or ""))

            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(2, 2, 2, 2)
            btn_layout.setSpacing(4)

            edit_btn = QPushButton("✏️")
            edit_btn.setFixedWidth(40)
            edit_btn.clicked.connect(lambda _, r=dict(row), s=site["site_name"]: self._on_edit(r, s))

            del_btn = QPushButton("🗑️")
            del_btn.setFixedWidth(40)
            del_btn.clicked.connect(lambda _, r=dict(row): self._on_delete(r))

            btn_layout.addWidget(edit_btn)
            btn_layout.addWidget(del_btn)
            self.table.setCellWidget(i, 6, btn_widget)
            self.table.setRowHeight(i, 36)

    def _on_add_site(self):
        dlg = SiteEditDialog(self)
        if dlg.exec_():
            self.refresh()
            self.data_changed.emit(self.MODULE_ID, -1, "add")

    def _on_add_batch(self):
        if self._current_site_id is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "提示", "请先选择一个站点")
            return
        site = next(s for s in self._site_rows if s["site_id"] == self._current_site_id)
        dlg = BatchEditDialog(self, site_id=self._current_site_id, site_name=site["site_name"])
        if dlg.exec_():
            self._load_batches(self._current_site_id)
            self.data_changed.emit(self.MODULE_ID, -1, "add")

    def _on_edit(self, row, site_name):
        dlg = BatchEditDialog(self, batch_row=row, site_id=row["site_id"], site_name=site_name)
        if dlg.exec_():
            self._load_batches(self._current_site_id)
            self.data_changed.emit(self.MODULE_ID, row["batch_id"], "update")

    def _on_delete(self, row):
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除批次「{row['batch_name']}」吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            conn = sqlite3.connect(str(DB_PATH))
            from datetime import datetime
            conn.execute(
                "UPDATE survey_batch SET is_active = 0, updated_at = ? WHERE batch_id = ?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row["batch_id"])
            )
            conn.commit()
            conn.close()
            self._load_batches(self._current_site_id)
            self.data_changed.emit(self.MODULE_ID, row["batch_id"], "delete")
```

- [ ] **Step 1: 编写 batch_module.py**

（代码同上，写入 `ui_adapter/data_management/batch_module.py`）

- [ ] **Step 2: 提交**

```bash
git add ui_adapter/data_management/batch_module.py
git commit -m "feat(data-management): 添加 BatchManagementModule 批次管理模块"
```

---

## Task 8: 集成到主界面 (main.py 新增按钮)

**Files:**
- Modify: `main.py:489-509` (在顶部控制面板的工具栏区域)

在 `top_layout` 中的"导入Excel"按钮后面，新增一个"数据管理"按钮，点击弹出 `DataManagementWindow`，并注册 `SiteManagementModule` 和 `BatchManagementModule`。

```python
# main.py import 区域加入:
from ui_adapter.data_management.window import DataManagementWindow
from ui_adapter.data_management.site_module import SiteManagementModule
from ui_adapter.data_management.batch_module import BatchManagementModule
```

在 `init_ui` 方法的顶部工具栏区域，在 `import_btn` 后面加入：

```python
# 数据管理按钮
self.data_mgmt_btn = QPushButton('🗂️ 数据管理')
self.data_mgmt_btn.setStyleSheet("font-size:14px;")
self.data_mgmt_btn.setObjectName("dataMgmtBtn")
self.data_mgmt_btn.clicked.connect(self.open_data_management)
top_layout.addWidget(self.data_mgmt_btn)
```

新增方法：

```python
def open_data_management(self):
    window = DataManagementWindow(self)
    window.register_module(SiteManagementModule())
    window.register_module(BatchManagementModule())
    window.exec_()
```

- [ ] **Step 1: 修改 main.py 的 import 区域**，加入上述两个 import 语句

- [ ] **Step 2: 在 init_ui 的顶部工具栏**，在 `self.import_btn` 后加入数据管理按钮的创建代码

- [ ] **Step 3: 新增 open_data_management 方法**

- [ ] **Step 4: 提交**

```bash
git add main.py
git commit -m "feat(main): 集成数据管理弹窗入口"
```

---

## Task 9: 冒烟测试验证

- [ ] **Step 1: 启动应用**

```bash
cd /Users/newly/pyproject/预测软件-1.3 && python main.py
```

预期：PyQt5 窗口正常打开，顶部工具栏右侧有"数据管理"按钮。

- [ ] **Step 2: 点击"数据管理"按钮**

预期：弹出 1000x600 的新窗口，左侧 sidebar 有"站点管理"和"批次管理"两项。

- [ ] **Step 3: 点击"站点管理"**

预期：显示站点搜索框、新增按钮，下方表格列出所有 is_active=1 的站点。

- [ ] **Step 4: 点击"新增站点"**

预期：弹出站点编辑弹窗，填写后点保存，数据正确入库。

- [ ] **Step 5: 切换到"批次管理"**

预期：左侧列出站点，点击站点右侧刷新该站点下的批次。点"新增批次"可正常添加。

- [ ] **Step 6: 提交**

```bash
git commit -m "test: 冒烟测试验证数据管理功能"
```

---

## 自检清单

- [ ] spec 覆盖检查：每个 spec 章节都能找到对应任务
- [ ] 占位符扫描：所有代码块均为完整可运行代码，无 TBD/TODO
- [ ] 类型一致性：
  - `register_module()` 接收 `ManagementModule` 子类
  - `ManagementModule.MODULE_NAME` / `MODULE_ID` 在各子类中正确设置
  - `BatchEditDialog.__init__` 参数 `site_id` 和 `site_name` 与调用处一致
  - `SiteManagementModule.data_changed` 信号参数 `(str, int, str)` 与 `window.py` 广播处一致
