from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QPushButton, QHeaderView,
    QMessageBox, QListWidget, QListWidgetItem, QLabel
)
from PyQt5.QtCore import Qt
import sqlite3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import pyinstaller_utils as pkgutil
from .base import ManagementModule
from .dialogs.site_edit_dialog import SiteEditDialog
from .dialogs.batch_edit_dialog import BatchEditDialog

DB_PATH = pkgutil.get_db_path()


class BatchManagementModule(ManagementModule):
    MODULE_NAME = "批次管理"
    MODULE_ID = "batch"

    COLUMNS = ["批次名称", "批次编码", "作物品种", "播种日期", "调查开始", "调查结束", "操作"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_site_id = None
        self._site_rows = []
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
        add_site_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; border: none; padding: 6px; border-radius: 4px;")
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
        add_batch_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; border: none; padding: 6px 12px; border-radius: 4px;")
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

        # 【修改点 1】第一列自动拉伸，最后一列操作列根据内容大小调整
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        # 将第6列设为固定模式 Fixed
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Fixed)
        # 现在这行代码可以完美生效了，120px 足够放下两个最小宽度 45 的按钮 + 间距
        self.table.setColumnWidth(6, 150)
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
                            SELECT site_id, site_name
                            FROM site_info
                            WHERE is_active = 1
                            ORDER BY site_id ASC
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
                            SELECT batch_id,
                                   batch_name,
                                   batch_code,
                                   crop_variety,
                                   sowing_date,
                                   survey_start_date,
                                   survey_end_date
                            FROM survey_batch
                            WHERE site_id = ?
                              AND is_active = 1
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

            # 【修改点 2】优化按钮布局
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(1, 2, 1, 2)  # 增加内边距
            btn_layout.setSpacing(6)  # 增加间距
            btn_layout.setAlignment(Qt.AlignCenter)  # 居中对齐

            # 编辑按钮
            edit_btn = QPushButton("编辑")
            edit_btn.setMinimumWidth(45)
            edit_btn.setStyleSheet("""
                QPushButton {
                    padding: 4px 1px; 
                    font-size: 12px; 
                    border: 1px solid #CBD5E1; 
                    border-radius: 4px;
                    background-color: white;
                    color: #334155;
                }
                QPushButton:hover { background-color: #F8FAFC; }
            """)
            edit_btn.clicked.connect(lambda _, r=dict(row), s=self.site_label.text(): self._on_edit(r, s))

            # 删除按钮
            del_btn = QPushButton("删除")
            del_btn.setMinimumWidth(45)
            del_btn.setStyleSheet("""
                QPushButton {
                    padding: 4px 1px; 
                    font-size: 12px; 
                    border: 1px solid #CBD5E1; 
                    border-radius: 4px;
                    background-color: white;
                    color: #EF4444;
                }
                QPushButton:hover { background-color: #FEF2F2; }
            """)
            del_btn.clicked.connect(lambda _, r=dict(row): self._on_delete(r))

            btn_layout.addWidget(edit_btn)
            btn_layout.addWidget(del_btn)
            # 取消掉原有的 btn_layout.addStretch()

            self.table.setCellWidget(i, 6, btn_widget)
            self.table.setRowHeight(i, 40)  # 将行高从36调大到40，让按钮显示更舒适

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
            from datetime import datetime
            conn = sqlite3.connect(str(DB_PATH))
            conn.execute(
                "UPDATE survey_batch SET is_active = 0, updated_at = ? WHERE batch_id = ?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row["batch_id"])
            )
            conn.commit()
            conn.close()
            self._load_batches(self._current_site_id)
            self.data_changed.emit(self.MODULE_ID, row["batch_id"], "delete")