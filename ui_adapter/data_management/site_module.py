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
        self._all_rows = []
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
        add_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; border: none; padding: 6px 12px; border-radius: 4px;")
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

        # 【修改点 1】设置列宽调整策略
        # 站点名称列自动拉伸填满空白
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        # 操作列根据内容（按钮）自动调整宽度，防止被挤压
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(6, 120)  # 强制操作列最小宽度确保按钮完整显示



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

            # 【修改点 2】优化操作按钮的布局与样式
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(4, 2, 4, 2)  # 增加内边距
            btn_layout.setSpacing(8)  # 增加按钮之间的间距
            btn_layout.setAlignment(Qt.AlignCenter)  # 居中对齐，取代 addStretch()

            # 编辑按钮
            edit_btn = QPushButton("编辑")
            edit_btn.setMinimumWidth(45)  # 限制最小宽度
            edit_btn.setStyleSheet("""
                QPushButton {
                    padding: 4px 8px; 
                    font-size: 12px; 
                    border: 1px solid #CBD5E1; 
                    border-radius: 4px;
                    background-color: white;
                    color: #334155;
                }
                QPushButton:hover { background-color: #F8FAFC; }
            """)
            edit_btn.clicked.connect(lambda _, r=row: self._on_edit(r))

            # 删除按钮
            del_btn = QPushButton("删除")
            del_btn.setMinimumWidth(45)  # 限制最小宽度
            del_btn.setStyleSheet("""
                QPushButton {
                    padding: 4px 8px; 
                    font-size: 12px; 
                    border: 1px solid #CBD5E1; 
                    border-radius: 4px;
                    background-color: white;
                    color: #EF4444;
                }
                QPushButton:hover { background-color: #FEF2F2; }
            """)
            del_btn.clicked.connect(lambda _, r=row: self._on_delete(r))

            btn_layout.addWidget(edit_btn)
            btn_layout.addWidget(del_btn)
            # 注意：这里已经去掉了原来的 btn_layout.addStretch()

            self.table.setCellWidget(i, 6, btn_widget)
            self.table.setRowHeight(i, 40)  # 增加行高，让按钮显示更美观

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
            from datetime import datetime
            conn = sqlite3.connect(str(DB_PATH))
            conn.execute(
                "UPDATE site_info SET is_active = 0, updated_at = ? WHERE site_id = ?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row["site_id"])
            )
            conn.commit()
            conn.close()
            self.refresh()
            self.data_changed.emit(self.MODULE_ID, row["site_id"], "delete")