from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QDateEdit, QPushButton, QLabel
)
from PyQt5.QtCore import Qt, QDate
import sqlite3
from pathlib import Path
from datetime import datetime

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
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _dt(edit):
            ds = edit.date().toString("yyyy-MM-dd")
            return ds if ds else None

        conn = sqlite3.connect(str(DB_PATH))
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
                _dt(self.start_date_edit),
                _dt(self.end_date_edit),
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
                _dt(self.start_date_edit),
                _dt(self.end_date_edit),
                now_str,
                now_str,
            ))
        conn.commit()
        conn.close()