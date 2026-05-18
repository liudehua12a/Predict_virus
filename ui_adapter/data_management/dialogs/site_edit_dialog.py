from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QDoubleSpinBox, QPushButton, QLabel
)
from PyQt5.QtCore import Qt
import sqlite3
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import pyinstaller_utils as pkgutil

DB_PATH = pkgutil.get_db_path()


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
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(str(DB_PATH))
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