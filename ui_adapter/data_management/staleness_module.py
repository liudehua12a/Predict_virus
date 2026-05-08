from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QGroupBox, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt
from .base import ManagementModule


class DataStalenessModule(ManagementModule):
    MODULE_NAME = "数据时效"
    MODULE_ID = "staleness"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # 说明文字
        desc_label = QLabel(
            "数据时效用于控制预测启动时对历史数据新鲜度的要求。\n"
            "当最近一次预测记录距今超过设定天数时，系统将拒绝启动预测并提示更新数据。"
        )
        desc_label.setStyleSheet("color: #64748B; font-size: 13px; line-height: 1.6;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # 设置卡片
        card = QGroupBox("时效阈值设置")
        card.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #334155;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 16px;
                margin-top: 8px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
        """)
        card_layout = QVBoxLayout(card)

        row = QHBoxLayout()
        row.setSpacing(12)

        row.addWidget(QLabel("允许的最大时间差:"))

        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(["3天", "7天"])
        self.threshold_combo.setStyleSheet("""
            QComboBox {
                padding: 6px 12px;
                font-size: 14px;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                background: white;
                min-width: 100px;
            }
            QComboBox:hover {
                border-color: #4CAF50;
            }
        """)
        self.threshold_combo.currentTextChanged.connect(self._on_threshold_changed)
        row.addWidget(self.threshold_combo)

        row.addWidget(QLabel("（超过此天数则提示数据过久）"))
        row.addStretch()

        card_layout.addLayout(row)
        layout.addWidget(card)

        # 当前状态
        status_card = QGroupBox("当前配置")
        status_card.setStyleSheet(card.styleSheet())
        status_layout = QVBoxLayout(status_card)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-size: 14px; color: #334155;")
        status_layout.addWidget(self.status_label)

        layout.addWidget(status_card)
        layout.addStretch()

    def refresh(self):
        from algorithm.k_weather_data_storage import get_data_staleness_threshold
        current = get_data_staleness_threshold()
        self.threshold_combo.blockSignals(True)
        self.threshold_combo.setCurrentText("3天" if current == 3 else "7天")
        self.threshold_combo.blockSignals(False)
        self.status_label.setText(
            f"当前配置：数据时效阈值为 {current} 天"
        )

    def _on_threshold_changed(self, text):
        value = 3 if text == "3天" else 7
        from algorithm.k_weather_data_storage import set_data_staleness_threshold
        set_data_staleness_threshold(value)
        self.status_label.setText(f"当前配置：数据时效阈值为 {value} 天")