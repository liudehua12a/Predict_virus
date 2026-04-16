import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton, 
                             QFileDialog, QMessageBox, QGroupBox, QStackedWidget)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from ui_adapter import adapter


def ensure_qt_platform_plugin_path():
    """确保 Qt 平台插件路径可用（Windows 常见问题修复）"""
    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        return
    try:
        import PyQt5
        plugin_path = os.path.join(
            os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms"
        )
        if os.path.isdir(plugin_path):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
    except Exception:
        pass



# 步骤卡片类
class StepCard(QWidget):
    def __init__(self, number, icon, description, parent=None):
        super().__init__(parent)
        self.setObjectName("StepCard")
        self.setProperty("class", "StepCard")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # 步骤编号
        number_label = QLabel(str(number), self)
        number_label.setObjectName("StepNumber")
        number_label.setProperty("class", "StepNumber")
        number_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(number_label)
        
        # 步骤图标
        icon_label = QLabel(icon, self)
        icon_label.setObjectName("StepIcon")
        icon_label.setProperty("class", "StepIcon")
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        # 步骤描述
        desc_label = QLabel(description, self)
        desc_label.setObjectName("StepDescription")
        desc_label.setProperty("class", "StepDescription")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

# 欢迎引导页面类
class WelcomeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化引导页面"""
        # 设置布局
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(30)
        
        # 顶部图标和标题
        header_layout = QVBoxLayout()
        header_layout.setAlignment(Qt.AlignCenter)
        header_layout.setSpacing(15)
        
        # 玉米图标
        icon_label = QLabel('🌽', self)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 80px;")
        header_layout.addWidget(icon_label)
        
        # 欢迎标题
        title_label = QLabel('欢迎使用玉米病害智能预测软件', self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #1B5E20;
        """)
        header_layout.addWidget(title_label)
        
        layout.addLayout(header_layout)
        
        # 操作步骤提示 - 卡片式引导流
        steps_layout = QHBoxLayout()
        steps_layout.setAlignment(Qt.AlignCenter)
        steps_layout.setSpacing(20)
        
        # 步骤1
        step1 = StepCard(
            number="1",
            icon="📊",
            description="点击左上角【导入Excel数据】加载历史危害数据"
        )
        steps_layout.addWidget(step1)
        
        # 步骤2
        step2 = StepCard(
            number="2",
            icon="🔍",
            description="在下拉框选择对应的【预测模型】和【目标区域】"
        )
        steps_layout.addWidget(step2)
        
        # 步骤3
        step3 = StepCard(
            number="3",
            icon="▶",
            description="点击【开始预测】获取未来一周的疾病危害趋势图表"
        )
        steps_layout.addWidget(step3)
        
        layout.addLayout(steps_layout)
        
        # 底部版权信息
        footer_label = QLabel('© 2026 玉米病害预测软件', self)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("""
            font-size: 12px;
            color: #666666;
            margin-top: 20px;
        """)
        layout.addWidget(footer_label)
        
        # 设置样式
        self.setStyleSheet("""
            WelcomeWidget {
                background-color: #F5F7FA;
            }
        """)

# 主应用类
class DiseasePredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.site_batch_rows = []
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('玉米病害预测软件')
        # 设置合理的默认窗口大小和最小尺寸，确保界面元素清晰可见
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)

        # 设置全局样式表 - 现代简约浅色主题
        self.setStyleSheet("""
            /* 主窗口和全局背景 */
            QMainWindow {
                background-color: #F5F7FA;
            }
            QWidget {
                background-color: #F5F7FA;
                color: #333333;
                font-family: "Microsoft YaHei UI", "PingFang SC", "SimHei", sans-serif;
                font-size: 12px;
            }

            /* 功能区卡片样式 */
            #controlPanel {
                background-color: #FFFFFF;
                border-radius: 8px;
                padding: 16px;
                border: 1px solid #E8E8E8;
            }

            /* 按钮样式 */
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #D9D9D9;
                border-radius: 6px;
                padding: 8px 16px;
                color: #333333;
                font-weight: 500;
                min-width: 100px;
            }
            QPushButton:hover {
                border-color: #1890FF;
                color: #1890FF;
            }
            QPushButton:pressed {
                background-color: #E6F7FF;
            }
            QPushButton:focus {
                outline: none;
                border-color: #1890FF;
            }
            QPushButton#predictBtn {
                background-color: #1890FF;
                border-color: #1890FF;
                color: #FFFFFF;
            }
            QPushButton#predictBtn:hover {
                background-color: #40A9FF;
                border-color: #40A9FF;
            }
            QPushButton#predictBtn:pressed {
                background-color: #096DD9;
                border-color: #096DD9;
            }
            QPushButton#importBtn {
                border-color: #1890FF;
                color: #1890FF;
            }
            QPushButton#importBtn:hover {
                background-color: #E6F7FF;
            }

            /* 下拉框样式 */
            QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #D9D9D9;
                border-radius: 6px;
                padding: 6px 12px;
                min-width: 120px;
                color: #333333;
            }
            QComboBox:hover {
                border-color: #1890FF;
            }
            QComboBox:focus {
                border-color: #1890FF;
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
                border-radius: 0 6px 6px 0;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #999999;
                width: 0;
                height: 0;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                border: 1px solid #D9D9D9;
                border-radius: 6px;
                selection-background-color: #E6F7FF;
                selection-color: #1890FF;
                padding: 4px;
            }

            /* 标签样式 */
            QLabel {
                background-color: transparent;
                color: #333333;
                font-weight: 500;
            }

            /* 消息框样式 */
            QMessageBox {
                background-color: #F5F7FA;
            }
            QMessageBox QLabel {
                color: #333333;
            }
            QMessageBox QPushButton {
                min-width: 80px;
            }

            /* 文件对话框样式 */
            QFileDialog {
                background-color: #F5F7FA;
            }

            /* 工具提示样式 */
            QToolTip {
                background-color: #FFFFFF;
                border: 1px solid #D9D9D9;
                border-radius: 4px;
                color: #333333;
                padding: 5px 10px;
            }

            /* 步骤卡片样式 */
            .StepCard {
                background-color: #FFFFFF;
                border-radius: 8px;
                padding: 20px;
                border: 1px solid #E8E8E8;
                margin: 10px;
                min-width: 200px;
            }
            .StepCard QLabel {
                color: #333333;
            }
            .StepNumber {
                font-size: 24px;
                font-weight: bold;
                color: #1890FF;
                margin-bottom: 10px;
            }
            .StepIcon {
                font-size: 32px;
                margin-bottom: 10px;
            }
            .StepDescription {
                font-size: 14px;
                line-height: 1.4;
                color: #666666;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 顶部功能区 - 卡片化设计
        control_panel = QWidget()
        control_panel.setObjectName("controlPanel")
        top_layout = QHBoxLayout(control_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(16)
        
        # Excel导入按钮
        self.import_btn = QPushButton('📊 导入Excel数据')
        self.import_btn.setObjectName("importBtn")
        self.import_btn.clicked.connect(self.import_excel)
        top_layout.addWidget(self.import_btn)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.setSpacing(8)
        model_label = QLabel('预测模型:')
        self.model_combo = QComboBox()
        self.model_combo.addItems(['XGBoost', 'LSTM', 'LSTM-XGBoost 融合模型'])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        top_layout.addLayout(model_layout)
        
        # 区域选择
        area_layout = QHBoxLayout()
        area_layout.setSpacing(8)
        area_label = QLabel('目标区域:')
        self.area_combo = QComboBox()
        self.batch_combo = QComboBox()
        
        # 初始化区域选项
        self.update_area_options()
        
        # 区域选择联动
        self.area_combo.currentIndexChanged.connect(self.update_batch_options)
        
        area_layout.addWidget(area_label)
        area_layout.addWidget(self.area_combo)
        area_layout.addWidget(self.batch_combo)
        top_layout.addLayout(area_layout)
        
        # 预测按钮
        self.predict_btn = QPushButton('▶ 开始预测')
        self.predict_btn.setObjectName("predictBtn")
        self.predict_btn.clicked.connect(self.start_prediction)
        top_layout.addWidget(self.predict_btn)
        
        main_layout.addWidget(control_panel)
        
        # 创建堆叠窗口用于管理引导页和图表
        self.stacked_widget = QStackedWidget()
        
        # 创建欢迎引导页
        self.welcome_widget = WelcomeWidget()
        
        # 创建图表容器
        self.chart_widget = QWidget()
        chart_layout = QVBoxLayout(self.chart_widget)
        
        # 图表区域
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)
        
        # 添加到堆叠窗口
        self.stacked_widget.addWidget(self.welcome_widget)  # 索引 0
        self.stacked_widget.addWidget(self.chart_widget)    # 索引 1
        
        # 默认显示欢迎页
        self.stacked_widget.setCurrentIndex(0)
        
        main_layout.addWidget(self.stacked_widget)
    
    def update_area_options(self):
        """更新区域选项（改为从 adapter 读取真实 site + batch）"""
        self.site_batch_rows = adapter.list_site_batch_pairs()

        # 一级下拉框显示 site_name
        site_names = sorted(list({row["site_name"] for row in self.site_batch_rows}))

        self.area_combo.blockSignals(True)
        self.area_combo.clear()
        self.area_combo.addItems(site_names)
        self.area_combo.blockSignals(False)

        self.update_batch_options()
    
    def update_batch_options(self):
        """根据选中的点位更新批次选项"""
        selected_site_name = self.area_combo.currentText()

        matched_rows = [
            row for row in self.site_batch_rows
            if row["site_name"] == selected_site_name
        ]

        self.batch_combo.clear()

        for row in matched_rows:
            # 下拉框显示 batch_name，但把真正的 site_id / batch_id 挂在 itemData 上
            self.batch_combo.addItem(
                row["batch_name"],
                {
                    "site_id": row["site_id"],
                    "site_name": row["site_name"],
                    "batch_id": row["batch_id"],
                    "batch_name": row["batch_name"],
                }
            )
    
    def import_excel(self):
        """导入真实调查Excel，并写入 disease_observation"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择真实调查Excel文件',
            'algorithm/data',
            'Excel文件 (*.xlsx *.xls)'
        )

        if not file_path:
            return

        try:
            result = adapter.import_observation(file_path)

            QMessageBox.information(
                self,
                '导入成功',
                f"真实调查数据导入成功！\n"
                f"结果：{result}"
            )

            # 导入真实调查后，点位/批次通常不变，但保留刷新
            self.update_area_options()

        except Exception as e:
            import traceback
            traceback.print_exc()

            QMessageBox.critical(
                self,
                '导入失败',
                f"真实调查数据导入失败：\n{str(e)}"
            )
    

    def start_prediction(self):
        """开始预测（接入真实算法）"""
        selected_data = self.batch_combo.currentData()

        if not selected_data:
            QMessageBox.warning(self, '警告', '请选择目标批次！')
            return

        site_id = selected_data["site_id"]
        batch_id = selected_data["batch_id"]
        model_type = self.model_combo.currentText()

        try:
            result = adapter.run_prediction(site_id, batch_id, model_type=model_type)

            # 统一结果结构，兼容不同预测引擎返回格式
            from ui_adapter.adapter import normalize_prediction_result_for_ui
            result = normalize_prediction_result_for_ui(
                result=result,
                site_id=site_id,
                batch_id=batch_id,
                model_type=model_type,
            )

            print("===== 预测结果 =====")
            print(f"model_type={model_type}")
            print(result)

            # 切换到图表界面
            self.stacked_widget.setCurrentIndex(1)

            # 保存预测结果和site_id到实例变量
            self.prediction_result = result
            self.current_site_id = site_id

            # 调用可视化方法
            self.update_visualization()

            QMessageBox.information(
                self,
                '预测完成',
                f"预测完成！\nrun_id: {result.get('prediction_run_id')}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()

            QMessageBox.critical(
                self,
                '错误',
                f"预测失败：\n{str(e)}"
            )


    def update_visualization(self):
        """更新可视化图表，显示三种病害的预测结果"""
        if not hasattr(self, 'prediction_result'):
            return
        
        # 调用 adapter 中的可视化方法
        from ui_adapter.adapter import visualize_prediction
        
        # 确保不会重复绑定事件
        if hasattr(self, 'hover_cid'):
            self.figure.canvas.mpl_disconnect(self.hover_cid)
        
        # 调用可视化函数
        self.hover_cid = visualize_prediction(
            figure=self.figure,
            canvas=self.canvas,
            prediction_result=self.prediction_result,
            site_id=self.current_site_id
        )

if __name__ == '__main__':
    ensure_qt_platform_plugin_path()
    app = QApplication(sys.argv)
    window = DiseasePredictionApp()
    window.show()
    sys.exit(app.exec_())
