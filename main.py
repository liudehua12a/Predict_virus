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
        # 给当前页面设置专属的 ObjectName，这是背景图能生效的关键
        self.setObjectName("myWelcomePage")
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

        title_label = QLabel('欢迎使用玉米病害智能预测软件', self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 30px;
            font-weight: bold;
            color: #1B5E20; /* 深绿色 */
            letter-spacing: 2px;
        """)
        # 设置 header_layout 的边距：左, 上, 右, 下
        header_layout.setContentsMargins(0, 70, 0, 0)
        header_layout.addWidget(title_label)
        layout.addLayout(header_layout)

        # 操作步骤提示 - 卡片式引导流
        steps_layout = QHBoxLayout()
        steps_layout.setAlignment(Qt.AlignCenter)
        steps_layout.setSpacing(20)

        # 步骤1
        step1 = StepCard(
            number="1",
            icon="<span style='font-size: 50px;'>📊</span>",
            description="点击左上角【导入Excel数据】加载历史危害数据"
        )
        step1.setStyleSheet("font-size: 20px;")
        steps_layout.addWidget(step1)

        # 步骤2
        step2 = StepCard(
            number="2",
            icon="<span style='font-size: 50px;'>🔍</span>",
            description="在下拉框选择对应的【预测模型】和【目标区域】"
        )
        step2.setStyleSheet("font-size: 20px;")
        steps_layout.addWidget(step2)

        # 步骤3
        step3 = StepCard(
            number="3",
            icon="<span style='font-size: 50px;'>▶</span>",
            description="点击【开始预测】获取未来一周的疾病危害趋势图表"
        )
        step3.setStyleSheet("font-size: 20px;")
        steps_layout.addWidget(step3)

        layout.addLayout(steps_layout)

        # 底部版权信息
        footer_label = QLabel('© 2026病害预测软件', self)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("""
            font-size: 15px;
            color: #000000; /* 适应浅色背景的深灰色 */
            margin-top: 150px;
        """)
        layout.addWidget(footer_label)

        # 1. 获取当前运行脚本的绝对目录
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. 拼接图片的绝对路径
        bg_path = os.path.join(base_dir, 'algorithm', 'data', 'imgs', 'background', '66.jpg')

        # 3. 替换反斜杠
        bg_path = bg_path.replace('\\', '/')

        # 4. 强制生效背景
        self.setAttribute(Qt.WA_StyledBackground, True)

        # 5. 设置针对白色系背景的 QSS
        self.setStyleSheet(f"""
            #myWelcomePage {{
                border-image: url('{bg_path}');
            }}

            #myWelcomePage QLabel {{
                border-image: none;
                background-color: transparent;
                color: #333333; /* 文字改回深色 */
            }}

            /* 白色玻璃拟物化卡片 */
            #myWelcomePage .StepCard {{
                border-image: none;
                background-color: rgba(255, 255, 255, 0.85); 
                border: 1px solid #E0E0E0;
                border-radius: 8px;
            }}

            #myWelcomePage .StepCard:hover {{
                border-image: none;
                background-color: rgba(255, 255, 255, 0.95); 
                border: 1px solid #4CAF50;
            }}

            /* 步骤数字改为绿色 */
            #myWelcomePage .StepNumber {{
                color: #4CAF50;
            }}
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
        # 获取屏幕对象
        screen = QApplication.primaryScreen().geometry()
        # 设置窗口为屏幕宽度的 80%，高度的 80%
        width = int(screen.width() * 0.8)
        height = int(screen.height() * 0.8)
        self.setGeometry(100, 100, width, height)

        # 设置全局样式表 - 现代简约生态风主题 (去除了所有冲突的深色配置)
        self.setStyleSheet("""
            /* 主窗口和全局背景：极浅的生态灰绿 */
            QMainWindow, QWidget {
                background-color: #F4F7F4; 
                color: #2C3E50; 
                font-family: "Microsoft YaHei UI", "PingFang SC", "SimHei", sans-serif;
                font-size: 20px;
            }

            /* 顶部控制面板：纯白卡片 + 细微边框 */
            #controlPanel {
                background-color: #FFFFFF;
                border-radius: 8px;
                padding: 16px;
                border: 1px solid #E2E8F0;
            }

            /* 下拉框样式：干净的白底灰边 */
            QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 10px 20px;
                color: #334155;
                font-size: 18px;
            }
            QComboBox:hover, QComboBox:focus {
                border-color: #4CAF50; /* 悬浮时显示自然绿 */
            }
            QComboBox::drop-down { border: none; width: 30px; }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #4CAF50;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                border: 1px solid #4CAF50;
                selection-background-color: #E8F5E9;
                selection-color: #1B5E20;
            }

            /* 普通按钮：白底绿字 */
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #4CAF50;
                border-radius: 6px;
                padding: 8px 16px;
                color: #4CAF50;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #E8F5E9; /* 浅绿悬浮背景 */
            }

            /* 返回首页按钮：白底绿字 */
            QPushButton#backBtn {
                background-color: #FFFFFF;
                border: 1px solid #4CAF50;
                border-radius: 6px;
                padding: 8px 16px;
                color: #4CAF50;
                font-weight: bold;
            }
            QPushButton#backBtn:hover {
                background-color: #E8F5E9;
            }

            /* 标签样式 */
            QLabel {
                background-color: transparent;
                color: #334155;
                font-weight: bold;
            }

            /* 弹出框和文件对话框 */
            QMessageBox, QFileDialog {
                background-color: #F4F7F4;
            }
            QMessageBox QLabel {
                color: #2C3E50;
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
        control_panel.setMinimumHeight(60)
        top_layout = QHBoxLayout(control_panel)
        top_layout.setContentsMargins(10, 10, 10, 10)
        top_layout.setSpacing(24)

        # 返回首页按钮
        self.back_btn = QPushButton('🔙 返回')
        self.back_btn.setObjectName("backBtn")
        self.back_btn.clicked.connect(self.go_home)
        self.back_btn.setVisible(False)
        top_layout.addWidget(self.back_btn)

        # Excel导入按钮
        self.import_btn = QPushButton('📊 导入Excel')
        self.import_btn.setStyleSheet("font-size:14px;")
        self.import_btn.setObjectName("importBtn")
        self.import_btn.clicked.connect(self.import_excel)
        top_layout.addWidget(self.import_btn)

        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.setSpacing(8)
        model_label = QLabel('预测模型:')
        model_label.setStyleSheet("font-weight:bold;font-size:20px;")
        self.model_combo = QComboBox()
        self.model_combo.addItems(['XGBoost', 'LSTM', 'LSTM-XGBoost 融合模型'])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        top_layout.addLayout(model_layout)

        # 区域选择
        area_layout = QHBoxLayout()
        area_layout.setSpacing(8)
        area_label = QLabel('目标区域:')
        area_label.setStyleSheet("font-weight:bold;font-size:20px;")
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
        self.predict_btn.setStyleSheet("font-size:14px;")
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
        # [新增这一行] 消除 PyQt 布局自带的四周空白边距
        chart_layout.setContentsMargins(0, 0, 0, 0)

        # 图表区域
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)

        # 添加到堆叠窗口
        self.stacked_widget.addWidget(self.welcome_widget)  # 索引 0
        self.stacked_widget.addWidget(self.chart_widget)  # 索引 1

        # 默认显示欢迎页
        self.stacked_widget.setCurrentIndex(0)
        self.stacked_widget.currentChanged.connect(self.on_page_changed)

        main_layout.addWidget(self.stacked_widget)

    def go_home(self):
        """返回首页"""
        self.stacked_widget.setCurrentIndex(0)

    def on_page_changed(self, index):
        """页面切换时控制返回按钮的显示"""
        self.back_btn.setVisible(index != 0)

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
        """弹出选择对话框：下载模板或直接导入"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle('导入Excel数据')
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #F4F7F4;
            }
            QLabel#titleLabel {
                font-size: 18px;
                font-weight: bold;
                color: #1B5E20;
            }
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #4CAF50;
                border-radius: 6px;
                padding: 10px 20px;
                color: #4CAF50;
                font-weight: bold;
                font-size: 14px;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: #E8F5E9;
            }
            QPushButton#cancelBtn {
                border-color: #CBD5E1;
                color: #64748B;
            }
            QPushButton#cancelBtn:hover {
                background-color: #F1F5F9;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)

        title = QLabel('请选择操作')
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        desc = QLabel('您可以下载标准 Excel 模板后再填充数据，\n也可以直接导入已有的数据文件。')
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("font-size: 14px; color: #334155;")
        layout.addWidget(desc)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        download_btn = QPushButton('📥 下载Excel模板')
        import_btn = QPushButton('📤 直接导入数据')
        cancel_btn = QPushButton('取消')
        cancel_btn.setObjectName("cancelBtn")

        btn_layout.addWidget(download_btn)
        btn_layout.addWidget(import_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def on_download():
            dialog.accept()
            base_dir = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.join(base_dir, 'algorithm', 'data', '调查数据表--模板.xlsx')
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                '保存Excel模板',
                '调查数据表--模板.xlsx',
                'Excel文件 (*.xlsx)'
            )
            if not save_path:
                return
            try:
                import shutil
                shutil.copy(template_path, save_path)
                QMessageBox.information(self, '下载成功', 'Excel 模板已保存到指定位置！')
            except Exception as e:
                QMessageBox.critical(self, '下载失败', f'模板保存失败：\n{str(e)}')

        def on_import():
            dialog.accept()
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                '选择真实调查Excel文件',
                'algorithm/data',
                'Excel文件 (*.xlsx *.xls)'
            )
            if not file_path:
                return
            try:
                adapter.import_observation(file_path)
                QMessageBox.information(self, '导入成功', '真实调查数据导入成功！')
                self.update_area_options()
            except Exception as e:
                QMessageBox.critical(self, '导入失败', f'真实调查数据导入失败：\n{str(e)}')

        download_btn.clicked.connect(on_download)
        import_btn.clicked.connect(on_import)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec_()

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
                f"预测完成！"
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