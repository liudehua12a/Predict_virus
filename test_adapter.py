import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['QT_MAC_WANTS_LAYER'] = '1'


import sys
import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QFileDialog, QMessageBox, QGroupBox, QStackedWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
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


# 数据库操作类
class DatabaseManager:
    def __init__(self, db_path='disease_db.db'):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._init_db()

    def _init_db(self):
        """初始化数据库，创建必要的表"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # 创建病害观测表
        self.cursor.execute('''
                            CREATE TABLE IF NOT EXISTS disease_observation
                            (
                                id
                                INTEGER
                                PRIMARY
                                KEY
                                AUTOINCREMENT,
                                site_id
                                TEXT,
                                survey_date
                                DATE,
                                gray_incidence
                                REAL,
                                gray_index
                                REAL,
                                blight_incidence
                                REAL,
                                blight_index
                                REAL,
                                white_incidence
                                REAL,
                                white_index
                                REAL
                            )
                            ''')

        # 创建病害预测表
        self.cursor.execute('''
                            CREATE TABLE IF NOT EXISTS disease_prediction
                            (
                                id
                                INTEGER
                                PRIMARY
                                KEY
                                AUTOINCREMENT,
                                site_id
                                TEXT,
                                predict_date
                                DATE,
                                gray_risk_level
                                REAL,
                                blight_risk_level
                                REAL,
                                white_risk_level
                                REAL
                            )
                            ''')

        self.conn.commit()

    def import_excel(self, excel_path):
        """从Excel导入数据到disease_observation表"""

        try:
            df = pd.read_excel(excel_path, header=1)
            for _, row in df.iterrows():
                survey_date = row.get('survey_date', pd.NaT)
                if pd.isna(survey_date):
                    survey_date_str = ''
                else:
                    survey_date_str = survey_date.strftime('%Y-%m-%d')  # 或 '%Y-%m-%d %H:%M:%S'
                self.cursor.execute('''
                                    INSERT INTO disease_observation
                                    (site_id, survey_date, gray_incidence, gray_index, blight_incidence, blight_index,
                                     white_incidence, white_index)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    ''', (
                                        row.get('site_id', ''),
                                        survey_date_str,
                                        row.get('gray_incidence', 0),
                                        row.get('gray_index', 0),
                                        row.get('blight_incidence', 0),
                                        row.get('blight_index', 0),
                                        row.get('white_incidence', 0),
                                        row.get('white_index', 0)
                                    ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"导入Excel失败: {e}")
            return False

    def get_observation_data(self, site_id, date):
        """获取指定站点和日期的观测数据"""
        self.cursor.execute('''
                            SELECT *
                            FROM disease_observation
                            WHERE site_id = ?
                              AND survey_date = ?
                            ''', (site_id, date))
        return self.cursor.fetchone()

    def get_prediction_data(self, site_id, date):
        """获取指定站点和日期的预测数据"""
        self.cursor.execute('''
                            SELECT *
                            FROM disease_prediction
                            WHERE site_id = ?
                              AND predict_date = ?
                            ''', (site_id, date))
        return self.cursor.fetchone()

    def save_prediction(self, site_id, predict_date, gray_risk, blight_risk, white_risk):
        """保存预测结果到数据库"""
        # 先检查是否已存在该预测记录
        self.cursor.execute('''
                            SELECT id
                            FROM disease_prediction
                            WHERE site_id = ?
                              AND predict_date = ?
                            ''', (site_id, predict_date))
        existing = self.cursor.fetchone()

        if existing:
            # 更新现有记录
            self.cursor.execute('''
                                UPDATE disease_prediction
                                SET gray_risk_level   = ?,
                                    blight_risk_level = ?,
                                    white_risk_level  = ?
                                WHERE id = ?
                                ''', (gray_risk, blight_risk, white_risk, existing[0]))
        else:
            # 插入新记录
            self.cursor.execute('''
                                INSERT INTO disease_prediction
                                (site_id, predict_date, gray_risk_level, blight_risk_level, white_risk_level)
                                VALUES (?, ?, ?, ?, ?)
                                ''', (site_id, predict_date, gray_risk, blight_risk, white_risk))

        self.conn.commit()

    def get_all_sites(self):
        """获取所有站点信息"""
        self.cursor.execute('SELECT DISTINCT site_id FROM disease_observation')
        return [row[0] for row in self.cursor.fetchall()]

    def get_weather_data(self, site_id, days=7):
        """从weather_daily表获取指定站点未来几天的天气数据"""
        # 连接到天气数据库（使用相对 main.py 的稳定路径）
        project_root = Path(__file__).resolve().parent
        candidate_paths = [
            project_root / "algorithm" / "data" / "nky-CornPre.db",
            project_root / "nky-CornPre.db",
        ]

        weather_db_path = next((p for p in candidate_paths if p.exists()), None)
        if weather_db_path is None:
            raise FileNotFoundError("未找到天气数据库文件 nky-CornPre.db")

        weather_conn = sqlite3.connect(str(weather_db_path))
        weather_cursor = weather_conn.cursor()

        try:
            # 计算今天的日期
            today = datetime.now().strftime('%Y-%m-%d')

            # 获取从今天开始的天气数据
            weather_cursor.execute('''
                                   SELECT date, temp_avg_c, relative_humidity
                                   FROM weather_daily
                                   WHERE site_id = ? AND date >= ?
                                   ORDER BY date ASC
                                       LIMIT ?
                                   ''', (site_id, today, days,))

            weather_data = []
            for row in weather_cursor.fetchall():
                date, temp, humidity = row
                # 根据土壤湿度确定天气类型
                if humidity > 80:
                    weather_icon = f'☂'  # 雨天
                elif humidity > 50:
                    weather_icon = f'☁'  # 多云}'  # 多云
                else:
                    weather_icon = '☀'  # 晴天
                weather_data.append({
                    'date': date,
                    'temp': temp,
                    'humidity': humidity,
                    'icon': weather_icon
                })

            return weather_data
        finally:
            weather_conn.close()

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


# 预测模型类（占位，实际模型后续接入）
class PredictionModel:
    def __init__(self, model_type):
        self.model_type = model_type

    def predict(self, input_data):
        """根据输入数据预测未来7天的病害风险"""
        # 这里是模型预测逻辑，目前返回随机数据作为示例
        import random
        predictions = []
        for i in range(7):
            predictions.append({
                'date': (datetime.now() + timedelta(days=i + 1)).strftime('%Y-%m-%d'),
                'gray_risk': random.uniform(0, 100),
                'blight_risk': random.uniform(0, 100),
                'white_risk': random.uniform(0, 100)
            })
        return predictions


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

        # 顶部图标
        icon_label = QLabel('🌽', self)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 80px; margin-bottom: 20px;")
        layout.addWidget(icon_label)

        # 欢迎标题
        title_label = QLabel('欢迎使用玉米病害智能预测系统', self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #2E7D32;
            margin-bottom: 30px;
        """)
        layout.addWidget(title_label)

        # 操作步骤提示
        steps_text = """
        <div style='line-height: 1.8; font-size: 14px; color: #388E3C;'>
        <p style='margin: 10px 0;'>① 点击左上角【导入Excel数据】加载历史危害数据。</p>
        <p style='margin: 10px 0;'>② 在下拉框选择对应的【预测模型】和【目标区域】。</p>
        <p style='margin: 10px 0;'>③ 点击【开始预测】获取未来一周的疾病危害趋势图表。</p>
        </div>
        """
        steps_label = QLabel(steps_text, self)
        steps_label.setAlignment(Qt.AlignCenter)
        steps_label.setWordWrap(True)
        steps_label.setStyleSheet("""
            padding: 20px;
            background-color: #F1F8E9;
            border-radius: 10px;
            border: 1px solid #A5D6A7;
            max-width: 600px;
        """)
        layout.addWidget(steps_label)

        # 底部版权信息
        footer_label = QLabel('© 2026 玉米病害预测系统', self)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("""
            font-size: 12px;
            color: #66BB6A;
            margin-top: 40px;
        """)
        layout.addWidget(footer_label)

        # 设置样式
        self.setStyleSheet("""
            WelcomeWidget {
                background-color: #E8F5E9;
            }
        """)


# 主应用类
class DiseasePredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager()
        self.site_batch_rows = []
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('玉米病害预测软件')
        self.setGeometry(100, 100, 1000, 700)

        # 设置全局样式表 - 浅绿色农业主题
        self.setStyleSheet("""
            /* 主窗口和全局背景 */
            QMainWindow {
                background-color: #E8F5E9;
            }
            QWidget {
                background-color: #E8F5E9;
                color: #2E7D32;
                font-family: "Microsoft YaHei", "SimHei", sans-serif;
                font-size: 12px;
            }

            /* 分组框样式 */
            QGroupBox {
                background-color: #F1F8E9;
                border: 2px solid #81C784;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #1B5E20;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                background-color: #F1F8E9;
            }

            /* 按钮样式 */
            QPushButton {
                background-color: #FFFFFF;
                border: 2px solid #66BB6A;
                border-radius: 6px;
                padding: 8px 16px;
                color: #2E7D32;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #C8E6C9;
                border-color: #4CAF50;
            }
            QPushButton:pressed {
                background-color: #A5D6A7;
                border-color: #388E3C;
            }
            QPushButton:focus {
                outline: none;
                border-color: #2E7D32;
            }

            /* 下拉框样式 */
            QComboBox {
                background-color: #FFFFFF;
                border: 2px solid #81C784;
                border-radius: 5px;
                padding: 5px 10px;
                min-width: 120px;
                color: #2E7D32;
            }
            QComboBox:hover {
                border-color: #66BB6A;
                background-color: #F1F8E9;
            }
            QComboBox:focus {
                border-color: #4CAF50;
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #4CAF50;
                width: 0;
                height: 0;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                border: 1px solid #81C784;
                selection-background-color: #C8E6C9;
                selection-color: #1B5E20;
            }

            /* 标签样式 */
            QLabel {
                background-color: transparent;
                color: #2E7D32;
                font-weight: 500;
            }

            /* 消息框样式 */
            QMessageBox {
                background-color: #E8F5E9;
            }
            QMessageBox QLabel {
                color: #2E7D32;
            }
            QMessageBox QPushButton {
                min-width: 80px;
            }

            /* 文件对话框样式 */
            QFileDialog {
                background-color: #E8F5E9;
            }

            /* 滚动区域样式 */
            QScrollArea {
                background-color: #F1F8E9;
                border: 1px solid #A5D6A7;
            }

            /* 工具提示样式 */
            QToolTip {
                background-color: #FFFFFF;
                border: 1px solid #81C784;
                border-radius: 4px;
                color: #2E7D32;
                padding: 5px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # 顶部功能区
        top_group = QGroupBox('功能区')
        top_layout = QHBoxLayout()

        # Excel导入按钮
        self.import_btn = QPushButton('导入Excel数据')
        self.import_btn.clicked.connect(self.import_excel)
        top_layout.addWidget(self.import_btn)

        # 模型选择
        model_layout = QHBoxLayout()
        model_label = QLabel('选择模型:')
        self.model_combo = QComboBox()
        self.model_combo.addItems(['XGBoost', 'LSTM', 'LSTM-XGBoost 融合模型'])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        top_layout.addLayout(model_layout)

        # 区域选择
        area_layout = QHBoxLayout()
        area_label = QLabel('选择区域:')
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
        self.predict_btn = QPushButton('开始预测')
        self.predict_btn.clicked.connect(self.start_prediction)
        top_layout.addWidget(self.predict_btn)

        top_group.setLayout(top_layout)
        main_layout.addWidget(top_group)

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
        self.stacked_widget.addWidget(self.chart_widget)  # 索引 1

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
            '',
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
            result = self._normalize_prediction_result_for_ui(
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

    def _risk_label_from_value(self, value):
        try:
            v = float(value)
        except Exception:
            v = 0.0
        if v < 10:
            return '低风险'
        if v < 20:
            return '中风险'
        return '高风险'

    def _build_ui_result_from_dataframe(self, df, site_id, batch_id, model_type):
        """把 prediction.py 风格 DataFrame 转成前端可视化需要的 results_by_disease 结构。"""
        if df is None or len(df) == 0:
            return {
                'site_id': site_id,
                'batch_id': batch_id,
                'model_type': model_type,
                'predict_dates': [],
                'prediction_run_id': f'ui_{datetime.now().strftime("%Y%m%d%H%M%S")}',
                'results_by_disease': {'gray': [], 'blight': [], 'white': []},
            }

        work = df.copy()
        work['date'] = pd.to_datetime(work['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        work = work.sort_values('date')

        mapping = {
            'gray': {'cn': '灰斑病', 'prob_col': '灰斑病_发病概率(%)'},
            'blight': {'cn': '大斑病', 'prob_col': '大斑病_发病概率(%)'},
            'white': {'cn': '白斑病', 'prob_col': '白斑病_发病概率(%)'},
        }

        results_by_disease = {}
        for disease_key, conf in mapping.items():
            col = conf['prob_col']
            disease_rows = []
            prev_value = 0.0
            for _, row in work.iterrows():
                cur_value = float(row.get(col, 0.0) or 0.0)
                disease_rows.append({
                    'date': row['date'],
                    'site_id': site_id,
                    'disease_key': disease_key,
                    'disease_cn': conf['cn'],
                    'prev_target_1_name': f'{disease_key}_incidence',
                    'prev_target_2_name': f'{disease_key}_index',
                    'prev_target_1_value': round(prev_value, 4),
                    'prev_target_2_value': round(prev_value, 4),
                    'pred_target_1_name': f'{disease_key}_incidence',
                    'pred_target_2_name': f'{disease_key}_index',
                    'pred_target_1_value': round(cur_value, 4),
                    'pred_target_2_value': round(cur_value, 4),
                    'pred_target_1_risk': self._risk_label_from_value(cur_value),
                    'pred_target_2_risk': self._risk_label_from_value(cur_value),
                    'pred_overall_risk': self._risk_label_from_value(cur_value),
                })
                prev_value = cur_value
            results_by_disease[disease_key] = disease_rows

        predict_dates = [d for d in work['date'].tolist() if isinstance(d, str)]
        return {
            'site_id': site_id,
            'batch_id': batch_id,
            'model_type': model_type,
            'predict_dates': predict_dates,
            'prediction_run_id': f'ui_{datetime.now().strftime("%Y%m%d%H%M%S")}',
            'results_by_disease': results_by_disease,
        }

    def _normalize_prediction_result_for_ui(self, result, site_id, batch_id, model_type):
        """
        统一预测结果格式，确保前端图表可显示。
        兼容：
        1) 在线服务返回的标准 dict（含 results_by_disease）
        2) prediction.py 返回的 DataFrame
        3) dict 中携带 records/predictions 列表
        """
        if isinstance(result, dict) and isinstance(result.get('results_by_disease'), dict):
            return result

        if isinstance(result, pd.DataFrame):
            return self._build_ui_result_from_dataframe(result, site_id, batch_id, model_type)

        if isinstance(result, dict):
            if isinstance(result.get('predictions'), pd.DataFrame):
                return self._build_ui_result_from_dataframe(result['predictions'], site_id, batch_id, model_type)

            records = result.get('predictions')
            if records is None:
                records = result.get('records')
            if isinstance(records, list) and records:
                try:
                    df = pd.DataFrame(records)
                    return self._build_ui_result_from_dataframe(df, site_id, batch_id, model_type)
                except Exception:
                    pass

        raise ValueError('预测结果格式不支持前端展示，请检查返回数据结构。')

    def update_visualization(self):
        """更新可视化图表，显示三种病害的预测结果"""
        if not hasattr(self, 'prediction_result'):
            return

        # 调用可视化方法，传入完整的预测结果
        self.visualize_prediction(self.prediction_result)

    def visualize_prediction(self, prediction_result):
        """可视化三种病害的预测结果为曲线图"""
        # 清空图表
        self.figure.clear()

        # 获取三种病害的数据
        results_by_disease = prediction_result.get('results_by_disease', {})

        # 病害配置
        diseases = {
            'blight': {'name': '大斑病', 'color': 'red'},
            'gray': {'name': '灰斑病', 'color': 'gray'},
            'white': {'name': '白斑病', 'color': 'blue'}
        }

        # 创建子图，设置双y轴
        ax1 = self.figure.add_subplot(111)
        ax2 = ax1.twinx()

        # 设置图表标题
        ax1.set_title('未来一周病害发病程度与天气预测')

        # 设置坐标轴
        ax1.set_xlabel('日期')
        ax1.set_ylabel('发病程度')
        ax2.set_ylabel('平均温度 (°C)', color='green')

        # 添加风险区划背景色带（在绘制曲线之前，确保色带在最底层）
        # 低风险区 (0 - 10)：柔和马卡龙绿色
        ax1.axhspan(0, 30, facecolor='#C8E6C9', alpha=0.5, zorder=0)
        # 中风险区 (10 - 20)：柔和马卡龙橙色
        ax1.axhspan(30, 70, facecolor='#FFE0B2', alpha=0.5, zorder=0)
        # 高风险区 (20 - 40)：柔和马卡龙红色
        ax1.axhspan(70, 100, facecolor='#FFCDD2', alpha=0.5, zorder=0)

        # 在色带边缘添加文字标签
        ax1.text(6.2, 15, '低风险', fontsize=9, color='#388E3C', alpha=0.8,
                 verticalalignment='center', horizontalalignment='left', zorder=0)
        ax1.text(6.2, 50, '中风险', fontsize=9, color='#F57C00', alpha=0.8,
                 verticalalignment='center', horizontalalignment='left', zorder=0)
        ax1.text(6.2, 85, '高风险', fontsize=9, color='#D32F2F', alpha=0.8,
                 verticalalignment='center', horizontalalignment='left', zorder=0)

        # 记录最大value
        max_val = 0
        # 准备数据并绘制曲线
        for disease_key, disease_info in diseases.items():
            disease_data = results_by_disease.get(disease_key, [])
            if disease_data:
                dates = [item['date'] for item in disease_data]
                values = [item['pred_target_2_value'] for item in disease_data]
                x = range(len(dates))

                # 绘制曲线
                ax1.plot(x, values, marker='o', linestyle='-', color=disease_info['color'], label=disease_info['name'])

                # 在曲线上添加数值标签
                # 在曲线上添加数值标签
                for i, value in enumerate(values):
                    if value >= 0.1:
                        ax1.text(x[i], value + 0.1, f'{value:.4f}', ha='center', va='bottom', fontsize=8)
                    max_val = max(max_val, value)

        # 获取天气数据
        weather_data = self.db_manager.get_weather_data(self.current_site_id, 7)
        if weather_data and disease_data:
            # 确保天气数据和预测数据日期匹配
            weather_dates = [item['date'] for item in weather_data]
            weather_temps = [item['temp'] for item in weather_data]
            weather_icons = [item['icon'] for item in weather_data]

            # 绘制温度曲线
            ax2.plot(x, weather_temps, marker='s', linestyle='--', color='green', label='平均温度')
            ax2.tick_params(axis='y', labelcolor='green')

            # 在温度曲线上添加数值标签，包含天气信息
            for i, temp in enumerate(weather_temps):
                if i < len(weather_icons):
                    weather_info = weather_icons[i]
                    ax2.text(x[i], temp + 0.05, f'{temp:.1f}°C\n{weather_info}', ha='center', va='bottom', fontsize=8,
                             color='green')
                else:
                    ax2.text(x[i], temp + 0.05, f'{temp:.1f}°C', ha='center', va='bottom', fontsize=8, color='green')

            # 设置x轴标签，只显示日期
            if disease_data:
                ax1.set_xticks(x)
                # 创建只包含日期的标签
                date_labels = []
                for date in dates:
                    date = date[5:]
                    date_labels.append(date)
                ax1.set_xticklabels(date_labels, rotation=0, ha='center')

        # 设置合理的y轴范围
        max_val = max_val - max_val % 10 + 15
        ax1.set_ylim(0, 100)
        # 设置温度轴范围
        if weather_data:
            min_temp = min(weather_temps) - 1
            max_temp = max(weather_temps) + 5
            ax2.set_ylim(min_temp, max_temp)

        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # 调整布局
        self.figure.tight_layout()

        # 更新画布
        self.canvas.draw()

    def closeEvent(self, event):
        """关闭窗口时关闭数据库连接"""
        self.db_manager.close()
        event.accept()


if __name__ == '__main__':
    ensure_qt_platform_plugin_path()
    app = QApplication(sys.argv)
    window = DiseasePredictionApp()
    window.show()
    sys.exit(app.exec_())
