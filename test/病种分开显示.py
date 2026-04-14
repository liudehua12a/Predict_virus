import sys
import platform
import matplotlib

# 声明使用 PyQt5 后台
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox, QLabel, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 自动配置中文字体
system = platform.system()
if system == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
elif system == 'Darwin':
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 模拟你提供的原始数据
RESULT_DATA={'batch_id': 2, 'forecast_end_date': '2026-04-08', 'predict_dates': ['2026-04-03', '2026-04-04', '2026-04-05', '2026-04-06', '2026-04-07', '2026-04-08'], 'prediction_run_id': 'predrun_20260402211319_4f686bb1', 'results_by_disease': {'blight': [{'date': '2026-04-03', 'disease_cn': '大斑病', 'disease_key': 'blight', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'blight_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'blight_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0042, 'prev_target_1_name': 'blight_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'blight_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-04', 'disease_cn': '大斑病', 'disease_key': 'blight', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'blight_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'blight_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0042, 'prev_target_1_name': 'blight_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'blight_index', 'prev_target_2_value': 0.0042, 'site_id': 7}, {'date': '2026-04-05', 'disease_cn': '大斑病', 'disease_key': 'blight', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'blight_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'blight_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0042, 'prev_target_1_name': 'blight_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'blight_index', 'prev_target_2_value': 0.0042, 'site_id': 7}, {'date': '2026-04-06', 'disease_cn': '大斑病', 'disease_key': 'blight', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'blight_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'blight_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.009, 'prev_target_1_name': 'blight_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'blight_index', 'prev_target_2_value': 0.0042, 'site_id': 7}, {'date': '2026-04-07', 'disease_cn': '大斑病', 'disease_key': 'blight', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'blight_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'blight_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0151, 'prev_target_1_name': 'blight_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'blight_index', 'prev_target_2_value': 0.009, 'site_id': 7}, {'date': '2026-04-08', 'disease_cn': '大斑病', 'disease_key': 'blight', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'blight_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'blight_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0342, 'prev_target_1_name': 'blight_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'blight_index', 'prev_target_2_value': 0.0151, 'site_id': 7}], 'gray': [{'date': '2026-04-03', 'disease_cn': '灰斑病', 'disease_key': 'gray', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'gray_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'gray_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 1.7262, 'prev_target_1_name': 'gray_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'gray_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-04', 'disease_cn': '灰斑病', 'disease_key': 'gray', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'gray_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'gray_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 3.66, 'prev_target_1_name': 'gray_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'gray_index', 'prev_target_2_value': 1.7262, 'site_id': 7}, {'date': '2026-04-05', 'disease_cn': '灰斑病', 'disease_key': 'gray', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'gray_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'gray_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 5.8926, 'prev_target_1_name': 'gray_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'gray_index', 'prev_target_2_value': 3.66, 'site_id': 7}, {'date': '2026-04-06', 'disease_cn': '灰斑病', 'disease_key': 'gray', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'gray_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'gray_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 7.8338, 'prev_target_1_name': 'gray_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'gray_index', 'prev_target_2_value': 5.8926, 'site_id': 7}, {'date': '2026-04-07', 'disease_cn': '灰斑病', 'disease_key': 'gray', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'gray_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'gray_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 8.871, 'prev_target_1_name': 'gray_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'gray_index', 'prev_target_2_value': 7.8338, 'site_id': 7}, {'date': '2026-04-08', 'disease_cn': '灰斑病', 'disease_key': 'gray', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'gray_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'gray_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 9.6449, 'prev_target_1_name': 'gray_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'gray_index', 'prev_target_2_value': 8.871, 'site_id': 7}], 'white': [{'date': '2026-04-03', 'disease_cn': '白斑病', 'disease_key': 'white', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'white_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'white_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0, 'prev_target_1_name': 'white_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'white_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-04', 'disease_cn': '白斑病', 'disease_key': 'white', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'white_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'white_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0, 'prev_target_1_name': 'white_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'white_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-05', 'disease_cn': '白斑病', 'disease_key': 'white', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'white_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'white_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0, 'prev_target_1_name': 'white_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'white_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-06', 'disease_cn': '白斑病', 'disease_key': 'white', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'white_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'white_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0, 'prev_target_1_name': 'white_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'white_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-07', 'disease_cn': '白斑病', 'disease_key': 'white', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'white_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'white_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0, 'prev_target_1_name': 'white_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'white_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-08', 'disease_cn': '白斑病', 'disease_key': 'white', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'white_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'white_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0, 'prev_target_1_name': 'white_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'white_index', 'prev_target_2_value': 0.0, 'site_id': 7}]}, 'site_id': 7, 'start_source_type': 'zero_init', 'today_date': '2026-04-02', 'yesterday_date': '2026-04-01'}


DISEASE_MAP = {
    '大斑病': 'blight',
    '灰斑病': 'gray',
    '白斑病': 'white'
}


class DiseaseForecastApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("玉米病害预测软件")
        self.resize(800, 600)

        # 主控面板 (Widget)
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # --- 顶部：下拉选择区 ---
        self.top_layout = QHBoxLayout()
        self.label = QLabel("请选择预测病种：")
        self.combo_box = QComboBox()
        self.combo_box.addItems(['大斑病', '灰斑病', '白斑病'])
        # 绑定下拉框的内容改变事件
        self.combo_box.currentTextChanged.connect(self.update_plot)

        self.top_layout.addWidget(self.label)
        self.top_layout.addWidget(self.combo_box)
        self.top_layout.addStretch(1)  # 把控件推到左边

        self.layout.addLayout(self.top_layout)

        # --- 底部：图表区 ---
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.layout.addWidget(self.canvas)

        # 初始化图表
        self.update_plot(self.combo_box.currentText())

    def get_color_by_value(self, value):
        """根据数值返回对应的颜色"""
        if 0 <= value <= 25:
            return '#2ca02c'  # 绿色 (Matplotlib默认绿，比较好看)
        elif 25 < value <= 50:
            return '#1f77b4'  # 蓝色
        elif 50 < value <= 75:
            return '#ff7f0e'  # 橙色
        elif 75 < value <= 100:
            return '#d62728'  # 红色
        else:
            return 'gray'  # 异常值显示为灰色

    def update_plot(self, disease_name):
        """更新图表内容"""
        # 清空之前的画布
        self.ax.clear()

        # 获取对应的数据
        disease_key = DISEASE_MAP[disease_name]
        dates = RESULT_DATA['predict_dates']
        values = RESULT_DATA['results_by_disease'][disease_key]

        # 根据数值获取每个柱子的颜色
        colors = [self.get_color_by_value(v) for v in values]

        # 绘制柱状图
        bars = self.ax.bar(dates, values, color=colors, width=0.5)

        # 设置图表标题和标签
        self.ax.set_title(f"未来6天 {disease_name} 发病程度预测")
        self.ax.set_ylabel("预测病情指数")

        # 动态设置 Y 轴范围。为了让你的 0-100 颜色逻辑在后续数据变大时能显示完整，这里做个自适应
        max_val = max(values) if values and max(values) > 0 else 1
        # 如果最大值很小，就留一点顶部空间；如果很大，也留一点空间
        self.ax.set_ylim(0, max_val * 1.2)

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            self.ax.annotate(f'{height:.4f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=9)

        # 刷新画布
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    # 解决高分屏显示模糊问题（针对Windows）
    if hasattr(sys, 'frozen') or hasattr(sys, 'importers') or platform.system() == 'Windows':
        QApplication.setAttribute(matplotlib.widgets.QtCore.Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(matplotlib.widgets.QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    ex = DiseaseForecastApp()
    ex.show()
    sys.exit(app.exec_())