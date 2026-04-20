from pathlib import Path
import sys
import sqlite3
import pandas as pd
from datetime import datetime
import os

# Matplotlib 图片标注组件
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# PIL 用于高质量图片读取
from PIL import Image

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
ALGORITHM_DIR = ROOT_DIR / "algorithm"

# 把 algorithm 目录加入导入路径
sys.path.insert(0, str(ALGORITHM_DIR))

# 统一数据库路径
DB_PATH = ALGORITHM_DIR / "data" / "nky-CornPre.db"


def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def import_site_batch(file_path: str):
    """
    导入点位 + 批次基础信息表
    这里改成函数内再导入，避免启动 UI 时加载整套算法
    """
    import q_site_batch_import_service as site_batch_service
    return site_batch_service.import_site_batch_excel(file_path)


def import_observation(file_path: str):
    """
    导入真实调查表：只入库，不触发预测/重算
    """
    import o_observation_import_service as observation_service
    return observation_service.import_observation_only(file_path)


def run_prediction(site_id: int, batch_id: int, model_type: str = "LSTM"):
    """
    触发单个 site + batch 的在线预测
    这里改成函数内再导入，避免启动 UI 时加载整套算法
    """

    import n_online_prediction_service as prediction_service
    return prediction_service.run_online_prediction_for_today(
        site_id=site_id,
        batch_id=batch_id,
        model_type=model_type,
        today_date=None,
        forecast_days=7,
    )


def list_site_batch_pairs():
    """
    启动 UI 时只读取 site + batch 列表，不导入 torch / 模型 / 天气模块
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                s.site_id,
                s.site_name,
                b.batch_id,
                b.batch_name
            FROM survey_batch b
            JOIN site_info s
              ON b.site_id = s.site_id
            WHERE b.is_active = 1
            ORDER BY s.site_name, b.batch_name
        """).fetchall()

    return [dict(r) for r in rows]


def risk_label_from_value(value):
    """
    根据数值生成风险标签
    
    参数:
        value: 数值
    
    返回:
        str: 风险标签 ('低风险', '中风险', '高风险')
    """
    try:
        v = float(value)
    except Exception:
        v = 0.0
    if v < 10:
        return '低风险'
    if v < 20:
        return '中风险'
    return '高风险'


def build_ui_result_from_dataframe(df, site_id, batch_id, model_type):
    """
    把 prediction.py 风格 DataFrame 转成前端可视化需要的 results_by_disease 结构
    
    参数:
        df: DataFrame 预测结果
        site_id: 站点 ID
        batch_id: 批次 ID
        model_type: 模型类型
    
    返回:
        dict: 前端可视化需要的结果结构
    """
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
                'pred_target_1_risk': risk_label_from_value(cur_value),
                'pred_target_2_risk': risk_label_from_value(cur_value),
                'pred_overall_risk': risk_label_from_value(cur_value),
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


def normalize_prediction_result_for_ui(result, site_id, batch_id, model_type):
    """
    统一预测结果格式，确保前端图表可显示
    兼容：
    1) 在线服务返回的标准 dict（含 results_by_disease）
    2) prediction.py 返回的 DataFrame
    3) dict 中携带 records/predictions 列表
    
    参数:
        result: 预测结果（可能是 dict、DataFrame 或其他格式）
        site_id: 站点 ID
        batch_id: 批次 ID
        model_type: 模型类型
    
    返回:
        dict: 统一格式的预测结果
    """
    if isinstance(result, dict) and isinstance(result.get('results_by_disease'), dict):
        return result

    if isinstance(result, pd.DataFrame):
        return build_ui_result_from_dataframe(result, site_id, batch_id, model_type)

    if isinstance(result, dict):
        if isinstance(result.get('predictions'), pd.DataFrame):
            return build_ui_result_from_dataframe(result['predictions'], site_id, batch_id, model_type)

        records = result.get('predictions')
        if records is None:
            records = result.get('records')
        if isinstance(records, list) and records:
            try:
                df = pd.DataFrame(records)
                return build_ui_result_from_dataframe(df, site_id, batch_id, model_type)
            except Exception:
                pass

    raise ValueError('预测结果格式不支持前端展示，请检查返回数据结构。')


def visualize_prediction(figure, canvas, prediction_result, site_id):
    """
    可视化三种病害的预测结果为曲线图 (现代生态农场风 - 浅色模式)
    """
    # 清空图表
    figure.clear()

    # 设置画板外围背景色为浅灰绿 (完美融入 PyQt 主窗口浅色背景)
    figure.patch.set_facecolor('#F4F7F4')

    # 获取三种病害的数据
    results_by_disease = prediction_result.get('results_by_disease', {})

    # [修改] 病害配置 - 针对纯白背景，使用自然、清晰的色彩
    diseases = {
        'blight': {'name': '大斑病', 'color': '#FA8C16'},  # 沉稳的深蓝色
        'gray': {'name': '灰斑病', 'color': '#64748B'},  # 生机盎然的绿色
        'white': {'name': '白斑病', 'color': '#E2E8F0'}  # 温暖的橘黄色
    }

    # 创建子图，设置双y轴
    ax1 = figure.add_subplot(111)
    ax2 = ax1.twinx()

    # 设置坐标系内部背景色 (纯白色，让数据展示更干净)
    ax1.set_facecolor('#FFFFFF')

    # 字体颜色改为深灰/浅灰，适应白底
    ax1.set_title('未来一周病害发病程度与天气预测', fontsize=18, fontweight='bold', color='#333333', pad=36)
    ax1.set_xlabel('日期', fontsize=12, color='#666666')
    ax1.set_ylabel('发病程度', fontsize=12, color='#666666')
    ax2.set_ylabel('平均温度 (°C)', color='#1890FF', fontsize=12)

    # --- 1. 绘制背景色带 (生态清新风格) ---
    # 浅蓝色 (安全/天空)
    ax1.axhspan(0, 30, facecolor='#312E81', alpha=0.3, zorder=0)
    # 浅黄色 (中度/麦穗)
    ax1.axhspan(30, 70, facecolor='#FDE047', alpha=0.5, zorder=0)
    # 浅粉红色 (高风险/警示)
    ax1.axhspan(70, 100, facecolor='#FCA5A5', alpha=1, zorder=0)

    # 风险区边界虚线改为浅灰
    #ax1.axhline(y=30, color='#D1D5DB', linestyle='--', linewidth=1, zorder=1)
    #ax1.axhline(y=70, color='#D1D5DB', linestyle='--', linewidth=1, zorder=1)

    max_val = 0
    global_dates = []
    global_x = []
    tooltip_data = []

    # --- 2. 绘制病害曲线 ---
    for disease_key, disease_info in diseases.items():
        disease_data = results_by_disease.get(disease_key, [])
        if disease_data:
            if not global_dates:
                global_dates = [item['date'] for item in disease_data]
                global_x = list(range(len(global_dates)))

            values = [item['pred_target_2_value'] for item in disease_data]

            edge_color = '#94A3B8' if disease_key == 'white' else disease_info['color']

            ax1.plot(global_x, values, marker='o', linestyle='-', color=disease_info['color'],
                     label=disease_info['name'], linewidth=2.5, markersize=7,
                     markerfacecolor='white', markeredgecolor=edge_color, markeredgewidth=2)

            tooltip_data.append({
                'name': disease_info['name'],
                'values': values,
                'color': disease_info['color']
            })

            if values:
                max_val = max(max_val, max(values))

    # --- 3. 绘制天气曲线 ---
    from algorithm.k_weather_data_storage import get_weather_data
    weather_data = get_weather_data(site_id, 7)
    weather_temps = []
    weather_icons = []
    if weather_data and global_x:
        weather_temps = [item['temp'] for item in weather_data]
        weather_icons = [item['icon'] for item in weather_data]

        # [修改] 温度虚线颜色改为标准的 '#1890FF'，点内部填充改为 'white'
        ax2.plot(global_x, weather_temps, marker='s', linestyle='--', color='#1890FF',
                 label='平均温度', linewidth=1.2, markersize=5, markerfacecolor='white',
                 markeredgewidth=1.2)
        ax2.tick_params(axis='y', colors='#1890FF')

        ICON_DIR = ALGORITHM_DIR / "data" / "weather"
        WEATHER_ICON_MAP = {
            '[云]': 'cloud1.png',
            '[雨]': 'rain1.png',
            '[晴]': 'sun.png',
        }
        ICON_ZOOM = 0.1
        _icon_cache = {}

        def _get_icon_image(icon_key):
            if icon_key not in _icon_cache:
                filename = WEATHER_ICON_MAP.get(icon_key)
                if filename:
                    icon_path = ICON_DIR / filename
                    if icon_path.exists():
                        from PIL import Image
                        from matplotlib.offsetbox import OffsetImage
                        img = Image.open(icon_path).convert('RGBA')
                        _icon_cache[icon_key] = OffsetImage(img, zoom=ICON_ZOOM)
                    else:
                        return None
                else:
                    return None
            return _icon_cache[icon_key]

        for i, temp in enumerate(weather_temps):
            # [修改] 天气文字颜色改为 '#1890FF'
            ax2.text(global_x[i], temp + 0.3, f'{temp:.1f}°C',
                     ha='center', va='bottom', fontsize=12, color='#1890FF', zorder=10)

            if i < len(weather_icons):
                from matplotlib.offsetbox import AnnotationBbox
                icon_img = _get_icon_image(weather_icons[i])
                if icon_img is not None:
                    ab = AnnotationBbox(
                        icon_img, xy=(global_x[i], temp), xybox=(0, -18),
                        xycoords='data', boxcoords='offset points',
                        frameon=False, zorder=10
                    )
                    ax2.add_artist(ab)

    # --- 4. 设置坐标轴和图例 ---
    if global_x:
        ax1.set_xticks(global_x)
        date_labels = [date[5:] for date in global_dates]
        ax1.set_xticklabels(date_labels, rotation=0, ha='center', fontsize=10)

    max_val = max_val - max_val % 10 + 15 if max_val > 0 else 100
    ax1.set_ylim(0, max(100, max_val))

    if weather_data:
        ax2.set_ylim(min(weather_temps) - 1, max(weather_temps) + 5)

    # 去除上和右边框
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # [修改] 将底边和左边的坐标轴线改为浅灰色
    ax1.spines['bottom'].set_color('#CCCCCC')
    ax1.spines['left'].set_color('#CCCCCC')
    ax2.spines['bottom'].set_color('#CCCCCC')
    ax2.spines['left'].set_color('#CCCCCC')

    # [修改] 刻度文字颜色改为深灰
    ax1.tick_params(axis='x', colors='#666666')
    ax1.tick_params(axis='y', colors='#666666')

    # [修改] 图例文字颜色改回深灰 '#333333'
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center',
               bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False,
               fontsize=12, handletextpad=0.5, columnspacing=1.5,
               labelcolor='#333333')

    figure.tight_layout(rect=[0, 0, 0.98, 1.0])

    # ====== 5. 交互式悬停提示框逻辑 ======
    # [修改] 游标线改为柔和的灰色
    v_line = ax1.axvline(x=0, color='#999999', linestyle=':', alpha=0.8, zorder=5)
    v_line.set_visible(False)

    # [修改] 提示框改为白底深色字，带绿色边框
    annot = ax1.annotate(
        "", xy=(0, 0), xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.6", fc="#FFFFFF", ec="#4CAF50", lw=1.5, alpha=0.95),
        zorder=100, fontsize=12, color="#333333"
    )
    annot.set_visible(False)

    def hover(event):
        if event.inaxes in (ax1, ax2) and global_x:
            x_idx = int(round(event.xdata))

            if 0 <= x_idx < len(global_x):
                v_line.set_xdata([x_idx, x_idx])
                v_line.set_visible(True)

                date_str = global_dates[x_idx]
                tooltip_text = f""

                for item in tooltip_data:
                    val = item['values'][x_idx]
                    tooltip_text += f"{item['name']}: {val:.4f}\n"

                annot.xy = (x_idx, event.ydata)
                annot.set_text(tooltip_text.strip())
                annot.set_visible(True)

                canvas.draw_idle()
                return

        if annot.get_visible():
            annot.set_visible(False)
            v_line.set_visible(False)
            canvas.draw_idle()

    hover_cid = figure.canvas.mpl_connect("motion_notify_event", hover)
    canvas.draw()

    return hover_cid