import matplotlib.pyplot as plt
import numpy as np

# 1. 填入你提供的 JSON/Dict 数据
result={'batch_id': 2, 'forecast_end_date': '2026-04-08', 'predict_dates': ['2026-04-03', '2026-04-04', '2026-04-05', '2026-04-06', '2026-04-07', '2026-04-08'], 'prediction_run_id': 'predrun_20260402211319_4f686bb1', 'results_by_disease': {'blight': [{'date': '2026-04-03', 'disease_cn': '大斑病', 'disease_key': 'blight', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'blight_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'blight_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0042, 'prev_target_1_name': 'blight_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'blight_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-04', 'disease_cn': '大斑病', 'disease_key': 'blight', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'blight_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'blight_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0042, 'prev_target_1_name': 'blight_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'blight_index', 'prev_target_2_value': 0.0042, 'site_id': 7}, {'date': '2026-04-05', 'disease_cn': '大斑病', 'disease_key': 'blight', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'blight_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'blight_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0042, 'prev_target_1_name': 'blight_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'blight_index', 'prev_target_2_value': 0.0042, 'site_id': 7}, {'date': '2026-04-06', 'disease_cn': '大斑病', 'disease_key': 'blight', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'blight_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'blight_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.009, 'prev_target_1_name': 'blight_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'blight_index', 'prev_target_2_value': 0.0042, 'site_id': 7}, {'date': '2026-04-07', 'disease_cn': '大斑病', 'disease_key': 'blight', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'blight_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'blight_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0151, 'prev_target_1_name': 'blight_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'blight_index', 'prev_target_2_value': 0.009, 'site_id': 7}, {'date': '2026-04-08', 'disease_cn': '大斑病', 'disease_key': 'blight', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'blight_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'blight_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0342, 'prev_target_1_name': 'blight_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'blight_index', 'prev_target_2_value': 0.0151, 'site_id': 7}], 'gray': [{'date': '2026-04-03', 'disease_cn': '灰斑病', 'disease_key': 'gray', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'gray_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'gray_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 1.7262, 'prev_target_1_name': 'gray_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'gray_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-04', 'disease_cn': '灰斑病', 'disease_key': 'gray', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'gray_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'gray_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 3.66, 'prev_target_1_name': 'gray_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'gray_index', 'prev_target_2_value': 1.7262, 'site_id': 7}, {'date': '2026-04-05', 'disease_cn': '灰斑病', 'disease_key': 'gray', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'gray_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'gray_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 5.8926, 'prev_target_1_name': 'gray_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'gray_index', 'prev_target_2_value': 3.66, 'site_id': 7}, {'date': '2026-04-06', 'disease_cn': '灰斑病', 'disease_key': 'gray', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'gray_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'gray_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 7.8338, 'prev_target_1_name': 'gray_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'gray_index', 'prev_target_2_value': 5.8926, 'site_id': 7}, {'date': '2026-04-07', 'disease_cn': '灰斑病', 'disease_key': 'gray', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'gray_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'gray_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 8.871, 'prev_target_1_name': 'gray_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'gray_index', 'prev_target_2_value': 7.8338, 'site_id': 7}, {'date': '2026-04-08', 'disease_cn': '灰斑病', 'disease_key': 'gray', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'gray_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'gray_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 9.6449, 'prev_target_1_name': 'gray_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'gray_index', 'prev_target_2_value': 8.871, 'site_id': 7}], 'white': [{'date': '2026-04-03', 'disease_cn': '白斑病', 'disease_key': 'white', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'white_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'white_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0, 'prev_target_1_name': 'white_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'white_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-04', 'disease_cn': '白斑病', 'disease_key': 'white', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'white_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'white_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0, 'prev_target_1_name': 'white_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'white_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-05', 'disease_cn': '白斑病', 'disease_key': 'white', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'white_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'white_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0, 'prev_target_1_name': 'white_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'white_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-06', 'disease_cn': '白斑病', 'disease_key': 'white', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'white_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'white_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0, 'prev_target_1_name': 'white_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'white_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-07', 'disease_cn': '白斑病', 'disease_key': 'white', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'white_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'white_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0, 'prev_target_1_name': 'white_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'white_index', 'prev_target_2_value': 0.0, 'site_id': 7}, {'date': '2026-04-08', 'disease_cn': '白斑病', 'disease_key': 'white', 'pred_overall_risk': '低风险', 'pred_target_1_name': 'white_incidence', 'pred_target_1_risk': '低风险', 'pred_target_1_value': 0.0, 'pred_target_2_name': 'white_index', 'pred_target_2_risk': '低风险', 'pred_target_2_value': 0.0, 'prev_target_1_name': 'white_incidence', 'prev_target_1_value': 0.0, 'prev_target_2_name': 'white_index', 'prev_target_2_value': 0.0, 'site_id': 7}]}, 'site_id': 7, 'start_source_type': 'zero_init', 'today_date': '2026-04-02', 'yesterday_date': '2026-04-01'}


# 2. 设置中文字体，防止图表中的中文显示为方块
# plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei'] # 根据你的系统选择可用中文字体
# plt.rcParams['axes.unicode_minus'] = False
import platform

# 自动检测操作系统并设置对应的中文字体
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
elif system == 'Darwin': # Mac
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

plt.rcParams['axes.unicode_minus'] = False
# 3. 数据提取与准备
dates = result['predict_dates']
diseases = ['blight', 'gray', 'white']
disease_names = ['大斑病', '灰斑病', '白斑病']

# 提取每天每种病害的病情指数(pred_target_2_value)
values = {d: [] for d in diseases}
for d in diseases:
    for entry in result['results_by_disease'][d]:
        values[d].append(entry['pred_target_2_value'])

# 4. 绘图设置
x = np.arange(len(dates))  # x轴的标签位置
width = 0.25               # 柱子的宽度

fig, ax = plt.subplots(figsize=(10, 6))

# 画分组柱状图
rects1 = ax.bar(x - width, values['blight'], width, label=disease_names[0], color='skyblue')
rects2 = ax.bar(x, values['gray'], width, label=disease_names[1], color='orange')
rects3 = ax.bar(x + width, values['white'], width, label=disease_names[2], color='lightgreen')

# 5. 添加图表修饰元素
ax.set_ylabel('预测病情指数 (Disease Index)')
ax.set_title('未来6天玉米病害发病程度预测')
ax.set_xticks(x)
ax.set_xticklabels(dates)
ax.legend() # 显示图例

# 6. 为柱状图添加具体数值标签（可选，让数据更易读）
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0: # 如果数值为0可以考虑不显示，保持图表整洁
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直向上偏移3个点
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 7. 调整布局并展示或保存
fig.tight_layout()
plt.show()
# plt.savefig('corn_disease_forecast.png') # 如果你想保存为图片可以使用这行