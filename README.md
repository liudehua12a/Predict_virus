# 玉米病害预测系统

基于 PyQt5 开发的玉米病害预测系统，用于利用多种模型（XGBoost、LSTM、LSTM-XGBoost 融合模型）预测目标区域未来7天内三种玉米病害（灰斑病、大斑病、白斑病）的发病程度，并以柱状图形式进行可视化展示。

## 功能特性

- **Excel 数据导入**：支持将 Excel 中的病害调查数据导入到数据库
- **模型选择**：提供 XGBoost、LSTM、LSTM-XGBoost 融合模型三种预测模型
- **目标区域选择**：采用联动下拉框，支持一级地点选择和二级批次选择
- **预测功能**：自动获取数据并执行预测，将结果保存到数据库
- **可视化展示**：使用柱状图展示预测结果，按风险级别着色
- **天气数据集成**：通过和风天气 API 获取实时和预测天气数据
- **历史数据处理**：支持历史数据填充和滚动预测

## 系统要求

- Python 3.8+
- PyQt5
- pandas
- openpyxl
- matplotlib
- torch
- requests
- numpy
- scikit-learn
- jwt
- xgboost

## 安装步骤

1. 克隆或下载项目到本地

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

3. 运行软件：
```bash
python main.py
```

## 使用说明

### 1. 数据导入

- 点击"导入Excel数据"按钮
- 选择包含病害调查数据的 Excel 文件
- 确保 Excel 文件包含以下列：
  - `site_id`：站点ID
  - `survey_date`：调查日期
  - `gray_incidence`：灰斑病发生率
  - `gray_index`：灰斑病指数
  - `blight_incidence`：大斑病发生率
  - `blight_index`：大斑病指数
  - `white_incidence`：白斑病发生率
  - `white_index`：白斑病指数

### 2. 模型选择

- 在"选择模型"下拉框中选择预测模型：
  - XGBoost
  - LSTM
  - LSTM-XGBoost 融合模型

### 3. 目标区域选择

- **一级选择**：选择地点（如德宏、雅安、泸定、新都等）
- **二级选择**：选择批次（如德宏-1、德宏-2 等）
- 二级选择会根据一级选择自动更新

### 4. 执行预测

- 点击"开始预测"按钮
- 系统会自动：
  1. 从数据库获取今天的病害数据
  2. 获取实时和预测天气数据
  3. 使用选择的模型进行预测
  4. 生成未来7天的病害预测结果
  5. 将结果保存到数据库
  6. 在图表区域显示可视化结果

### 5. 查看结果

- 柱状图会显示未来7天三种病害的预测结果
- 颜色说明：
  - **绿色**：0-25（低风险）
  - **蓝色**：25-50（轻度）
  - **橙色**：50-75（中度）
  - **红色**：75-100（高风险）

## 项目结构

```
├── algorithm/             # 算法核心模块
│   ├── data/              # 数据文件
│   ├── lstm_xgboost_fusion/  # 融合模型实现
│   ├── models/            # 预训练模型
│   ├── outputs/           # 输出文件
│   ├── 11_predict_disease.py  # 病害预测主文件
│   └── ...                # 其他辅助文件
├── test/                  # 测试文件
├── ui_adapter/            # UI适配器
├── README.md              # 项目说明
├── main.py                # 主程序入口
└── requirements.txt       # 依赖项
```

## 数据库结构

### disease_observation（病害观测表）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INTEGER | 主键，自增 |
| site_id | TEXT | 站点ID |
| survey_date | DATE | 调查日期 |
| gray_incidence | REAL | 灰斑病发生率 |
| gray_index | REAL | 灰斑病指数 |
| blight_incidence | REAL | 大斑病发生率 |
| blight_index | REAL | 大斑病指数 |
| white_incidence | REAL | 白斑病发生率 |
| white_index | REAL | 白斑病指数 |

### disease_prediction（病害预测表）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INTEGER | 主键，自增 |
| site_id | TEXT | 站点ID |
| predict_date | DATE | 预测日期 |
| gray_risk_level | REAL | 灰斑病发病程度 |
| blight_risk_level | REAL | 大斑病发病程度 |
| white_risk_level | REAL | 白斑病发病程度 |

## 注意事项

1. **数据准备**：在使用预测功能前，请确保已导入足够的观测数据
2. **模型文件**：确保 `algorithm/models/` 目录下存在预训练模型文件
3. **天气API**：系统使用和风天气API获取天气数据，需要确保网络连接正常
4. **数据备份**：建议定期备份数据库文件（nky-CornPre.db）
5. **日期格式**：Excel 中的日期格式应为 YYYY-MM-DD

## 故障排除

### 无法导入 Excel 文件
- 检查文件格式是否为 .xlsx 或 .xls
- 确认 Excel 文件包含所有必需的列
- 检查文件是否被其他程序占用

### 预测失败
- 确认已选择目标区域
- 检查数据库中是否有该站点的观测数据
- 检查网络连接是否正常（天气数据获取需要网络）
- 检查模型文件是否存在

### 图表显示异常
- 检查 matplotlib 是否正确安装
- 确认数据范围在 0-100 之间

## 技术栈

- **Python**：主要编程语言
- **PyQt5**：GUI 框架
- **SQLite**：本地数据库
- **pandas**：数据处理
- **matplotlib**：数据可视化
- **PyTorch**：深度学习框架（用于LSTM模型）
- **XGBoost**：梯度提升模型
- **scikit-learn**：机器学习工具
- **requests**：HTTP请求（用于天气API）
- **JWT**：身份验证（用于API调用）

## 后续开发计划

- [x] 接入实际的预测模型（XGBoost、LSTM、LSTM-XGBoost 融合模型）
- [ ] 添加历史数据查询和分析功能
- [ ] 支持导出预测结果为 Excel 或 PDF
- [ ] 添加数据校验和错误处理机制
- [ ] 优化界面设计，提高用户体验
- [ ] 增加更多病害类型的预测支持
- [ ] 实现模型自动更新机制

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，欢迎反馈。
