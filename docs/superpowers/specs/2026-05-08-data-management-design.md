# 数据管理模块设计方案

## 1. 目标

在 PyQt5 主界面新增一个"数据管理"按钮，点击弹出独立管理窗口，支持对 `site_info`（站点）和 `survey_batch`（批次）两个表进行完整 CRUD 操作，并预留未来扩展能力。

---

## 2. 整体架构

```
DataManagementWindow (QDialog, 弹窗)
├── LeftSidebar (QWidget, 160px)
│   └── QListWidget (模块导航)
│       ├── "站点管理"  ← 默认激活
│       ├── "批次管理"
│       └── ... (未来扩展)
│
└── ContentArea (QWidget)
    └── StackedWidget (各模块页面)
        ├── ModulePage_Site
        └── ModulePage_Batch
```

### 模块注册机制

`DataManagementWindow` 维护 `_modules: dict[str, ManagementModule]` 注册表。每个模块实现统一接口后调用 `register_module(name, module_instance)` 即可出现在左侧导航。扩展新功能 = 新写一个 widget 类 + 一行注册，无需改动主窗口。

---

## 3. ManagementModule 接口规范

```python
class ManagementModule(QWidget):
    """所有管理模块的基类/接口"""

    MODULE_NAME: str = ""  # 左侧 sidebar 显示名称
    MODULE_ID: str = ""    # 模块唯一标识

    def __init__(self, parent=None):
        super().__init__(parent)

    def refresh(self):
        """模块被激活时调用，用于刷新数据"""
        raise NotImplementedError
```

---

## 4. 站点管理模块 (SiteManagementModule)

### 布局：上方案具栏 + 下方表格

```
+--[🔍搜索框] [➕新增站点] [📥导入Excel] [📤导出]--------------+
| site_id | 站点名称 | 省   | 市   | 纬度  | 经度  | 操作  |
|---------|---------|------|------|-------|-------|-------|
| 1       | 河北站点 | 河北 | 石家庄 | 38.0  | 114.5 | ✏️ 🗑️ |
| ...     | ...     | ...  | ...   | ...   | ...   | ...   |
+------------------------------------------------------------+
```

- **表格**：`QTableWidget`，支持按列排序
- **搜索**：按 `site_name` 模糊过滤
- **操作列**：编辑（✏️）、删除（🗑️）
- **新增/编辑**：弹出 `SiteEditDialog`
- **删除**：二次确认后软删除（`is_active = 0`）

### 字段暴露策略

UI 暴露字段：`site_name`、`province`、`city`、`lat`、`lon`、`elevation`

不暴露字段：`site_id`（主键）、`location_id`、`is_active`、`created_at`、`updated_at`（内部管理字段）

---

## 5. 批次管理模块 (BatchManagementModule)

批次挂在站点下，采用主-从布局：

```
+--[站点选择 ▼]-----------------------------------------------+
|  【左侧：站点列表】      【右侧：该站点批次表格】             |
|  🔵 河北站点             +--[➕新增批次]--------------------+ |
|  🔵 河南站点             | batch_id | 批次名 | 作物 | 操作 |
|  🔵 山东站点             |----------|-------|------|------|
|  ...                     | ...      | ...   | ...  |✏️🗑️ |
                            +--------------------------------+
```

- **左侧站点列表**：`QListWidget`，点击选中站点，右侧刷新该站点下的批次
- **站点列表顶部**：`+` 按钮新增站点（复用 `SiteEditDialog`）
- **右侧批次表格**：上方 `➕新增批次` 按钮，操作列编辑/删除
- **新增/编辑批次**：弹出 `BatchEditDialog`

### 字段暴露策略

UI 暴露字段：`batch_name`、`batch_code`、`crop_variety`、`sowing_date`、`survey_start_date`、`survey_end_date`

不暴露字段：`batch_id`、`site_id`、`is_active`、`created_at`、`updated_at`

---

## 6. 编辑弹窗

### SiteEditDialog（站点编辑）

```
+------------------------------------------------+
|  编辑站点                                    [X]|
+------------------------------------------------+
|  站点名称:  [________________]  *必填           |
|  省份:     [________________]                   |
|  城市:     [________________]                   |
|  纬度:     [________________]  范围 -90~90       |
|  经度:     [________________]  范围 -180~180     |
|  海拔:     [________________]                   |
+------------------------------------------------+
|                        [取消]    [保存]         |
+------------------------------------------------+
```

- 弹窗宽 480px
- 保存时复用 `k_weather_data_storage.insert_site_info_row`；若为编辑则执行 UPDATE
- 校验：站点名必填，纬度 -90~90，经度 -180~180

### BatchEditDialog（批次编辑）

```
+------------------------------------------------+
|  编辑批次                                    [X]|
+------------------------------------------------+
|  所属站点:  [河北站点 (只读)]                   |
|  批次名称:  [________________]  *必填           |
|  批次编码:  [________________]                   |
|  作物品种:  [________________]                   |
|  播种日期:  [____-__-__]                        |
|  调查开始:  [____-__-__]                        |
|  调查结束:  [____-__-__]                        |
+------------------------------------------------+
|                        [取消]    [保存]         |
+------------------------------------------------+
```

- 所属站点字段只读，自动填入当前选中站点
- 保存时复用 `k_weather_data_storage.insert_survey_batch_row`；若为编辑则执行 UPDATE

---

## 7. 扩展机制（未来功能）

未来新增管理模块示例：

```python
class WeatherDataModule(ManagementModule):
    MODULE_NAME = "天气数据"
    MODULE_ID = "weather"

    def refresh(self):
        # 刷新天气数据表格
        ...

# 注册到窗口
window = DataManagementWindow(self)
window.register_module(SiteManagementModule())
window.register_module(BatchManagementModule())
window.register_module(WeatherDataModule())  # 未来扩展
window.exec_()
```

扩展新模块无需修改 `DataManagementWindow` 主代码，只需新建类 + 一行注册。

---

## 8. 数据同步

- 增/删/改后调用 `self.refresh()` 刷新当前模块
- 跨模块数据变更同步：引入 `pyqtsignal` 广播 `"data_changed"` 事件，各模块订阅后自行刷新

---

## 9. 数据库操作映射

| 操作 | 函数 |
|------|------|
| 查询所有站点（管理用） | `get_all_active_sites()` |
| 按 site_name 模糊搜索 | SQL `LIKE %site_name%` |
| 插入站点 | `insert_site_info_row()` |
| 更新站点 | UPDATE SQL |
| 软删除站点 | UPDATE `is_active = 0` |
| 查询某站点下所有批次 | SQL `SELECT * FROM survey_batch WHERE site_id = ?` |
| 插入批次 | `insert_survey_batch_row()` |
| 更新批次 | UPDATE SQL |
| 软删除批次 | UPDATE `is_active = 0` |

---

## 10. 文件结构

```
ui_adapter/
├── adapter.py                  # 现有代码（不动）
└── data_management/
    ├── __init__.py
    ├── window.py               # DataManagementWindow 主弹窗 + 模块注册机制
    ├── base.py                 # ManagementModule 基类
    ├── site_module.py          # 站点管理模块（站点表格 + SiteEditDialog）
    ├── batch_module.py         # 批次管理模块（站点列表 + 批次表格 + BatchEditDialog）
    └── dialogs/
        ├── __init__.py
        ├── site_edit_dialog.py # 站点编辑弹窗
        └── batch_edit_dialog.py# 批次编辑弹窗
```
