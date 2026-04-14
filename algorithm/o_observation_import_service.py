from __future__ import annotations

from pathlib import Path
from typing import Any

import m_observation_excel_reader as reader14
import k_weather_data_storage as storage


def import_observation_only(file_path: str | Path) -> dict[str, Any]:
    """
    导入真实调查 Excel：
    - 只解析
    - 只入库
    - 不触发预测
    - 不触发重算
    """
    file_path = Path(file_path)

    mapped_rows = reader14.read_and_map_observation_excel(file_path)

    inserted_ids: list[int] = []

    for row in mapped_rows:
        observation_id = storage.insert_disease_observation_row(row)
        inserted_ids.append(observation_id)

    return {
        "inserted_ids": inserted_ids,
        "insert_count": len(inserted_ids),
    }