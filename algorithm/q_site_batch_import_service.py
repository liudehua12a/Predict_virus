from __future__ import annotations

from pathlib import Path
from typing import Any

import k_weather_data_storage as storage
import p_site_batch_excel_reader as reader


def import_site_batch_excel(file_path: str | Path) -> dict[str, Any]:
    """
    导入点位+批次基础信息表：
    1. 读取 Excel
    2. 每行先写入/复用 1 个 site_info
    3. 再写入/复用该行拆出的多个 survey_batch
    """
    file_path = Path(file_path)

    mapped_rows = reader.read_and_map_site_batch_excel(file_path)

    site_ids: list[int] = []
    batch_ids: list[int] = []

    for row in mapped_rows:
        site_row = row["site_row"]
        batch_rows = row["batch_rows"]

        site_id = storage.insert_site_info_row(site_row)
        site_ids.append(site_id)

        for batch_row in batch_rows:
            batch_row["site_id"] = site_id
            batch_id = storage.insert_survey_batch_row(batch_row)
            batch_ids.append(batch_id)

    return {
        "site_ids": site_ids,
        "batch_ids": batch_ids,
    }


if __name__ == "__main__":
    demo_path = Path(__file__).resolve().parent / "data" / "点位批次基础信息表.xlsx"
    result = import_site_batch_excel(demo_path)

    print("===== 点位 + 批次导入完成 =====")
    print(result)