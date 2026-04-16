from pathlib import Path
import sys
import traceback

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.logger import log
import r_weather_schedule_service as service


if __name__ == "__main__":
    log("===== 历史覆盖任务开始 =====", "history")

    try:
        service.run_history_override_task_for_all_sites()
        log("===== 历史覆盖任务完成 =====", "history")

    except Exception as e:
        log("❌ 历史任务异常！", "history")
        log(traceback.format_exc(), "history")