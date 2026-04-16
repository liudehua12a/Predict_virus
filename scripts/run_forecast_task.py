from pathlib import Path
import sys
import traceback

ROOT_DIR = Path(__file__).resolve().parent.parent
ALGORITHM_DIR = ROOT_DIR / "algorithm"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ALGORITHM_DIR))

from scripts.utils.logger import log
import scripts.r_weather_schedule_service as service


if __name__ == "__main__":
    log("===== 预报任务开始 =====", "forecast")

    try:
        service.run_forecast_task_for_all_sites()
        log("===== 预报任务完成 =====", "forecast")

    except Exception as e:
        log("❌ 预报任务异常！", "forecast")
        log(traceback.format_exc(), "forecast")