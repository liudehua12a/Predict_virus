from algorithm.k_weather_data_storage import upsert_next_7_full_days_forecast_to_db
site_id = 9
lat = 30.67
lon = 104.14
history_end_date_str = "2026-04-08"
print("\n=== Step 1: 写入未来 168h 聚合日表 ===")
forecast_rows = upsert_next_7_full_days_forecast_to_db(
    site_id=site_id,
    lat=lat,
    lon=lon,
    history_end_date_str=history_end_date_str,
)