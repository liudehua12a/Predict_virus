from algorithm.h_qweather_api import get_next_7_full_days_forecast_by_latlon
lat=30.67
lon=104.14
last_history_row = {
    "soil_moisture": 26.5,
    "soil_rel_humidity": 72.0,
    "soil_temp_c": 21.4,
}
a=get_next_7_full_days_forecast_by_latlon(
        lat=lat,
        lon=lon,
        last_history_row=last_history_row,
        app_key="",
        app_secret="",)
print(a)