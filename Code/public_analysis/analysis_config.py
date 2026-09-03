"""Public analysis settings for the gridded mobility-exposure models."""

TARGETS = {
    "festival_pre": "festival_pre_exposure_change",
    "post_festival": "post_festival_exposure_change",
}

BASELINE_FEATURES = {
    "festival_pre": [
        "festival_pre_count_change",
        "pre_mean_count",
        "pre_mean_pm25",
        "gdp_2018",
        "road_density_m_per_km2",
    ],
    "post_festival": [
        "post_festival_count_change",
        "festival_mean_count",
        "festival_mean_pm25",
        "road_density_m_per_km2",
    ],
}

FULL_FEATURES = {
    "festival_pre": [
        "festival_pre_count_change",
        "pre_mean_count",
        "pre_count_cv",
        "pre_day_night_ratio",
        "pre_daily_relative_amplitude",
        "pre_mean_pm25",
        "festival_pre_pm25_change",
        "festival_pre_temperature_change",
        "festival_pre_precipitation_change",
        "festival_pre_wind_speed_change",
        "road_density_m_per_km2",
        "built_up_ratio",
        "nighttime_light_2018",
    ],
    "post_festival": [
        "post_festival_count_change",
        "festival_mean_count",
        "festival_count_cv",
        "festival_day_night_ratio",
        "festival_daily_relative_amplitude",
        "festival_mean_pm25",
        "post_festival_pm25_change",
        "post_festival_temperature_change",
        "post_festival_precipitation_change",
        "post_festival_wind_speed_change",
        "road_density_m_per_km2",
        "built_up_ratio",
        "nighttime_light_2018",
    ],
}

MODEL_SETTINGS = {
    "test_size": 0.30,
    "random_state": 42,
    "spatial_block_size_m": 300_000,
    "spatial_folds": 5,
    "random_forest": {
        "n_estimators": 300,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    },
}

SPATIAL_SETTINGS = {
    "projected_crs": "EPSG:3857",
    "distance_threshold_m": 50_000,
    "permutations": 999,
    "significance_level": 0.05,
    "random_seed": 42,
}
