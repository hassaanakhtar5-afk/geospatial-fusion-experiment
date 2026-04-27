import numpy as np

RANDOM_SEED = 42
EPSG_CRS = "EPSG:32630"
PIXEL_RESOLUTION_M = 10
TOTAL_PIXELS = 861000
TRAIN_GRID_PIXELS = 71289
TEST_PIXELS = 15223
UNLABELLED_PIXELS = 789711

N_RASTER_MODALITIES = 14
N_TOTAL_MODALITIES = 35
N_PCA_COMPONENTS = 16
PCA_VARIANCE_RETAINED = 0.999

RISK_CLASSES = {0: "Negligible", 1: "Low", 2: "Moderate", 3: "High Risk"}
N_CLASSES = 4
HIGH_RISK_CLASS = 3
HIGH_RISK_TEST_PIXELS = 135
HIGH_RISK_FRACTION = 0.0044

QA_THRESHOLD_SATELLITE = 0.50
QA_THRESHOLD_STATIC = 0.70

FEEDBACK_MAX_ITER = 8
FEEDBACK_LR = 0.01
FEEDBACK_EPS = 0.01
FEEDBACK_DELTA = 0.005
FEEDBACK_CONVERGENCE_EPS = 1e-4

ISOLATION_FOREST_CONTAMINATION = 0.05

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}

GEOVIT_PARAMS = {
    "n_tokens": N_PCA_COMPONENTS,
    "embed_dim": 64,
    "n_blocks": 2,
    "n_heads": 4,
    "mlp_ratio": 2.0,
    "dropout": 0.1,
    "focal_gamma": 2.0,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 80,
    "batch_size": 512,
    "early_stopping_patience": 10,
}

MODALITY_NAMES = [
    "NDVI_t1", "NDVI_t2", "NDVI_t3",
    "NDBI_t1", "NDBI_t2", "NDBI_t3",
    "NDWI_t1", "NDWI_t2",
    "RGB_t1", "RGB_t2", "RGB_t3",
    "LiDAR_DEM", "LiDAR_slope", "LiDAR_aspect", "LiDAR_flow_accum",
    "Landuse_t1", "Landuse_t2", "Landuse_t3",
    "River_Network_V1",
    "Road_Network_V2",
    "Flood_Risk_V3", "Flood_Risk_V4", "Flood_Risk_V5",
    "Sentinel2_B2", "Sentinel2_B3", "Sentinel2_B4",
    "Sentinel2_B8", "Sentinel2_B11", "Sentinel2_B12",
    "IoT_rainfall_t1", "IoT_rainfall_t2",
    "IoT_groundwater_t1", "IoT_groundwater_t2",
    "Soil_saturation_t1", "Soil_saturation_t2",
]

SATELLITE_MODALITY_INDICES = list(range(0, 12)) + list(range(23, 30))
STATIC_MODALITY_INDICES = list(range(12, 23)) + list(range(30, 35))

CLASS_DISTRIBUTION = {
    0: 0.232,
    1: 0.432,
    2: 0.327,
    3: 0.0044,
}

OUTPUT_DIR = "outputs"
FIGURE_DPI = 150
