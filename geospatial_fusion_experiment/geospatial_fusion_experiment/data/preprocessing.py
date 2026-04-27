import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedShuffleSplit
import config


def generate_chelmsford_modalities(n_pixels, random_state=42):
    rng = np.random.default_rng(random_state)

    labels = _generate_stratified_labels(n_pixels, rng)

    X = np.zeros((n_pixels, config.N_TOTAL_MODALITIES), dtype=np.float32)

    ndvi_base = np.where(labels == 3, 0.55, np.where(labels == 2, 0.35, np.where(labels == 1, 0.25, 0.15)))
    for i, col in enumerate([0, 1, 2]):
        X[:, col] = ndvi_base + rng.normal(0, 0.08, n_pixels) + rng.normal(0, 0.02, n_pixels) * i

    ndbi_base = np.where(labels == 0, 0.45, np.where(labels == 1, 0.35, np.where(labels == 2, 0.20, 0.10)))
    for i, col in enumerate([3, 4, 5]):
        X[:, col] = ndbi_base + rng.normal(0, 0.07, n_pixels) + rng.normal(0, 0.01, n_pixels) * i

    ndwi_base = np.where(labels == 3, 0.55, np.where(labels == 2, 0.35, np.where(labels == 1, 0.15, -0.05)))
    for i, col in enumerate([6, 7]):
        X[:, col] = ndwi_base + rng.normal(0, 0.09, n_pixels)

    rgb_base = np.where(labels == 0, 0.6, np.where(labels == 1, 0.5, np.where(labels == 2, 0.4, 0.3)))
    for i, col in enumerate([8, 9, 10]):
        X[:, col] = rgb_base + rng.normal(0, 0.10, n_pixels)

    dem_base = np.where(labels == 3, 10.0, np.where(labels == 2, 20.0, np.where(labels == 1, 40.0, 80.0)))
    X[:, 11] = dem_base + rng.normal(0, 5.0, n_pixels)
    X[:, 12] = np.where(labels == 3, 1.5, np.where(labels == 2, 4.0, 8.0)) + rng.normal(0, 1.5, n_pixels)
    X[:, 13] = rng.uniform(0, 360, n_pixels)
    flow_base = np.where(labels == 3, 500.0, np.where(labels == 2, 200.0, 50.0))
    X[:, 14] = flow_base + rng.exponential(30.0, n_pixels)

    landuse_base = np.where(labels == 0, 0.1, np.where(labels == 1, 0.3, np.where(labels == 2, 0.5, 0.7)))
    for i, col in enumerate([15, 16, 17]):
        X[:, col] = landuse_base + rng.normal(0, 0.06, n_pixels)

    river_base = np.where(labels == 3, 0.9, np.where(labels == 2, 0.55, 0.20))
    X[:, 18] = np.clip(river_base + rng.normal(0, 0.12, n_pixels), 0, 1)

    road_density = np.where(labels == 0, 0.7, np.where(labels == 1, 0.55, 0.30))
    X[:, 19] = np.clip(road_density + rng.normal(0, 0.10, n_pixels), 0, 1)

    flood_risk_base = np.where(labels == 3, 0.95, np.where(labels == 2, 0.65, np.where(labels == 1, 0.30, 0.05)))
    for i, col in enumerate([20, 21, 22]):
        X[:, col] = np.clip(flood_risk_base + rng.normal(0, 0.06, n_pixels), 0, 1)

    for i, col in enumerate(range(23, 30)):
        band_base = rng.uniform(0.1, 0.9, n_pixels)
        X[:, col] = np.clip(band_base + (labels == 3).astype(float) * 0.15 + rng.normal(0, 0.05, n_pixels), 0, 1)

    iot_rain_base = np.where(labels == 3, 15.0, np.where(labels == 2, 9.0, 4.0))
    for col in [29, 30]:
        X[:, col] = np.clip(iot_rain_base + rng.normal(0, 3.0, n_pixels), 0, 50)

    gw_base = np.where(labels == 3, 0.8, np.where(labels == 2, 0.5, 0.2))
    for col in [31, 32]:
        X[:, col] = np.clip(gw_base + rng.normal(0, 0.12, n_pixels), 0, 1)

    sat_base = np.where(labels == 3, 0.75, np.where(labels == 2, 0.50, 0.25))
    for col in [33, 34]:
        X[:, col] = np.clip(sat_base + rng.normal(0, 0.10, n_pixels), 0, 1)

    X = _inject_cloud_masking_nulls(X, rng)
    X = _inject_sensor_anomalies(X, rng)

    return X, labels


def _generate_stratified_labels(n_pixels, rng):
    labels = np.zeros(n_pixels, dtype=np.int32)
    dist = config.CLASS_DISTRIBUTION
    counts = {
        0: int(dist[0] * n_pixels),
        1: int(dist[1] * n_pixels),
        2: int(dist[2] * n_pixels),
        3: int(dist[3] * n_pixels),
    }
    counts[0] += n_pixels - sum(counts.values())

    idx = 0
    for cls, count in counts.items():
        labels[idx:idx + count] = cls
        idx += count

    rng.shuffle(labels)
    return labels


def _inject_cloud_masking_nulls(X, rng):
    sentinel2_cols = list(range(0, 12)) + list(range(23, 30))
    cloud_fraction = rng.uniform(0.35, 0.55, len(sentinel2_cols))
    for i, col in enumerate(sentinel2_cols):
        mask = rng.random(X.shape[0]) < cloud_fraction[i]
        X[mask, col] = np.nan
    return X


def _inject_sensor_anomalies(X, rng):
    n_anomalies = int(0.03 * X.shape[0])
    anomaly_indices = rng.choice(X.shape[0], n_anomalies, replace=False)
    anomaly_cols = rng.integers(0, X.shape[1], n_anomalies)
    for idx, col in zip(anomaly_indices, anomaly_cols):
        X[idx, col] = rng.choice([-9999.0, 9999.0, np.nan])
    return X


def compute_qa_scores(X):
    n_pixels, n_modalities = X.shape
    qa_scores = np.zeros(n_modalities)

    for col in range(n_modalities):
        col_data = X[:, col]
        valid = np.isfinite(col_data) & (col_data > -999) & (col_data < 9999)
        qa_scores[col] = valid.sum() / n_pixels

    return qa_scores


def apply_dual_threshold_qa(X, qa_scores):
    satellite_indices = config.SATELLITE_MODALITY_INDICES
    static_indices = config.STATIC_MODALITY_INDICES

    retained_mask = np.ones(X.shape[1], dtype=bool)

    for col in range(X.shape[1]):
        if col in satellite_indices:
            threshold = config.QA_THRESHOLD_SATELLITE
        else:
            threshold = config.QA_THRESHOLD_STATIC

        if qa_scores[col] < threshold:
            retained_mask[col] = False

    return retained_mask


def apply_static_threshold_qa(X, qa_scores, threshold=0.70):
    retained_mask = qa_scores >= threshold
    return retained_mask


def run_isolation_forest(X):
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    iso = IsolationForest(
        contamination=config.ISOLATION_FOREST_CONTAMINATION,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X_clean)
    inlier_mask = preds == 1
    return inlier_mask, iso


def impute_missing_values(X):
    X_imp = X.copy()
    for col in range(X_imp.shape[1]):
        col_data = X_imp[:, col]
        valid_mask = np.isfinite(col_data) & (col_data > -999) & (col_data < 9999)
        if valid_mask.sum() == 0:
            X_imp[:, col] = 0.0
        else:
            col_mean = col_data[valid_mask].mean()
            X_imp[~valid_mask, col] = col_mean
    return X_imp


def harmonise_modalities(X, psi_spatial=None, psi_temporal=None, psi_semantic=None):
    X_harm = X.copy()

    if psi_spatial is not None:
        for i, w in enumerate(psi_spatial):
            if i < X_harm.shape[1]:
                X_harm[:, i] *= (1.0 + 0.1 * (w - 1.0))

    if psi_temporal is not None:
        for i, coef in enumerate(psi_temporal):
            if i < X_harm.shape[1]:
                X_harm[:, i] = X_harm[:, i] * coef

    if psi_semantic is not None:
        for i, weight in enumerate(psi_semantic):
            if i < X_harm.shape[1]:
                X_harm[:, i] *= weight

    return X_harm


def fit_pca(X_train, n_components=None):
    if n_components is None:
        n_components = config.N_PCA_COMPONENTS
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
    pca.fit(X_scaled)
    return scaler, pca


def transform_pca(X, scaler, pca):
    X_scaled = scaler.transform(X)
    return pca.transform(X_scaled)


def create_train_test_split(X, y, test_size=None):
    if test_size is None:
        test_size = config.TEST_PIXELS / (config.TRAIN_GRID_PIXELS + config.TEST_PIXELS)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=config.RANDOM_SEED)
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def build_single_source_datasets(X_raw, modality_groups):
    datasets = {}
    for name, col_indices in modality_groups.items():
        datasets[name] = X_raw[:, col_indices]
    return datasets


def get_modality_groups():
    return {
        "DEM Only": [11, 12, 13, 14],
        "Landuse Only": [15, 16, 17],
        "Sentinel-2 Only": list(range(0, 12)) + list(range(23, 30)),
        "Road Network": [19],
        "Flood Risk Vectors": [20, 21, 22],
    }
