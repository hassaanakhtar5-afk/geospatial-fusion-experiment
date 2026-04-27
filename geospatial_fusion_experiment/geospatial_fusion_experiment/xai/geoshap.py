import numpy as np
import shap
from scipy.stats import spearmanr
import config


def compute_shap_values_xgb(model, X, background_size=100, random_state=42):
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(X), min(background_size, len(X)), replace=False)
    background = X[bg_idx]
    explainer = shap.TreeExplainer(model.model, data=background)
    shap_values = explainer.shap_values(X)
    return shap_values


def compute_shap_values_rf(model, X, background_size=100, random_state=42):
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(X), min(background_size, len(X)), replace=False)
    background = X[bg_idx]
    explainer = shap.TreeExplainer(model.model, data=background)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
        shap_values = shap_values.transpose(2, 0, 1)
    return shap_values


def compute_shap_values_geovit(model, X, background_size=50, random_state=42):
    import torch
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(X), min(background_size, len(X)), replace=False)
    background = torch.tensor(X[bg_idx], dtype=torch.float32).to(model.device)

    def f(x_np):
        import torch.nn.functional as F
        x_t = torch.tensor(x_np, dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            logits = model.model(x_t)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    explainer = shap.GradientExplainer(
        (model.model, model.model.head),
        background,
    )
    try:
        x_t = torch.tensor(X[:min(500, len(X))], dtype=torch.float32).to(model.device)
        shap_values_raw = explainer.shap_values(x_t)
        if isinstance(shap_values_raw, list):
            shap_values = np.stack(shap_values_raw, axis=0)
        else:
            shap_values = shap_values_raw
    except Exception:
        shap_values = _permutation_shap_fallback(f, X, background_size, random_state)

    return shap_values


def _permutation_shap_fallback(predict_fn, X, background_size=50, random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples = min(200, len(X))
    idx = rng.choice(len(X), n_samples, replace=False)
    X_sub = X[idx]
    n_features = X_sub.shape[1]
    n_classes = predict_fn(X_sub[:1]).shape[1]
    shap_vals = np.zeros((n_classes, n_samples, n_features))

    baseline_probs = predict_fn(X_sub)

    for f in range(n_features):
        X_perm = X_sub.copy()
        X_perm[:, f] = rng.choice(X_sub[:, f], n_samples)
        perm_probs = predict_fn(X_perm)
        diff = baseline_probs - perm_probs
        for c in range(n_classes):
            shap_vals[c, :, f] = diff[:, c]

    return shap_vals


def compute_mean_abs_shap_per_modality(shap_values, class_idx=None):
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        if class_idx is not None:
            sv = shap_values[class_idx]
        else:
            sv = np.abs(shap_values).mean(axis=0)
    elif isinstance(shap_values, list):
        sv = np.abs(shap_values[class_idx]) if class_idx is not None else np.mean([np.abs(s) for s in shap_values], axis=0)
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    return mean_abs


def compute_shap_spatial_entropy(shap_values, class_idx, grid_shape=(40, 30)):
    if shap_values.ndim == 3:
        sv_class = np.abs(shap_values[class_idx])
    else:
        sv_class = np.abs(shap_values)

    pixel_importance = sv_class.mean(axis=1)
    n_pixels = min(len(pixel_importance), grid_shape[0] * grid_shape[1])
    pixel_importance = pixel_importance[:n_pixels]

    total = pixel_importance.sum() + 1e-12
    p = pixel_importance / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p + 1e-12))
    return entropy


def backproject_attention_to_modalities(attn_weights_mean, pca_components):
    projected = attn_weights_mean @ np.abs(pca_components)
    projected = projected / (projected.sum() + 1e-12)
    return projected


def compute_geoshap_stability(model_fn, X, y, n_splits=5, random_state=42):
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    top_k = 5
    top_features_per_fold = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_val_fold = X[val_idx]
        shap_vals = _permutation_shap_fallback(model_fn, X_val_fold, random_state=random_state + fold_idx)
        if shap_vals.ndim == 3:
            mean_abs = np.abs(shap_vals).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx = set(np.argsort(mean_abs)[-top_k:])
        top_features_per_fold.append(top_idx)

    jaccard_scores = []
    for i in range(len(top_features_per_fold)):
        for j in range(i + 1, len(top_features_per_fold)):
            inter = len(top_features_per_fold[i] & top_features_per_fold[j])
            union = len(top_features_per_fold[i] | top_features_per_fold[j])
            jaccard_scores.append(inter / union if union > 0 else 0.0)

    return np.mean(jaccard_scores) if jaccard_scores else 0.0


def compute_spatial_sensitivity_map(predict_fn, X_ref, grid_shape=(40, 30), n_perturb=20, random_state=42):
    rng = np.random.default_rng(random_state)
    n_pixels = grid_shape[0] * grid_shape[1]
    n_samples = min(n_pixels, len(X_ref))
    X_sample = X_ref[:n_samples]

    baseline_probs = predict_fn(X_sample)
    hr_class = config.HIGH_RISK_CLASS

    sensitivity = np.zeros(n_samples)
    for _ in range(n_perturb):
        X_perturbed = X_sample + rng.normal(0, 0.1, X_sample.shape)
        perturbed_probs = predict_fn(X_perturbed)
        sensitivity += np.abs(perturbed_probs[:, hr_class] - baseline_probs[:, hr_class])

    sensitivity /= n_perturb
    sensitivity_map = sensitivity[:min(n_samples, n_pixels)]
    pad_size = n_pixels - len(sensitivity_map)
    if pad_size > 0:
        sensitivity_map = np.concatenate([sensitivity_map, np.zeros(pad_size)])
    return sensitivity_map.reshape(grid_shape)


def compute_ndwi_attention_map(attn_weights, grid_shape=(37, 22), ndwi_token_indices=None):
    if ndwi_token_indices is None:
        ndwi_token_indices = [6, 7]

    n_samples = len(attn_weights)
    n_pixels = min(n_samples, grid_shape[0] * grid_shape[1])

    ndwi_attn = attn_weights[:n_pixels, ndwi_token_indices].mean(axis=1)
    ndwi_attn_norm = (ndwi_attn - ndwi_attn.min()) / (ndwi_attn.max() - ndwi_attn.min() + 1e-12)

    pad_size = grid_shape[0] * grid_shape[1] - len(ndwi_attn_norm)
    if pad_size > 0:
        ndwi_attn_norm = np.concatenate([ndwi_attn_norm, np.zeros(pad_size)])

    return ndwi_attn_norm.reshape(grid_shape)
