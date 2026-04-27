import numpy as np
from sklearn.metrics import log_loss
import config
from data.preprocessing import harmonise_modalities, impute_missing_values


class HarmonisationParameters:
    def __init__(self, n_modalities):
        self.n_modalities = n_modalities
        self.psi_spatial = np.ones(n_modalities, dtype=np.float64)
        self.psi_temporal = np.ones(n_modalities, dtype=np.float64)
        self.psi_semantic = np.ones(n_modalities, dtype=np.float64)

    def apply(self, X):
        return harmonise_modalities(X, self.psi_spatial, self.psi_temporal, self.psi_semantic)

    def copy(self):
        new_params = HarmonisationParameters(self.n_modalities)
        new_params.psi_spatial = self.psi_spatial.copy()
        new_params.psi_temporal = self.psi_temporal.copy()
        new_params.psi_semantic = self.psi_semantic.copy()
        return new_params


def compute_categorical_crossentropy(y_true, y_prob):
    eps = 1e-12
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    return log_loss(y_true, y_prob_clipped)


def finite_difference_gradient(X_harm, y, psi_vec, psi_type, classifier, scaler, pca, delta=None):
    if delta is None:
        delta = config.FEEDBACK_DELTA

    grad = np.zeros(len(psi_vec))

    for j in range(len(psi_vec)):
        psi_plus = psi_vec.copy()
        psi_plus[j] += delta

        psi_minus = psi_vec.copy()
        psi_minus[j] -= delta

        if psi_type == "spatial":
            X_plus = harmonise_modalities(X_harm, psi_spatial=psi_plus)
            X_minus = harmonise_modalities(X_harm, psi_spatial=psi_minus)
        elif psi_type == "temporal":
            X_plus = harmonise_modalities(X_harm, psi_temporal=psi_plus)
            X_minus = harmonise_modalities(X_harm, psi_temporal=psi_minus)
        else:
            X_plus = harmonise_modalities(X_harm, psi_semantic=psi_plus)
            X_minus = harmonise_modalities(X_harm, psi_semantic=psi_minus)

        X_plus_imp = impute_missing_values(X_plus)
        X_minus_imp = impute_missing_values(X_minus)

        from sklearn.preprocessing import StandardScaler
        X_plus_scaled = scaler.transform(X_plus_imp)
        X_minus_scaled = scaler.transform(X_minus_imp)

        X_plus_pca = pca.transform(X_plus_scaled)
        X_minus_pca = pca.transform(X_minus_scaled)

        prob_plus = classifier.predict_proba(X_plus_pca)
        prob_minus = classifier.predict_proba(X_minus_pca)

        L_plus = compute_categorical_crossentropy(y, prob_plus)
        L_minus = compute_categorical_crossentropy(y, prob_minus)

        grad[j] = (L_plus - L_minus) / (2 * delta)

    return grad


class FeedbackLoop:
    def __init__(self, max_iter=None, lr=None, convergence_eps=None, delta=None):
        self.max_iter = max_iter or config.FEEDBACK_MAX_ITER
        self.lr = lr or config.FEEDBACK_LR
        self.convergence_eps = convergence_eps or config.FEEDBACK_CONVERGENCE_EPS
        self.delta = delta or config.FEEDBACK_DELTA
        self.loss_history = []
        self.convergence_k = None

    def run(self, X_raw, y, classifier, scaler, pca, psi_init=None):
        n_modalities = X_raw.shape[1]
        if psi_init is None:
            psi = HarmonisationParameters(n_modalities)
        else:
            psi = psi_init.copy()

        lr = self.lr
        loss_prev = np.inf
        self.loss_history = []

        for k in range(self.max_iter):
            X_harm = psi.apply(X_raw)
            X_imp = impute_missing_values(X_harm)
            X_scaled = scaler.transform(X_imp)
            X_pca = pca.transform(X_scaled)

            prob = classifier.predict_proba(X_pca)
            loss_k = compute_categorical_crossentropy(y, prob)
            self.loss_history.append(loss_k)

            if abs(loss_k - loss_prev) < self.convergence_eps and k > 0:
                self.convergence_k = k
                break

            cosine_decay = 0.5 * (1 + np.cos(np.pi * k / self.max_iter))
            lr_k = lr * cosine_decay

            grad_spatial = finite_difference_gradient(
                X_raw, y, psi.psi_spatial, "spatial", classifier, scaler, pca, self.delta
            )
            grad_temporal = finite_difference_gradient(
                X_raw, y, psi.psi_temporal, "temporal", classifier, scaler, pca, self.delta
            )
            grad_semantic = finite_difference_gradient(
                X_raw, y, psi.psi_semantic, "semantic", classifier, scaler, pca, self.delta
            )

            psi.psi_spatial -= lr_k * grad_spatial
            psi.psi_temporal -= lr_k * grad_temporal
            psi.psi_semantic -= lr_k * grad_semantic

            psi.psi_spatial = np.clip(psi.psi_spatial, 0.5, 2.0)
            psi.psi_temporal = np.clip(psi.psi_temporal, 0.5, 2.0)
            psi.psi_semantic = np.clip(psi.psi_semantic, 0.0, 2.0)

            loss_prev = loss_k

        if self.convergence_k is None:
            self.convergence_k = self.max_iter - 1

        return psi, self.loss_history

    def get_loss_reduction_pct(self):
        if len(self.loss_history) < 2:
            return 0.0
        return (self.loss_history[0] - self.loss_history[-1]) / self.loss_history[0] * 100
