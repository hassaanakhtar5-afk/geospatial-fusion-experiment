import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import config


class RFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None):
        self.params = params or config.RF_PARAMS
        self.model = RandomForestClassifier(**self.params)
        self.classes_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importances(self):
        return self.model.feature_importances_


class XGBFusionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None):
        self.params = params or config.XGBOOST_PARAMS
        self.model = XGBClassifier(**self.params)
        self.classes_ = None

    def fit(self, X, y):
        self.model.fit(X, y, eval_set=[(X, y)], verbose=False)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importances(self):
        return self.model.feature_importances_


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = residual + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class GeoViTModel(nn.Module):
    def __init__(self, n_tokens, embed_dim, n_blocks, n_heads, mlp_ratio, dropout, n_classes):
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.token_embed = nn.Linear(1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_attention=False):
        B, T = x.shape
        x_tokens = x.unsqueeze(-1)
        x_tokens = self.token_embed(x_tokens)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls_tokens, x_tokens], dim=1)
        x_seq = x_seq + self.pos_embed
        x_seq = self.dropout(x_seq)

        attn_weights_all = []
        for block in self.blocks:
            x_seq, attn_w = block(x_seq)
            attn_weights_all.append(attn_w)

        x_seq = self.norm(x_seq)
        cls_out = x_seq[:, 0]
        logits = self.head(cls_out)

        if return_attention:
            return logits, attn_weights_all
        return logits


class GeoViTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None):
        self.params = params or config.GEOVIT_PARAMS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes_ = None
        self.train_loss_history = []
        self.val_loss_history = []

    def _build_model(self, n_classes):
        p = self.params
        return GeoViTModel(
            n_tokens=p["n_tokens"],
            embed_dim=p["embed_dim"],
            n_blocks=p["n_blocks"],
            n_heads=p["n_heads"],
            mlp_ratio=p["mlp_ratio"],
            dropout=p["dropout"],
            n_classes=n_classes,
        ).to(self.device)

    def fit(self, X, y, X_val=None, y_val=None):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        class_counts = np.bincount(y, minlength=n_classes).astype(float)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights /= class_weights.sum() / n_classes
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        self.model = self._build_model(n_classes)
        criterion = FocalLoss(gamma=self.params["focal_gamma"], weight=weight_tensor)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.params["epochs"]
        )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True)

        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)

        self.train_loss_history = []
        self.val_loss_history = []
        best_val_loss = np.inf
        patience_counter = 0
        best_state = None

        for epoch in range(self.params["epochs"]):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(dataset)
            self.train_loss_history.append(epoch_loss)
            scheduler.step()

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                self.val_loss_history.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.params["early_stopping_patience"]:
                        if best_state is not None:
                            self.model.load_state_dict(best_state)
                        break

        if best_state is not None and X_val is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

    def get_attention_weights(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, attn_weights_all = self.model(X_t, return_attention=True)
        last_attn = attn_weights_all[-1].cpu().numpy()
        cls_attn = last_attn[:, 0, 1:]
        return cls_attn
