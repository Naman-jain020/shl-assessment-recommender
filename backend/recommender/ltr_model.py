# backend/recommender/ltr_model.py
import numpy as np
import lightgbm as lgb

class LTRModel:
    def __init__(self, model_txt_path: str | None):
        self.model = None
        self.expected = None
        if model_txt_path:
            self.model = lgb.Booster(model_file=model_txt_path)
            # read expected num_feature from model attributes (fallback 10)
            self.expected = int(self.model.num_feature()) if hasattr(self.model, "num_feature") else 10

    def _fit_dim(self, X: np.ndarray):
        if self.model is None:
            return X  # no LTR model—return raw features
        exp = self.expected or X.shape[1]
        cur = X.shape[1]
        if cur == exp:
            return X
        if cur < exp:
            pad = np.zeros((X.shape[0], exp - cur), dtype=X.dtype)
            return np.hstack([X, pad])
        # cur > exp
        return X[:, :exp]

    def predict(self, feats: np.ndarray) -> np.ndarray:
        if self.model is None:
            # fall back to first column if you want, or a mean of features
            return feats.mean(axis=1)
        X = self._fit_dim(feats)
        return self.model.predict(X, raw_score=False)