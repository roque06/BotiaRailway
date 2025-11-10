# AI_MODEL/online_learner.py
import os
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

MODEL_PATH = "AI_MODEL/model_online.pkl"


class OnlineLearner:
    """
    Clasificador on-line (partial_fit) con 5 features:
    [ema_fast, ema_slow, ema_long, rsi, atr]
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.clf = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            penalty="l2",
            max_iter=1,
            learning_rate="optimal",
            random_state=42,
        )
        self.is_warmed = False  # hasta ver primeras clases
        X_init = np.array([[0, 0, 0, 50, 1]])
        y_init = np.array([0])
        self.scaler.fit(X_init)
        self.clf.partial_fit(X_init, y_init, classes=[0, 1])
        self.is_warmed = True
        # intenta cargar
        if os.path.exists(self.model_path):
            try:
                obj = joblib.load(self.model_path)
                self.scaler = obj["scaler"]
                self.clf = obj["clf"]
                self.is_warmed = obj.get("is_warmed", False)
            except Exception:
                pass

    def _save(self):
        joblib.dump(
            {"scaler": self.scaler, "clf": self.clf, "is_warmed": self.is_warmed},
            self.model_path,
        )

    def partial_update(self, X, y):
        """
        X: np.array shape (n, 5)
        y: np.array shape (n,)  (0/1)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # scaler incremental
        self.scaler.partial_fit(X)
        Xs = self.scaler.transform(X)

        # warm start: necesita clases para inicializar
        if not self.is_warmed:
            self.clf.partial_fit(Xs, y, classes=np.array([0, 1], dtype=int))
            self.is_warmed = True
        else:
            self.clf.partial_fit(Xs, y)

        self._save()

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # si aún no está caliente, devuelve 0.5
        if not self.is_warmed:
            return np.array([[0.5, 0.5] for _ in range(X.shape[0])])
        Xs = self.scaler.transform(X)
        try:
            p1 = self.clf.predict_proba(Xs)  # [:,1]
        except Exception:
            # SGD sin calibración puede no tener predict_proba en versiones antiguas
            z = self.clf.decision_function(Xs)
            # sigmoide
            p = 1.0 / (1.0 + np.exp(-z))
            p1 = np.vstack([1 - p, p]).T
        return p1
