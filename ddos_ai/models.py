"""The four detectors of the original project: Gaussian Naive Bayes, LightGBM, FNN and LSTM.

Every `fit_*` function takes the prepared data set and the row numbers of the training and
validation scenarios, and returns a `Scorer` whose `score(rows)` gives the attack probability of any
rows of that data set. The validation rows are used for early stopping only; the test rows are never
seen here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .data import FlowData
from .features import FEATURES, Preprocessor, make_sequences

MODEL_NAMES = ("GNB", "LightGBM", "FNN", "LSTM")


@dataclass
class Scorer:
    name: str
    score: Callable[[np.ndarray], np.ndarray]     # row numbers -> attack probability
    info: dict = field(default_factory=dict)
    booster: object = None                        # LightGBM only: the trained booster


# ------------------------------------------------------------------------------ classical models
def fit_gnb(fd: FlowData, train: np.ndarray, val: np.ndarray, seed: int = 0) -> Scorer:
    """Gaussian Naive Bayes on log-scaled, standardised features (deterministic, `seed` unused)."""
    from sklearn.naive_bayes import GaussianNB

    pre = Preprocessor().fit(fd.X[train])
    Xs = pre.transform(fd.X)
    model = GaussianNB().fit(Xs[train], fd.y[train])
    return Scorer("GNB", lambda rows: model.predict_proba(Xs[rows])[:, 1], {"parameters": 2 * Xs.shape[1] * 2})


LIGHTGBM_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, lambda_l2=1.0,
    bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8,
    deterministic=True, force_row_wise=True, verbose=-1,
)


def fit_lightgbm(fd: FlowData, train: np.ndarray, val: np.ndarray, seed: int = 0) -> Scorer:
    """Gradient-boosted trees on the raw features, with early stopping on the validation scenarios."""
    import lightgbm as lgb

    train_set = lgb.Dataset(fd.X[train], fd.y[train], feature_name=FEATURES)
    val_set = lgb.Dataset(fd.X[val], fd.y[val], reference=train_set)
    booster = lgb.train({**LIGHTGBM_PARAMS, "seed": seed}, train_set, num_boost_round=600,
                        valid_sets=[val_set], callbacks=[lgb.early_stopping(30, verbose=False)])
    best = booster.best_iteration
    return Scorer("LightGBM", lambda rows: booster.predict(fd.X[rows], num_iteration=best),
                  {"trees": int(best)}, booster=booster)


# ------------------------------------------------------------------------------ neural networks
def _keras():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import tensorflow as tf
    from tensorflow import keras

    tf.config.experimental.enable_op_determinism()   # same seed -> same weights
    return keras


def _train_keras(model, X_train, y_train, X_val, y_val, seed: int, epochs: int = 30,
                 batch_size: int = 512, patience: int = 4) -> dict:
    keras = _keras()
    n = len(y_train)
    weight = {0: n / (2 * max((y_train == 0).sum(), 1)), 1: n / (2 * max((y_train == 1).sum(), 1))}
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
                        batch_size=batch_size, class_weight=weight, callbacks=[stop], verbose=0)
    return {"epochs": len(history.history["loss"]), "parameters": int(model.count_params())}


def fit_fnn(fd: FlowData, train: np.ndarray, val: np.ndarray, seed: int = 0) -> Scorer:
    """Feed-forward network (64-32-1) on log-scaled, standardised features of the single record."""
    keras = _keras()
    keras.utils.set_random_seed(seed)
    pre = Preprocessor().fit(fd.X[train])
    Xs = pre.transform(fd.X)
    model = keras.Sequential([
        keras.layers.Input((Xs.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    info = _train_keras(model, Xs[train], fd.y[train], Xs[val], fd.y[val], seed)
    return Scorer("FNN", lambda rows: model.predict(Xs[rows], batch_size=4096, verbose=0).ravel(), info)


def fit_lstm(fd: FlowData, train: np.ndarray, val: np.ndarray, seed: int = 0) -> Scorer:
    """LSTM over the flow's last windows: the record itself plus up to 4 earlier windows of that flow.

    Flows without history (a spoofed source is seen once) get padded steps; a mask channel tells the
    network which steps are real.
    """
    keras = _keras()
    keras.utils.set_random_seed(seed)
    pre = Preprocessor().fit(fd.X[train])
    Xs = pre.transform(fd.X)
    model = keras.Sequential([
        keras.layers.Input((fd.hist.shape[1] + 1, Xs.shape[1] + 1)),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    info = _train_keras(model, make_sequences(Xs, fd.hist, train), fd.y[train],
                        make_sequences(Xs, fd.hist, val), fd.y[val], seed)

    def score(rows: np.ndarray) -> np.ndarray:
        out = np.empty(len(rows), dtype=np.float32)
        for start in range(0, len(rows), 50_000):         # keep the sequence tensor small
            chunk = rows[start:start + 50_000]
            out[start:start + len(chunk)] = model.predict(make_sequences(Xs, fd.hist, chunk),
                                                          batch_size=4096, verbose=0).ravel()
        return out

    return Scorer("LSTM", score, info)


FITTERS = {"GNB": fit_gnb, "LightGBM": fit_lightgbm, "FNN": fit_fnn, "LSTM": fit_lstm}
#: GNB has no randomness; the other three are trained once per seed.
STOCHASTIC = {"GNB": False, "LightGBM": True, "FNN": True, "LSTM": True}


def save_lightgbm(scorer: Scorer, threshold: float, directory: str | Path = "models") -> None:
    """Write the LightGBM model and its decision threshold for the controller runtime."""
    import json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scorer.booster.save_model(str(directory / "lightgbm.txt"), num_iteration=scorer.info["trees"])
    (directory / "lightgbm_threshold.json").write_text(
        json.dumps({"threshold": threshold, "features": FEATURES}, indent=2) + "\n")
