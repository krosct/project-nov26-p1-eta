"""Treina e avalia os modelos de ETA.

Uso: uv run --group train python scripts/train.py

1. Separa um teste final (20%) agrupado por entregador.
2. Compara baselines e LightGBM com GroupKFold (5 folds) nos 80% restantes.
3. Registra cada experimento no MLflow.
4. Treina o modelo escolhido em todo o conjunto de desenvolvimento,
   avalia uma única vez no teste final e salva em models/modelo.joblib.
"""

import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from project_nov26_p1_eta.data import load_raw
from project_nov26_p1_eta.features import (
    CATEGORICAS,
    FEATURES,
    GROUP,
    NUMERICAS,
    TARGET,
    build_features,
)

SEED = 42
RAIZ = Path(__file__).resolve().parents[1]
METRICAS = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
PERFIL_ENTREGADOR = ["Delivery_person_Age", "Delivery_person_Ratings"]


def lgbm() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1
    )


def linear() -> object:
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERICAS),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                list(CATEGORICAS),
            ),
        ]
    )
    return make_pipeline(pre, LinearRegression())


# nome -> (modelo, colunas usadas, descrição)
EXPERIMENTOS = {
    "baseline_media": (DummyRegressor(strategy="mean"), FEATURES, "prevê sempre a média"),
    "baseline_linear": (linear(), FEATURES, "regressão linear + imputação + one-hot"),
    "lightgbm": (lgbm(), FEATURES, "todas as features"),
    "lightgbm_sem_multiplas": (
        lgbm(),
        [c for c in FEATURES if c != "multiple_deliveries"],
        "sem multiple_deliveries",
    ),
    "lightgbm_sem_perfil": (
        lgbm(),
        [c for c in FEATURES if c not in PERFIL_ENTREGADOR],
        "sem idade e nota do entregador",
    ),
}
FINAL = "lightgbm"


def main() -> None:
    df = load_raw(RAIZ / "data/raw/train.csv")
    X, y, grupos = build_features(df), df[TARGET], df[GROUP]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    idx_dev, idx_teste = next(gss.split(X, y, grupos))
    X_dev, y_dev, g_dev = X.iloc[idx_dev], y.iloc[idx_dev], grupos.iloc[idx_dev]
    X_teste, y_teste = X.iloc[idx_teste], y.iloc[idx_teste]
    assert not set(g_dev) & set(grupos.iloc[idx_teste]), "Entregador em treino e teste."

    mlflow.set_tracking_uri(f"sqlite:///{RAIZ / 'mlflow.db'}")
    mlflow.set_experiment("eta-p1")
    cv = GroupKFold(n_splits=5)
    linhas = []
    for nome, (modelo, colunas, descricao) in EXPERIMENTOS.items():
        r = cross_validate(modelo, X_dev[colunas], y_dev, groups=g_dev, cv=cv, scoring=METRICAS)
        metricas = {
            "mae_cv": -r["test_neg_mean_absolute_error"].mean(),
            "mae_cv_std": r["test_neg_mean_absolute_error"].std(),
            "rmse_cv": -r["test_neg_root_mean_squared_error"].mean(),
            "r2_cv": r["test_r2"].mean(),
        }
        with mlflow.start_run(run_name=nome):
            mlflow.log_params({"modelo": nome, "descricao": descricao, "n_features": len(colunas)})
            mlflow.log_metrics(metricas)
        linhas.append({"experimento": nome, "descricao": descricao, **metricas})
        print(f"{nome:26} MAE {metricas['mae_cv']:.2f} ± {metricas['mae_cv_std']:.2f}")

    final = lgbm().fit(X_dev, y_dev)
    pred = final.predict(X_teste)
    teste = {
        "mae": mean_absolute_error(y_teste, pred),
        "rmse": root_mean_squared_error(y_teste, pred),
        "r2": r2_score(y_teste, pred),
    }
    print(f"Teste final ({FINAL}): MAE {teste['mae']:.2f} | RMSE {teste['rmse']:.2f}")

    (RAIZ / "models").mkdir(exist_ok=True)
    joblib.dump(final, RAIZ / "models/modelo.joblib")

    reports = RAIZ / "reports"
    reports.mkdir(exist_ok=True)
    pd.DataFrame(linhas).round(3).to_csv(reports / "resultados_cv.csv", index=False)
    importancia = pd.Series(final.booster_.feature_importance("gain"), index=FEATURES)
    resumo = {
        "modelo_final": FINAL,
        "n_dev": len(idx_dev),
        "n_teste": len(idx_teste),
        "teste": {k: round(v, 3) for k, v in teste.items()},
        "importancia_gain": (importancia / importancia.sum()).round(3).sort_values().to_dict(),
    }
    (reports / "resultado_teste.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
