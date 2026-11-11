"""Treina e avalia os modelos de ETA.

Uso: uv run --group train python scripts/train.py

1. Separa um teste final (20%) agrupado por entregador.
2. Compara baselines e LightGBM com GroupKFold (5 folds) nos 80% restantes.
3. Registra cada experimento no MLflow.
4. Treina o modelo escolhido em todo o conjunto de desenvolvimento,
   avalia uma única vez no teste final e salva em models/modelo.joblib.
5. Teste temporal (treino até a data de corte, teste depois) e IC95 do MAE por bootstrap.
O modelo principal NÃO usa a nota do entregador (features.FEATURE_SUSPEITA); a versão com a nota
roda como análise de sensibilidade ("lightgbm_com_nota" e "teste_com_nota").
"""

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
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
    FEATURES_MODELO,
    GROUP,
    NUMERICAS,
    TARGET,
    build_features,
)

SEED = 42  # semente fixa: mesmos resultados a cada execução
RAIZ = Path(__file__).resolve().parents[1]  # pasta do projeto
# Métricas medidas em cada fold da validação cruzada
METRICAS = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
REAMOSTRAGENS_IC = 2000  # reamostragens do bootstrap do intervalo de confiança
FRACAO_TREINO_TEMPORAL = 0.8  # fração dos DIAS usada no treino do teste temporal


def lgbm() -> LGBMRegressor:
    # Árvores de gradiente: pegam interações e tratam ausentes sozinhas
    return LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1
    )


def linear(colunas: list[str]) -> object:
    # Baseline linear: imputa ausentes, codifica categorias (one-hot) e ajusta a reta
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), [c for c in NUMERICAS if c in colunas]),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                [c for c in CATEGORICAS if c in colunas],
            ),
        ]
    )
    return make_pipeline(pre, LinearRegression())


# nome -> (modelo, colunas usadas, descrição)
EXPERIMENTOS = {
    "baseline_media": (DummyRegressor(strategy="mean"), FEATURES_MODELO, "prevê sempre a média"),
    "baseline_linear": (
        linear(FEATURES_MODELO),
        FEATURES_MODELO,
        "regressão linear + imputação + one-hot",
    ),
    "lightgbm": (lgbm(), FEATURES_MODELO, "modelo principal (sem a nota do entregador)"),
    "lightgbm_com_nota": (lgbm(), FEATURES, "sensibilidade: com a nota do entregador"),
    "lightgbm_sem_multiplas": (
        lgbm(),
        [c for c in FEATURES_MODELO if c != "multiple_deliveries"],
        "sem multiple_deliveries",
    ),
    "lightgbm_sem_idade": (
        lgbm(),
        [c for c in FEATURES_MODELO if c != "Delivery_person_Age"],
        "sem idade do entregador",
    ),
}
FINAL = "lightgbm"  # experimento que vira o modelo salvo


def main() -> None:
    # Dados limpos, features, alvo e grupos (entregador) para dividir sem misturar
    df = load_raw(RAIZ / "data/raw/train.csv")
    X, y, grupos = build_features(df), df[TARGET], df[GROUP]

    # Separa 20% de teste final: o mesmo ID nunca fica nos dois lados
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    idx_dev, idx_teste = next(gss.split(X, y, grupos))
    X_dev, y_dev, g_dev = X.iloc[idx_dev], y.iloc[idx_dev], grupos.iloc[idx_dev]
    X_teste, y_teste = X.iloc[idx_teste], y.iloc[idx_teste]
    # Confere a regra acima: nenhum ID em treino e teste ao mesmo tempo
    assert not set(g_dev) & set(grupos.iloc[idx_teste]), "Entregador em treino e teste."

    # MLflow guarda cada experimento num banco local (mlflow.db)
    mlflow.set_tracking_uri(f"sqlite:///{RAIZ / 'mlflow.db'}")
    mlflow.set_experiment("eta-p1")
    cv = GroupKFold(n_splits=5)  # 5 folds, cada ID só aparece em um
    linhas = []
    # Valida cada experimento por validação cruzada agrupada, só nos 80% de desenvolvimento
    for nome, (modelo, colunas, descricao) in EXPERIMENTOS.items():
        r = cross_validate(modelo, X_dev[colunas], y_dev, groups=g_dev, cv=cv, scoring=METRICAS)
        metricas = {
            "mae_cv": -r["test_neg_mean_absolute_error"].mean(),
            "mae_cv_std": r["test_neg_mean_absolute_error"].std(),
            "rmse_cv": -r["test_neg_root_mean_squared_error"].mean(),
            "r2_cv": r["test_r2"].mean(),
        }
        # Registra parâmetros e métricas deste experimento no MLflow
        with mlflow.start_run(run_name=nome):
            mlflow.log_params({"modelo": nome, "descricao": descricao, "n_features": len(colunas)})
            mlflow.log_metrics(metricas)
        linhas.append({"experimento": nome, "descricao": descricao, **metricas})
        print(f"{nome:26} MAE {metricas['mae_cv']:.2f} ± {metricas['mae_cv_std']:.2f}")

    # Treina o modelo final em todo o desenvolvimento e mede UMA vez no teste
    final = lgbm().fit(X_dev[FEATURES_MODELO], y_dev)
    pred = final.predict(X_teste[FEATURES_MODELO])
    teste = {
        "mae": mean_absolute_error(y_teste, pred),
        "rmse": root_mean_squared_error(y_teste, pred),
        "r2": r2_score(y_teste, pred),
    }
    # Erro absoluto de cada pedido, base do MAE e do bootstrap do IC95
    erros = np.abs(y_teste.to_numpy() - pred)
    rng = np.random.default_rng(SEED)  # sorteio fixo do bootstrap
    maes = [erros[rng.integers(0, len(erros), len(erros))].mean() for _ in range(REAMOSTRAGENS_IC)]
    teste["mae_ic95"] = [float(v) for v in np.percentile(maes, [2.5, 97.5])]
    # Sensibilidade: mesmo teste, treinando com a nota do entregador
    com_nota = lgbm().fit(X_dev, y_dev).predict(X_teste)
    teste_com_nota = {
        "mae": mean_absolute_error(y_teste, com_nota),
        "rmse": root_mean_squared_error(y_teste, com_nota),
        "r2": r2_score(y_teste, com_nota),
    }
    print(f"Teste final agrupado ({FINAL}): MAE {teste['mae']:.2f} | RMSE {teste['rmse']:.2f}")

    # Teste temporal: treina nos primeiros 80% dos DIAS e testa nos seguintes (IDs se repetem)
    dias = np.sort(df["Order_Date"].unique())
    data_corte = dias[int(len(dias) * FRACAO_TREINO_TEMPORAL)]
    # True nos pedidos de antes do corte
    no_treino = (df["Order_Date"] < data_corte).to_numpy()
    modelo_temporal = lgbm().fit(X[FEATURES_MODELO][no_treino], y[no_treino])
    pred_temporal = modelo_temporal.predict(X[FEATURES_MODELO][~no_treino])
    y_t = y[~no_treino]
    teste_temporal = {
        "mae": mean_absolute_error(y_t, pred_temporal),
        "rmse": root_mean_squared_error(y_t, pred_temporal),
        "r2": r2_score(y_t, pred_temporal),
    }
    info_temporal = {
        "data_corte": str(pd.Timestamp(data_corte).date()),
        "n_treino": int(no_treino.sum()),
        "n_teste": int((~no_treino).sum()),
        "dias_teste": int(len(dias) - int(len(dias) * FRACAO_TREINO_TEMPORAL)),
    }
    print(
        f"Teste temporal (a partir de {info_temporal['data_corte']}): "
        f"MAE {teste_temporal['mae']:.2f} | RMSE {teste_temporal['rmse']:.2f}"
    )

    # Salva o modelo para a API carregar
    (RAIZ / "models").mkdir(exist_ok=True)
    joblib.dump(final, RAIZ / "models/modelo.joblib")

    # Grava a tabela de CV, o resumo do teste e a importância de cada feature
    reports = RAIZ / "reports"
    reports.mkdir(exist_ok=True)
    pd.DataFrame(linhas).round(3).to_csv(reports / "resultados_cv.csv", index=False)
    # Ganho de cada feature nas árvores, normalizado para somar 1
    importancia = pd.Series(final.booster_.feature_importance("gain"), index=FEATURES_MODELO)
    resumo = {
        "modelo_final": FINAL,
        "n_dev": len(idx_dev),
        "n_teste": len(idx_teste),
        "features_modelo": FEATURES_MODELO,
        "teste": {
            k: ([round(x, 3) for x in v] if isinstance(v, list) else round(v, 3))
            for k, v in teste.items()
        },
        "teste_com_nota_sensibilidade": {k: round(v, 3) for k, v in teste_com_nota.items()},
        "teste_temporal": {k: round(v, 3) for k, v in teste_temporal.items()},
        "teste_temporal_info": info_temporal,
        "importancia_gain": (importancia / importancia.sum()).round(3).sort_values().to_dict(),
    }
    (reports / "resultado_teste.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
