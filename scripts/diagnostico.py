"""Diagnóstico dos dados: o que o ID do entregador é, a nota e os sinais de dado sintético.

Uso: uv run --group train python scripts/diagnostico.py
Grava reports/diagnostico_dados.json, citado no README. Sem custo (CPU, poucos segundos).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict, cross_val_score

from project_nov26_p1_eta.data import load_raw
from project_nov26_p1_eta.features import (
    FEATURE_SUSPEITA,
    FEATURES,
    FEATURES_MODELO,
    GROUP,
    TARGET,
    build_features,
)

RAIZ = Path(__file__).resolve().parents[1]
SEED = 42


def lgbm() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1
    )


def main() -> None:
    df = load_raw(RAIZ / "data/raw/train.csv")
    X, y, g = build_features(df), df[TARGET], df[GROUP]
    por_id = df.groupby(GROUP)
    pedidos_dia = df["Order_Date"].value_counts().sort_index()
    dia_semana = pedidos_dia.groupby(pedidos_dia.index.dayofweek).mean().round(0)

    # O ID é uma pessoa estável? Se fosse, idade e nota quase não variariam dentro dele
    identidade = {
        "n_ids": int(df[GROUP].nunique()),
        "pedidos_por_id": round(len(df) / df[GROUP].nunique(), 1),
        "idades_distintas_por_id_mediana": float(por_id["Delivery_person_Age"].nunique().median()),
        "notas_distintas_por_id_mediana": float(
            por_id["Delivery_person_Ratings"].nunique().median()
        ),
        "desvio_nota_dentro_do_id": round(float(por_id["Delivery_person_Ratings"].std().mean()), 3),
        "desvio_nota_total": round(float(df["Delivery_person_Ratings"].std()), 3),
    }

    # A nota carrega informação sobre o tempo além das demais features?
    cv = GroupKFold(n_splits=5)
    pred_sem = cross_val_predict(lgbm(), X[FEATURES_MODELO], y, groups=g, cv=cv)
    nota = {
        "corr_nota_tempo": round(
            float(df[["Delivery_person_Ratings", TARGET]].corr().iloc[0, 1]), 3
        ),
        "corr_nota_residuo_do_modelo_sem_nota": round(
            float(pd.Series(y - pred_sem).corr(df[FEATURE_SUSPEITA])), 3
        ),
        "mae_cv_sem_nota": round(float(np.abs(y - pred_sem).mean()), 3),
        "mae_cv_com_nota": round(
            float(
                -cross_val_score(
                    lgbm(), X[FEATURES], y, groups=g, cv=cv, scoring="neg_mean_absolute_error"
                ).mean()
            ),
            3,
        ),
        "leitura": (
            "A nota muda pedido a pedido e ainda explica parte do resíduo; não dá para provar "
            "que é conhecida na hora do pedido, então fica fora do modelo principal."
        ),
    }

    # Agrupar por ID importa? Compara com um split aleatório (mesmo modelo, mesmas features)
    aleatorio = -cross_val_score(
        lgbm(),
        X[FEATURES_MODELO],
        y,
        cv=KFold(5, shuffle=True, random_state=SEED),
        scoring="neg_mean_absolute_error",
    ).mean()
    split = {
        "mae_cv_agrupado_por_id": nota["mae_cv_sem_nota"],
        "mae_cv_aleatorio": round(float(aleatorio), 3),
    }

    sinais_sinteticos = {
        "dias_com_pedidos": int(len(pedidos_dia)),
        "dias_sem_pedidos_entre_o_primeiro_e_o_ultimo": int(
            (pedidos_dia.index.max() - pedidos_dia.index.min()).days + 1 - len(pedidos_dia)
        ),
        "media_pedidos_por_dia_da_semana_seg_a_dom": [int(v) for v in dia_semana.tolist()],
        "primeiro_dia": str(pedidos_dia.index.min().date()),
        "ultimo_dia": str(pedidos_dia.index.max().date()),
    }
    saida = {
        "identidade_do_id": identidade,
        "nota_do_entregador": nota,
        "split_agrupado_x_aleatorio": split,
        "calendario": sinais_sinteticos,
    }
    (RAIZ / "reports/diagnostico_dados.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False)
    )
    print(json.dumps(saida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
