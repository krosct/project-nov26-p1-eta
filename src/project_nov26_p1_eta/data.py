"""Carregamento e limpeza do Food Delivery Dataset (Kaggle)."""

from pathlib import Path

import numpy as np
import pandas as pd

# Colunas numéricas que o CSV traz como texto (têm o texto "NaN " no meio)
NUMERICAS_EM_TEXTO = ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]
COORD_RESTAURANTE = ["Restaurant_latitude", "Restaurant_longitude"]
COORD_ENTREGA = ["Delivery_location_latitude", "Delivery_location_longitude"]
TARGET = "Time_taken(min)"  # alvo: tempo de entrega em minutos


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos, ausentes e valores impossíveis. Não imputa nada."""
    # Quase todo texto vem com espaço sobrando no fim ("Snack ", "High ")
    df = df.apply(lambda col: col.str.strip() if col.dtype == "str" else col)
    df["Weatherconditions"] = df["Weatherconditions"].str.removeprefix("conditions ")
    df = df.replace("NaN", np.nan)  # o texto "NaN" vira ausente de verdade

    # O alvo vem como "(min) 24": fica só o número
    if df[TARGET].dtype == "str":
        df[TARGET] = df[TARGET].str.extract(r"(\d+)")[0].astype(int)

    for col in NUMERICAS_EM_TEXTO:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Nota acima de 5 é erro de registro (a escala é 1–5)
    df.loc[df["Delivery_person_Ratings"] > 5, "Delivery_person_Ratings"] = np.nan

    # As cidades ficam no hemisfério norte, a leste de Greenwich: negativo é sinal invertido
    for col in COORD_RESTAURANTE:
        df[col] = df[col].abs()
    # Coordenada perto de 0 é dado ausente disfarçado
    for col in COORD_RESTAURANTE + COORD_ENTREGA:
        df.loc[df[col] < 1, col] = np.nan

    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")
    return df


def load_raw(path: str | Path) -> pd.DataFrame:
    # Lê o CSV bruto e já devolve limpo
    return clean(pd.read_csv(path))
