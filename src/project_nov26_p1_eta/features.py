"""Transformação dos dados limpos em features do modelo.

`build_features` só transforma (não aprende nada com os dados), por isso pode
rodar antes do split sem vazar informação. `Time_Order_picked` nunca é usada:
ela só existe depois que o pedido sai do restaurante.
"""

import numpy as np
import pandas as pd

TARGET = "Time_taken(min)"
GROUP = "Delivery_person_ID"  # usado só para agrupar a validação cruzada

# Categorias fixas: garantem a mesma codificação no treino e na API
CATEGORICAS = {
    "Weatherconditions": ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"],
    "Road_traffic_density": ["Low", "Medium", "High", "Jam"],
    "Type_of_order": ["Snack", "Meal", "Drinks", "Buffet"],
    "Type_of_vehicle": ["motorcycle", "scooter", "electric_scooter", "bicycle"],
    "Festival": ["No", "Yes"],
    "City": ["Urban", "Metropolitian", "Semi-Urban"],
}
NUMERICAS_BASE = [  # colunas numéricas que passam direto do dado limpo
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
]
NUMERICAS = NUMERICAS_BASE + ["distancia_km", "hora_pedido", "dia_semana"]
FEATURES = NUMERICAS + list(CATEGORICAS)  # todas as colunas que build_features devolve
# A nota muda a cada pedido e pode ser a avaliação da entrega (só existe depois dela)
FEATURE_SUSPEITA = "Delivery_person_Ratings"
# Features do modelo principal: tudo menos a nota (ela só entra na análise de sensibilidade)
FEATURES_MODELO = [c for c in FEATURES if c != FEATURE_SUSPEITA]


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em linha reta sobre a superfície da Terra, em km."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))  # graus → radianos
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371 * np.arcsin(np.sqrt(a))  # 6371 km = raio médio da Terra


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    # Distância entre restaurante e entrega
    X["distancia_km"] = haversine_km(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )
    hora = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce")
    X["hora_pedido"] = hora.dt.hour  # hora do pedido (0–23); ausente se a hora faltar
    X["dia_semana"] = df["Order_Date"].dt.day_of_week  # 0 = segunda
    for col in NUMERICAS_BASE:
        X[col] = df[col]
    for col, cats in CATEGORICAS.items():
        X[col] = pd.Categorical(df[col], categories=cats)  # categorias fixas, nunca aprendidas
    return X[FEATURES]
