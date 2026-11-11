"""API de previsão de tempo de entrega.

Executar: uv run uvicorn project_nov26_p1_eta.api:app
"""

import os
from datetime import date, time
from functools import lru_cache
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from project_nov26_p1_eta.features import FEATURES_MODELO, build_features

# Aplicação FastAPI: a documentação interativa fica em /docs
app = FastAPI(
    title="ETA de entregas",
    description="Previsão do tempo de entrega (minutos) a partir das informações do pedido.",
    version="1.0.0",
)


class Pedido(BaseModel):
    # Entrada da API; faixas e categorias inválidas viram erro 422
    restaurante_lat: float = Field(ge=-90, le=90, examples=[22.745049])
    restaurante_lon: float = Field(ge=-180, le=180, examples=[75.892471])
    entrega_lat: float = Field(ge=-90, le=90, examples=[22.765049])
    entrega_lon: float = Field(ge=-180, le=180, examples=[75.912471])
    data_pedido: date = Field(examples=["2022-03-19"])
    hora_pedido: time = Field(examples=["11:30:00"])
    clima: Literal["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"]
    transito: Literal["Low", "Medium", "High", "Jam"]
    tipo_pedido: Literal["Snack", "Meal", "Drinks", "Buffet"]
    veiculo: Literal["motorcycle", "scooter", "electric_scooter", "bicycle"]
    condicao_veiculo: int = Field(ge=0, le=3)
    cidade: Literal["Urban", "Metropolitian", "Semi-Urban"]
    festival: bool = False  # feriado ou festival no dia
    idade_entregador: float | None = Field(default=None, ge=15, le=70)  # opcional
    entregas_multiplas: int | None = Field(default=None, ge=0, le=3)  # opcional


class Previsao(BaseModel):
    # Saída da API: tempo previsto em minutos
    eta_min: float


def pedido_para_dataframe(pedido: Pedido) -> pd.DataFrame:
    """Converte o pedido para o mesmo esquema que `data.clean` produz."""
    linha = {
        "Restaurant_latitude": pedido.restaurante_lat,
        "Restaurant_longitude": pedido.restaurante_lon,
        "Delivery_location_latitude": pedido.entrega_lat,
        "Delivery_location_longitude": pedido.entrega_lon,
        "Order_Date": pd.Timestamp(pedido.data_pedido),
        "Time_Orderd": pedido.hora_pedido.strftime("%H:%M:%S"),
        "Weatherconditions": pedido.clima,
        "Road_traffic_density": pedido.transito,
        "Type_of_order": pedido.tipo_pedido,
        "Type_of_vehicle": pedido.veiculo,
        "Vehicle_condition": pedido.condicao_veiculo,
        "City": pedido.cidade,
        "Festival": "Yes" if pedido.festival else "No",
        "Delivery_person_Age": pedido.idade_entregador,
        "Delivery_person_Ratings": float("nan"),  # fora do modelo (ver features.FEATURE_SUSPEITA)
        "multiple_deliveries": pedido.entregas_multiplas,
    }
    df = pd.DataFrame([linha])  # uma linha só: o pedido
    # Ausente (None) vira NaN, que o modelo trata sozinho
    numericas = ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]
    df[numericas] = df[numericas].astype(float)
    return df


@lru_cache
def get_model():
    # Carrega o modelo uma vez só; o caminho pode vir da variável MODEL_PATH
    return joblib.load(os.environ.get("MODEL_PATH", "models/modelo.joblib"))


@app.get("/health")
def health() -> dict:
    # Usado por Docker e orquestradores para saber se a API está de pé
    return {"status": "ok"}


@app.post("/predict", response_model=Previsao)
def predict(pedido: Pedido, model: Annotated[object, Depends(get_model)]) -> Previsao:
    # Mesma função de features do treino, só com as colunas que o modelo usa
    X = build_features(pedido_para_dataframe(pedido))[FEATURES_MODELO]
    return Previsao(eta_min=round(float(model.predict(X)[0]), 1))
