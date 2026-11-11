import numpy as np
import pandas as pd
import pytest

from project_nov26_p1_eta.data import clean
from project_nov26_p1_eta.features import CATEGORICAS, FEATURES, build_features, haversine_km


@pytest.fixture
def bruto() -> pd.DataFrame:
    """Duas linhas no formato original do CSV, com os problemas conhecidos."""
    return pd.DataFrame(
        {
            "ID": ["0x4607 ", "0xb379 "],
            "Delivery_person_ID": ["INDORES13DEL02 ", "BANGRES18DEL02 "],
            "Delivery_person_Age": ["37", "NaN "],
            "Delivery_person_Ratings": ["4.9", "6"],
            "Restaurant_latitude": [-22.745049, 0.0],
            "Restaurant_longitude": [75.892471, 0.0],
            "Delivery_location_latitude": [22.765049, 13.043041],
            "Delivery_location_longitude": [75.912471, 77.813237],
            "Order_Date": ["19-03-2022", "25-03-2022"],
            "Time_Orderd": ["11:30:00", "NaN "],
            "Time_Order_picked": ["11:45:00", "19:50:00"],
            "Weatherconditions": ["conditions Sunny", "conditions NaN"],
            "Road_traffic_density": ["High ", "Jam "],
            "Vehicle_condition": [2, 2],
            "Type_of_order": ["Snack ", "Meal "],
            "Type_of_vehicle": ["motorcycle ", "scooter "],
            "multiple_deliveries": ["0", "1"],
            "Festival": ["No ", "NaN "],
            "City": ["Urban ", "Metropolitian "],
            "Time_taken(min)": ["(min) 24", "(min) 33"],
        }
    )


def test_haversine_mesmo_ponto():
    # Mesmo ponto: distância zero
    assert haversine_km(0, 0, 0, 0) == 0


def test_haversine_um_grau_de_latitude():
    # 1 grau de latitude vale cerca de 111 km
    assert haversine_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.1)


def test_clean_normaliza_texto_e_alvo(bruto):
    # Tira espaços, o prefixo "conditions", o "NaN" em texto e o "(min)" do alvo
    df = clean(bruto)
    assert df["Road_traffic_density"].tolist() == ["High", "Jam"]
    assert df["Weatherconditions"].iloc[0] == "Sunny"
    assert pd.isna(df["Weatherconditions"].iloc[1])
    assert df["Time_taken(min)"].tolist() == [24, 33]


def test_clean_trata_valores_invalidos(bruto):
    # Coordenada, nota e idade impossíveis viram ausentes
    df = clean(bruto)
    assert df["Restaurant_latitude"].iloc[0] == pytest.approx(22.745049)  # sinal corrigido
    assert np.isnan(df["Restaurant_latitude"].iloc[1])  # coordenada ~0 vira ausente
    assert np.isnan(df["Delivery_person_Ratings"].iloc[1])  # nota 6 fora da escala
    assert np.isnan(df["Delivery_person_Age"].iloc[1])


def test_build_features_sem_vazamento(bruto):
    # Só as features previstas; a hora da retirada (que vaza o alvo) fica de fora
    X = build_features(clean(bruto))
    assert list(X.columns) == FEATURES
    assert "Time_Order_picked" not in X.columns


def test_build_features_categorias_fixas(bruto):
    # As categorias são as da lista fixa, na mesma ordem, e a hora vem do pedido
    X = build_features(clean(bruto))
    for col, cats in CATEGORICAS.items():
        assert list(X[col].cat.categories) == cats
    assert X["hora_pedido"].iloc[0] == 11
    assert np.isnan(X["hora_pedido"].iloc[1])
