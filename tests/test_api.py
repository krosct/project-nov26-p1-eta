import pytest
from fastapi.testclient import TestClient

from project_nov26_p1_eta.api import app, get_model
from project_nov26_p1_eta.features import FEATURES


class ModeloFalso:
    """Substitui o modelo treinado: o teste verifica o contrato da API, não a qualidade."""

    def predict(self, X):
        assert list(X.columns) == FEATURES
        return [30.04] * len(X)


PEDIDO = {
    "restaurante_lat": 22.745049,
    "restaurante_lon": 75.892471,
    "entrega_lat": 22.765049,
    "entrega_lon": 75.912471,
    "data_pedido": "2022-03-19",
    "hora_pedido": "11:30:00",
    "clima": "Sunny",
    "transito": "High",
    "tipo_pedido": "Snack",
    "veiculo": "motorcycle",
    "condicao_veiculo": 2,
    "cidade": "Urban",
    "idade_entregador": 37,
    "nota_entregador": 4.9,
    "entregas_multiplas": 0,
}


@pytest.fixture
def client():
    app.dependency_overrides[get_model] = lambda: ModeloFalso()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_ok(client):
    r = client.post("/predict", json=PEDIDO)
    assert r.status_code == 200
    assert r.json() == {"eta_min": 30.0}


def test_campos_opcionais_ausentes(client):
    pedido = {k: v for k, v in PEDIDO.items() if k not in {"idade_entregador", "nota_entregador"}}
    assert client.post("/predict", json=pedido).status_code == 200


@pytest.mark.parametrize(
    "campo,valor",
    [
        ("clima", "Tornado"),
        ("restaurante_lat", 500),
        ("nota_entregador", 6),
        ("hora_pedido", "25h"),
    ],
)
def test_entrada_invalida_retorna_422(client, campo, valor):
    assert client.post("/predict", json={**PEDIDO, campo: valor}).status_code == 422
