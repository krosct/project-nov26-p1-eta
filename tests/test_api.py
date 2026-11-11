import pytest
from fastapi.testclient import TestClient

from project_nov26_p1_eta.api import app, get_model
from project_nov26_p1_eta.features import FEATURES_MODELO


# Testes da API: usam um modelo falso e checam só o contrato (entrada, saída e erros)
class ModeloFalso:
    """Substitui o modelo treinado: o teste verifica o contrato da API, não a qualidade."""

    def predict(self, X):
        assert list(X.columns) == FEATURES_MODELO  # a API manda exatamente as colunas do modelo
        return [30.04] * len(X)


# Pedido válido de exemplo; cada teste altera um campo
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
    "entregas_multiplas": 0,
}


@pytest.fixture
def client():
    # Troca o modelo real pelo falso durante o teste
    app.dependency_overrides[get_model] = lambda: ModeloFalso()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    # A rota de saúde responde ok
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_ok(client):
    # Pedido válido devolve a previsão arredondada em 1 casa
    r = client.post("/predict", json=PEDIDO)
    assert r.status_code == 200
    assert r.json() == {"eta_min": 30.0}


def test_nota_do_entregador_nao_influencia_o_modelo(client):
    # A nota saiu do contrato da API: se vier, é ignorada em vez de chegar ao modelo
    r = client.post("/predict", json={**PEDIDO, "nota_entregador": 4.9})
    assert r.status_code == 200


def test_campos_opcionais_ausentes(client):
    # Idade é opcional: sem ela a API ainda responde
    pedido = {k: v for k, v in PEDIDO.items() if k not in {"idade_entregador"}}
    assert client.post("/predict", json=pedido).status_code == 200


@pytest.mark.parametrize(
    "campo,valor",
    [
        ("clima", "Tornado"),
        ("restaurante_lat", 500),
        ("idade_entregador", 10),
        ("hora_pedido", "25h"),
    ],
)
def test_entrada_invalida_retorna_422(client, campo, valor):
    # Categoria, faixa ou hora fora do permitido devolve erro 422
    assert client.post("/predict", json={**PEDIDO, campo: valor}).status_code == 422
