# 🗃️ Dados

## 📌 Fonte

| | |
|---|---|
| **Dataset** | [Food Delivery Dataset](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset), de Gaurav Malik (Kaggle) |
| **Contexto** | Entregas de comida em 22 cidades da Índia, de 11/02/2022 a 06/04/2022 (44 dias) |
| **Licença** | "Other (specified in description)", mas a descrição não especifica nenhuma. Por isso os dados **não são redistribuídos** neste repositório. |
| **Tamanho** | ~9,9 MB |

## ⬇️ Como baixar

A partir da raiz do projeto:

```bash
mkdir -p data/raw
curl -L -o data/raw/food-delivery.zip \
  https://www.kaggle.com/api/v1/datasets/download/gauravmalik26/food-delivery-dataset
unzip -o data/raw/food-delivery.zip -d data/raw && rm data/raw/food-delivery.zip
```

Tudo dentro de `data/` (exceto este README) é ignorado pelo Git.

## 📁 Arquivos

| Arquivo | Linhas | Uso no projeto |
|---|---|---|
| `raw/train.csv` | 45.593 | **Único arquivo usado.** Tem o alvo; dele saem treino, validação e teste |
| `raw/test.csv` | 11.399 | Não usado: **não tem o alvo**, então não serve para avaliar |
| `raw/Sample_Submission.csv` | — | Não usado (formato de submissão do Kaggle) |

## 📖 Dicionário de colunas (`train.csv`)

| Coluna | Tipo | Descrição | Disponível na hora do pedido? |
|---|---|---|---|
| `ID` | texto | ID do pedido (ex.: `0x4607`) | — (identificador) |
| `Delivery_person_ID` | texto | ID do entregador. Começa com o código da cidade (ex.: `INDORES13DEL02` → `INDO`) | ✅ (agrupa o split, mas **não identifica uma pessoa estável**: 17 idades e 9 notas distintas por ID, em mediana) |
| `Delivery_person_Age` | número | Idade do entregador | ✅ |
| `Delivery_person_Ratings` | número | Nota do entregador | ⚠️ **Não garantido**: varia pedido a pedido dentro do mesmo ID e pode ser a avaliação da própria entrega; fora do modelo principal |
| `Restaurant_latitude` / `Restaurant_longitude` | número | Coordenadas do restaurante | ✅ |
| `Delivery_location_latitude` / `Delivery_location_longitude` | número | Coordenadas da entrega | ✅ |
| `Order_Date` | data `dd-mm-aaaa` | Data do pedido | ✅ |
| `Time_Orderd` | hora | Hora em que o pedido foi feito | ✅ |
| `Time_Order_picked` | hora | Hora em que o entregador **retirou** o pedido | ❌ **VAZAMENTO** — ver abaixo |
| `Weatherconditions` | categoria | Clima: Sunny, Stormy, Sandstorms, Cloudy, Fog, Windy | ✅ |
| `Road_traffic_density` | categoria | Trânsito: Low, Medium, High, Jam | ✅ |
| `Vehicle_condition` | número 0–3 | Condição do veículo | ✅ |
| `Type_of_order` | categoria | Snack, Meal, Drinks, Buffet | ✅ |
| `Type_of_vehicle` | categoria | motorcycle, scooter, electric_scooter, bicycle | ✅ |
| `multiple_deliveries` | número 0–3 | Quantas entregas o entregador faz na mesma rota | ⚠️ Depende de quando o ETA é calculado (decidir e documentar) |
| `Festival` | Yes/No | Se havia feriado ou festival | ✅ |
| `City` | categoria | Metropolitian, Urban, Semi-Urban | ✅ |
| `Time_taken(min)` | texto → número | **Alvo:** tempo de entrega em minutos (vem como `"(min) 24"`) | 🎯 |

## 🧹 Problemas conhecidos (levantados em 01/11/2026)

**Formatação**
- Quase todos os textos têm **espaço sobrando no fim** (`"Snack "`, `"High "`). Aplique `.str.strip()` em tudo.
- Valores faltantes vêm como o **texto** `"NaN "`, e não como nulo de verdade. No clima, aparecem como `"conditions NaN"`.
- O alvo vem como `"(min) 24"`; o clima vem com o prefixo `"conditions "`.

**Valores faltantes**

| Coluna | Faltantes |
|---|---|
| `Delivery_person_Ratings` | 1.908 |
| `Delivery_person_Age` | 1.854 |
| `Time_Orderd` | 1.731 |
| `City` | 1.200 |
| `multiple_deliveries` | 993 |
| `Weatherconditions` | 616 |
| `Road_traffic_density` | 601 |
| `Festival` | 228 |

**Valores impossíveis**
- ~3.640 restaurantes com coordenadas perto de `0` (dado ausente disfarçado).
- 431 latitudes e 162 longitudes de restaurante **negativas** (sinal invertido: as cidades ficam no hemisfério norte, a leste de Greenwich).
- 327 locais de entrega com coordenada `0.01`.
- Notas acima de 5 (ex.: `6.0`), em escala que deveria ir até 5.
- Nenhum ID de pedido duplicado (verificado na EDA).

## ⚠️ Decisões de modelagem que dependem dos dados

1. **Vazamento:** `Time_Order_picked` só existe **depois** que o pedido sai do restaurante. Um ETA calculado no momento do pedido não conhece esse valor, então a coluna **fica de fora das features**. Isso inclui qualquer feature derivada dela, como o tempo de espera até a retirada.
2. **Split agrupado por entregador:** são 1.320 IDs de entregador, com 34,5 pedidos cada. Se o mesmo ID aparece no treino e no teste, o modelo poderia "decorá-lo"; `GroupKFold` com `groups=Delivery_person_ID` evita isso por cautela (na prática, o MAE é igual ao do split aleatório, porque o ID não é uma pessoa estável).
3. **Alternativa de split temporal:** como os dados cobrem 44 dias, dá para treinar nos primeiros dias e testar nos últimos, o que simula a produção. Se for usar, documente a escolha no README principal.
