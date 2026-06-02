# US Flights ML — Tech Challenge

Este projeto foi desenvolvido como proposta de solução ao Tech Challenge, especificamente ao desafio da 3ª Fase da Pós Tech em Machine Learning Engineering. O desafio consiste em desenvolver um pipeline completo de ciência de dados, abrangendo exploração, modelagem, e interpretação dos resultados. O conjunto de dados utilizado foi o de voos nos EUA. Mais detalhes sobre o estudo conduzido podem ser encontrados neste [vídeo](https://www.youtube.com/watch?v=-_q5-EOjX-g).

## Configuração do ambiente

Requisitos:
- [git](https://git-scm.com/)
- [Python](https://www.python.org/) >= 3.12 (recomenda-se o uso do [pyenv](https://github.com/pyenv/pyenv) para gerenciar diferentes versões de Python)
- [Poetry](https://python-poetry.org/)

Início rápido:
1. Clonar repositório: `git clone https://github.com/yuriperim/FIAP-tech_challenge-us_flights_ml.git us_flights_ml`
2. Entrar no diretório do projeto: `cd us_flights_ml`
3. Instalar dependências: `poetry install`
4. Ativar ambiente virtual: `poetry shell` (em caso de erro, verificar comando [`poetry env activate`](https://python-poetry.org/docs/cli/#env-activate))

## Principais arquivos e responsabilidades
- `us_flights_ml_eda_discarded.ipynb` — análise exploratória dos voos cancelados e desviados
- `us_flights_ml_eda_delayed.ipynb` — análise exploratória dos voos atrasados
- `us_flights_ml_eda_sampled.ipynb` — análise dos voos amostrados e geração de arquivo `.parquet` para modelagem
- `us_flights_ml_supervised.ipynb` — modelagem supervisionada (classificação e regressão)
- `us_flights_ml_unsupervised.ipynb` — modelagem não supervisionada (PCA)
