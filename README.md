# EDA — Netflix Daily Top 10

Desafio Rocketseat de Análise Exploratória de Dados: tipos de dados, período, tamanho da base, dados nulos e outliers.

## Conteúdo
- `eda_netflix_top10.py` — análise completa, com as conclusões no final do arquivo
- `netflix_daily_top_10.csv` — dataset
- `outliers.png`, `nulos_tipos.png`, `cocomelon.png` — gráficos

## Como executar
```bash
pip install -r requirements.txt
python eda_netflix_top10.py
```

## Resumo
| Item | Resultado |
|---|---|
| Tamanho | 7.100 linhas × 10 colunas |
| Período | 01/04/2020 a 11/03/2022 (710 dias, 10 linhas por dia) |
| Tipos | 3 inteiros, 2 datas (texto convertido), 2 ranks numéricos lidos como texto, 3 categóricas |
| Nulos | `Netflix Exclusive`: 2.501 (35,2%) = "não exclusivo"; "-" em `Last Week Rank` (55,9%) e `Year to Date Rank` (12,1%) |
| Outliers | `Days In Top 10` e `Viewership Score` muito assimétricas; principal outlier: Cocomelon (428 dias, score 1.474) |
| Qualidade | 505 títulos truncados, 276 com encoding corrompido, 28 lançamentos posteriores à data do ranking |
