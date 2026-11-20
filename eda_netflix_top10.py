# %% [markdown]
# # Desafio EDA — Netflix Daily Top 10
# Analisa: tipos de dados, período, tamanho da base, dados nulos e outliers.
# %%
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("netflix_daily_top_10.csv")

# %% 1. Tamanho da base
print("Linhas x colunas:", df.shape)
print(f"Memória: {df.memory_usage(deep=True).sum() / 1024:.0f} KB")
print(df.head(3).to_string())

# %% 2. Tipos de dados
print(df.dtypes)
# Todas as colunas de data (As of, Netflix Release Date) vêm como texto; Rank, Days In Top 10 e Viewership Score são inteiros.
# 'Year to Date Rank' e 'Last Week Rank' são numéricas, mas lidas como texto por causa do "-" (sem posição).
df["As of"] = pd.to_datetime(df["As of"])
df["Netflix Release Date"] = pd.to_datetime(df["Netflix Release Date"], format="%b %d, %Y")
for c in ["Year to Date Rank", "Last Week Rank"]:
    df[c] = pd.to_numeric(df[c].replace("-", np.nan))          # "-" vira nulo de verdade
print(df.dtypes)
print(df.nunique())

# %% 3. Período da análise
ini, fim = df["As of"].min(), df["As of"].max()
print(f"Período: {ini.date()} a {fim.date()} ({(fim - ini).days + 1} dias corridos)")
print("Datas distintas:", df["As of"].nunique(), "| datas faltando no intervalo:", len(set(pd.date_range(ini, fim)) - set(df["As of"])))
print("Linhas por data:", df.groupby("As of").size().unique())   # sempre 10 (o Top 10 de cada dia)
print("Lançamentos dos títulos:", df["Netflix Release Date"].min().date(), "a", df["Netflix Release Date"].max().date())

# %% 4. Dados nulos
nulos = df.isna().sum().to_frame("nulos").assign(pct=lambda d: (d["nulos"] / len(df) * 100).round(1))
print(nulos[nulos["nulos"] > 0])
# 'Netflix Exclusive' só tem "Yes" ou vazio: o vazio significa "não é exclusivo", não é dado perdido
print(df["Netflix Exclusive"].value_counts(dropna=False))
print(df.groupby("Type")["Netflix Exclusive"].apply(lambda s: s.isna().mean()).round(2))
# '-' em Year to Date Rank / Last Week Rank = título que não estava no ranking: também é uma informação, não um erro
print("'Last Week Rank' nulo:", df["Last Week Rank"].isna().sum(), "| 'Year to Date Rank' nulo:", df["Year to Date Rank"].isna().sum())
df["Netflix Exclusive"] = df["Netflix Exclusive"].fillna("No")

# %% 5. Qualidade dos dados (duplicados e inconsistências)
print("Linhas duplicadas:", df.duplicated().sum(), "| (data, rank) duplicados:", df.duplicated(["As of", "Rank"]).sum())
print("Linhas com lançamento posterior à data do ranking:", (df["Netflix Release Date"] > df["As of"]).sum())
print("Títulos truncados com '…':", df["Title"].str.contains("…").sum(), "linhas | Tipos truncados:", df.loc[df["Type"].str.contains("…"), "Type"].unique())
print("Títulos com caracteres corrompidos (encoding):", df["Title"].str.contains("â|Ã").sum(), "linhas")
print(df["Type"].value_counts())

# %% 6. Estatísticas e outliers
num = ["Days In Top 10", "Viewership Score"]
print(df[num].describe().round(1))
print("Assimetria:", df[num].skew().round(2).to_dict())

def iqr(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return q1, q3, q3 + 1.5 * i, q3 + 3 * i
for c in num:
    q1, q3, lim, lim3 = iqr(df[c])
    print(f"{c}: Q1={q1:.0f} Q3={q3:.0f} limite (1,5×IQR)={lim:.1f} -> {(df[c] > lim).sum()} linhas ({(df[c] > lim).mean()*100:.1f}%) | extremos (3×IQR)={(df[c] > lim3).sum()}")
# 'Days In Top 10' é cumulativo: a mesma série reaparece em muitos dias. Olhar por título evita contar o mesmo título várias vezes.
por_titulo = df.groupby("Title").agg(dias_max=("Days In Top 10", "max"), score_max=("Viewership Score", "max"), tipo=("Type", "first"))
q1, q3, lim, _ = iqr(por_titulo["dias_max"])
print(f"\nPor título ({len(por_titulo)} títulos): mediana de dias no Top 10 = {por_titulo['dias_max'].median():.0f}; limite 1,5×IQR = {lim:.1f} -> {(por_titulo['dias_max'] > lim).sum()} títulos")
print(por_titulo.sort_values("dias_max", ascending=False).head(8))
print(df.nlargest(3, "Viewership Score")[["As of", "Title", "Days In Top 10", "Viewership Score"]])
print("Correlação de Spearman:", df[["Days In Top 10", "Viewership Score", "Rank"]].corr(method="spearman").round(2).to_dict())
print("Sem Cocomelon: máximo de dias =", df.loc[df["Title"] != "Cocomelon", "Days In Top 10"].max(), "| máximo de score =", df.loc[df["Title"] != "Cocomelon", "Viewership Score"].max())

# %% 7. Gráficos
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for j, c in enumerate(num):
    ax[0, j].boxplot(df[c], vert=False); ax[0, j].set_title(f"Boxplot — {c}"); ax[0, j].set_xscale("log")
    ax[1, j].hist(df[c], bins=60, color="#e50914"); ax[1, j].set_yscale("log"); ax[1, j].set_title(f"Distribuição — {c} (eixo y em log)")
plt.tight_layout(); plt.savefig("outliers.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
nulos["nulos"].plot.bar(ax=ax[0], color="#e50914"); ax[0].set_title("Valores nulos por coluna (após converter '-')"); ax[0].tick_params(axis="x", rotation=70)
df["Type"].value_counts().plot.bar(ax=ax[1], color="#e50914"); ax[1].set_title("Linhas por tipo"); ax[1].tick_params(axis="x", rotation=0)
df.groupby(df["As of"].dt.to_period("M")).size().plot(ax=ax[2], color="#e50914"); ax[2].set_title("Linhas por mês (constante: 10 por dia)")
plt.tight_layout(); plt.savefig("nulos_tipos.png", dpi=120); plt.close()

c = df[df["Title"] == "Cocomelon"].sort_values("As of")
plt.figure(figsize=(8, 4)); plt.plot(c["As of"], c["Days In Top 10"], color="#e50914"); plt.title("Cocomelon: dias acumulados no Top 10"); plt.ylabel("dias"); plt.tight_layout()
plt.savefig("cocomelon.png", dpi=120); plt.close()

# %% [markdown]
# ## 8. Conclusões da EDA
# - **Tamanho:** 7.100 linhas × 10 colunas (~2,8 MB em memória). Cada linha é uma posição do Top 10 em um dia: 710 dias × 10 posições.
# - **Tipos de dados:** 3 inteiros (`Rank`, `Days In Top 10`, `Viewership Score`) e 7 colunas de texto. Duas delas são datas (`As of`, `Netflix Release Date`) e foram convertidas para datetime;
#   `Year to Date Rank` e `Last Week Rank` são numéricas, mas vinham como texto por causa do "-", e foram convertidas para número.
#   `Type` e `Netflix Exclusive` são categóricas (4 e 1 valores distintos).
# - **Período:** 01/04/2020 a 11/03/2022 (710 dias corridos, sem nenhuma data faltando e com exatamente 10 linhas por dia). Os títulos listados foram lançados entre 2007 e 03/2022.
# - **Nulos:** apenas `Netflix Exclusive` tem nulos "reais" no arquivo (2.501 linhas, 35,2%), e eles significam "não é exclusivo da Netflix" (só existe o valor "Yes"), então foram preenchidos com "No".
#   O "-" em `Last Week Rank` (55,9%) e `Year to Date Rank` (12,1%) é um nulo disfarçado: indica que o título não estava no ranking naquela referência, e não um dado perdido.
# - **Outliers:** `Days In Top 10` e `Viewership Score` são muito assimétricas (assimetria 4,7 e 3,8; mediana 7 dias e 50 pontos, contra máximos de 428 e 1.474).
#   Pelo critério de 1,5×IQR há 743 linhas (10,5%) e 698 linhas (9,8%) acima do limite. O principal outlier é **Cocomelon**: 428 dias no Top 10 e score 1.474, contra 89 dias e 590
#   do segundo colocado em dias (Ozark). Como `Days In Top 10` é acumulado, um mesmo título aparece em muitos dias; por título, 56 dos 645 títulos passam do limite (24 dias).
#   Esses valores **não são erros**: são títulos que ficaram muito tempo no ranking, e `Days In Top 10` e `Viewership Score` têm correlação de Spearman de 0,96 (o score parece derivar dos dias e da posição).
#   Recomenda-se manter os dados e usar escala logarítmica ou medianas, ou tratar o Cocomelon à parte em modelos sensíveis a extremos.
# - **Outros problemas de qualidade:** 505 linhas com títulos truncados com "…" (e o tipo "Concert/Perf…" truncado), 276 linhas com caracteres corrompidos por encoding (ex.: "Queenâs Gambit"),
#   e 28 linhas em que a data de lançamento é posterior à data do ranking. Não há linhas duplicadas.
