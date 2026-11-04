import timesfm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch

# Compilação do modelo
torch.set_float32_matmul_precision("high")

tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

tfm.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
        return_backcast=True,  # Added this line to fix the error
    )
)


# --- 1. PREPARAÇÃO DOS DADOS ---

# Carregar o arquivo
df = pd.read_csv("./assets/csv/protocolos_ultimos_3anos.csv", sep=";")

df_feriados_csv = pd.read_csv("./assets/csv/feriados.csv", sep=";") 
datas_feriados = pd.to_datetime(df_feriados_csv['Data']) # Garanta que é data

# Converter data
df["DataEntrada"] = pd.to_datetime(df["DataEntrada"])

# Agregação (Contagem diária)
df = df.set_index("DataEntrada")

df_diario = df.resample("D")["id"].count().to_frame(name="qtd_protocolos")

# Criar a covariável (Fim de semana) para TODO o período disponível
df_diario["is_weekend"] = (df_diario.index.dayofweek >= 5).astype(int)

# Criar covariavel feriado
df_diario["is_holiday"] = df_diario.index.normalize().isin(datas_feriados).astype(int)
holiday_historico = df_diario["is_holiday"].values


# # --- 2. PREVISÃO REAL ---
# horizonte = 60 # Quantos dias vamos prever para frente

# # Criar as datas futuras (começando do dia seguinte ao último dia do seu arquivo)
# ultima_data_real = df_diario.index[-1]
# datas_futuras = pd.date_range(
#     start=ultima_data_real + pd.Timedelta(days=1), periods=horizonte, freq="D"
# )

# # Criar a covariável de Fim de Semana para o FUTURO
# weekend_futuro = (datas_futuras.dayofweek >= 5).astype(int)

# # Criar a covariável de feriados para o FUTURO
# holiday_futuro = datas_futuras.normalize().isin(datas_feriados).astype(int)

# # --- 3. RODANDO O TIMESFM ---
# # Transformar em arrays numpy (que o TimesFM gosta)
# target_historico = df_diario["qtd_protocolos"].values
# cov_historico = df_diario["is_weekend"].values
# cov_futuro = weekend_futuro

# weekend_total = np.concatenate([cov_historico, cov_futuro])
# holiday_total = np.concatenate([holiday_historico, holiday_futuro])

# # Executar a previsão
# forecast_result, forecast_quantis = tfm.forecast_with_covariates(
#     inputs=[target_historico],
#     dynamic_numerical_covariates={
#         "is_weekend": [weekend_total],
#         "is_holiday": [holiday_total],
#     },
#     # xreg_mode='xreg + timesfm'
# )
# previsao_final = forecast_result[0]
# previsao_final = np.maximum(previsao_final, 0)
# quantis = np.array(forecast_quantis)

# dados_brutos = quantis[0]

# # Mandamos o numpy varrer cada linha e achar quem é o menor e o maior de verdade
# # axis=1 significa "olhe entre as 10 opções de cada dia"
# limite_superior = np.max(dados_brutos, axis=1)  # O verdadeiro Teto

# limite_inferior = np.min(dados_brutos, axis=1)  # O verdadeiro Chão
# limite_inferior = np.maximum(limite_inferior, 0)  # Impede que seja menor que zero

# # --- 4. VISUALIZAÇÃO ---


# plt.figure(figsize=(15, 7))


# zoom_dias =180  # Personaliza a quantidade de dias a mostrar do histórico

# # --- PLOTAGEM (Seu código original) ---
# plt.plot(
#     df_diario.index[-zoom_dias:],
#     target_historico[-zoom_dias:],
#     label="Histórico Real",
#     color="black",
# )

# plt.fill_between(
#     datas_futuras,
#     limite_inferior,
#     limite_superior,
#     color="green",
#     alpha=0.2,
#     label="Intervalo de Confiança (10%-90%)",
# )

# plt.plot(
#     datas_futuras,
#     previsao_final,
#     label="Previsão TimesFM",
#     color="red",
#     linewidth=2,
#     linestyle="--",
# )


# plt.title(f"Previsão de Demanda de Protocolos (Próximos {horizonte} Dias)")
# plt.legend()
# plt.grid(True, alpha=0.3)

# # 1. Pegamos o "controle remoto" do eixo atual
# ax = plt.gca()

# # 2. Formatamos para mostrar Dia/Mês (Ex: 05/01)
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))

# # 3. Definimos o intervalo: Mostrar uma data a cada 2 dias (para não ficar tudo grudado)
# # Se quiser todos os dias, mude interval para 1
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

# # 4. Rotacionamos o texto em 45 graus para facilitar a leitura
# plt.xticks(rotation=45)

# # -------------------------------------
# plt.show()
# print("Previsão concluída!")
# print(
#     f"Média prevista para os próximos {horizonte} dias: {previsao_final.mean():.0f} protocolos/dia"
# )


# --- 1. CONFIGURAR O TESTE ---
horizonte = 60 # Tamanho do teste (30 dias)

# Pegamos os dados totais que temos
dados_totais = df_diario['qtd_protocolos'].values
covariavel_total = df_diario['is_weekend'].values
feriados_total = df_diario['is_holiday'].values

# CRUCIAL: Separação Treino vs Teste
# Escondemos os últimos 30 dias (input_treino vai até o dia -30)
input_treino = dados_totais[:-horizonte]

# Guardamos o "Gabarito" (os últimos 30 dias reais)
gabarito_teste = dados_totais[-horizonte:]

# --- 2. RODAR A PREVISÃO (COM COVARIÁVEIS) ---

# Nota: Passamos o 'input_treino' (curto) e a 'covariavel_total' (longa).
# O TimesFM faz a conta: Tamanho da Covariável - Tamanho do Input = Horizonte.
# Então ele vai prever exatamente os dias que faltam para completar a covariável.

forecast_result, forecast_quantiles = tfm.forecast_with_covariates(
    inputs=[input_treino],
    dynamic_numerical_covariates={
        "is_weekend": [covariavel_total],
        "is_holiday": [feriados_total]
    },
)

previsao_valores = forecast_result[0]
quantis = np.array(forecast_quantiles)

dados_brutos = quantis[0]

limite_superior = np.max(dados_brutos, axis=1)  # O verdadeiro Teto

limite_inferior = np.min(dados_brutos, axis=1)  # O verdadeiro Chão
limite_inferior = np.maximum(limite_inferior, 0)  # Impede que seja menor que zero

# Pegamos a previsão (garantindo que pegamos só a parte futura com o slice)
previsao_teste = previsao_valores[-horizonte:]

# --- 3. CALCULAR A PRECISÃO (RMSE) ---

rmse = np.sqrt(np.mean((gabarito_teste - previsao_teste)**2))

print(f"=== RESULTADO DO TESTE ===")
print(f"RMSE: {rmse:.2f}")
print(f"Interpretação: O modelo erra, em média, {rmse:.1f} protocolos por dia.")

# --- 4. VISUALIZAR A VERDADE ---
dias_visiveis = 200
plt.figure(figsize=(15, 7))

# Plotar os dados usados no treino (Preto)
# Usamos o índice (datas) correspondente
datas_treino = df_diario.index[:-horizonte]
plt.plot(datas_treino[-dias_visiveis:], input_treino[-dias_visiveis:], label="Dados de Treino (Visíveis)", color="black")

datas_teste = df_diario.index[-horizonte:]
plt.fill_between(datas_teste, limite_inferior, limite_superior, color='green', alpha=0.2, label="Intervalo de Confiança (10%-90%)")

# Plotar o Gabarito Real (Azul) - O que aconteceu de verdade
plt.plot(datas_teste, gabarito_teste, label="Realidade (Escondido)", color="blue", linewidth=2)

# Plotar a Previsão (Vermelho) - O que a IA achou que ia acontecer
plt.plot(datas_teste, previsao_teste, label="Previsão TimesFM", color="red", linestyle='--', linewidth=2)

plt.title("Previsão de Demanda de Protocolos")
plt.legend()

# # 1. Pegamos o "controle remoto" do eixo atual
ax = plt.gca()

# # 2. Formatamos para mostrar Dia/Mês (Ex: 05/01)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))

# # 3. Definimos o intervalo: Mostrar uma data a cada 2 dias (para não ficar tudo grudado)
# # Se quiser todos os dias, mude interval para 1
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

# # 4. Rotacionamos o texto em 45 graus para facilitar a leitura
plt.xticks(rotation=45, ha="right")

plt.grid(True, which='major', axis='x', linestyle='--', alpha=0.5)
# -------------------------------------
plt.show()

