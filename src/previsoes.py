import timesfm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import plotly.graph_objects as go

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

# Carregar o arquivo para treinamento 
df = pd.read_csv("./assets/csv/protocolos_ultimos_3anos.csv", sep=";") # < alterar para o caso necessário 

df["DataEntrada"] = pd.to_datetime(df["DataEntrada"]) # garante que é data

# Carregar arquivo de feriados
df_feriados_csv = pd.read_csv("./assets/csv/feriados.csv", sep=";") 

datas_feriados = pd.to_datetime(df_feriados_csv['Data']) # Garante que é data


# Agregação (Contagem diária)
df = df.set_index("DataEntrada")
df_diario = df.resample("D")["id"].count().to_frame(name="qtd_protocolos")

# Criar a covariável (Fim de semana) para TODO o período disponível
df_diario["is_weekend"] = (df_diario.index.dayofweek >= 5).astype(int)

# Criar covariavel feriado
df_diario["is_holiday"] = df_diario.index.normalize().isin(datas_feriados).astype(int)
holiday_historico = df_diario["is_holiday"].values


# # --- 2. PREVISÃO REAL ---
# horizonte = 30 # Quantos dias deseja que o modelo preveja

# # Criar as datas futuras (começando do dia seguinte ao último dia do arquivo)
# ultima_data_real = df_diario.index[-1]
# datas_futuras = pd.date_range( start=ultima_data_real + pd.Timedelta(days=1), periods=horizonte, freq="D")

# # Criar a covariável de Fim de Semana para o FUTURO
# weekend_futuro = (datas_futuras.dayofweek >= 5).astype(int)

# # Criar a covariável de feriados para o FUTURO
# holiday_futuro = datas_futuras.normalize().isin(datas_feriados).astype(int)

# # --- 3. RODANDO O TIMESFM ---
# # Transformar em arrays numpy 
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
# previsao_final = np.maximum(previsao_final, 0) # Evita que a linha da previsão do modelo vá abaixo de zero
# quantis = np.array(forecast_quantis)

# dados_brutos = quantis[0]


# # Numpy varre cada linha e achar quem é o menor e o maior 
# limite_superior = np.max(dados_brutos, axis=1)  
# limite_inferior = np.min(dados_brutos, axis=1)  # O verdadeiro Chão
# limite_inferior = np.maximum(limite_inferior, 0)  # Impede que seja menor que zero



# # --- 4. VISUALIZAÇÃO ---
# plt.figure(figsize=(15, 7))


# zoom_dias = 180  # Personaliza a quantidade de dias a mostrar do histórico

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

# # 1. Pegamos o eixo atual
# ax = plt.gca()

# # 2. Formatamos para mostrar Mês/Ano (Ex: 09/2025)
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))

# # 3. Definimos o intervalo
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
horizonte = 90 # Tamanho do teste (30 dias)

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

limite_superior = np.max(dados_brutos, axis=1) * 1.5 # O verdadeiro Teto

limite_inferior = np.min(dados_brutos, axis=1) * 0.5  # O verdadeiro Chão
limite_inferior = np.maximum(limite_inferior, 0)  # Impede que seja menor que zero

# Pegamos a previsão (garantindo que pegamos só a parte futura com o slice)
previsao_teste = previsao_valores[-horizonte:]

# --- 3. CALCULAR A PRECISÃO (RMSE) ---

rmse = np.sqrt(np.mean((gabarito_teste - previsao_teste)**2))

print(f"=== RESULTADO DO TESTE ===")
print(f"RMSE: {rmse:.2f}")
print(f"Interpretação: O modelo erra, em média, {rmse:.1f} protocolos por dia.")

# --- 4. VISUALIZAR A VERDADE (Versão Interativa HTML) ---

# Configuração de quantos dias mostrar no passado
dias_visiveis = 200

# Recalculando os índices exatamente como no seu código original
datas_treino = df_diario.index[:-horizonte]
datas_teste = df_diario.index[-horizonte:]

fig = go.Figure()

# 1. Plotar os dados usados no treino (Preto)
# Mantendo o mesmo slice [-dias_visiveis:]
fig.add_trace(go.Scatter(
    x=datas_treino[-dias_visiveis:],
    y=input_treino[-dias_visiveis:],
    mode='lines',
    name='Dados de Treino (Visíveis)',
    line=dict(color='black')
))

# 2. Plotar o Intervalo de Confiança (Sombra Verde)
# Plotly precisa concatenar ida e volta para fechar a forma (fill='toself')
fig.add_trace(go.Scatter(
    x=list(datas_teste) + list(datas_teste)[::-1],
    y=list(limite_superior) + list(limite_inferior)[::-1],
    fill='toself',
    fillcolor='rgba(0, 128, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Intervalo de Confiança (10%-90%)',
    hoverinfo="skip"
))

# 3. Plotar o Gabarito Real (Azul)
fig.add_trace(go.Scatter(
    x=datas_teste,
    y=gabarito_teste,
    mode='lines',
    name='Realidade (Escondido)',
    line=dict(color='blue', width=2)
))

# 4. Plotar a Previsão (Vermelho)
fig.add_trace(go.Scatter(
    x=datas_teste,
    y=previsao_teste,
    mode='lines',
    name='Previsão TimesFM',
    line=dict(color='red', width=2, dash='dash')
))

# --- PONTOS DE ATENÇÃO (ANOMALIAS) ---
# Usando a mesma lógica sua para criar a máscara
mask_outliers = (gabarito_teste > limite_superior) | (gabarito_teste < limite_inferior)

# Filtrando com as mesmas variáveis
datas_outliers = datas_teste[mask_outliers]
valores_outliers = gabarito_teste[mask_outliers]

if len(datas_outliers) > 0:
    # Adiciona X e Texto na mesma camada
    fig.add_trace(go.Scatter(
        x=datas_outliers,
        y=valores_outliers,
        mode='markers+text',
        name='Anomalia (Fora da Margem)',
        text=[d.strftime('%d/%m') for d in datas_outliers], # Formata a data para texto
        textposition="top center",
        marker=dict(symbol='x', size=12, color='darkred', line=dict(width=2)),
        textfont=dict(color='darkred', size=10, family="Arial Black")
    ))
    
    # Adiciona o círculo vermelho ao redor (Destaque)
    fig.add_trace(go.Scatter(
        x=datas_outliers,
        y=valores_outliers,
        mode='markers',
        name='Destaque',
        marker=dict(symbol='circle-open', size=18, color='red', line=dict(width=2)),
        showlegend=False,
        hoverinfo="skip"
    ))

# --- LAYOUT FINAL ---
fig.update_layout(
    title="Previsão de Demanda de Protocolos (Com Detecção de Anomalias)",
    xaxis_title="Data",
    yaxis_title="Qtd Protocolos",
    template="plotly_white",
    hovermode="x unified", # Mostra todos os valores ao passar o mouse
    height=900,            # Equivalente ao figsize=(15, 7)
    xaxis=dict(
        tickformat="%m/%Y", # Seu formato de data
        tickangle=-45
    )
)

# Salva o arquivo HTML interativo
# fig.write_html("previsao_interativa.html")
# print("✅ Gráfico salvo como 'previsao_interativa.html'")

# Se quiser exibir dentro do notebook ou abrir no browser:
fig.show()

