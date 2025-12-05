import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import numpy as np
import os
import datetime as dt
import pytz
import warnings

# --- LIMPEZA DE LOGS ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Monitor Pro BDRs", layout="wide", page_icon="📉")

# --- FUNÇÃO DE SEGREDOS ---
def get_secret(key):
    env_var = os.environ.get(key)
    if env_var: return env_var
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except: pass
    return None

# --- MODO ROBÔ VS HUMANO ---
if os.environ.get("GITHUB_ACTIONS") == "true":
    MODO_ROBO = True
    FILTRO_QUEDA = -0.01
    USAR_BOLLINGER = False
else:
    MODO_ROBO = False
    
# --- BARRA LATERAL (APENAS SITE) ---
if not MODO_ROBO:
    st.sidebar.title("🎛️ Painel de Controle")
    st.sidebar.info("Modo: Tabela Detalhada (Hora a Hora)")
    
    filtro_visual = st.sidebar.slider("Mínimo de Queda (%)", -15, 0, -3, 1) / 100
    bollinger_visual = st.sidebar.checkbox("Abaixo da Banda de Bollinger?", value=True)
    
    FILTRO_QUEDA = filtro_visual
    USAR_BOLLINGER = bollinger_visual

# --- CREDENCIAIS ---
WHATSAPP_PHONE = get_secret('WHATSAPP_PHONE')
WHATSAPP_APIKEY = get_secret('WHATSAPP_APIKEY')
BRAPI_API_TOKEN = get_secret('BRAPI_API_TOKEN')

PERIODO_HISTORICO_DIAS = "60d"
TERMINACOES_BDR = ('31', '32', '33', '34', '35', '39')

# --- FUNÇÕES ---

@st.cache_data(ttl=3600)
def obter_lista_bdrs_da_brapi():
    if not BRAPI_API_TOKEN: return []
    try:
        url = f"https://brapi.dev/api/quote/list?token={BRAPI_API_TOKEN}"
        r = requests.get(url, timeout=30)
        dados = r.json().get('stocks', [])
        df = pd.DataFrame(dados)
        return df[df['stock'].str.endswith(TERMINACOES_BDR, na=False)]['stock'].tolist()
    except: return []

def buscar_dados(tickers):
    if not tickers: return pd.DataFrame()
    sa_tickers = [f"{t}.SA" for t in tickers]
    try:
        if MODO_ROBO: print(f"Baixando dados de {len(tickers)} ativos...")
        df = yf.download(sa_tickers, period=PERIODO_HISTORICO_DIAS, auto_adjust=True, progress=False, ignore_tz=True)
        if df.empty: return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_tuples([(c[0], c[1].replace(".SA", "")) for c in df.columns])
        elif isinstance(df.index, pd.DatetimeIndex) and len(tickers) == 1:
            df.columns = pd.MultiIndex.from_product([df.columns, [tickers[0]]])
            
        return df.dropna(axis=1, how='all')
    except: return pd.DataFrame()

# --- NOVA FUNÇÃO: EVOLUÇÃO HORÁRIA ---
def obter_evolucao_horaria(ticker):
    try:
        # Baixa apenas o dia de hoje, intervalo de 1 hora
        df = yf.download(f"{ticker}.SA", period="1d", interval="1h", progress=False, ignore_tz=True)
        if df.empty: return "-"
        
        # Converte índice para Horário de Brasília (se necessário ajustar manual)
        # O yfinance geralmente traz UTC. Vamos simplificar pegando as horas.
        
        evolucao_txt = []
        # Preço de abertura do dia (primeira barra)
        abertura_dia = df['Open'].iloc[0]
        
        for hora, row in df.iterrows():
            # Converte UTC para Brasília (aproximado -3h se estiver em UTC)
            # Nota: O yfinance varia dependendo do servidor, mas vamos tentar formatar a hora
            hora_str = str(hora.hour - 3) if hora.hour >= 3 else str(hora.hour + 21) # Ajuste manual simples fuso
            
            # Se o timestamp já estiver correto (às vezes vem certo), usamos direto:
            if df.index.tz is None: 
                # Se não tem fuso, assume que já é local ou UTC
                hora_display = hora.hour 
            else:
                 # Converte corretamente
                 fuso_br = pytz.timezone('America/Sao_Paulo')
                 hora_display = hora.astimezone(fuso_br).hour

            # Filtra apenas horário de pregão comum (10h às 18h)
            if 10 <= hora_display <= 18:
                # Calcula variação em relação à ABERTURA DO DIA
                var_momento = ((row['Close'] / abertura_dia) - 1)
                evolucao_txt.append(f"{hora_display}h: {var_momento:+.1%}")
        
        return " ➡ ".join(evolucao_txt)
    except:
        return "-"

def calcular_indicadores(df):
    df = df.copy()
    tickers = df.columns.get_level_values(1).unique()
    inds = {}
    for t in tickers:
        try:
            close = df[('Close', t)]
            vol = df[('Volume', t)]
            variacao = close.pct_change(fill_method=None)
            delta = close.diff()
            ganho = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
            perda = -delta.where(delta < 0, 0).ewm(com=13, adjust=False).mean()
            ifr = 100 - (100 / (1 + (ganho/perda)))
            inds[('IFR14', t)] = ifr.fillna(50)
            inds[('VolMedio', t)] = vol.rolling(10).mean()
            inds[('Variacao', t)] = variacao
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            inds[('BandaInf', t)] = sma - (std * 2)
        except: continue
    if not inds: return pd.DataFrame()
    return df.join(pd.DataFrame(inds), how='left').sort_index(axis=1)

def analisar_sinal(row, t):
    try:
        vol = row[('Volume', t)]
        vol_med = row[('VolMedio', t)]
        ifr = row[('IFR14', t)]
        tem_vol = vol > vol_med if (not pd.isna(vol) and not pd.isna(vol_med)) else False
        tem_ifr = ifr < 30 if not pd.isna(ifr) else False
        
        if tem_vol and tem_ifr: return "★★★ Forte", "Vol Explosivo + IFR Baixo", 3
        elif tem_vol: return "★★☆ Médio", "Volume Alto", 2
        elif tem_ifr: return "★★☆ Médio", "IFR Sobrevenda", 2
        else: return "★☆☆ Atenção", "Apenas Queda", 1
    except: return "Erro", "-", 0

def enviar_whatsapp(msg):
    if not WHATSAPP_PHONE or not WHATSAPP_APIKEY: return
    try:
        texto_codificado = requests.utils.quote(msg)
        url_whatsapp = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={texto_codificado}&apikey={WHATSAPP_APIKEY}"
        headers = { "User-Agent": "Mozilla/5.0" }
        requests.get(url_whatsapp, headers=headers, timeout=20)
    except: pass

# --- VISUAL (SITE) ---
if not MODO_ROBO:
    st.title("📉 Monitor BDRs (Detalhado)")
    
    with st.expander("ℹ️ Legenda da Tabela"):
        st.markdown("""
        * **Evolução Hoje:** Mostra a variação percentual acumulada em cada hora (em relação à abertura do dia).
        * **Exemplo:** `10h: -0.5% ➡ 11h: -1.2%` significa que às 10h caía 0.5% e às 11h a queda piorou para 1.2%.
        """)

# --- EXECUÇÃO ---
botao_analisar = st.button("🔄 Rodar Análise") if not MODO_ROBO else True

if botao_analisar:
    bdrs = obter_lista_bdrs_da_brapi()
    
    if bdrs:
        df = buscar_dados(bdrs)
        if not df.empty:
            df_calc = calcular_indicadores(df)
            last = df_calc.iloc[-1]
            resultados = []
            
            for t in df_calc.columns.get_level_values(1).unique():
                try:
                    var = last.get(('Variacao', t), np.nan)
                    low = last.get(('Low', t), np.nan)
                    banda = last.get(('BandaInf', t), np.nan)
                    
                    if pd.isna(var) or var > FILTRO_QUEDA: continue
                    if USAR_BOLLINGER and (pd.isna(low) or low >= banda): continue
                    
                    classif, motivo, score = analisar_sinal(last, t)
                    
                    # --- NOVIDADE: Busca evolução horária AQUI ---
                    # Só faz isso se estiver no modo Site (para não deixar o Robô lento)
                    evolucao_str = "-"
                    if not MODO_ROBO:
                        evolucao_str = obter_evolucao_horaria(t)

                    resultados.append({
                        'Ticker': t, 
                        'Variação': var, 
                        'Preço': last[('Close', t)],
                        'IFR14': last[('IFR14', t)], 
                        'Classificação': classif,
                        'Motivo': motivo, 
                        'Score': score,
                        'Evolução Hoje': evolucao_str # Nova coluna
                    })
                except: continue

            if resultados:
                resultados.sort(key=lambda x: x['Variação'])
                
                # --- VISUALIZAÇÃO NO SITE ---
                if not MODO_ROBO:
                    st.metric("Oportunidades Encontradas", len(resultados))

                    df_show = pd.DataFrame(resultados)
                    df_show['Variação'] = df_show['Variação'].apply(lambda x: f"{x:.2%}")
                    df_show['Preço'] = df_show['Preço'].apply(lambda x: f"R$ {x:.2f}")
                    df_show['IFR14'] = df_show['IFR14'].apply(lambda x: f"{x:.1f}")
                    
                    st.subheader("📋 Tabela Detalhada")
                    # Mostra a tabela com a nova coluna
                    st.dataframe(
                        df_show[['Ticker', 'Variação', 'Preço', 'IFR14', 'Classificação', 'Evolução Hoje']], 
                        use_container_width=True,
                        column_config={
                            "Evolução Hoje": st.column_config.TextColumn("Variação Hora-a-Hora", width="large"),
                        }
                    )
                    
                    if st.checkbox("Enviar WhatsApp Manual?"):
                        fuso = pytz.timezone('America/Sao_Paulo')
                        hora = dt.datetime.now(fuso).strftime("%H:%M")
                        msg = f"🚨 *Manual* ({hora})\n\n"
                        for item in resultados[:10]:
                            msg += f"-> *{item['Ticker']}*: {item['Variação']:.2%} | {item['Classificação']}\n"
                        enviar_whatsapp(msg)
                        st.success("Enviado!")

                # --- MODO ROBÔ ---
                if MODO_ROBO:
                    print(f"Encontradas {len(resultados)} oportunidades.")
                    fuso = pytz.timezone('America/Sao_Paulo')
                    hora = dt.datetime.now(fuso).strftime("%H:%M")
                    msg = f"🚨 *Top 10* ({hora})\n\n"
                    for item in resultados[:10]:
                        icone = "🔥" if item['Score'] == 3 else "🔻"
                        msg += f"{icone} *{item['Ticker']}*: {item['Variação']:.2%} | {item['Classificação']}\n"
                    msg += f"\nSite: share.streamlit.io"
                    enviar_whatsapp(msg)
            else:
                if MODO_ROBO: print("Sem oportunidades.")
                else: st.info("Nenhuma oportunidade encontrada.")
