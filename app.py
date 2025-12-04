import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import numpy as np
import os
import datetime as dt
import pytz
import warnings

# --- LIMPEZA ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Monitor Pro v11", layout="wide", page_icon="📊")

# --- FUNÇÃO DE SEGREDOS ---
def get_secret(key):
    env_var = os.environ.get(key)
    if env_var: return env_var
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except: pass
    return None

# --- MODO ROBÔ ---
if os.environ.get("GITHUB_ACTIONS") == "true":
    MODO_ROBO = True
    FILTRO_QUEDA = -0.01
    USAR_BOLLINGER = False
else:
    MODO_ROBO = False
    
# --- BARRA LATERAL (SITE) ---
if not MODO_ROBO:
    st.sidebar.title("🎛️ Painel v11") # Mudança no título para confirmar atualização
    st.sidebar.info("Versão com Gráficos Ativada")
    
    filtro_visual = st.sidebar.slider("Mínimo de Queda (%)", -15, 0, -3, 1) / 100
    bollinger_visual = st.sidebar.checkbox("Abaixo da Banda?", value=True)
    
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

# --- NOVA FUNÇÃO DE GRÁFICOS (ROBUSTA) ---
def buscar_dados_intraday(ticker):
    try:
        # Pega 5 dias para garantir que o gráfico não fica vazio
        # Intervalo de 15m mostra bem a tendência recente
        df = yf.download(f"{ticker}.SA", period="5d", interval="15m", progress=False, ignore_tz=True)
        if df.empty: return pd.Series()
        return df['Close']
    except:
        return pd.Series()

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
        
        if tem_vol and tem_ifr: return "★★★ Forte", "Vol+IFR", 3
        elif tem_vol: return "★★☆ Médio", "Volume", 2
        elif tem_ifr: return "★★☆ Médio", "IFR", 2
        else: return "★☆☆ Atenção", "Queda", 1
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
    st.title("📊 Monitor BDRs v11") # Título novo para confirmar atualização
    
    botao_analisar = st.button("🔄 Rodar Análise")
    
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
                        resultados.append({
                            'Ticker': t, 'Variação': var, 'Preço': last[('Close', t)],
                            'IFR14': last[('IFR14', t)], 'Classificação': classif,
                            'Motivo': motivo, 'Score': score
                        })
                    except: continue

                if resultados:
                    resultados.sort(key=lambda x: x['Variação'])
                    
                    col1, col2 = st.columns(2)
                    col1.metric("BDRs Analisados", len(bdrs))
                    col2.metric("Oportunidades", len(resultados))

                    # 1. Tabela
                    st.subheader("📋 Resultados")
                    df_show = pd.DataFrame(resultados)
                    df_show['Variação'] = df_show['Variação'].apply(lambda x: f"{x:.2%}")
                    df_show['Preço'] = df_show['Preço'].apply(lambda x: f"R$ {x:.2f}")
                    df_show['IFR14'] = df_show['IFR14'].apply(lambda x: f"{x:.1f}")
                    st.dataframe(df_show[['Ticker', 'Variação', 'Preço', 'IFR14', 'Classificação', 'Motivo']], use_container_width=True)
                    
                    # 2. Gráficos (AQUI ESTÁ A NOVIDADE)
                    st.markdown("---")
                    st.subheader("📈 Tendência Recente (5 Dias - 15min)")
                    
                    # Layout de 3 colunas
                    cols = st.columns(3)
                    for i, item in enumerate(resultados):
                        ticker = item['Ticker']
                        with cols[i % 3]:
                            st.caption(f"**{ticker}** | Variação: {item['Variação']:.2%}")
                            
                            # Baixar dados específicos
                            chart_data = buscar_dados_intraday(ticker)
                            
                            if not chart_data.empty:
                                # Usa area_chart para ficar mais visual
                                st.area_chart(chart_data, height=150, color="#FF4B4B")
                            else:
                                st.warning(f"Sem gráfico para {ticker}")

                    # 3. WhatsApp Manual
                    st.markdown("---")
                    if st.checkbox("Enviar WhatsApp?"):
                        fuso = pytz.timezone('America/Sao_Paulo')
                        hora = dt.datetime.now(fuso).strftime("%H:%M")
                        msg = f"🚨 *Manual* ({hora})\n\n"
                        for item in resultados[:10]:
                            msg += f"-> *{item['Ticker']}*: {item['Variação']:.2%} | {item['Classificação']}\n"
                        enviar_whatsapp(msg)
                        st.success("Enviado!")
                else:
                    st.info("Nenhuma oportunidade encontrada.")
            else:
                st.warning("Sem dados históricos.")

# --- MODO ROBÔ ---
if MODO_ROBO:
    # Lógica do robô simplificada para economizar espaço aqui
    # (Mantém a lógica que já funcionou no passo anterior)
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
                    if pd.isna(var) or var > FILTRO_QUEDA: continue
                    classif, motivo, score = analisar_sinal(last, t)
                    resultados.append({'Ticker': t, 'Variação': var, 'Classificação': classif, 'Score': score})
                except: continue
            
            if resultados:
                resultados.sort(key=lambda x: x['Variação'])
                fuso = pytz.timezone('America/Sao_Paulo')
                hora = dt.datetime.now(fuso).strftime("%H:%M")
                msg = f"🚨 *Top 10* ({hora})\n\n"
                for item in resultados[:10]:
                    icone = "🔥" if item['Score'] == 3 else "🔻"
                    msg += f"{icone} *{item['Ticker']}*: {item['Variação']:.2%} | {item['Classificação']}\n"
                msg += f"\nSite: share.streamlit.io"
                enviar_whatsapp(msg)
