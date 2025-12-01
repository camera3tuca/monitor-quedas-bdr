import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import numpy as np
import os
import datetime as dt
import pytz

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Monitor de Quedas BDRs", layout="wide")

# --- FUNÇÃO DE SEGREDOS ---
def get_secret(key):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key)

# --- CONFIGURAÇÃO DA BARRA LATERAL (VISUAL - USUÁRIO) ---
st.sidebar.header("🎛️ Configurações (Site)")

# Estes são os controlos que tu vês no tablet (Versão 3.0)
filtro_visual = st.sidebar.slider("Mínimo de Queda (%)", -15, 0, -3, 1) / 100
bollinger_visual = st.sidebar.checkbox("Exigir estar abaixo da Banda?", value=True)

# --- LÓGICA DE DECISÃO (CÉREBRO) ---
# Se for o ROBÔ rodando no GitHub -> Usa regras fixas (-1%, Top 10, Sem Bollinger)
# Se for TU no site -> Usa o que escolheste na barra lateral
if os.environ.get("GITHUB_ACTIONS") == "true":
    FILTRO_QUEDA = -0.01  # -1% (Regra do Robô)
    USAR_BOLLINGER = False # Robô quer ver tudo
    MODO_ROBO = True
else:
    FILTRO_QUEDA = filtro_visual
    USAR_BOLLINGER = bollinger_visual
    MODO_ROBO = False

# --- CREDENCIAIS ---
WHATSAPP_PHONE = get_secret('WHATSAPP_PHONE')
WHATSAPP_APIKEY = get_secret('WHATSAPP_APIKEY')
BRAPI_API_TOKEN = get_secret('BRAPI_API_TOKEN')

PERIODO_HISTORICO_DIAS = "60d"
TERMINACOES_BDR = ('31', '32', '33', '34', '35', '39')

# --- LÓGICA E DADOS ---

@st.cache_data(ttl=3600)
def obter_lista_bdrs_da_brapi():
    if not BRAPI_API_TOKEN:
        st.error("Token BRAPI ausente.")
        return []
    try:
        url = f"https://brapi.dev/api/quote/list?token={BRAPI_API_TOKEN}"
        r = requests.get(url, timeout=30)
        df = pd.DataFrame(r.json().get('stocks', []))
        return df[df['stock'].str.endswith(TERMINACOES_BDR, na=False)]['stock'].tolist()
    except Exception as e:
        st.error(f"Erro BRAPI: {e}")
        return []

def buscar_dados(tickers):
    if not tickers: return pd.DataFrame()
    sa_tickers = [f"{t}.SA" for t in tickers]
    try:
        with st.spinner(f"Analisando {len(tickers)} ativos..."):
            df = yf.download(sa_tickers, period=PERIODO_HISTORICO_DIAS, auto_adjust=True, progress=False, ignore_tz=True)
        if df.empty: return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_tuples([(c[0], c[1].replace(".SA", "")) for c in df.columns])
        elif isinstance(df.index, pd.DatetimeIndex) and len(tickers) == 1:
            df.columns = pd.MultiIndex.from_product([df.columns, [tickers[0]]])
            
        return df.dropna(axis=1, how='all')
    except:
        return pd.DataFrame()

def calcular_indicadores(df):
    df = df.copy()
    tickers = df.columns.get_level_values(1).unique()
    inds = {}
    
    for t in tickers:
        try:
            close = df[('Close', t)]
            vol = df[('Volume', t)]
            
            # IFR 14
            delta = close.diff()
            ganho = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
            perda = -delta.where(delta < 0, 0).ewm(com=13, adjust=False).mean()
            ifr = 100 - (100 / (1 + (ganho/perda)))
            inds[('IFR14', t)] = ifr.fillna(50)
            
            # Outros
            inds[('VolMedio', t)] = vol.rolling(10).mean()
            inds[('Variacao', t)] = close.pct_change()
            
            # Bollinger
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            inds[('BandaInf', t)] = sma - (std * 2)
            
        except: continue
        
    if not inds: return pd.DataFrame()
    df_inds = pd.DataFrame(inds)
    return df.join(df_inds, how='left').sort_index(axis=1)

def analisar_sinal(row, t):
    try:
        vol = row[('Volume', t)]
        vol_med = row[('VolMedio', t)]
        ifr = row[('IFR14', t)]
        
        tem_vol = vol > vol_med if (not pd.isna(vol) and not pd.isna(vol_med)) else False
        tem_ifr = ifr < 30 if not pd.isna(ifr) else False
        
        if tem_vol and tem_ifr:
            return "★★★ Forte", "Volume Explosivo + IFR Baixo", 3
        elif tem_vol:
            return "★★☆ Médio", "Volume Alto", 2
        elif tem_ifr:
            return "★★☆ Médio", "IFR Baixo (Sobrevenda)", 2
        else:
            return "★☆☆ Atenção", "Apenas Queda", 1
    except:
        return "Erro", "-", 0

def enviar_whatsapp(msg):
    if not WHATSAPP_PHONE or not WHATSAPP_APIKEY: return
    try:
        texto_codificado = requests.utils.quote(msg)
        url = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={texto_codificado}&apikey={WHATSAPP_APIKEY}"
        requests.get(url, timeout=20)
    except: pass

# --- APP VISUAL ---
st.title("📉 Monitor Inteligente de BDRs")

# Mostra status
if MODO_ROBO:
    st.info(f"🤖 MODO ROBÔ: Buscando Top 10 Maiores Quedas (Min -1%)")
else:
    st.info(f"👤 MODO VISUAL: Filtro {FILTRO_QUEDA:.1%} | Bollinger {'Ligado' if USAR_BOLLINGER else 'Desligado'}")

if st.button("🔄 Analisar Mercado") or MODO_ROBO:
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
                    
                    # 1. Filtro de Queda
                    if pd.isna(var) or var > FILTRO_QUEDA: continue
                    
                    # 2. Filtro de Bollinger (Depende do modo)
                    if USAR_BOLLINGER:
                         if pd.isna(low) or low >= banda: continue
                    
                    classif, motivo, score = analisar_sinal(last, t)
                    
                    resultados.append({
                        'Ticker': t,
                        'Variação': var,
                        'Preço': last[('Close', t)],
                        'IFR14': last[('IFR14', t)],
                        'Classificação': classif,
                        'Motivo': motivo,
                        'Score': score
                    })
                except: continue

            if resultados:
                # ORDENAÇÃO: Sempre pela maior queda (número mais negativo primeiro)
                resultados.sort(key=lambda x: x['Variação'])
                
                # Exibição Visual (Tabela Bonita)
                df_show = pd.DataFrame(resultados)
                df_tela = df_show.copy()
                df_tela['Variação'] = df_tela['Variação'].apply(lambda x: f"{x:.2%}")
                df_tela['Preço'] = df_tela['Preço'].apply(lambda x: f"R$ {x:.2f}")
                df_tela['IFR14'] = df_tela['IFR14'].apply(lambda x: f"{x:.1f}")
                
                st.subheader(f"🚨 {len(resultados)} Oportunidades Encontradas")
                st.dataframe(
                    df_tela[['Ticker', 'Variação', 'Classificação', 'Motivo', 'Preço', 'IFR14']], 
                    use_container_width=True,
                    hide_index=True
                )
                
                # ENVIO WHATSAPP (Lógica do Robô: Top 10 Maiores Quedas)
                if MODO_ROBO:
                    fuso = pytz.timezone('America/Sao_Paulo')
                    hora = dt.datetime.now(fuso).strftime("%H:%M")
                    
                    msg = f"🚨 *Monitor Top 10 Quedas* ({hora})\nCritério: Queda > 1% (Sem Bollinger)\n\n"
                    
                    # Pega apenas os 10 primeiros (que já ordenamos pela maior queda)
                    top_10 = resultados[:10]
                    
                    for item in top_10:
                        # Icone muda conforme a força, mas a ordem é pela queda
                        icone = "🔥" if item['Score'] == 3 else "🔻"
                        msg += f"{icone} *{item['Ticker']}*: {item['Variação']:.2%} | {item['Classificação']}\n"
                    
                    if len(resultados) > 10:
                        msg += f"\n...e mais {len(resultados)-10} no site."
                    
                    msg += f"\nLink: https://share.streamlit.io"
                    enviar_whatsapp(msg)
                    st.success("Relatório Top 10 enviado para o WhatsApp!")
                
            else:
                st.info("Nenhuma oportunidade com os filtros atuais.")
        else:
            st.warning("Sem dados históricos.")
