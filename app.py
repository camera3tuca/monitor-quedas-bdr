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

# --- SIDEBAR (FILTROS) ---
st.sidebar.header("🎛️ Configurações")
FILTRO_QUEDA = st.sidebar.slider("Mínimo de Queda (%)", -15, 0, -3, 1) / 100
USAR_BOLLINGER = st.sidebar.checkbox("Exigir estar abaixo da Banda?", value=True)

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
        
        # Ajuste de colunas (MultiIndex)
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
    # Retorna: (Texto, Motivo, Score Numérico para ordenar)
    try:
        vol = row[('Volume', t)]
        vol_med = row[('VolMedio', t)]
        ifr = row[('IFR14', t)]
        
        tem_vol = vol > vol_med if (not pd.isna(vol) and not pd.isna(vol_med)) else False
        tem_ifr = ifr < 30 if not pd.isna(ifr) else False
        
        if tem_vol and tem_ifr:
            return "★★★ Forte", "Volume Alto + IFR < 30", 3
        elif tem_vol:
            return "★★☆ Médio", "Volume Alto", 2
        elif tem_ifr:
            return "★★☆ Médio", "IFR < 30 (Sobrevenda)", 2
        else:
            return "★☆☆ Atenção", "Apenas Queda (Bandas)", 1
    except:
        return "Erro", "-", 0

# --- APP VISUAL ---
st.title("📉 Monitor Inteligente de BDRs")

# Legenda Explicativa
with st.expander("ℹ️ Entenda a Classificação (Critérios)"):
    st.markdown("""
    * **★★★ Forte:** A ação caiu, furou a banda de Bollinger, o **Volume explodiu** (pânico) e o **IFR está abaixo de 30** (muito barato). É o cenário ideal de reversão.
    * **★★☆ Médio:** A ação caiu e tem **ou** Volume alto **ou** IFR baixo. É um sinal bom, mas falta um dos confirmadores.
    * **★☆☆ Atenção:** A ação caiu abaixo da banda de Bollinger, mas sem volume expressivo ou IFR extremo. Pode continuar caindo (faca caindo).
    """)

if st.button("🔄 Analisar Mercado") or os.environ.get("GITHUB_ACTIONS") == "true":
    bdrs = obter_lista_bdrs_da_brapi()
    st.write(f"🔍 {len(bdrs)} BDRs na lista. Baixando dados...")
    
    if bdrs:
        df = buscar_dados(bdrs)
        if not df.empty:
            df_calc = calcular_indicadores(df)
            last = df_calc.iloc[-1]
            
            resultados = []
            
            # Loop nos ativos
            for t in df_calc.columns.get_level_values(1).unique():
                try:
                    var = last.get(('Variacao', t), np.nan)
                    low = last.get(('Low', t), np.nan)
                    banda = last.get(('BandaInf', t), np.nan)
                    
                    # Filtros principais
                    if pd.isna(var) or var > FILTRO_QUEDA: continue
                    if USAR_BOLLINGER and (pd.isna(low) or low >= banda): continue
                    
                    # Análise detalhada
                    classif, motivo, score = analisar_sinal(last, t)
                    
                    resultados.append({
                        'Ticker': t,
                        'Variação': var, # Mantém numérico para ordenar
                        'Preço': last[('Close', t)],
                        'IFR14': last[('IFR14', t)],
                        'Classificação': classif,
                        'Motivo': motivo,
                        'Score': score
                    })
                except: continue

            if resultados:
                # ORDENAÇÃO DUPLA: 
                # 1º Pelo Score (3 estrelas primeiro)
                # 2º Pelo tamanho da queda (maior queda primeiro, ou seja, menor número negativo)
                
                # Primeiro ordenamos pela variação (ascendente: -10% vem antes de -5%)
                resultados.sort(key=lambda x: x['Variação'])
                # Depois ordenamos pelo Score (descendente: 3 antes de 1). 
                # O Python mantém a ordem anterior dentro dos grupos (Estabilidade).
                resultados.sort(key=lambda x: x['Score'], reverse=True)
                
                # Preparar para exibir (Formatar números)
                df_show = pd.DataFrame(resultados)
                # Guardar valores originais para envio e formatar para tela
                df_tela = df_show.copy()
                df_tela['Variação'] = df_tela['Variação'].apply(lambda x: f"{x:.2%}")
                df_tela['Preço'] = df_tela['Preço'].apply(lambda x: f"R$ {x:.2f}")
                df_tela['IFR14'] = df_tela['IFR14'].apply(lambda x: f"{x:.1f}")
                
                # Remove colunas técnicas da tela
                st.subheader(f"🚨 {len(resultados)} Oportunidades")
                st.dataframe(
                    df_tela[['Ticker', 'Variação', 'Classificação', 'Motivo', 'Preço', 'IFR14']], 
                    use_container_width=True,
                    hide_index=True
                )
                
                # WhatsApp
                fuso = pytz.timezone('America/Sao_Paulo')
                hora = dt.datetime.now(fuso).strftime("%H:%M")
                msg = f"🚨 *Robô BDRs* ({hora})\n\n"
                
                for item in resultados[:10]:
                    icone = "🔥" if item['Score'] == 3 else "⚠️"
                    msg += f"{icone} *{item['Ticker']}*: {item['Variação']:.2%} | {item['Classificação']}\n   ↳ {item['Motivo']}\n"
                
                msg += f"\nLink: https://share.streamlit.io"
                
                check = st.checkbox("Enviar WhatsApp?", value=(os.environ.get("GITHUB_ACTIONS") == "true"))
                if check: enviar_whatsapp(msg)
                
            else:
                st.info("Nenhuma oportunidade com os filtros atuais.")
        else:
            st.warning("Sem dados históricos.")
