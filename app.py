import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import numpy as np
import os
import datetime as dt
import pytz
import warnings

# --- CONFIGURAÇÃO VISUAL PROFISSIONAL ---
st.set_page_config(
    page_title="Monitor BDR Pro", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- LIMPEZA DE LOGS ---
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# --- CREDENCIAIS ---
WHATSAPP_PHONE = get_secret('WHATSAPP_PHONE')
WHATSAPP_APIKEY = get_secret('WHATSAPP_APIKEY')
BRAPI_API_TOKEN = get_secret('BRAPI_API_TOKEN')

PERIODO_HISTORICO_DIAS = "60d"
TERMINACOES_BDR = ('31', '32', '33', '34', '35', '39')

# --- SIDEBAR (CONTROLES) ---
if not MODO_ROBO:
    st.sidebar.header("🎛️ Filtros de Análise")
    
    filtro_visual = st.sidebar.slider("Mínimo de Queda (%)", -15, 0, -3, 1) / 100
    bollinger_visual = st.sidebar.checkbox("Abaixo da Banda de Bollinger?", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Monitoramento Profissional de BDRs v15.0")
    
    FILTRO_QUEDA = filtro_visual
    USAR_BOLLINGER = bollinger_visual

# --- FUNÇÕES ---

@st.cache_data(ttl=3600)
def obter_dados_brapi():
    """Retorna uma tupla: (lista_tickers, dicionario_nomes)"""
    if not BRAPI_API_TOKEN: return [], {}
    try:
        url = f"https://brapi.dev/api/quote/list?token={BRAPI_API_TOKEN}"
        r = requests.get(url, timeout=30)
        dados = r.json().get('stocks', [])
        
        # Filtra BDRs
        bdrs_raw = [d for d in dados if d['stock'].endswith(TERMINACOES_BDR)]
        
        lista_tickers = [d['stock'] for d in bdrs_raw]
        # Cria mapa { 'AAPL34': 'Apple Inc.', ... }
        mapa_nomes = {d['stock']: d.get('name', d['stock']) for d in bdrs_raw}
        
        return lista_tickers, mapa_nomes
    except: return [], {}

def buscar_dados(tickers):
    if not tickers: return pd.DataFrame()
    sa_tickers = [f"{t}.SA" for t in tickers]
    try:
        if MODO_ROBO: print(f"Baixando dados de {len(tickers)} ativos...")
        
        # Download otimizado
        df = yf.download(sa_tickers, period=PERIODO_HISTORICO_DIAS, auto_adjust=True, progress=False, ignore_tz=True)
        if df.empty: return pd.DataFrame()
        
        # Tratamento MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_tuples([(c[0], c[1].replace(".SA", "")) for c in df.columns])
        elif isinstance(df.index, pd.DatetimeIndex) and len(tickers) == 1:
            df.columns = pd.MultiIndex.from_product([df.columns, [tickers[0]]])
            
        return df.dropna(axis=1, how='all')
    except: return pd.DataFrame()

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
            
            # Outros indicadores
            inds[('IFR14', t)] = ifr.fillna(50)
            inds[('VolMedio', t)] = vol.rolling(10).mean()
            inds[('Variacao', t)] = close.pct_change(fill_method=None)
            
            # Bollinger
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
        
        if tem_vol and tem_ifr: return "★★★ Forte", "Volume Explosivo + IFR Baixo", 3
        elif tem_vol: return "★★☆ Médio", "Volume Acima da Média", 2
        elif tem_ifr: return "★★☆ Médio", "IFR < 30 (Sobrevenda)", 2
        else: return "★☆☆ Atenção", "Queda Abaixo da Banda", 1
    except: return "Erro", "-", 0

def enviar_whatsapp(msg):
    if not WHATSAPP_PHONE or not WHATSAPP_APIKEY: return
    try:
        texto_codificado = requests.utils.quote(msg)
        url_whatsapp = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={texto_codificado}&apikey={WHATSAPP_APIKEY}"
        headers = { "User-Agent": "Mozilla/5.0" }
        requests.get(url_whatsapp, headers=headers, timeout=20)
    except: pass

# --- UI PRINCIPAL ---

# 1. Header com Hora de Brasília
fuso_br = pytz.timezone('America/Sao_Paulo')
hora_atual = dt.datetime.now(fuso_br).strftime("%H:%M")

if not MODO_ROBO:
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.title("📉 Monitor de Oportunidades BDR")
        st.markdown(f"**Status:** Sistema Operacional | **Filtro:** Queda > {FILTRO_QUEDA:.0%}")
    with col_t2:
        st.metric("🕒 Atualizado às", hora_atual)
    
    st.markdown("---")

# --- EXECUÇÃO ---
botao_analisar = st.button("🔄 Rodar Análise Agora", type="primary") if not MODO_ROBO else True

if botao_analisar:
    # Spinner elegante para carregar
    with st.status("Analisando Mercado...", expanded=True) as status:
        st.write("Conectando à Brapi para obter lista de BDRs...")
        lista_bdrs, mapa_nomes = obter_dados_brapi()
        
        if lista_bdrs:
            st.write(f"Baixando dados históricos de {len(lista_bdrs)} ativos...")
            df = buscar_dados(lista_bdrs)
            
            if not df.empty:
                st.write("Calculando indicadores técnicos (IFR, Bollinger)...")
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
                        nome_empresa = mapa_nomes.get(t, t) # Pega o nome ou usa ticker se falhar
                        
                        resultados.append({
                            'Ticker': t, 
                            'Empresa': nome_empresa,
                            'Variação': var, 
                            'Preço': last[('Close', t)],
                            'IFR (0-100)': last[('IFR14', t)], 
                            'Sinal': classif,
                            'Motivo Técnico': motivo, 
                            'Score': score
                        })
                    except: continue

                status.update(label="Análise Concluída!", state="complete", expanded=False)
                
                if resultados:
                    resultados.sort(key=lambda x: x['Variação'])
                    df_res = pd.DataFrame(resultados)
                    
                    # --- MODO SITE (VISUALIZAÇÃO PRO) ---
                    if not MODO_ROBO:
                        # KPIs de Topo
                        kpi1, kpi2, kpi3 = st.columns(3)
                        kpi1.metric("BDRs Rastreados", len(lista_bdrs))
                        kpi2.metric("Oportunidades", len(resultados))
                        kpi3.metric("Maior Queda", f"{df_res['Variação'].min():.2%}")

                        st.subheader("📋 Relatório de Oportunidades")
                        
                        # CONFIGURAÇÃO DA TABELA PROFISSIONAL
                        st.dataframe(
                            df_res[['Ticker', 'Empresa', 'Preço', 'Variação', 'IFR (0-100)', 'Sinal', 'Motivo Técnico']],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Ticker": st.column_config.TextColumn("Código", width="small"),
                                "Empresa": st.column_config.TextColumn("Nome da Empresa", width="medium"),
                                "Preço": st.column_config.NumberColumn("Preço Atual", format="R$ %.2f"),
                                "Variação": st.column_config.NumberColumn(
                                    "Queda Hoje", 
                                    format="%.2f%%",
                                    help="Variação percentual em relação ao fechamento anterior"
                                ),
                                "IFR (0-100)": st.column_config.ProgressColumn(
                                    "IFR (Sobrevenda)",
                                    format="%.0f",
                                    min_value=0,
                                    max_value=100,
                                    help="Quanto menor, mais 'barato' (sobrevendido) está o ativo."
                                ),
                                "Sinal": st.column_config.TextColumn("Classificação", width="small"),
                            }
                        )
                        
                        # Legenda e Envio
                        with st.expander("ℹ️ Entenda os Sinais"):
                             st.markdown("* **Forte:** Queda + Volume + IFR Baixo.\n* **Médio:** Queda + Volume OU IFR.\n* **IFR:** Abaixo de 30 indica que caiu 'demais'.")

                        if st.checkbox("Enviar Relatório Manual (WhatsApp)"):
                            msg = f"🚨 *Manual* ({hora_atual})\n\n"
                            for item in resultados[:10]:
                                msg += f"-> *{item['Ticker']}*: {item['Variação']:.2%} | {item['Sinal']}\n"
                            enviar_whatsapp(msg)
                            st.success("Enviado!")

                    # --- MODO ROBÔ ---
                    if MODO_ROBO:
                        print(f"Encontradas {len(resultados)} oportunidades.")
                        msg = f"🚨 *Top 10* ({hora_atual})\n\n"
                        for item in resultados[:10]:
                            icone = "🔥" if item['Score'] == 3 else "🔻"
                            # Incluímos o nome da empresa se couber
                            msg += f"{icone} *{item['Ticker']}*: {item['Variação']:.2%} | {item['Sinal']}\n"
                        msg += f"\nSite: share.streamlit.io"
                        enviar_whatsapp(msg)
                else:
                    if not MODO_ROBO: st.info("Nenhuma oportunidade encontrada com os filtros atuais.")
                    else: print("Sem oportunidades.")
            else:
                st.error("Erro ao baixar dados do Yahoo Finance.")
        else:
            st.error("Erro ao conectar na Brapi API.")
