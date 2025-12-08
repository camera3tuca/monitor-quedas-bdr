import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import numpy as np
import os
import datetime as dt
import pytz
import warnings

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Monitor BDR v17", layout="wide", page_icon="📉")
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

# --- MODO ROBÔ ---
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

# --- FUNÇÕES ---

@st.cache_data(ttl=3600)
def obter_dados_brapi():
    """Retorna Tickers e Mapa de Nomes"""
    if not BRAPI_API_TOKEN: return [], {}
    try:
        url = f"https://brapi.dev/api/quote/list?token={BRAPI_API_TOKEN}"
        r = requests.get(url, timeout=30)
        dados = r.json().get('stocks', [])
        
        bdrs_raw = [d for d in dados if d['stock'].endswith(TERMINACOES_BDR)]
        lista_tickers = [d['stock'] for d in bdrs_raw]
        mapa_nomes = {d['stock']: d.get('name', d['stock']) for d in bdrs_raw}
        
        return lista_tickers, mapa_nomes
    except: return [], {}

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

# --- LÓGICA V14 (SEGURA) ---
def obter_resumo_dia(ticker, open_daily, close_daily):
    # Tenta dados horários
    try:
        df = yf.download(f"{ticker}.SA", period="1d", interval="1h", progress=False, ignore_tz=True)
        if not df.empty and len(df) > 1:
            txt_partes = []
            for hora_ts, row in df.iterrows():
                h = hora_ts.hour
                val = row['Close']
                var_vs_open = (val / df['Open'].iloc[0]) - 1
                txt_partes.append(f"{h}h: {var_vs_open:+.1%}")
            return " ➡ ".join(txt_partes[-4:])
    except: pass
    
    # Fallback (Plano B)
    try:
        var_dia = (close_daily / open_daily) - 1
        return f"Abertura: {open_daily:.2f} ➡ Atual: {close_daily:.2f} ({var_dia:+.1%})"
    except: return "-"

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
        
        if tem_vol and tem_ifr: return "★★★ Forte", "Volume Explosivo + IFR Baixo", 3
        elif tem_vol: return "★★☆ Médio", "Volume Acima da Média", 2
        elif tem_ifr: return "★★☆ Médio", "IFR < 30 (Sobrevenda)", 2
        else: return "★☆☆ Atenção", "Apenas Queda (Furou Banda)", 1
    except: return "Erro", "-", 0

def enviar_whatsapp(msg):
    if not WHATSAPP_PHONE or not WHATSAPP_APIKEY: return
    try:
        texto_codificado = requests.utils.quote(msg)
        url_whatsapp = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={texto_codificado}&apikey={WHATSAPP_APIKEY}"
        headers = { "User-Agent": "Mozilla/5.0" }
        requests.get(url_whatsapp, headers=headers, timeout=20)
    except: pass

# --- UI VISUAL (SITE) ---

# 1. Cabeçalho com Relógio
fuso = pytz.timezone('America/Sao_Paulo')
hora_atual = dt.datetime.now(fuso).strftime("%H:%M")

if not MODO_ROBO:
    # Sidebar
    st.sidebar.title("🎛️ Painel v17")
    filtro_visual = st.sidebar.slider("Mínimo de Queda (%)", -15, 0, -3, 1) / 100
    bollinger_visual = st.sidebar.checkbox("Abaixo da Banda de Bollinger?", value=True)
    FILTRO_QUEDA = filtro_visual
    USAR_BOLLINGER = bollinger_visual

    # Header
    col_a, col_b = st.columns([3, 1])
    col_a.title("📉 Monitor BDR")
    col_b.metric("🕒 Hora Brasília", hora_atual)
    
    # 2. Legenda Explicativa
    with st.expander("ℹ️ Entenda a Classificação dos Sinais"):
        st.markdown("""
        * **★★★ Forte:** Queda acentuada + Volume alto (pânico) + IFR abaixo de 30 (barato).
        * **★★☆ Médio:** Queda acentuada + (Volume alto OU IFR baixo).
        * **★☆☆ Atenção:** Ação caiu e furou a Banda de Bollinger, mas sem volume expressivo.
        """)

# --- EXECUÇÃO ---
botao_analisar = st.button("🔄 Rodar Análise Agora", type="primary") if not MODO_ROBO else True

if botao_analisar:
    lista_bdrs, mapa_nomes = obter_dados_brapi()
    
    if lista_bdrs:
        df = buscar_dados(lista_bdrs)
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
                    
                    # Nome Curto (Apenas o primeiro nome)
                    nome_completo = mapa_nomes.get(t, t)
                    nome_curto = nome_completo.split()[0] if nome_completo else t
                    
                    # Evolução v14
                    resumo_dia = "-"
                    if not MODO_ROBO:
                         resumo_dia = obter_resumo_dia(t, last[('Open', t)], last[('Close', t)])

                    resultados.append({
                        'Ticker': t, 
                        'Empresa': nome_curto, # Só o primeiro nome
                        'Variação': var, 
                        'Preço': last[('Close', t)],
                        'IFR14': last[('IFR14', t)], 
                        'Classificação': classif,
                        'Motivo': motivo, 
                        'Score': score,
                        'Evolução do Dia': resumo_dia
                    })
                except: continue

            if resultados:
                resultados.sort(key=lambda x: x['Variação'])
                
                # --- VISUALIZAÇÃO SITE ---
                if not MODO_ROBO:
                    st.success(f"{len(resultados)} oportunidades encontradas.")
                    
                    df_show = pd.DataFrame(resultados)
                    
                    # TABELA CONFIGURADA
                    st.dataframe(
                        df_show[['Ticker', 'Empresa', 'Variação', 'Preço', 'Classificação', 'Motivo', 'Evolução do Dia']], 
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Ticker": st.column_config.TextColumn("Código", width="small"),
                            "Empresa": st.column_config.TextColumn("Nome", width="small"),
                            "Variação": st.column_config.NumberColumn("Queda", format="%.2f%%"),
                            "Preço": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
                            "Motivo": st.column_config.TextColumn("Explicação Técnica", width="medium"),
                            "Evolução do Dia": st.column_config.TextColumn("Tendência Intraday", width="large"),
                        }
                    )
                    
                    if st.checkbox("Enviar WhatsApp Manual?"):
                        msg = f"🚨 *Manual* ({hora_atual})\n\n"
                        for item in resultados[:10]:
                            msg += f"-> *{item['Ticker']}*: {item['Variação']:.2%} | {item['Classificação']}\n"
                        enviar_whatsapp(msg)
                        st.success("Enviado!")

                # --- MODO ROBÔ ---
                if MODO_ROBO:
                    print(f"Encontradas {len(resultados)} oportunidades.")
                    msg = f"🚨 *Top 10* ({hora_atual})\n\n"
                    for item in resultados[:10]:
                        icone = "🔥" if item['Score'] == 3 else "🔻"
                        msg += f"{icone} *{item['Ticker']}* ({item['Empresa']}): {item['Variação']:.2%} | {item['Classificação']}\n"
                    msg += f"\nSite: share.streamlit.io"
                    enviar_whatsapp(msg)
            else:
                if MODO_ROBO: print("Sem oportunidades.")
                else: st.info("Nenhuma oportunidade encontrada.")
