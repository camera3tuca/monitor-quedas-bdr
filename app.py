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
st.set_page_config(page_title="Monitor BDR v29", layout="wide", page_icon="📉")
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
    USAR_FIBO = False
    USAR_DONCHIAN = False
else:
    MODO_ROBO = False

# --- CREDENCIAIS ---
WHATSAPP_PHONE = get_secret('WHATSAPP_PHONE')
WHATSAPP_APIKEY = get_secret('WHATSAPP_APIKEY')
BRAPI_API_TOKEN = get_secret('BRAPI_API_TOKEN')

PERIODO_HISTORICO_DIAS = "250d"
TERMINACOES_BDR = ('31', '32', '33', '34', '35', '39')

# --- SIDEBAR ---
if not MODO_ROBO:
    st.sidebar.title("🎛️ Painel v29")
    st.sidebar.markdown("---")
    
    st.sidebar.header("1. Day/Swing Trade (Quedas)")
    filtro_visual = st.sidebar.slider("Mínimo de Queda Total (%)", -15, 0, -3, 1) / 100
    bollinger_visual = st.sidebar.checkbox("Abaixo da Banda de Bollinger?", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. Estratégias de Tendência")
    fibo_visual = st.sidebar.checkbox("💎 Fibo Golden Zone", value=False)
    donchian_visual = st.sidebar.checkbox("🐢 Donchian 10 (Semanal)", value=False)
    
    st.sidebar.info("Ordenação: Sempre pela Maior Queda")
    
    FILTRO_QUEDA = filtro_visual
    USAR_BOLLINGER = bollinger_visual
    USAR_FIBO = fibo_visual
    USAR_DONCHIAN = donchian_visual

# --- FUNÇÕES ---

@st.cache_data(ttl=3600)
def obter_dados_brapi():
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

# Resumo Simples
def obter_resumo_simples(p_open, p_atual):
    return f"Abertura: {p_open:.2f} ➡ Atual: {p_atual:.2f}"

# ESTRATÉGIA 1: FIBO
def verificar_padrao_fibo(df_asset):
    try:
        if len(df_asset) < 70: return None
        close = df_asset['Close']; high = df_asset['High']; low = df_asset['Low']
        ema_trend = close.ewm(span=50).mean()
        if close.iloc[-1] < ema_trend.iloc[-1]: return None
        recorte_topo = high.tail(20)
        topo_val = recorte_topo.max(); topo_idx = recorte_topo.idxmax()
        df_antes = df_asset.loc[:topo_idx].iloc[:-1]
        if len(df_antes) < 60: return None
        fundo_val = df_antes['Low'].tail(60).min()
        diff = topo_val - fundo_val
        if diff <= 0 or (diff/fundo_val) < 0.08: return None
        fibo_618 = topo_val - (diff * 0.618)
        fibo_500 = topo_val - (diff * 0.500)
        low_hj = low.iloc[-1]
        if low_hj <= fibo_500*1.01 and low_hj >= fibo_618*0.99:
            return f"Golden Zone"
        return None
    except: return None

# ESTRATÉGIA 2: DONCHIAN 10
def verificar_donchian_semanal(df_daily):
    try:
        df_w = df_daily.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        })
        if len(df_w) < 12: return None
        donchian_high = df_w['High'].rolling(10).max().shift(1)
        preco_atual = df_w['Close'].iloc[-1]
        banda_superior = donchian_high.iloc[-1]
        if pd.isna(banda_superior): return None
        if preco_atual > banda_superior:
            return f"Rompimento Semanal (Preço: {preco_atual:.2f} > Banda: {banda_superior:.2f})"
        return None
    except: return None

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

def analisar_sinal_classico(row, t):
    try:
        vol = row[('Volume', t)]
        vol_med = row[('VolMedio', t)]
        ifr = row[('IFR14', t)]
        tem_vol = vol > vol_med if (not pd.isna(vol) and not pd.isna(vol_med)) else False
        tem_ifr = ifr < 30 if not pd.isna(ifr) else False
        
        if tem_vol and tem_ifr: return "★★★ Forte", "Vol + IFR", 3
        elif tem_vol: return "★★☆ Médio", "Volume", 2
        elif tem_ifr: return "★★☆ Médio", "IFR", 2
        else: return "★☆☆ Atenção", "Queda", 1
    except: return "Erro", "-", 0

def enviar_whatsapp(msg):
    print("--- TENTANDO ENVIAR WHATSAPP ---")
    if not WHATSAPP_PHONE or not WHATSAPP_APIKEY: return
    try:
        texto_codificado = requests.utils.quote(msg)
        url_whatsapp = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={texto_codificado}&apikey={WHATSAPP_APIKEY}"
        headers = { "User-Agent": "Mozilla/5.0" }
        requests.get(url_whatsapp, headers=headers, timeout=25)
    except: pass

# --- UI VISUAL ---
fuso = pytz.timezone('America/Sao_Paulo')
hora_atual = dt.datetime.now(fuso).strftime("%H:%M")

if not MODO_ROBO:
    col_a, col_b = st.columns([3, 1])
    col_a.title("📉 Monitor BDR v29")
    col_b.metric("🕒 Hora Brasília", hora_atual)
    
    with st.expander("ℹ️ Detalhes das Estratégias"):
        st.markdown("""
        * **Day/Swing (Queda):** Busca ativos sobrevendidos.
        * **Fibo Golden Zone:** Busca ativos em tendência de alta corrigindo.
        * **Donchian 10:** Busca rompimento de máxima semanal (Position).
        """)

# --- EXECUÇÃO ---
start_btn = True if MODO_ROBO else st.button("🔄 Rodar Análise Agora", type="primary")

if start_btn:
    lista_bdrs, mapa_nomes = obter_dados_brapi()
    
    if not MODO_ROBO and lista_bdrs:
        st.write(f"Analisando {len(lista_bdrs)} ativos...")
        
    if lista_bdrs:
        df = buscar_dados(lista_bdrs)
        if not df.empty:
            df_calc = calcular_indicadores(df)
            last = df_calc.iloc[-1]
            resultados = []
            
            for t in df_calc.columns.get_level_values(1).unique():
                try:
                    # DADOS BÁSICOS
                    var_total = last.get(('Variacao', t), np.nan)
                    p_atual = last[('Close', t)]
                    p_open = last[('Open', t)]
                    
                    # Gap e Intraday
                    p_ontem = p_atual / (1 + var_total)
                    gap_pct = (p_open / p_ontem) - 1
                    intraday_pct = (p_atual / p_open) - 1
                    
                    status_movimento = "Neutro"
                    if gap_pct < -0.005:
                        if intraday_pct > 0.002: status_movimento = "♻️ Recuperando"
                        elif intraday_pct < -0.002: status_movimento = "📉 Afundando"
                        else: status_movimento = "↔️ Lateral"
                    elif intraday_pct < -0.01:
                         status_movimento = "🔻 Queda Intraday"

                    low = last.get(('Low', t), np.nan)
                    banda = last.get(('BandaInf', t), np.nan)
                    
                    # CHECAGEM DE ESTRATÉGIAS
                    sinal_fibo = None
                    if USAR_FIBO:
                        try:
                            df_ticker = df.xs(t, axis=1, level=1).dropna()
                            sinal_fibo = verificar_padrao_fibo(df_ticker)
                        except: pass
                    
                    sinal_donchian = None
                    if USAR_DONCHIAN:
                        try:
                            df_ticker = df.xs(t, axis=1, level=1).dropna()
                            sinal_donchian = verificar_donchian_semanal(df_ticker)
                        except: pass
                    
                    passou_queda = False
                    if not USAR_FIBO and not USAR_DONCHIAN:
                        passou_queda = True
                        if USAR_BOLLINGER and (pd.isna(low) or low >= banda): passou_queda = False
                        if pd.isna(var_total) or var_total > FILTRO_QUEDA: passou_queda = False
                    
                    # LOGICA DE INCLUSÃO
                    incluir = False
                    if USAR_FIBO and sinal_fibo: incluir = True
                    if USAR_DONCHIAN and sinal_donchian: incluir = True
                    if not USAR_FIBO and not USAR_DONCHIAN and passou_queda: incluir = True
                    
                    if not incluir: continue
                    
                    # CLASSIFICAÇÃO
                    if sinal_donchian:
                        classif = "🐢 DONCHIAN"
                        motivo = sinal_donchian
                        score = 6
                    elif sinal_fibo:
                        classif = "💎 FIBO"
                        motivo = sinal_fibo
                        score = 5
                    else:
                        classif, motivo, score = analisar_sinal_classico(last, t)
                    
                    nome_completo = mapa_nomes.get(t, t)
                    primeiro_nome = nome_completo.split()[0] if nome_completo else t
                    resumo_simples = obter_resumo_simples(p_open, p_atual)

                    resultados.append({
                        'Ticker': t, 'Empresa': primeiro_nome,
                        'Variação Total': var_total, 'Gap Abertura': gap_pct,
                        'Força Intraday': intraday_pct, 'Preço': p_atual,
                        'IFR14': last[('IFR14', t)], 'Classificação': classif,
                        'Status': status_movimento, 'Motivo': motivo, 
                        'Score': score, 'Evolução': resumo_simples
                    })
                except: continue

            if resultados:
                # ORDENAÇÃO CORRIGIDA: SEMPRE PELA MAIOR QUEDA (Variação Total)
                # Mesmo que seja um sinal Donchian, se a queda for grande, vai pro topo
                resultados.sort(key=lambda x: x['Variação Total'])
                
                if not MODO_ROBO:
                    st.success(f"{len(resultados)} oportunidades.")
                    df_show = pd.DataFrame(resultados)
                    
                    df_show['Variação Total'] = df_show['Variação Total'].apply(lambda x: f"{x:.2%}")
                    df_show['Gap Abertura'] = df_show['Gap Abertura'].apply(lambda x: f"{x:.2%}")
                    df_show['Força Intraday'] = df_show['Força Intraday'].apply(lambda x: f"{x:.2%}")
                    df_show['Preço'] = df_show['Preço'].apply(lambda x: f"R$ {x:.2f}")
                    df_show['IFR14'] = df_show['IFR14'].apply(lambda x: f"{x:.1f}")
                    
                    st.dataframe(
                        df_show[['Ticker', 'Empresa', 'Variação Total', 'Gap Abertura', 'Força Intraday', 'Status', 'Preço', 'IFR14', 'Classificação', 'Evolução']], 
                        use_container_width=True, hide_index=True,
                        column_config={
                            "Variação Total": st.column_config.TextColumn("Total", width="small"),
                            "Gap Abertura": st.column_config.TextColumn("Gap", width="small"),
                            "Status": st.column_config.TextColumn("Diagnóstico", width="medium"),
                            "Evolução": st.column_config.TextColumn("Abertura ➡ Atual", width="medium"),
                        }
                    )
                    
                    if st.checkbox("Enviar WhatsApp Manual?"):
                        msg = f"🚨 *Manual* ({hora_atual})\n\n"
                        for item in resultados[:10]:
                            msg += f"-> *{item['Ticker']}*: {item['Variação Total']:.2%} | {item['Classificação']}\n"
                        enviar_whatsapp(msg)
                        st.success("Enviado!")

                if MODO_ROBO:
                    print(f"Encontradas {len(resultados)} oportunidades.")
                    msg = f"🚨 *Top Quedas* ({hora_atual})\n\n"
                    # WhatsApp Limpo (Ticker + Queda + Classificação)
                    for item in resultados[:10]:
                        icone = "🐢" if "DONCHIAN" in item['Classificação'] else ("💎" if "FIBO" in item['Classificação'] else "🔻")
                        queda_txt = f"{item['Variação Total']:.2%}"
                        msg += f"{icone} *{item['Ticker']}*: {queda_txt} | {item['Classificação']}\n"
                    msg += f"\nDetalhes: share.streamlit.io"
                    enviar_whatsapp(msg)
            else:
                if MODO_ROBO: print("Sem oportunidades.")
                else: st.info("Nenhuma oportunidade encontrada.")
