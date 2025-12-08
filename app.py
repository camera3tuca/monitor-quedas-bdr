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
st.set_page_config(page_title="Monitor BDR v19 (Fibo)", layout="wide", page_icon="📈")
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
    # Configuração Padrão do Robô (Pode ajustar aqui se quer Fibo ou Queda)
    FILTRO_QUEDA = -0.01
    USAR_BOLLINGER = False
    USAR_FIBO = False # Robô por padrão usa Queda Simples (mais frequente)
else:
    MODO_ROBO = False

# --- CREDENCIAIS ---
WHATSAPP_PHONE = get_secret('WHATSAPP_PHONE')
WHATSAPP_APIKEY = get_secret('WHATSAPP_APIKEY')
BRAPI_API_TOKEN = get_secret('BRAPI_API_TOKEN')

# Aumentamos o histórico para a EMA 50 funcionar bem na estratégia Fibo
PERIODO_HISTORICO_DIAS = "250d" 
TERMINACOES_BDR = ('31', '32', '33', '34', '35', '39')

# --- SIDEBAR (FILTROS) ---
if not MODO_ROBO:
    st.sidebar.title("🎛️ Painel v19")
    st.sidebar.markdown("---")
    
    st.sidebar.header("1. Estratégia Clássica")
    filtro_visual = st.sidebar.slider("Mínimo de Queda (%)", -15, 0, -3, 1) / 100
    bollinger_visual = st.sidebar.checkbox("Abaixo da Banda de Bollinger?", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. Estratégia Gráfica")
    # A NOVA CAIXA DE FILTRO FIBO
    fibo_visual = st.sidebar.checkbox("💎 Fibo Golden Zone (Tendência)", value=False, help="Busca ativos em tendência de alta corrigindo entre 50% e 61.8% de Fibo.")
    
    FILTRO_QUEDA = filtro_visual
    USAR_BOLLINGER = bollinger_visual
    USAR_FIBO = fibo_visual

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

# --- LÓGICA V14 (EVOLUÇÃO) ---
def obter_resumo_dia(ticker, open_daily, close_daily):
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
    try:
        var_dia = (close_daily / open_daily) - 1
        return f"Abertura: {open_daily:.2f} ➡ Atual: {close_daily:.2f} ({var_dia:+.1%})"
    except: return "-"

# --- NOVA LÓGICA: FIBO GOLDEN ZONE ---
def verificar_padrao_fibo(df_asset):
    # Parâmetros da estratégia
    EMA_TENDENCIA = 50
    JANELA_TOPO = 20
    JANELA_FUNDO = 60
    
    try:
        # Precisa de dados suficientes
        if len(df_asset) < JANELA_FUNDO + 10: return None
        
        close = df_asset['Close']
        high = df_asset['High']
        low = df_asset['Low']
        
        # 1. Filtro de Tendência (EMA 50)
        ema_trend = close.ewm(span=EMA_TENDENCIA).mean()
        # Preço atual deve estar ACIMA da média (tendência de alta) ou muito próximo
        if close.iloc[-1] < ema_trend.iloc[-1]: return None
        
        # 2. Identificar Topo Recente e Fundo Anterior
        recorte_topo = high.tail(JANELA_TOPO)
        topo_val = recorte_topo.max()
        topo_idx = recorte_topo.idxmax()
        
        # O Fundo deve ser ANTES do topo
        df_antes_topo = df_asset.loc[:topo_idx].iloc[:-1]
        if len(df_antes_topo) < JANELA_FUNDO: return None
        
        recorte_fundo = df_antes_topo['Low'].tail(JANELA_FUNDO)
        fundo_val = recorte_fundo.min()
        
        diff = topo_val - fundo_val
        # Se a diferença for muito pequena (lateralização), ignora
        if diff <= 0 or (diff / fundo_val) < 0.08: return None
        
        # 3. Calcular Níveis Fibo
        fibo_500 = topo_val - (diff * 0.500) # Retração de 50%
        fibo_618 = topo_val - (diff * 0.618) # Retração de 61.8%
        
        # 4. Verificar se preço atual está na Zona
        # Aceitamos se a mínima do dia tocou a zona
        low_hoje = low.iloc[-1]
        close_hoje = close.iloc[-1]
        
        # Margem de tolerância de 1%
        zona_topo = fibo_500 * 1.01
        zona_fundo = fibo_618 * 0.99
        
        if low_hoje <= zona_topo and low_hoje >= zona_fundo:
            return f"Golden Zone (R$ {fibo_618:.2f}-{fibo_500:.2f})"
        
        return None
    except:
        return None

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

# --- UI VISUAL ---
fuso = pytz.timezone('America/Sao_Paulo')
hora_atual = dt.datetime.now(fuso).strftime("%H:%M")

if not MODO_ROBO:
    col_a, col_b = st.columns([3, 1])
    col_a.title("📉 Monitor BDR Pro")
    col_b.metric("🕒 Hora Brasília", hora_atual)
    
    with st.expander("ℹ️ Legenda"):
        st.markdown("""
        * **Estratégia Clássica:** Foca em ativos que caíram muito hoje (Setup de Queda).
        * **Fibo Golden Zone:** Foca em ativos em **Tendência de Alta** que estão apenas a "respirar" (corrigir) para voltar a subir.
        """)

# --- EXECUÇÃO ---
botao_analisar = st.button("🔄 Rodar Análise Agora", type="primary") if not MODO_ROBO else True

if botao_analisar:
    lista_bdrs, mapa_nomes = obter_dados_brapi()
    
    if not MODO_ROBO and lista_bdrs:
        st.write(f"Analisando {len(lista_bdrs)} ativos (Histórico: 250 dias)...")
        
    if lista_bdrs:
        df = buscar_dados(lista_bdrs)
        if not df.empty:
            df_calc = calcular_indicadores(df)
            last = df_calc.iloc[-1]
            resultados = []
            
            for t in df_calc.columns.get_level_values(1).unique():
                try:
                    # 1. DADOS BÁSICOS
                    var = last.get(('Variacao', t), np.nan)
                    low = last.get(('Low', t), np.nan)
                    banda = last.get(('BandaInf', t), np.nan)
                    
                    # --- FILTRO 1: ESTRATÉGIA CLÁSSICA (QUEDA) ---
                    passou_queda = True
                    if USAR_BOLLINGER and (pd.isna(low) or low >= banda): passou_queda = False
                    if pd.isna(var) or var > FILTRO_QUEDA: passou_queda = False
                    
                    # --- FILTRO 2: ESTRATÉGIA FIBO (NOVO) ---
                    sinal_fibo = None
                    if USAR_FIBO:
                        # Extrai dados históricos deste ticker específico para análise gráfica
                        try:
                            df_ticker = df.xs(t, axis=1, level=1).dropna()
                            sinal_fibo = verificar_padrao_fibo(df_ticker)
                        except: pass
                    
                    # LÓGICA DE DECISÃO FINAL
                    # Se Fibo estiver ativado, só passa se tiver sinal Fibo
                    # Se Bollinger estiver ativado, só passa se tiver Bollinger
                    # Se ambos ativados, precisa dos dois (muito raro)
                    # Se nenhum filtro (só % queda), usa queda
                    
                    if USAR_FIBO and not sinal_fibo: continue
                    if not USAR_FIBO and not passou_queda: continue
                    
                    # Se passou, define a classificação
                    if sinal_fibo:
                        classif = "💎 FIBO"
                        motivo = sinal_fibo
                        score = 5 # Prioridade máxima
                    else:
                        classif, motivo, score = analisar_sinal_classico(last, t)

                    # Nome e Evolução
                    nome_completo = mapa_nomes.get(t, t)
                    primeiro_nome = nome_completo.split()[0] if nome_completo else t
                    
                    resumo_dia = "-"
                    if not MODO_ROBO:
                         resumo_dia = obter_resumo_dia(t, last[('Open', t)], last[('Close', t)])

                    resultados.append({
                        'Ticker': t, 
                        'Empresa': primeiro_nome,
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
                # Ordena por Score (Fibo no topo) e depois por Variação
                resultados.sort(key=lambda x: (-x['Score'], x['Variação']))
                
                if not MODO_ROBO:
                    st.success(f"{len(resultados)} oportunidades encontradas.")
                    
                    df_show = pd.DataFrame(resultados)
                    df_show['Variação'] = df_show['Variação'].apply(lambda x: f"{x:.2%}")
                    df_show['Preço'] = df_show['Preço'].apply(lambda x: f"R$ {x:.2f}")
                    df_show['IFR14'] = df_show['IFR14'].apply(lambda x: f"{x:.1f}")
                    
                    st.dataframe(
                        df_show[['Ticker', 'Empresa', 'Variação', 'Preço', 'IFR14', 'Classificação', 'Motivo', 'Evolução do Dia']], 
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Evolução do Dia": st.column_config.TextColumn("Tendência Intraday", width="medium"),
                            "Motivo": st.column_config.TextColumn("Análise", width="medium"),
                        }
                    )
                    
                    if st.checkbox("Enviar WhatsApp Manual?"):
                        msg = f"🚨 *Manual* ({hora_atual})\n\n"
                        for item in resultados[:10]:
                            msg += f"-> *{item['Ticker']}*: {item['Variação']} | {item['Classificação']}\n"
                        enviar_whatsapp(msg)
                        st.success("Enviado!")

                if MODO_ROBO:
                    print(f"Encontradas {len(resultados)} oportunidades.")
                    msg = f"🚨 *Top 10* ({hora_atual})\n\n"
                    for item in resultados[:10]:
                        icone = "💎" if "FIBO" in item['Classificação'] else "🔻"
                        msg += f"{icone} *{item['Ticker']}*: {item['Variação']:.2%} | {item['Classificação']}\n"
                    msg += f"\nSite: share.streamlit.io"
                    enviar_whatsapp(msg)
            else:
                if MODO_ROBO: print("Sem oportunidades.")
                else: st.info("Nenhuma oportunidade encontrada.")
