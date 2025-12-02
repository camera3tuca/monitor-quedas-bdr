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

# --- FUNÇÃO DE SEGREDOS (VERSÃO ROBUSTA) ---
# Esta função evita o erro "Exit code 1" no GitHub
def get_secret(key):
    # 1. Tenta pegar das variáveis de ambiente (GitHub Actions)
    env_var = os.environ.get(key)
    if env_var:
        return env_var
    
    # 2. Se não achar, tenta pegar dos segredos do Streamlit (Site)
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except:
        pass
        
    return None

# --- LÓGICA DE MODO (HUMANO vs ROBÔ) ---
# Se estiver rodando no GitHub Actions, ativa o modo Robô
if os.environ.get("GITHUB_ACTIONS") == "true":
    MODO_ROBO = True
else:
    MODO_ROBO = False

# --- BARRA LATERAL (APENAS PARA O MODO VISUAL) ---
if not MODO_ROBO:
    st.sidebar.header("🎛️ Configurações (Site)")
    # Usuário escolhe no tablet. Padrão: -3% e com Bollinger
    filtro_visual = st.sidebar.slider("Mínimo de Queda (%)", -15, 0, -3, 1) / 100
    bollinger_visual = st.sidebar.checkbox("Exigir estar abaixo da Banda?", value=True)
    
    FILTRO_QUEDA = filtro_visual
    USAR_BOLLINGER = bollinger_visual
else:
    # Configuração Fixa do Robô (Para o WhatsApp)
    # Pega tudo que caiu mais de 1%, sem exigir Bollinger
    FILTRO_QUEDA = -0.01 
    USAR_BOLLINGER = False 

# --- CREDENCIAIS ---
WHATSAPP_PHONE = get_secret('WHATSAPP_PHONE')
WHATSAPP_APIKEY = get_secret('WHATSAPP_APIKEY')
BRAPI_API_TOKEN = get_secret('BRAPI_API_TOKEN')

PERIODO_HISTORICO_DIAS = "60d"
TERMINACOES_BDR = ('31', '32', '33', '34', '35', '39')

# --- FUNÇÕES DE DADOS E CÁLCULO ---

@st.cache_data(ttl=3600)
def obter_lista_bdrs_da_brapi():
    if not BRAPI_API_TOKEN:
        print("ERRO: Token BRAPI não encontrado.")
        if not MODO_ROBO: st.error("Token BRAPI ausente.")
        return []
    try:
        url = f"https://brapi.dev/api/quote/list?token={BRAPI_API_TOKEN}"
        r = requests.get(url, timeout=30)
        dados = r.json().get('stocks', [])
        df = pd.DataFrame(dados)
        return df[df['stock'].str.endswith(TERMINACOES_BDR, na=False)]['stock'].tolist()
    except Exception as e:
        print(f"ERRO BRAPI: {e}")
        return []

def buscar_dados(tickers):
    if not tickers: return pd.DataFrame()
    sa_tickers = [f"{t}.SA" for t in tickers]
    try:
        # No modo robô, imprimimos no console. No site, usamos spinner.
        if not MODO_ROBO:
            with st.spinner(f"Analisando {len(tickers)} ativos..."):
                df = yf.download(sa_tickers, period=PERIODO_HISTORICO_DIAS, auto_adjust=True, progress=False, ignore_tz=True)
        else:
            print(f"Baixando dados de {len(tickers)} ativos...")
            df = yf.download(sa_tickers, period=PERIODO_HISTORICO_DIAS, auto_adjust=True, progress=False, ignore_tz=True)
            
        if df.empty: return pd.DataFrame()
        
        # Ajuste de colunas (MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_tuples([(c[0], c[1].replace(".SA", "")) for c in df.columns])
        elif isinstance(df.index, pd.DatetimeIndex) and len(tickers) == 1:
            df.columns = pd.MultiIndex.from_product([df.columns, [tickers[0]]])
            
        return df.dropna(axis=1, how='all')
    except Exception as e:
        print(f"ERRO YFINANCE: {e}")
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
    print("--- INICIANDO ENVIO WHATSAPP ---")
    if not WHATSAPP_PHONE or not WHATSAPP_APIKEY:
        print("ERRO: Credenciais de WhatsApp não encontradas.")
        return

    try:
        texto_codificado = requests.utils.quote(msg)
        url = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={texto_codificado}&apikey={WHATSAPP_APIKEY}"
        
        # Timeout curto para não travar o robô
        response = requests.get(url, timeout=25)
        
        if response.status_code == 200:
            print("SUCESSO: Mensagem enviada para a API CallMeBot.")
        else:
            print(f"FALHA: Código {response.status_code} - Resposta: {response.text}")
            
    except Exception as e:
        print(f"ERRO DE CONEXÃO WHATSAPP: {e}")

# --- EXECUÇÃO PRINCIPAL ---

st.title("📉 Monitor Inteligente de BDRs")

if MODO_ROBO:
    st.info("🤖 MODO ROBÔ ATIVO: Buscando Top 10 Quedas (> 1%)")
else:
    st.info(f"👤 MODO VISUAL: Filtro {FILTRO_QUEDA:.1%} | Bollinger {'Ligado' if USAR_BOLLINGER else 'Desligado'}")

# Botão no site OU execução automática no GitHub
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
                    
                    # 2. Filtro de Bollinger
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
                # Ordenar pela maior queda (valor mais negativo primeiro)
                resultados.sort(key=lambda x: x['Variação'])
                
                # Exibição no Site (Bonita)
                if not MODO_ROBO:
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

                # Envio WhatsApp (Apenas Robô ou Checkbox Manual)
                # No modo Robô, envia automático
                if MODO_ROBO:
                    print(f"Encontradas {len(resultados)} oportunidades. Preparando mensagem...")
                    fuso = pytz.timezone('America/Sao_Paulo')
                    hora = dt.datetime.now(fuso).strftime("%H:%M")
                    
                    msg = f"🚨 *Monitor Top 10* ({hora})\nQuedas > 1% (Sem Bollinger)\n\n"
                    
                    # Top 10 Maiores Quedas
                    for item in resultados[:10]:
                        icone = "🔥" if item['Score'] == 3 else "🔻"
                        msg += f"{icone} *{item['Ticker']}*: {item['Variação']:.2%} | {item['Classificação']}\n"
                    
                    if len(resultados) > 10:
                        msg += f"\n...e mais {len(resultados)-10} no site."
                    
                    msg += f"\nLink: https://share.streamlit.io"
                    
                    enviar_whatsapp(msg)
                
            else:
                if MODO_ROBO:
                    print("Nenhuma oportunidade encontrada com os filtros atuais.")
                else:
                    st.info("Nenhuma oportunidade com os filtros atuais.")
        else:
            if not MODO_ROBO: st.warning("Sem dados históricos.")
            else: print("Erro: DataFrame de dados vazio.")
