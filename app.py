import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import numpy as np
import os
import datetime as dt
import pytz

# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(page_title="Monitor de Quedas BDRs", layout="wide")

# --- FUNÇÃO PARA GERIR SEGREDOS ---
def get_secret(key):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key)

# --- CONFIGURAÇÕES LATERAIS (SIDEBAR) ---
st.sidebar.header("🎛️ Configurações do Filtro")

# Slider para definir a queda mínima (Padrão: -3%)
FILTRO_QUEDA = st.sidebar.slider(
    "Mínimo de Queda (%)", 
    min_value=-15, 
    max_value=0, 
    value=-3,  # Agora o padrão é 3% (mais sensível)
    step=1
) / 100

# Opção para ignorar filtro de Bollinger (para ver tudo o que cai)
USAR_BOLLINGER = st.sidebar.checkbox("Exigir estar abaixo da Banda de Bollinger?", value=True)

st.sidebar.info(f"Procurando ativos com queda maior que {FILTRO_QUEDA:.0%}...")

# --- CREDENCIAIS ---
WHATSAPP_PHONE = get_secret('WHATSAPP_PHONE')
WHATSAPP_APIKEY = get_secret('WHATSAPP_APIKEY')
BRAPI_API_TOKEN = get_secret('BRAPI_API_TOKEN')

PERIODO_HISTORICO_DIAS = "60d"
TERMINACOES_BDR = ('31', '32', '33', '34', '35', '39')

# --- FUNÇÕES DE LÓGICA ---

@st.cache_data(ttl=3600)
def obter_lista_bdrs_da_brapi():
    if not BRAPI_API_TOKEN:
        st.error("BRAPI_API_TOKEN não configurado.")
        return []
    try:
        url = f"https://brapi.dev/api/quote/list?token={BRAPI_API_TOKEN}"
        response = requests.get(url, timeout=30)
        dados = response.json().get('stocks', [])
        df = pd.DataFrame(dados)
        bdrs = df[df['stock'].str.endswith(TERMINACOES_BDR, na=False)]['stock'].tolist()
        return bdrs
    except Exception as e:
        st.error(f"Erro ao buscar lista de BDRs: {e}")
        return []

def buscar_dados_historicos_completos(tickers, periodo):
    tickers_sa = [f"{ticker}.SA" for ticker in tickers]
    try:
        with st.spinner(f'Baixando dados de {len(tickers)} ativos...'):
            dados = yf.download(tickers_sa, period=periodo, auto_adjust=True, progress=False, ignore_tz=True)
        
        if dados.empty: return pd.DataFrame()

        if isinstance(dados.columns, pd.MultiIndex):
             dados.columns = pd.MultiIndex.from_tuples([(col[0], col[1].replace(".SA", "")) for col in dados.columns])
        elif isinstance(dados.index, pd.DatetimeIndex) and len(tickers) == 1:
             ticker_name = tickers[0]
             dados.columns = pd.MultiIndex.from_product([dados.columns, [ticker_name]])
        
        dados = dados.dropna(axis=1, how='all')
        return dados
    except Exception as e:
        st.error(f"Erro ao buscar dados históricos: {e}")
        return pd.DataFrame()

def calcular_indicadores(df):
    df_completo = df.copy()
    tickers = df.columns.get_level_values(1).unique()
    dict_indicadores = {}

    for ticker in tickers:
        try:
            close_df = df[('Close', ticker)]
            volume_df = df[('Volume', ticker)]
            
            # IFR14
            delta = close_df.diff()
            ganhos = delta.where(delta > 0, 0).ewm(com=14 - 1, adjust=False).mean()
            perdas = -delta.where(delta < 0, 0).ewm(com=14 - 1, adjust=False).mean()
            rs = ganhos / perdas
            ifr14 = 100 - (100 / (1 + rs))
            dict_indicadores[('IFR14', ticker)] = ifr14.replace([np.inf, -np.inf], 100).fillna(50)

            # Volume Médio 10 e Variação
            dict_indicadores[('VolumeMedio10', ticker)] = volume_df.rolling(window=10).mean()
            dict_indicadores[('Variacao%', ticker)] = close_df.pct_change()

            # Bandas de Bollinger
            sma_20 = close_df.rolling(window=20).mean()
            std_20 = close_df.rolling(window=20).std()
            dict_indicadores[('BandaSuperior', ticker)] = sma_20 + (std_20 * 2)
            dict_indicadores[('BandaInferior', ticker)] = sma_20 - (std_20 * 2)

        except Exception:
            continue

    if not dict_indicadores: return pd.DataFrame()
    
    df_indicadores = pd.DataFrame(dict_indicadores)
    df_completo = df_completo.join(df_indicadores, how='left')
    return df_completo.sort_index(axis=1)

def avaliar_sinal_queda(ultimo_candle, ticker):
    try:
        has_volume = ('Volume', ticker) in ultimo_candle and ('VolumeMedio10', ticker) in ultimo_candle
        has_ifr = ('IFR14', ticker) in ultimo_candle

        volume_alto = has_volume and not pd.isna(ultimo_candle[('VolumeMedio10', ticker)]) and ultimo_candle[('Volume', ticker)] > ultimo_candle[('VolumeMedio10', ticker)]
        em_sobrevenda = has_ifr and ultimo_candle[('IFR14', ticker)] < 30
        
        if volume_alto and em_sobrevenda:
            return "★★★ Sinal Forte"
        elif volume_alto or em_sobrevenda:
            return "★★☆ Sinal Bom"
        else:
            return "★☆☆ Sinal de Atenção"
    except:
         return "☆☆☆ Erro"

def enviar_whatsapp(msg):
    if not WHATSAPP_PHONE or not WHATSAPP_APIKEY:
        st.warning("Credenciais WhatsApp não configuradas.")
        return
    try:
        texto_codificado = requests.utils.quote(msg)
        url = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={texto_codificado}&apikey={WHATSAPP_APIKEY}"
        requests.get(url, timeout=20)
        st.success("Notificação enviada para o WhatsApp!")
    except Exception as e: 
        st.error(f"Erro ao enviar WhatsApp: {e}")

# --- INTERFACE PRINCIPAL ---

st.title("📉 Monitor de Quedas BDRs")
st.markdown(f"**Filtro Atual:** Queda de {FILTRO_QUEDA:.0%} | Bollinger: {'Sim' if USAR_BOLLINGER else 'Não'}")

if st.button("🔄 Rodar Análise Agora") or os.environ.get("GITHUB_ACTIONS") == "true":
    
    bdrs = obter_lista_bdrs_da_brapi()
    st.write(f"🔍 **Total de BDRs encontrados:** {len(bdrs)}")
    
    if len(bdrs) > 0:
        dados = buscar_dados_historicos_completos(bdrs, PERIODO_HISTORICO_DIAS)
        
        if not dados.empty:
            df_calc = calcular_indicadores(dados)
            ultimo_dia = df_calc.iloc[-1]
            variacoes = ultimo_dia['Variacao%']
            
            # Filtro 1: Percentual de Queda (Controlado pelo Slider)
            quedas = variacoes[variacoes <= FILTRO_QUEDA]
            
            sinais_finais = []
            
            for ticker in quedas.index:
                # Filtro 2: Bandas de Bollinger (Opcional via Checkbox)
                low = ultimo_dia[('Low', ticker)]
                b_inf = ultimo_dia[('BandaInferior', ticker)]
                
                passou_bollinger = False
                if not pd.isna(low) and not pd.isna(b_inf) and low < b_inf:
                    passou_bollinger = True
                
                # Só adiciona se o filtro Bollinger estiver desligado OU se passou no filtro
                if not USAR_BOLLINGER or passou_bollinger:
                    rating = avaliar_sinal_queda(ultimo_dia, ticker)
                    var_val = ultimo_dia[('Variacao%', ticker)]
                    ifr_val = ultimo_dia[('IFR14', ticker)]
                    close_val = ultimo_dia[('Close', ticker)]
                    
                    sinais_finais.append({
                        'Ticker': ticker,
                        'Variação': f"{var_val:.2%}",
                        'IFR14': f"{ifr_val:.1f}",
                        'Classificação': rating,
                        'Preço Fecho': f"R$ {close_val:.2f}" if not pd.isna(close_val) else "N/A"
                    })

            if sinais_finais:
                # Ordenar: Fortes primeiro
                sinais_finais.sort(key=lambda x: x['Classificação'], reverse=True)
                
                df_resultado = pd.DataFrame(sinais_finais)
                st.subheader(f"🚨 {len(sinais_finais)} Oportunidades Encontradas")
                st.dataframe(df_resultado, use_container_width=True)
                
                fuso = pytz.timezone('America/Sao_Paulo')
                agora = dt.datetime.now(fuso).strftime("%d/%m %H:%M")
                
                msg = f"🚨 *Robô BDRs* ({agora})\nFiltro: {FILTRO_QUEDA:.0%}\n\n"
                for item in sinais_finais[:10]: # Limita a 10 no Whatsapp para não ficar gigante
                    msg += f"-> *{item['Ticker']}*: {item['Variação']} | {item['Classificação']}\n"
                
                if len(sinais_finais) > 10:
                    msg += f"\n...e mais {len(sinais_finais)-10} no App."
                
                msg += "\nVer detalhes no WebApp."
                
                # Envio automático só no GitHub Actions ou se marcado
                is_github = os.environ.get("GITHUB_ACTIONS") == "true"
                enviar = st.checkbox("Enviar notificação WhatsApp?", value=is_github)
                
                if enviar:
                    enviar_whatsapp(msg)
            else:
                st.info("Nenhum ativo corresponde aos filtros atuais. Tente diminuir a % de queda na barra lateral.")
        else:
            st.warning("Não foi possível obter dados históricos.")
