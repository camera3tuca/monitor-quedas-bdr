import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import urllib.parse
from datetime import datetime
import pytz
import warnings

# --- CONFIGURAÇÃO DA PÁGINA ---
VERSAO_APP = "v2.2 (Botão Manual + Teste)"

st.set_page_config(
    page_title=f"Monitor BDR {VERSAO_APP}",
    page_icon="🦅",
    layout="wide"
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

PERIODO = "1y" 
TERMINACOES_BDR = ('31', '32', '33', '34', '35', '39')

# --- GERENCIAMENTO DE SEGREDOS (SECRETS) ---
try:
    WHATSAPP_PHONE = st.secrets["WHATSAPP_PHONE"]
    WHATSAPP_APIKEY = st.secrets["WHATSAPP_APIKEY"]
    BRAPI_API_TOKEN = st.secrets["BRAPI_API_TOKEN"]
except Exception:
    st.error("❌ ERRO CRÍTICO: Chaves de API não encontradas!")
    st.info("Configure os 'Secrets' no painel do Streamlit.")
    st.stop()

# --- INICIALIZAÇÃO DA SESSÃO (MEMÓRIA) ---
if 'dados_carregados' not in st.session_state:
    st.session_state.dados_carregados = False
if 'df_resultado' not in st.session_state:
    st.session_state.df_resultado = pd.DataFrame()
if 'df_calculado' not in st.session_state:
    st.session_state.df_calculado = pd.DataFrame()

# --- FUNÇÕES AUXILIARES ---

def obter_hora_brasil():
    fuso = pytz.timezone('America/Sao_Paulo')
    return datetime.now(fuso).strftime('%d/%m/%Y %H:%M:%S')

def enviar_whatsapp(mensagem):
    try:
        texto_encoded = urllib.parse.quote(mensagem)
        url = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={texto_encoded}&apikey={WHATSAPP_APIKEY}"
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            return True, "Enviado com sucesso!"
        else:
            return False, f"Erro API: {r.status_code}"
    except Exception as e:
        return False, str(e)

# --- FUNÇÕES DE DADOS ---

@st.cache_data(ttl=3600)
def obter_dados_brapi():
    try:
        url = f"https://brapi.dev/api/quote/list?token={BRAPI_API_TOKEN}"
        r = requests.get(url, timeout=30)
        dados = r.json().get('stocks', [])
        bdrs_raw = [d for d in dados if d['stock'].endswith(TERMINACOES_BDR)]
        lista_tickers = [d['stock'] for d in bdrs_raw]
        mapa_nomes = {d['stock']: d.get('name', d['stock']) for d in bdrs_raw}
        return lista_tickers, mapa_nomes
    except Exception as e:
        st.error(f"Erro Brapi: {e}")
        return [], {}

@st.cache_data(ttl=1800)
def buscar_dados(tickers):
    if not tickers: return pd.DataFrame()
    sa_tickers = [f"{t}.SA" for t in tickers]
    try:
        df = yf.download(sa_tickers, period=PERIODO, auto_adjust=True, progress=False, timeout=60)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_tuples([(c[0], c[1].replace(".SA", "")) for c in df.columns])
        return df.dropna(axis=1, how='all')
    except Exception: return pd.DataFrame()

def calcular_indicadores(df):
    df_calc = df.copy()
    tickers = df_calc.columns.get_level_values(1).unique()
    
    progresso = st.progress(0)
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        progresso.progress((i + 1) / total)
        try:
            close = df_calc[('Close', ticker)]
            high = df_calc[('High', ticker)]
            low = df_calc[('Low', ticker)]
            
            # RSI 14
            delta = close.diff()
            ganho = delta.clip(lower=0).rolling(14).mean()
            perda = -delta.clip(upper=0).rolling(14).mean()
            rs = ganho / perda
            df_calc[('RSI14', ticker)] = 100 - (100 / (1 + rs))

            # ESTOCÁSTICO 14 (%K)
            lowest_low = low.rolling(window=14).min()
            highest_high = high.rolling(window=14).max()
            stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            df_calc[('Stoch_K', ticker)] = stoch_k

            # MÉDIAS
            df_calc[('EMA20', ticker)] = close.ewm(span=20).mean()
            df_calc[('SMA200', ticker)] = close.rolling(window=200).mean()

            # Bollinger
            sma20 = close.rolling(20).mean()
            std = close.rolling(20).std()
            df_calc[('BB_Lower', ticker)] = sma20 - (std * 2)
            df_calc[('BB_Upper', ticker)] = sma20 + (std * 2)
        except: continue
            
    progresso.empty()
    return df_calc

def calcular_fibonacci(df_ticker):
    try:
        if len(df_ticker) < 50: return None
        high = df_ticker['High'].max()
        low = df_ticker['Low'].min()
        diff = high - low
        return {'61.8%': low + (diff * 0.618)} 
    except: return None

def gerar_sinal(row_ticker, df_ticker):
    sinais = []
    score = 0
    
    def classificar(s):
        if s >= 4: return "💎 Ouro"
        if s >= 2: return "🥈 Prata"
        if s >= 1: return "🥉 Bronze"
        return "⚪ Neutro"

    try:
        close = row_ticker.get('Close')
        sma200 = row_ticker.get('SMA200')
        rsi = row_ticker.get('RSI14')
        stoch = row_ticker.get('Stoch_K')
        bb_lower = row_ticker.get('BB_Lower')
        
        tendencia_alta = False
        if pd.notna(sma200) and pd.notna(close):
            if close > sma200:
                tendencia_alta = True
                sinais.append("Trend Alta")
                score += 3
            else:
                sinais.append("Trend Baixa")

        if pd.notna(rsi):
            if rsi < 30:
                sinais.append("RSI Baixo")
                score += 3
            elif rsi < 40:
                score += 1
        
        if pd.notna(stoch) and stoch < 20:
            sinais.append("Stoch Fundo")
            score += 2
            
        if pd.notna(close) and pd.notna(bb_lower):
            if close < bb_lower * 1.02:
                sinais.append("BB Suporte")
                score += 1

        fibo = calcular_fibonacci(df_ticker)
        if fibo and (fibo['61.8%'] * 0.99 <= close <= fibo['61.8%'] * 1.01):
            sinais.append("Fibo 61.8")
            score += 2

        return sinais, score, classificar(score), tendencia_alta
    except:
        return [], 0, "Indefinida", False

def analisar_oportunidades(df_calc, mapa_nomes):
    resultados = []
    tickers = df_calc.columns.get_level_values(1).unique()

    for ticker in tickers:
        try:
            df_ticker = df_calc.xs(ticker, axis=1, level=1).dropna()
            if len(df_ticker) < 200: continue

            last = df_ticker.iloc[-1]
            anterior = df_ticker.iloc[-2]
            
            preco = last.get('Close')
            preco_ant = anterior.get('Close')
            volume = last.get('Volume')
            sma200 = last.get('SMA200')
            
            if pd.isna(preco) or pd.isna(preco_ant): continue

            queda_dia = ((preco - preco_ant) / preco_ant) * 100
            if queda_dia >= 0: continue 

            sinais, score, classificacao, tendencia_alta = gerar_sinal(last, df_ticker)

            rsi = last.get('RSI14', 50)
            stoch = last.get('Stoch_K', 50)
            is_index = ((100 - rsi) + (100 - stoch)) / 2
            
            dist_sma200 = ((preco - sma200) / sma200) * 100 if pd.notna(sma200) else 0

            nome_completo = mapa_nomes.get(ticker, ticker)
            palavras = nome_completo.split()
            ignore_list = ['INC', 'CORP', 'LTD', 'S.A.', 'GMBH', 'PLC', 'GROUP', 'HOLDINGS']
            palavras_uteis = [p for p in palavras if p.upper().replace('.', '') not in ignore_list]
            nome_curto = " ".join(palavras_uteis[:2]) if len(palavras_uteis) > 0 else ticker
            nome_curto = nome_curto.replace(',', '').title()

            status_visual = "⭐ STRATEGY" if tendencia_alta else "⚠️ Contra-Tend."

            resultados.append({
                'Ticker': ticker,
                'Empresa': nome_curto,
                'Preco': preco,
                'Volume': volume,
                'Queda_Dia': queda_dia,
                'IS': is_index,
                'Dist_SMA200': dist_sma200,
                'RSI14': rsi,
                'Setup': status_visual,
                'Tendencia_Alta': tendencia_alta,
                'Potencial': classificacao,
                'Score': score,
                'Sinais': ", ".join(sinais)
            })
        except: continue
    return resultados

def plotar_grafico(df_ticker, ticker, empresa, is_val, tendencia_alta):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    close = df_ticker['Close']
    sma200 = df_ticker['SMA200']
    
    ax1 = axes[0]
    ax1.plot(close.index, close.values, label='Preço', color='#333333', linewidth=1.5)
    cor_sma = '#FFD700' if tendencia_alta else '#FF5252'
    ax1.plot(close.index, sma200, label='SMA 200', color=cor_sma, linewidth=2.5)
    ax1.plot(close.index, df_ticker['EMA20'], label='EMA 20', color='blue', alpha=0.5)
    ax1.fill_between(close.index, df_ticker['BB_Lower'], df_ticker['BB_Upper'], alpha=0.1, color='gray')
    
    titulo_status = "✅ TENDÊNCIA ALTA" if tendencia_alta else "❌ TENDÊNCIA BAIXA"
    cor_titulo = "green" if tendencia_alta else "red"
    ax1.set_title(f'{ticker} - {empresa} | {titulo_status} | I.S.: {is_val:.0f}', fontweight='bold', color=cor_titulo)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(close.index, df_ticker['RSI14'], color='orange')
    ax2.axhline(30, color='red', linestyle='--')
    ax2.axhline(70, color='green', linestyle='--')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    if 'Stoch_K' in df_ticker.columns:
        ax3.plot(close.index, df_ticker['Stoch_K'], color='purple')
        ax3.axhline(20, color='red', linestyle='--')
        ax3.axhline(80, color='green', linestyle='--')
    ax3.set_ylabel('Stoch')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Funções de Estilo
def estilizar_is(val):
    if val >= 75: return 'background-color: #d32f2f; color: white; font-weight: bold'
    elif val >= 60: return 'background-color: #ffa726; color: black'
    return 'color: #888888'

def estilizar_setup(val):
    if "STRATEGY" in val:
        return 'background-color: #1b5e20; color: white; font-weight: bold; border-radius: 5px'
    return 'color: #757575'

def formatar_msg_whatsapp(df_res, hora):
    top = df_res.head(10)
    msg = f"🦅 *BDR ALERT {VERSAO_APP}*\n"
    msg += f"🗓️ {hora}\n"
    
    qtd_strategy = df_res[df_res['Tendencia_Alta'] == True].shape[0]
    msg += f"🚨 *{len(df_res)}* Quedas | ⭐ *{qtd_strategy}* Estratégia\n\n"
    msg += "*🏆 TOP 10 MAIORES QUEDAS:*\n"
    
    for _, row in top.iterrows():
        icon = "⭐" if row['Tendencia_Alta'] else "🔻"
        msg += f"{icon} *{row['Ticker']}* ({row['Queda_Dia']:.1f}%)\n"
        msg += f"   💵 R$ {row['Preco']:.2f} | 📊 I.S. {row['IS']:.0f}\n"
        msg += f"   📋 {row['Sinais']}\n" 
        msg += "   - - - - - - - -\n"
        
    msg += "\n🔗 _Ver detalhes no App_"
    return msg

# --- LAYOUT BARRA LATERAL (Teste) ---

with st.sidebar:
    st.title("Configurações")
    st.info(f"App Versão: {VERSAO_APP}")
    st.write(f"⏰ Hora Brasil:\n**{obter_hora_brasil()}**")
    st.divider()
    st.write("🔧 **Teste de Notificação**")
    
    # BOTÃO 1: Teste Rápido (Sidebar)
    if st.button("🔔 Testar Conexão WhatsApp"):
        with st.spinner("Enviando teste..."):
            sucesso, msg = enviar_whatsapp(f"🤖 *Teste do Bot BDR*\nOlá! A conexão está funcionando perfeitamente.\nHora: {obter_hora_brasil()}")
            if sucesso:
                st.success("Teste enviado!")
            else:
                st.error(f"Falha: {msg}")

# --- LAYOUT PRINCIPAL ---

st.title(f"🦅 Monitor BDR - Swing Trade")
st.caption(f"Versão: {VERSAO_APP} | Hora Execução: {obter_hora_brasil()}")

# BOTÃO 2: Rastrear Mercado
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    btn_analisar = st.button("🔄 Rastrear Mercado", type="primary")

# Lógica de Análise (Popula a Sessão)
if btn_analisar:
    with st.spinner("Analisando mercado..."):
        lista_bdrs, mapa_nomes = obter_dados_brapi()
        df = buscar_dados(lista_bdrs)
        
    if not df.empty:
        df_calc = calcular_indicadores(df)
        oportunidades = analisar_oportunidades(df_calc, mapa_nomes)
        
        if oportunidades:
            df_res = pd.DataFrame(oportunidades)
            # Ordenação
            df_res = df_res.sort_values(by=['Tendencia_Alta', 'Queda_Dia'], ascending=[False, True])
            
            # SALVA NA MEMÓRIA (SESSION STATE)
            st.session_state.dados_carregados = True
            st.session_state.df_resultado = df_res
            st.session_state.df_calculado = df_calc
            st.toast("Análise concluída!", icon="✅")
        else:
            st.warning("Nenhuma oportunidade encontrada.")
            st.session_state.dados_carregados = False

# SE TIVER DADOS NA MEMÓRIA, MOSTRA O CONTEÚDO
if st.session_state.dados_carregados:
    df_res = st.session_state.df_resultado
    df_calc = st.session_state.df_calculado
    
    # --- ÁREA DE ENVIO DE RELATÓRIO ---
    st.markdown("### 🚀 Ações")
    col_envio, col_vazia = st.columns([2, 5])
    
    # BOTÃO 3: Enviar Relatório Real (Só aparece se tiver dados)
    with col_envio:
        if st.button("📱 Enviar Relatório para WhatsApp"):
            hora_atual = obter_hora_brasil()
            msg_zap = formatar_msg_whatsapp(df_res, hora_atual)
            with st.spinner("Enviando relatório..."):
                sucesso, retorno = enviar_whatsapp(msg_zap)
                if sucesso:
                    st.success("Relatório enviado com sucesso!")
                else:
                    st.error(f"Erro no envio: {retorno}")
    
    # --- TABELA E DADOS ---
    qtd_strategy = df_res[df_res['Tendencia_Alta'] == True].shape[0]
    st.divider()
    st.success(f"{len(df_res)} quedas encontradas. {qtd_strategy} na Estratégia Principal!")
    
    st.dataframe(
        df_res.style.map(estilizar_setup, subset=['Setup'])
                    .map(estilizar_is, subset=['IS'])
        .format({
            'Preco': 'R$ {:.2f}',
            'Volume': '{:,.0f}',
            'Queda_Dia': '{:.2f}%',
            'Dist_SMA200': '{:.2f}%',
            'IS': '{:.0f}'
        }),
        column_order=("Ticker", "Empresa", "Setup", "Preco", "Queda_Dia", "IS", "Dist_SMA200", "Volume", "Sinais"),
        column_config={
            "Setup": st.column_config.Column("Estratégia", width="medium"),
            "Dist_SMA200": st.column_config.NumberColumn("Dist. SMA200"),
            "IS": st.column_config.NumberColumn("I.S."),
            "Sinais": st.column_config.TextColumn("Motivos", width="large")
        },
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    st.subheader("🎯 Destaques da Estratégia (Top 5)")
    top_graficos = df_res.head(5)
    
    for _, row in top_graficos.iterrows():
        try:
            df_ticker = df_calc.xs(row['Ticker'], axis=1, level=1).dropna()
            icon = "⭐" if row['Tendencia_Alta'] else "⚠️"
            st.markdown(f"### {icon} {row['Ticker']} - {row['Empresa']}")
            fig = plotar_grafico(df_ticker, row['Ticker'], row['Empresa'], row['IS'], row['Tendencia_Alta'])
            st.pyplot(fig)
            col1, col2, col3 = st.columns(3)
            col1.metric("Queda", f"{row['Queda_Dia']:.2f}%")
            col2.metric("Distância SMA200", f"{row['Dist_SMA200']:.2f}%")
            col3.metric("Sobrevenda (I.S.)", f"{row['IS']:.0f}/100")
            st.divider()
        except: continue
