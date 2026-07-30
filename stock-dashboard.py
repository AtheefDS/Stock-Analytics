import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StockAI — Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  (dark finance terminal theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root palette ───────────────────────────── */
:root {
  --bg-base:      #0a0e1a;
  --bg-surface:   #111827;
  --bg-card:      #151d2e;
  --bg-elevated:  #1a2438;
  --border:       #1e2d45;
  --border-light: #243551;

  --teal:         #00d4aa;
  --teal-dim:     #00896e;
  --teal-glow:    rgba(0,212,170,0.12);

  --coral:        #ff5f6d;
  --coral-dim:    #c23f4b;
  --coral-glow:   rgba(255,95,109,0.12);

  --gold:         #f59e0b;
  --gold-dim:     #b87411;

  --blue:         #3b82f6;
  --blue-dim:     #1d4ed8;
  --purple:       #8b5cf6;

  --text-primary:   #e8edf5;
  --text-secondary: #8899b4;
  --text-muted:     #4a5f7a;

  --mono: 'JetBrains Mono', monospace;
  --sans: 'Inter', sans-serif;
}

/* ── Base ───────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: var(--sans) !important;
}

[data-testid="stAppViewContainer"] > .main {
  background: var(--bg-base) !important;
}

/* ── Sidebar ────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
  color: var(--text-primary) !important;
  font-family: var(--sans) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-light) !important;
  color: var(--text-primary) !important;
  border-radius: 8px !important;
}

/* ── Metrics ────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 1.1rem 1.3rem !important;
}
[data-testid="stMetric"] label {
  color: var(--text-secondary) !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  font-family: var(--sans) !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-family: var(--mono) !important;
  font-size: 1.45rem !important;
  font-weight: 600 !important;
  color: var(--text-primary) !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  font-family: var(--mono) !important;
  font-size: 0.8rem !important;
}
[data-testid="stMetricDeltaIcon-Up"]   { color: var(--teal)  !important; }
[data-testid="stMetricDeltaIcon-Down"] { color: var(--coral) !important; }

/* ── Buttons ────────────────────────────────── */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--border-light) !important;
  color: var(--text-primary) !important;
  border-radius: 8px !important;
  font-family: var(--sans) !important;
  font-weight: 500 !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  border-color: var(--teal) !important;
  color: var(--teal) !important;
  background: var(--teal-glow) !important;
}
.stButton > button[kind="primary"] {
  background: var(--teal) !important;
  border-color: var(--teal) !important;
  color: #0a0e1a !important;
  font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--teal-dim) !important;
  color: #fff !important;
}

/* ── Tabs ───────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  border: none !important;
  font-family: var(--sans) !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  padding: 0.6rem 1.2rem !important;
  border-bottom: 2px solid transparent !important;
  transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
  color: var(--teal) !important;
  border-bottom: 2px solid var(--teal) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: transparent !important;
  padding-top: 1.5rem !important;
}

/* ── Dataframes ─────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
  background: var(--bg-elevated) !important;
  color: var(--text-secondary) !important;
  font-family: var(--sans) !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] td {
  font-family: var(--mono) !important;
  font-size: 0.82rem !important;
  color: var(--text-primary) !important;
  border-bottom: 1px solid var(--border) !important;
}

/* ── Alert boxes ────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: 10px !important;
  border-left-width: 3px !important;
  font-family: var(--sans) !important;
  font-size: 0.87rem !important;
}

/* ── Info / Success / Warning / Error ────────── */
.stInfo    { background: rgba(59,130,246,0.08) !important; border-left-color: var(--blue)  !important; color: var(--text-primary) !important; }
.stSuccess { background: var(--teal-glow)      !important; border-left-color: var(--teal)  !important; color: var(--text-primary) !important; }
.stWarning { background: rgba(245,158,11,0.08) !important; border-left-color: var(--gold)  !important; color: var(--text-primary) !important; }
.stError   { background: var(--coral-glow)     !important; border-left-color: var(--coral) !important; color: var(--text-primary) !important; }

/* ── Spinner ─────────────────────────────────── */
[data-testid="stSpinner"] { color: var(--teal) !important; }

/* ── Dividers / horizontal rules ─────────────── */
hr { border-color: var(--border) !important; }

/* ── Download button ─────────────────────────── */
[data-testid="stDownloadButton"] > button {
  background: transparent !important;
  border: 1px solid var(--border-light) !important;
  color: var(--text-secondary) !important;
  border-radius: 8px !important;
  font-family: var(--sans) !important;
  font-size: 0.82rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
  border-color: var(--teal) !important;
  color: var(--teal) !important;
}

/* ── Checkbox ────────────────────────────────── */
.stCheckbox label {
  font-size: 0.85rem !important;
  color: var(--text-secondary) !important;
}

/* ── Section headers ──────────────────────────── */
h1, h2, h3 {
  font-family: var(--sans) !important;
  color: var(--text-primary) !important;
  letter-spacing: -0.02em !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CUSTOM COMPONENT HELPERS
# ─────────────────────────────────────────────
def ticker_banner(symbol: str, price: float, change: float, change_pct: float, company_name: str = ""):
    """
    Streamlit's markdown sanitizer strips flex/inline-flex and many layout CSS props.
    Fix: use st.components.v1.html() which renders in a full iframe — no sanitization.
    """
    import streamlit.components.v1 as components

    up    = change >= 0
    arrow = "▲" if up else "▼"
    color = "#22c55e" if up else "#ef4444"
    bg    = "rgba(0,212,170,0.08)" if up else "rgba(255,95,109,0.08)"
    sign  = "+" if up else ""

    html = f"""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
    <div style="
        background:#111827;
        border:1px solid #1e2d45;
        border-top:3px solid {color};
        border-radius:14px;
        padding:1.2rem 1.8rem;
        display:flex;
        align-items:center;
        justify-content:space-between;
        font-family:'Inter',sans-serif;
        box-sizing:border-box;
        width:100%;
    ">
        <div>
            <div style="font-size:11px;font-weight:600;letter-spacing:0.1em;color:#8899b4;text-transform:uppercase;margin-bottom:4px;">
                {company_name if company_name else "Stock"}
            </div>
            <div style="font-size:2rem;font-weight:700;color:#e8edf5;font-family:'JetBrains Mono',monospace;">
                {symbol}
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:2.2rem;font-weight:600;color:#e8edf5;font-family:'JetBrains Mono',monospace;">
                ${price:,.2f}
            </div>
            <div style="
                display:inline-block;
                background:{bg};
                border:1px solid {color};
                border-radius:8px;
                padding:4px 14px;
                margin-top:6px;
            ">
                <span style="color:{color};font-size:13px;font-weight:600;font-family:'JetBrains Mono',monospace;">
                    {arrow} {sign}{change:.2f} ({sign}{change_pct:.2f}%)
                </span>
            </div>
        </div>
    </div>
    """
    components.html(html, height=110, scrolling=False)


def section_header(title: str, subtitle: str = ""):
    sub_html = f"<div style='font-size:0.78rem;color:#8899b4;margin-top:3px;padding-left:14px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div style="margin:1.8rem 0 1rem 0;font-family:'Inter',sans-serif;">
        <div style="border-left:3px solid #00d4aa;padding-left:10px;">
            <span style="font-size:1rem;font-weight:600;color:#e8edf5;">{title}</span>
        </div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def info_badge(label: str, value: str, color: str = "#00d4aa"):
    st.markdown(f"""
    <div style="
        background:#151d2e; border:1px solid #1e2d45; border-radius:10px;
        padding:0.85rem 1.1rem; font-family:'Inter',sans-serif;
    ">
        <div style="font-size:0.68rem; font-weight:600; letter-spacing:0.09em; color:#8899b4;
                    text-transform:uppercase; margin-bottom:5px;">{label}</div>
        <div style="font-size:1rem; font-weight:600; color:{color};
                    font-family:'JetBrains Mono',monospace;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def kv_row(key: str, value: str):
    # Use table layout — flex gets stripped by Streamlit sanitizer
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;border-bottom:1px solid #1e2d45;font-family:'Inter',sans-serif;font-size:0.85rem;">
      <tr>
        <td style="padding:9px 0;color:#8899b4;font-weight:400;width:50%;">{key}</td>
        <td style="padding:9px 0;color:#e8edf5;font-weight:500;text-align:right;font-family:'JetBrains Mono',monospace;">{value}</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CHART THEME
# ─────────────────────────────────────────────
CHART_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1520",
    font=dict(family="JetBrains Mono, monospace", color="#8899b4", size=11),
    xaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45", linecolor="#1e2d45"),
    yaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45", linecolor="#1e2d45"),
    legend=dict(
        bgcolor="rgba(17,24,39,0.8)",
        bordercolor="#1e2d45",
        borderwidth=1,
        font=dict(size=10),
    ),
    margin=dict(l=10, r=10, t=36, b=10),
    hoverlabel=dict(
        bgcolor="#151d2e",
        bordercolor="#1e2d45",
        font=dict(family="JetBrains Mono", size=12, color="#e8edf5"),
    ),
)

TEAL   = "#00d4aa"
CORAL  = "#ff5f6d"
GOLD   = "#f59e0b"
BLUE   = "#3b82f6"
PURPLE = "#8b5cf6"


# ─────────────────────────────────────────────
#  CORE ANALYZER
# ─────────────────────────────────────────────
class StockAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = RandomForestRegressor(n_estimators=100, random_state=42)

    def fetch_stock_data(self, symbol, period="1y"):
        try:
            stock = yf.Ticker(symbol)
            data  = stock.history(period=period)
            info  = stock.info
            return data, info
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None, None

    def calculate_technical_indicators(self, data):
        df = data.copy()
        df['SMA_20']  = df['Close'].rolling(20).mean()
        df['SMA_50']  = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['EMA_12']  = df['Close'].ewm(span=12).mean()
        df['EMA_26']  = df['Close'].ewm(span=26).mean()
        df['MACD']          = df['EMA_12'] - df['EMA_26']
        df['MACD_signal']   = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram']= df['MACD'] - df['MACD_signal']
        delta = df['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))
        df['BB_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + bb_std * 2
        df['BB_lower'] = df['BB_middle'] - bb_std * 2
        df['Volume_SMA']   = df['Volume'].rolling(20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        df['High_Low']   = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Low_Close']  = np.abs(df['Low']  - df['Close'].shift())
        df['True_Range'] = df[['High_Low','High_Close','Low_Close']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(14).mean()
        low_14  = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        return df

    def prepare_ml_features(self, data):
        df = data.copy()
        df['Returns']    = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_10d']= df['Close'].pct_change(10)
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_lag_{lag}']   = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}']  = df['Volume'].shift(lag)
            df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
        for window in [5, 10, 20, 50]:
            df[f'Close_mean_{window}']  = df['Close'].rolling(window).mean()
            df[f'Close_std_{window}']   = df['Close'].rolling(window).std()
            df[f'Volume_mean_{window}'] = df['Volume'].rolling(window).mean()
            df[f'High_mean_{window}']   = df['High'].rolling(window).mean()
            df[f'Low_mean_{window}']    = df['Low'].rolling(window).mean()
        df['Price_vs_SMA20']       = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
        df['Price_vs_SMA50']       = (df['Close'] - df['SMA_50']) / df['SMA_50'] * 100
        df['Price_volatility_10d'] = df['Returns'].rolling(10).std()
        df['Price_volatility_20d'] = df['Returns'].rolling(20).std()
        return df

    def train_prediction_model(self, data):
        df = self.prepare_ml_features(data)
        df = df.dropna()
        if len(df) < 100:
            return None
        exclude_cols = ['Open','High','Low','Close','Volume','Dividends','Stock Splits',
                        'Returns','Returns_5d','Returns_10d']
        feature_cols = [c for c in df.columns if not any(e in c for e in exclude_cols)]
        feature_cols = [c for c in feature_cols if
                        'lag' in c or 'mean' in c or 'std' in c or
                        c in ['RSI','MACD','Price_vs_SMA20','Price_vs_SMA50',
                              'Price_volatility_10d','Price_volatility_20d','ATR']]
        if len(feature_cols) < 5:
            return None
        X = df[feature_cols].ffill().bfill()
        y = df['Close'].shift(-1)
        X, y = X[:-1], y[:-1]
        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[mask], y[mask]
        if len(X) < 50:
            return None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        Xs_tr = self.scaler.fit_transform(X_tr)
        Xs_te = self.scaler.transform(X_te)
        self.model.fit(Xs_tr, y_tr)
        return {
            'train_score': self.model.score(Xs_tr, y_tr),
            'test_score':  self.model.score(Xs_te, y_te),
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_)),
            'last_features': X.iloc[-1:],
            'feature_cols':  feature_cols,
        }

    def predict_next_price(self, model_info):
        if model_info is None:
            return None
        scaled = self.scaler.transform(model_info['last_features'])
        return self.model.predict(scaled)[0]

    def generate_market_analysis(self, data, info, symbol):
        latest = data.iloc[-1]
        prev   = data.iloc[-2]
        price_change     = latest['Close'] - prev['Close']
        price_change_pct = price_change / prev['Close'] * 100
        rsi       = latest.get('RSI', 50)
        sma_20    = latest.get('SMA_20', latest['Close'])
        sma_50    = latest.get('SMA_50', latest['Close'])
        bb_upper  = latest.get('BB_upper', latest['Close'])
        bb_lower  = latest.get('BB_lower', latest['Close'])
        avg_vol   = data['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = latest['Volume'] / avg_vol if avg_vol > 0 else 1
        macd      = latest.get('MACD', 0)
        macd_sig  = latest.get('MACD_signal', 0)
        analysis  = []

        if price_change_pct > 3:
            analysis.append(("bullish", f"{symbol} shows exceptional bullish momentum with a {price_change_pct:.2f}% surge"))
        elif price_change_pct > 1:
            analysis.append(("bullish", f"{symbol} demonstrates strong upward movement (+{price_change_pct:.2f}%)"))
        elif price_change_pct > 0:
            analysis.append(("neutral", f"{symbol} shows modest gains (+{price_change_pct:.2f}%)"))
        elif price_change_pct > -1:
            analysis.append(("neutral", f"{symbol} experiences slight decline ({price_change_pct:.2f}%)"))
        elif price_change_pct > -3:
            analysis.append(("bearish", f"{symbol} shows moderate bearish pressure ({price_change_pct:.2f}%)"))
        else:
            analysis.append(("bearish", f"{symbol} faces significant selling pressure ({price_change_pct:.2f}%)"))

        if rsi > 80:
            analysis.append(("bearish", f"RSI at {rsi:.1f} — severely overbought, potential reversal ahead"))
        elif rsi > 70:
            analysis.append(("neutral", f"RSI at {rsi:.1f} — overbought territory, exercise caution"))
        elif rsi < 20:
            analysis.append(("bullish", f"RSI at {rsi:.1f} — severely oversold, strong buying signal"))
        elif rsi < 30:
            analysis.append(("bullish", f"RSI at {rsi:.1f} — oversold conditions, potential bounce"))
        else:
            analysis.append(("neutral", f"RSI at {rsi:.1f} — balanced momentum"))

        if latest['Close'] > sma_20 > sma_50:
            analysis.append(("bullish", "Strong bullish alignment — price above both 20-day and 50-day MAs"))
        elif latest['Close'] < sma_20 < sma_50:
            analysis.append(("bearish", "Bearish trend confirmed — price below key moving averages"))
        elif latest['Close'] > sma_20 and sma_20 < sma_50:
            analysis.append(("neutral", "Mixed signals — short-term bullish, longer-term bearish"))
        else:
            analysis.append(("neutral", "Consolidation phase — awaiting directional breakout"))

        if latest['Close'] > bb_upper:
            analysis.append(("neutral", "Price above upper Bollinger Band — potential overbought extension"))
        elif latest['Close'] < bb_lower:
            analysis.append(("bullish", "Price near lower Bollinger Band — potential mean-reversion bounce"))

        if macd > macd_sig and macd > 0:
            analysis.append(("bullish", "MACD above signal line in positive territory — strong bullish momentum"))
        elif macd < macd_sig and macd < 0:
            analysis.append(("bearish", "MACD below signal in negative territory — confirmed bearish momentum"))
        elif macd > macd_sig:
            analysis.append(("bullish", "MACD bullish crossover — momentum shifting upward"))
        else:
            analysis.append(("neutral", "MACD bearish crossover — momentum weakening"))

        if vol_ratio > 2:
            analysis.append(("bullish", f"Volume {vol_ratio:.1f}x above average — exceptional conviction behind move"))
        elif vol_ratio > 1.5:
            analysis.append(("neutral", f"Volume {vol_ratio:.1f}x above average — validates price movement"))
        elif vol_ratio < 0.5:
            analysis.append(("neutral", "Below-average volume — weak conviction, treat move with caution"))

        return analysis


# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────
def create_main_chart(data, symbol):
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.52, 0.14, 0.19, 0.15],
        subplot_titles=("", "", "", ""),
    )

    # ── Candlesticks
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'], high=data['High'],
        low=data['Low'],   close=data['Close'],
        name="Price",
        increasing=dict(line=dict(color=TEAL,  width=1), fillcolor=TEAL),
        decreasing=dict(line=dict(color=CORAL, width=1), fillcolor=CORAL),
    ), row=1, col=1)

    # ── Moving averages
    ma_cfg = [('SMA_20','SMA 20',GOLD,1.5),('SMA_50','SMA 50',BLUE,1.5),('SMA_200','SMA 200',PURPLE,1.5)]
    for col, name, color, w in ma_cfg:
        if col in data.columns and not data[col].isna().all():
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col],
                line=dict(color=color, width=w), name=name, opacity=0.85,
            ), row=1, col=1)

    # ── Bollinger Bands
    if 'BB_upper' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_upper'],
            line=dict(color='rgba(139,92,246,0.4)', width=1, dash='dot'),
            name='BB Upper', showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_lower'],
            line=dict(color='rgba(139,92,246,0.4)', width=1, dash='dot'),
            fill='tonexty', fillcolor='rgba(139,92,246,0.05)',
            name='BB Lower', showlegend=False,
        ), row=1, col=1)

    # ── Volume
    v_colors = [TEAL if data['Close'].iloc[i] >= data['Open'].iloc[i] else CORAL
                for i in range(len(data))]
    fig.add_trace(go.Bar(
        x=data.index, y=data['Volume'],
        marker_color=v_colors, name='Volume', opacity=0.5, showlegend=False,
    ), row=2, col=1)
    if 'Volume_SMA' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Volume_SMA'],
            line=dict(color=GOLD, width=1.2), name='Vol SMA', showlegend=False,
        ), row=2, col=1)

    # ── MACD
    if 'MACD' in data.columns:
        hist_colors = [TEAL if v >= 0 else CORAL for v in data['MACD_histogram'].fillna(0)]
        fig.add_trace(go.Bar(
            x=data.index, y=data['MACD_histogram'],
            marker_color=hist_colors, name='Histogram', opacity=0.6, showlegend=False,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'],
            line=dict(color=BLUE, width=1.8), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'],
            line=dict(color=GOLD, width=1.8), name='Signal'), row=3, col=1)

    # ── RSI
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            line=dict(color=PURPLE, width=2), name='RSI',
        ), row=4, col=1)
        for lvl, color in [(70,"rgba(255,95,109,0.5)"), (30,"rgba(0,212,170,0.5)"), (50,"rgba(255,255,255,0.15)")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=color, line_width=1, row=4, col=1)

    fig.update_layout(
        **CHART_THEME,
        height=860,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        title=dict(
            text=f"<b>{symbol}</b> — Technical Analysis",
            font=dict(family="Inter, sans-serif", size=14, color="#e8edf5"),
            x=0.02, xanchor="left",
        ),
    )
    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    # Row labels
    annotations = [
        dict(x=0, y=1,  xref="paper", yref="paper", text="<b>PRICE</b>",  showarrow=False,
             font=dict(size=9, color="#4a5f7a"), xanchor="left"),
        dict(x=0, y=0.46, xref="paper", yref="paper", text="<b>VOLUME</b>", showarrow=False,
             font=dict(size=9, color="#4a5f7a"), xanchor="left"),
        dict(x=0, y=0.315, xref="paper", yref="paper", text="<b>MACD</b>",  showarrow=False,
             font=dict(size=9, color="#4a5f7a"), xanchor="left"),
        dict(x=0, y=0.12,  xref="paper", yref="paper", text="<b>RSI</b>",   showarrow=False,
             font=dict(size=9, color="#4a5f7a"), xanchor="left"),
    ]
    fig.update_layout(annotations=annotations)
    return fig


def create_returns_chart(data, symbol):
    data = data.copy()
    data['Daily_Returns']      = data['Close'].pct_change()
    data['Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod() - 1
    final = data['Cumulative_Returns'].iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Cumulative_Returns'] * 100,
        mode='lines',
        line=dict(color=TEAL if final >= 0 else CORAL, width=2),
        fill='tozeroy',
        fillcolor=f"{'rgba(0,212,170,0.07)' if final >= 0 else 'rgba(255,95,109,0.07)'}",
        name='Cumulative Return',
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.update_layout(
        **CHART_THEME,
        height=320,
        title=dict(text=f"<b>Cumulative Return</b>",
                   font=dict(family="Inter", size=13, color="#e8edf5"), x=0.02, xanchor="left"),
        yaxis_title=None, xaxis_title=None,
        showlegend=False,
    )
    return fig, data


def create_feature_importance_chart(model_info):
    imp = pd.DataFrame(
        list(model_info['feature_importance'].items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True).tail(12)

    fig = go.Figure(go.Bar(
        x=imp['Importance'],
        y=imp['Feature'],
        orientation='h',
        marker=dict(
            color=imp['Importance'],
            colorscale=[[0, "#1a2438"], [0.5, "#1d4ed8"], [1, TEAL]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_THEME,
        height=380,
        title=dict(text="<b>Feature Importance</b>",
                   font=dict(family="Inter", size=13, color="#e8edf5"), x=0.02, xanchor="left"),
        xaxis_title=None, yaxis_title=None,
        yaxis=dict(tickfont=dict(size=10)),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("""
    <div style="padding:1.2rem 0 1rem 0; font-family:'Inter',sans-serif;">
        <div style="font-size:1.1rem; font-weight:700; color:#e8edf5; letter-spacing:-0.02em;">
            StockAI
        </div>
        <div style="font-size:0.72rem; color:#8899b4; margin-top:2px;">
            Market Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:0.72rem; font-weight:600; letter-spacing:0.08em; "
        "color:#4a5f7a; text-transform:uppercase; margin-bottom:8px;'>Instrument</div>",
        unsafe_allow_html=True,
    )

    popular = {
        'Apple (AAPL)': 'AAPL', 'Microsoft (MSFT)': 'MSFT', 'Google (GOOGL)': 'GOOGL',
        'Amazon (AMZN)': 'AMZN', 'Tesla (TSLA)': 'TSLA',   'NVIDIA (NVDA)': 'NVDA',
        'Meta (META)': 'META',   'Netflix (NFLX)': 'NFLX',  'AMD': 'AMD', 'Intel (INTC)': 'INTC',
        'Custom ↗': 'CUSTOM',
    }
    choice = st.sidebar.selectbox("Stock", list(popular.keys()), index=0,
                                  label_visibility="collapsed")
    symbol = (st.sidebar.text_input("Symbol", value="AAPL", max_chars=10,
                                    label_visibility="collapsed").upper()
              if choice == 'Custom ↗' else popular[choice])

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:0.72rem; font-weight:600; letter-spacing:0.08em; "
        "color:#4a5f7a; text-transform:uppercase; margin-bottom:8px;'>Timeframe</div>",
        unsafe_allow_html=True,
    )
    period_map = {'1M':'1mo','3M':'3mo','6M':'6mo','1Y':'1y','2Y':'2y','5Y':'5y'}
    period_label = st.sidebar.radio("Period", list(period_map.keys()), index=3,
                                    horizontal=True, label_visibility="collapsed")
    period = period_map[period_label]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:0.72rem; font-weight:600; letter-spacing:0.08em; "
        "color:#4a5f7a; text-transform:uppercase; margin-bottom:8px;'>Modules</div>",
        unsafe_allow_html=True,
    )
    show_chart  = st.sidebar.checkbox("Technical Charts",   value=True)
    show_perf   = st.sidebar.checkbox("Performance",        value=True)
    show_ml     = st.sidebar.checkbox("ML Prediction",      value=True)
    show_ai     = st.sidebar.checkbox("AI Market Analysis", value=True)

    st.sidebar.markdown("---")
    refresh = st.sidebar.button("Refresh Data", type="primary", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("""
    <div style="font-size:0.7rem; color:#4a5f7a; text-align:center;
                padding:1.5rem 0 0.5rem 0; font-family:'Inter',sans-serif;">
        ⚠️ Not financial advice. Educational only.
    </div>
    """, unsafe_allow_html=True)

    return symbol, period, show_chart, show_perf, show_ml, show_ai


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    symbol, period, show_chart, show_perf, show_ml, show_ai = render_sidebar()

    # Page title (minimal) - table layout avoids flex stripping
    _now = datetime.now().strftime('%d %b %Y  %H:%M')
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;margin-bottom:0.5rem;font-family:'Inter',sans-serif;">
      <tr>
        <td style="font-size:0.75rem;color:#4a5f7a;letter-spacing:0.05em;text-transform:uppercase;">Market Intelligence</td>
        <td style="font-size:0.72rem;color:#4a5f7a;text-align:right;font-family:'JetBrains Mono',monospace;">{_now}</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    analyzer = StockAnalyzer()

    with st.spinner(f"Fetching {symbol}..."):
        data, info = analyzer.fetch_stock_data(symbol, period)

    if data is None or data.empty:
        st.error(f"Could not fetch data for **{symbol}**. Verify the ticker and try again.")
        return

    with st.spinner("Calculating indicators..."):
        data = analyzer.calculate_technical_indicators(data)

    # ── Derived metrics
    latest       = data['Close'].iloc[-1]
    prev         = data['Close'].iloc[-2]
    change       = latest - prev
    change_pct   = change / prev * 100
    company_name = info.get('longName', '')

    # ── Ticker banner
    ticker_banner(symbol, latest, change, change_pct, company_name)

    # ── Top metrics row
    vol       = data['Volume'].iloc[-1]
    avg_vol   = data['Volume'].rolling(20).mean().iloc[-1]
    vol_chg   = (vol - avg_vol) / avg_vol * 100 if avg_vol > 0 else 0
    rsi_val   = data['RSI'].iloc[-1] if 'RSI' in data.columns else None
    rsi_lbl   = ("Overbought" if rsi_val and rsi_val > 70
                 else "Oversold" if rsi_val and rsi_val < 30 else "Neutral")
    sma20     = data['SMA_20'].iloc[-1] if 'SMA_20' in data.columns else None
    sma_dist  = (latest - sma20) / sma20 * 100 if sma20 else None
    mktcap    = info.get('marketCap', 0)
    cap_str   = (f"${mktcap/1e12:.2f}T" if mktcap > 1e12
                 else f"${mktcap/1e9:.1f}B" if mktcap > 1e9
                 else f"${mktcap/1e6:.0f}M" if mktcap else "N/A")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Current Price", f"${latest:,.2f}",
                  f"{change:+.2f} ({change_pct:+.2f}%)")
    with c2:
        st.metric("Volume", f"{vol:,.0f}", f"{vol_chg:+.1f}% vs 20d avg")
    with c3:
        st.metric("RSI (14)", f"{rsi_val:.1f}" if rsi_val else "N/A", rsi_lbl)
    with c4:
        st.metric("vs SMA 20", f"{sma_dist:+.1f}%" if sma_dist else "N/A",
                  "Above" if sma_dist and sma_dist > 0 else "Below")
    with c5:
        st.metric("Market Cap", cap_str)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Technical chart
    if show_chart:
        section_header("Technical Analysis", "Candlestick · Moving Averages · MACD · RSI")
        st.plotly_chart(create_main_chart(data, symbol), use_container_width=True)

    # ── Performance
    if show_perf:
        section_header("Performance Metrics")
        fig_ret, data_with_ret = create_returns_chart(data, symbol)

        dr = data_with_ret['Daily_Returns'].dropna()
        total_ret  = data_with_ret['Cumulative_Returns'].iloc[-1] * 100
        volatility = dr.std() * np.sqrt(252) * 100
        sharpe     = (dr.mean() * 252) / (dr.std() * np.sqrt(252)) if dr.std() != 0 else 0
        max_dd     = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100

        p1, p2, p3, p4 = st.columns(4)
        with p1: st.metric("Total Return",       f"{total_ret:+.1f}%")
        with p2: st.metric("Volatility (Ann.)",  f"{volatility:.1f}%")
        with p3: st.metric("Sharpe Ratio",        f"{sharpe:.2f}")
        with p4: st.metric("Max Drawdown",        f"{max_dd:.1f}%")

        st.plotly_chart(fig_ret, use_container_width=True)

    # ── ML Prediction
    if show_ml:
        section_header("ML Price Prediction", "Random Forest — trained on 50+ technical features")
        mc1, mc2 = st.columns([1, 1])
        with mc1:
            with st.spinner("Training model..."):
                model_info = analyzer.train_prediction_model(data)
            if model_info:
                pred     = analyzer.predict_next_price(model_info)
                pred_chg = (pred - latest) / latest * 100
                conf     = model_info['test_score']
                conf_lbl = "High" if conf > 0.8 else "Medium" if conf > 0.6 else "Low"

                st.success("Model trained successfully")
                pm1, pm2 = st.columns(2)
                with pm1:
                    st.metric("Next Day Prediction", f"${pred:.2f}", f"{pred_chg:+.2f}%")
                with pm2:
                    st.metric("Model Confidence", f"{conf:.1%}", conf_lbl)

                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                st.info(f"Train accuracy: **{model_info['train_score']:.1%}** · Test accuracy: **{model_info['test_score']:.1%}**")
            else:
                st.warning("Insufficient historical data for reliable prediction.")
                model_info = None

        with mc2:
            if show_ml and 'model_info' in dir() and model_info:
                st.plotly_chart(create_feature_importance_chart(model_info),
                                use_container_width=True)

    # ── AI Analysis
    if show_ai:
        section_header("AI Market Analysis", "Rule-based signal interpretation across multiple indicators")
        analysis = analyzer.generate_market_analysis(data, info, symbol)

        SIGNAL_STYLE = {
            "bullish": ("var(--teal-glow, rgba(0,212,170,0.1))", "#00d4aa", "▲ BULLISH"),
            "bearish": ("var(--coral-glow, rgba(255,95,109,0.1))", "#ff5f6d", "▼ BEARISH"),
            "neutral": ("rgba(59,130,246,0.08)", "#3b82f6", "● NEUTRAL"),
        }

        cols = st.columns(2)
        for i, (signal, text) in enumerate(analysis):
            bg, border, badge = SIGNAL_STYLE.get(signal, SIGNAL_STYLE["neutral"])
            with cols[i % 2]:
                st.markdown(f"""
                <div style="
                    background:{bg};
                    border:1px solid {border}30;
                    border-left:3px solid {border};
                    border-radius:10px;
                    padding:0.75rem 1rem;
                    margin-bottom:10px;
                    font-family:'Inter',sans-serif;
                ">
                    <div style="font-size:0.62rem; font-weight:700; letter-spacing:0.1em;
                                color:{border}; margin-bottom:4px;">{badge}</div>
                    <div style="font-size:0.84rem; color:#c8d6e8; line-height:1.45;">{text}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Detail Tabs
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Company Info", "Raw Data", "Technical Indicators"])

    with tab1:
        if info:
            t1, t2 = st.columns(2)
            with t1:
                section_header("Company Details")
                for k, v in {
                    "Name":      info.get('longName', 'N/A'),
                    "Sector":    info.get('sector',   'N/A'),
                    "Industry":  info.get('industry', 'N/A'),
                    "Country":   info.get('country',  'N/A'),
                    "Website":   info.get('website',  'N/A'),
                    "Employees": f"{info['fullTimeEmployees']:,}" if info.get('fullTimeEmployees') else 'N/A',
                }.items():
                    kv_row(k, str(v))
            with t2:
                section_header("Valuation Metrics")
                for k, v in {
                    "P/E Ratio":    f"{info['trailingPE']:.2f}"    if info.get('trailingPE')     else 'N/A',
                    "Forward P/E":  f"{info['forwardPE']:.2f}"     if info.get('forwardPE')      else 'N/A',
                    "PEG Ratio":    f"{info['pegRatio']:.2f}"      if info.get('pegRatio')       else 'N/A',
                    "Price/Book":   f"{info['priceToBook']:.2f}"   if info.get('priceToBook')    else 'N/A',
                    "Div. Yield":   f"{info['dividendYield']*100:.2f}%" if info.get('dividendYield') else 'N/A',
                    "Beta":         f"{info['beta']:.2f}"          if info.get('beta')           else 'N/A',
                    "52W High":     f"${info['fiftyTwoWeekHigh']:.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A',
                    "52W Low":      f"${info['fiftyTwoWeekLow']:.2f}"  if info.get('fiftyTwoWeekLow')  else 'N/A',
                }.items():
                    kv_row(k, str(v))

    with tab2:
        section_header("Recent Price History", "Last 20 sessions")
        disp = data[['Open','High','Low','Close','Volume']].tail(20).copy()
        disp.index = disp.index.strftime('%Y-%m-%d')
        disp = disp.round(2)
        st.dataframe(disp, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=disp.to_csv(),
            file_name=f"{symbol}_data.csv",
            mime="text/csv",
        )

    with tab3:
        section_header("Technical Indicator Readings", "Last 10 sessions")
        tech_cols = ['Close','SMA_20','SMA_50','RSI','MACD','MACD_signal',
                     'BB_upper','BB_lower','ATR']
        avail = [c for c in tech_cols if c in data.columns]
        tech  = data[avail].tail(10).copy()
        tech.index = tech.index.strftime('%Y-%m-%d')
        st.dataframe(tech.round(3), use_container_width=True)

    # ── Footer
    st.markdown("""
    <table style="width:100%;margin-top:3rem;padding:1.5rem 0;border-top:1px solid #1e2d45;font-family:'Inter',sans-serif;font-size:0.72rem;color:#4a5f7a;border-collapse:collapse;">
      <tr>
        <td style="padding:1rem 0;"><b style="color:#8899b4;">StockAI</b> — Professional market intelligence</td>
        <td style="padding:1rem 0;text-align:right;">⚠️ For educational purposes only. Not financial advice.</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()