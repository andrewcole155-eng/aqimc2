import os
import sys

# Dynamically append root '/app' to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import json
import time
import re
import plotly.express as px
import alpaca_trade_api as tradeapi
import gspread
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
import yfinance as yf
import requests
import tzdata
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import warnings

# --- SUPPRESS THIRD-PARTY WARNINGS ---
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow is deprecated.*")

st.set_page_config(
    page_title="AQI Mission Control",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Path Configuration (Docker vs Local Routing) ===
if os.path.exists('/app'):
    ROOT_DIR = '/app'
else:
    ROOT_DIR = '/home/andrew/.ssh/Trading/Alpaca_V2'

CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
OBSERVATIONS_DIR = os.path.join(ROOT_DIR, 'observations')
MODELS_DIR = os.path.join(ROOT_DIR, 'Models')

ALPACA_CONFIG_PATH = os.path.join(CONFIG_DIR, 'config_Alpaca_REAL_V2.json')

# === STYLING ===
st.markdown("""
    <style>
    /* VS Code Terminal Theme */
    .terminal-box {
        background-color: #1e1e1e; /* VS Code Background */
        color: #cccccc;            /* Default Text */
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        padding: 10px;
        border: 1px solid #3c3c3c;
        border-radius: 4px;
        height: 600px;
        overflow-y: auto;
        font-size: 14px;           /* Larger Font */
        line-height: 1.5;
    }
    .log-line {
        display: block;            /* Forces each log to its own line */
        padding: 1px 0;
        border-bottom: 1px solid #2d2d2d; /* Subtle separator line */
    }
    .log-ts { color: #6a9955; }    /* VS Code Comment Green for Dates */
    .log-info { color: #569cd6; font-weight: bold; } /* VS Code Blue */
    .log-warn { color: #cca700; font-weight: bold; } /* Yellow */
    .log-err { color: #f44747; font-weight: bold; }  /* Red */
    .log-ticker { color: #c586c0; font-weight: bold;} /* Purple for Tickers */
    .log-neo4j { color: #00ff41; font-weight: bold; } /* Green for Graph DB */
    .log-stgnn { color: #f4b236; font-weight: bold; } /* Gold for Quantum/GNN ✨ */
    </style>
""", unsafe_allow_html=True)

# === CONNECTIONS (CACHED) ===

@st.cache_resource
def init_alpaca():
    """Connects to Alpaca using Streamlit Secrets."""
    try:
        api_key = st.secrets["alpaca"]["API_KEY"]
        secret_key = st.secrets["alpaca"]["SECRET_KEY"]
        base_url = st.secrets["alpaca"]["BASE_URL"]
        api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        return api
    except Exception as e:
        st.error(f"Alpaca Connection Error: {e}")
        return None

# --- NEW: TIMESCALEDB FAST TELEMETRY ADAPTER ---
@st.cache_data(ttl=30)
def fetch_timescaledb_telemetry():
    """Fetches execution reality directly from TimescaleDB, bypassing slow Alpaca order loops."""
    try:
        import psycopg2
        # Auto-detect Docker DNS vs Localhost
        db_host = os.environ.get("DB_HOST", "localhost")
        if 'timescaledb' in st.secrets:
            db_host = st.secrets["timescaledb"].get("HOST", db_host)
            
        conn = psycopg2.connect(
            host=db_host,
            port=os.environ.get("DB_PORT", "5432"),
            user=os.environ.get("DB_USER", "aqi_admin"),
            password=os.environ.get("AQI_DB_PASSWORD", "aqi_secure_db_pass_2026"),
            dbname=os.environ.get("DB_NAME", "aqi_telemetry")
        )
        
        query = """
        SELECT time as "Exit_Time", symbol as "Ticker", side as "Type", 
               intended_price as "Intended", actual_price as "Exit_Price", 
               pnl_pct * 100 as "PnL (%)", slippage_pct * 100 as "Slippage (%)", exit_reason 
        FROM execution_telemetry 
        ORDER BY time ASC
        """
        df_ex = pd.read_sql(query, conn)
        conn.close()
        
        if not df_ex.empty:
            df_ex['Result'] = df_ex['PnL (%)'].apply(lambda x: 'Win' if x > 0 else 'Loss')
            df_ex['Entry_Price'] = df_ex['Exit_Price'] / (1 + (df_ex['PnL (%)']/100))
            # Proxies for the scatter plot to prevent breaks until YF integration is ported
            df_ex['MAE (%)'] = np.minimum(df_ex['PnL (%)'], 0) - abs(df_ex['Slippage (%)'])
            df_ex['MFE (%)'] = np.maximum(df_ex['PnL (%)'], 0) + abs(df_ex['Slippage (%)'])
            
        return df_ex
    except ImportError:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"TimescaleDB Offline (Falling back to Alpaca): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def read_bot_logs():
    """Reads logs from Google Sheets (The Bridge)."""
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open("Angel_Bot_Logs")
        worksheet = sh.worksheet("logs")
        
        # Get all values, but filter out empty strings immediately
        logs = worksheet.col_values(1)
        clean_logs = [line for line in logs if line.strip()] 
        
        return clean_logs
    except Exception as e:
        return [f"Google Sheets Error: {e}"]

@st.cache_data(ttl=60)
def get_cloud_telemetry():
    """Reads distributed states from the Google Sheets bridge."""
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open("Angel_Bot_Logs")
        
        # 1. Get Trading Agent State (Positions/Locks)
        try:
            trading_ws = sh.worksheet("Trading_State")
            trading_str = trading_ws.acell('A1').value
            trading_state = json.loads(trading_str) if trading_str else {}
        except gspread.exceptions.WorksheetNotFound:
            trading_state = {}

        # 2. Get Daily Inference Agent State (Tensors, Health, Signals)
        try:
            inference_ws = sh.worksheet("Inference_State")
            inference_chunks = inference_ws.col_values(1)
            inference_str = "".join(inference_chunks)
            inference_state = json.loads(inference_str) if inference_str else {}
        except gspread.exceptions.WorksheetNotFound:
            inference_state = {}

        return trading_state, inference_state
        
    except Exception as e:
        st.warning(f"Telemetry Sync Warning: {e}")
        return {}, {}

@st.cache_data(ttl=60)
def get_account_data(_api):
    account = None
    positions = []
    all_orders = []
    
    # 1. Fetch Core Account Data Safely
    try:
        account = _api.get_account()._raw
        positions = [p._raw for p in _api.list_positions()]
    except Exception as e:
        print(f"Alpaca Account Fetch Error: {e}")
        return None, [], []
        
    # 2. Fetch Orders Safely (Decoupled from Account)
    try:
        until_dt = None
        for _ in range(4):  # Fetch up to 2,000 fills
            params = {'status': 'filled', 'limit': 500, 'direction': 'desc'}
            if until_dt:
                params['until'] = until_dt
                
            batch = _api.list_orders(**params)
            if not batch:
                break
                
            all_orders.extend([o._raw for o in batch])
            
            if len(batch) < 500:
                break
                
            # Safe string conversion for pagination
            until_dt = str(batch[-1].submitted_at)
            
    except Exception as e:
        print(f"Alpaca Orders Pagination Error (Safe Continue): {e}")
        # If pagination fails, we simply return the orders successfully fetched so far
        
    return account, positions, all_orders

@st.cache_data(ttl=3600)
def load_global_config(config_path='config_Alpaca_REAL_V2.json'):
    """Dynamically loads the master configuration for Universe Mapping."""
    import os
    
    fallback_config = {
        "asset_index_map": {
            # 1. TECHNOLOGY (XLK)
            "CSCO": "Tech/Hardware",
            "IONQ": "Tech/Quantum",
            "INTC": "Tech/Semis",
            
            # 2. COMMUNICATION SERVICES (XLC)
            "T": "Communication Services",
            "CMCSA": "Communication Services",
            "PINS": "Communication Services",
            
            # 3. ENERGY (XLE)
            "OXY": "Energy",
            "SLB": "Energy",
            "HAL": "Energy",
            
            # 4. HEALTHCARE (XLV)
            "PFE": "Healthcare",
            "VTRS": "Healthcare",
            "BMY": "Healthcare",
            
            # 5. INDUSTRIALS (XLI)
            "DAL": "Industrials",
            "AAL": "Industrials",
            "CSX": "Industrials",
            
            # 6. FINANCIALS (XLF)
            "BAC": "Financials",
            "SOFI": "Financials",
            "WFC": "Financials",
            
            # 7. CONSUMER STAPLES (XLP)
            "KO": "Consumer Defensive",
            "KR": "Consumer Defensive",
            "KHC": "Consumer Defensive",
            
            # 8. CONSUMER DISCRETIONARY (XLY)
            "CCL": "Consumer Cyclical",
            "F": "Consumer Cyclical",
            "GM": "Consumer Cyclical",
            
            # 9. BASIC MATERIALS & MINING (XME)
            "FCX": "Basic Materials",
            "CLF": "Basic Materials",
            "VALE": "Basic Materials",
        }
    }

    try:
        if os.path.exists(ALPACA_CONFIG_PATH):
            target_path = ALPACA_CONFIG_PATH
        else:
            return fallback_config 

        with open(target_path, 'r') as f:
            data = json.load(f)
            if "asset_index_map" not in data:
                data["asset_index_map"] = fallback_config["asset_index_map"]
            return data
            
    except Exception as e:
        print(f"Config Load Warning: {e}. Defaulting to hardcoded map.") 
        return fallback_config

def extract_bot_states(logs):
    """Extracts the exact number of tickers in each state from the end-of-cycle log."""
    for line in reversed(logs):
        if "Current states count" in line:
            match = re.search(r"Counter\(\{([^}]+)\}\)", line)
            if match:
                state_str = match.group(1)
                try:
                    return dict((k.strip("' "), int(v)) for k, v in (item.split(':') for item in state_str.split(',')))
                except:
                    pass
    return {}

@st.cache_data(ttl=60)
def get_portfolio_history(_api):
    try:
        history = _api.get_portfolio_history(period='all', timeframe='1D')
        if not history.timestamp: 
            return pd.DataFrame()
            
        df = pd.DataFrame({'timestamp': history.timestamp, 'equity': history.equity})
        df['equity'] = pd.to_numeric(df['equity'], errors='coerce').ffill()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        
        inception_date = pd.to_datetime('2025-05-24', utc=True)
        df = df[df['timestamp'] >= inception_date]
        df = df.sort_values('timestamp')
        
        return df
    except Exception as e:
        print(f"Portfolio History API Error: {e}") 
        return pd.DataFrame()

def apply_twr_adjustments(hist_df):
    """
    Calculates True Time-Weighted Return (TWR).
    Isolates actual trading performance by explicitly neutralizing capital injections (deposits/withdrawals)
    rather than relying on dangerous statistical smoothing.
    """
    if hist_df.empty:
        return hist_df

    # 1. Fetch exact cash-flow ledger from the Trading Agent state
    trading_state, _ = get_cloud_telemetry()
    cash_flows = trading_state.get('cash_flows', {})
    
    # Ensure UTC timezone alignment
    hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], utc=True)
    
    # 2. Calculate daily raw equity changes
    hist_df['equity_change'] = hist_df['equity'].diff().fillna(0)
    
    # 3. Map precise cash flows to the exact days they occurred
    hist_df['net_cash_flow'] = 0.0
    for date_str, flow_amount in cash_flows.items():
        try:
            flow_date = pd.to_datetime(date_str, utc=True).floor('D')
            mask = hist_df['timestamp'].dt.floor('D') == flow_date
            if mask.any():
                hist_df.loc[mask, 'net_cash_flow'] += float(flow_amount)
        except Exception:
            continue
            
    # 4. Calculate True Daily Return (HPR - Holding Period Return)
    # Formula: R_t = (E_t - E_{t-1} - CF_t) / (E_{t-1} + CF_{in})
    # Assumes deposits happen at the start of the day, withdrawals at the end
    
    hist_df['twr_return'] = 0.0
    
    for i in range(1, len(hist_df)):
        prev_equity = hist_df['equity'].iloc[i-1]
        curr_equity = hist_df['equity'].iloc[i]
        net_cf = hist_df['net_cash_flow'].iloc[i]
        
        # If no previous equity exists (day 1), return is 0
        if prev_equity <= 0:
            hist_df.loc[hist_df.index[i], 'twr_return'] = 0.0
            continue
            
        # Denominator adjusts based on cash flow direction to prevent division distortion
        if net_cf > 0:
            denominator = prev_equity + net_cf
        else:
            denominator = prev_equity
            
        twr = (curr_equity - prev_equity - net_cf) / denominator
        hist_df.loc[hist_df.index[i], 'twr_return'] = twr

    # Replace infinite/NaN values with 0
    hist_df['twr_return'] = hist_df['twr_return'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Map the TWR back to the standard 'daily_return' column expected by the rest of the dashboard
    hist_df['daily_return'] = hist_df['twr_return']
    
    # 5. Reconstruct the clean TWR Equity Curve
    true_starting_principal = hist_df['equity'].iloc[0] if not pd.isna(hist_df['equity'].iloc[0]) else 100.0
    hist_df['twr_equity'] = true_starting_principal * (1 + hist_df['daily_return']).cumprod()
    hist_df['twr_equity'] = hist_df['twr_equity'].fillna(true_starting_principal)
    
    # Overwrite raw equity with the TWR curve for all downstream metric calculations
    hist_df['equity'] = hist_df['twr_equity']
    
    # Cleanup intermediate columns
    hist_df.drop(columns=['equity_change', 'net_cash_flow', 'twr_return'], inplace=True)
    
    return hist_df

def parse_latest_run_logic(logs, bot_state=None, df_ex=None):
    if bot_state is None:
        bot_state = {}

    live_metrics = {}
    if df_ex is not None and not df_ex.empty:
        for t, group in df_ex.groupby('Ticker'):
            trades = len(group)
            wins = len(group[group['Result'] == 'Win'])
            wr = (wins / trades) * 100.0 if trades > 0 else 0.0
            
            mean_pnl = group['PnL (%)'].mean()
            std_pnl = group['PnL (%)'].std()
            ir = (mean_pnl / std_pnl) * (50 ** 0.5) if std_pnl > 0 else 0.0
            
            live_metrics[t] = {'Live WR': wr, 'Live IR': ir, 'Trades': trades}

    signals = {}
    watchlist = [] 
    neural_conviction = {} 
    model_health = {} 
    last_run_timestamp = None
    last_run_str = "Unknown"
    neo4j_status = "Unknown" 

    json_signals = bot_state.get("tickers", bot_state.get("signals", {}))
    action_map = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "CLOSE"}

    for ticker, data in json_signals.items():
        if isinstance(data, dict):
            raw_action = data.get("action", 0)
            if isinstance(raw_action, str):
                try: raw_action = int(raw_action)
                except ValueError: raw_action = 0

            raw_conf = data.get("confidence_score", data.get("confidence", 0.0))
            conf_val = raw_conf * 100.0  
            sig_text = data.get("signal", "HOLD (Unknown)")
            mapped_action = action_map.get(raw_action, "HOLD")
            current_atr = data.get("atr_norm", 0.03)

            neural_conviction[ticker] = {
                "Confidence": conf_val, 
                "Action": mapped_action,
                "ATR": current_atr  
            }

            if "Hold" in sig_text or "Suppressed" in sig_text:
                signals[ticker] = "⏸️ " + sig_text
            elif "Error" in sig_text:
                signals[ticker] = "❌ " + sig_text
            else:
                signals[ticker] = "✅ " + sig_text

            if conf_val > 40.0 and mapped_action != "HOLD":
                tag = "🔥 Screaming Setup" if conf_val > 80.0 else "⚡ High Conviction"
                watchlist.append({"Ticker": ticker, "Conf": f"{conf_val:.1f}%", "Status": tag})

            if "base_ir" in data:
                base_ir = data.get("base_ir", 0.0)
                base_wr = data.get("base_wr", 0.0) * 100.0
                base_mdd = data.get("base_mdd_days", 0)

                live_ir = live_metrics.get(ticker, {}).get('Live IR', 0.0)
                live_wr = live_metrics.get(ticker, {}).get('Live WR', 0.0)
                live_trades = live_metrics.get(ticker, {}).get('Trades', 0)

                mdd_val = data.get("mdd_days", 0)
                readiness_status = data.get("readiness_status", "UNKNOWN")

                # --- FIX: Respect Backend Readiness & Empirical Override Flags ---
                if readiness_status == "SUSPENDED_BASE_EDGE" or (base_ir <= 0.0 and readiness_status != "READY"):
                    status_clean = "🔴 QUARANTINED (Negative Base Edge)"
                    decay_val = 0.0
                    lifecycle_stage = "🔴 HALTED"
                elif readiness_status == "READY" and base_ir <= 0.0:
                    status_clean = "🟢 OPTIMAL (Empirical Override)"
                    decay_val = 1.0 # Bypass decay penalty
                    lifecycle_stage = "🟢 ACTIVE (Production)"
                else:
                    if live_trades >= 5 and base_ir > 0:
                        decay_val = live_ir / base_ir
                    else:
                        decay_val = 1.0 

                    if live_trades >= 5:
                        if decay_val >= 0.7: status_clean = "🟢 OPTIMAL"
                        elif decay_val >= 0.4: status_clean = "🟡 STABLE"
                        else: status_clean = "🔴 DEGRADED"
                    else:
                        if live_trades >= 2 and live_ir < -3.0:
                            status_clean = "🔴 ABORTED (Critical Early Failure)"
                        else:
                            status_clean = "🟢 OPTIMAL (Warming Up)"
                    
                    if "OPTIMAL" in status_clean:
                        lifecycle_stage = "🟢 ACTIVE (Production)"
                    elif "STABLE" in status_clean:
                        lifecycle_stage = "🟡 MATURE (Monitoring)"
                    elif "DEGRADED" in status_clean:
                        lifecycle_stage = "🔴 DEPRECATED (Pending Rollback)" if mdd_val > 42 else "🟠 DRIFTING (Requires Retraining)"
                    elif "ABORTED" in status_clean:
                        lifecycle_stage = "🔴 HALTED"

                model_health[ticker] = {
                    "Status": status_clean,
                    "Lifecycle": lifecycle_stage,
                    "Base IR": base_ir,
                    "Live IR": live_ir,
                    "Base WR": base_wr,
                    "Live WR": live_wr,
                    "Decay": decay_val,
                    "MDD": mdd_val,
                    "Base MDD": base_mdd,
                    "Trades": live_trades
                }

    ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})')
    for line in reversed(logs):
        if "Successfully connected to Neo4j" in line:
            if neo4j_status == "Unknown": neo4j_status = "🟢 Connected"
        elif "Failed to connect to Neo4j" in line:
            if neo4j_status == "Unknown": neo4j_status = "🔴 Disconnected"

        if last_run_str == "Unknown":
            match = ts_pattern.search(line)
            if match:
                last_run_str = match.group(1)
                try: last_run_timestamp = datetime.strptime(last_run_str, '%Y-%m-%d %H:%M:%S')
                except: pass

    if last_run_str == "Unknown" and len(logs) > 0:
        last_run_str = "Sheet Stream Live"
        last_run_timestamp = datetime.now()

    if not model_health and 'saved_model_health' in st.session_state:
        model_health = st.session_state['saved_model_health']

    for ticker, data in neural_conviction.items():
        if data["Action"] == "": data["Action"] = "HOLD"

    final_conviction = {k: v for k, v in neural_conviction.items() if v["Confidence"] > 0}
    unique_watchlist = {v['Ticker']:v for v in watchlist}.values()

    return last_run_str, last_run_timestamp, signals, list(unique_watchlist), final_conviction, model_health, neo4j_status

@st.cache_data(ttl=300)
def get_market_benchmark():
    try:
        hist = yf.download("SPY", period="2d", interval="1d", progress=False, threads=False)
        if isinstance(hist.columns, pd.MultiIndex):
            close_col = hist['Close'].iloc[:, 0]
        else:
            close_col = hist['Close']
            
        if len(close_col) >= 2:
            return ((float(close_col.iloc[-1]) - float(close_col.iloc[-2])) / float(close_col.iloc[-2])) * 100
        return 0.0
    except Exception:
        return 0.0

@st.cache_data(ttl=3600)
def get_trade_excursions(_api, orders):
    """Fallback method if TimescaleDB is offline. Reconstructs all lifetime round-trip trades."""
    if not orders:
        return pd.DataFrame()
        
    filled_orders = sorted(
        [o for o in orders if isinstance(o, dict) and o.get('status') == 'filled'],
        key=lambda x: x.get('filled_at', '')
    )
    
    trades = []
    inventory = {}
    
    for o in filled_orders:
        sym = o.get('symbol')
        side = o.get('side')
        qty = float(o.get('filled_qty', 0))
        price = float(o.get('filled_avg_price', 0))
        
        try:
            t = pd.to_datetime(o.get('filled_at')).tz_convert('UTC')
        except Exception:
            continue
            
        if sym not in inventory:
            inventory[sym] = {'qty': 0.0, 'cost': 0.0, 'entry_time': t, 'side': None}
        inv = inventory[sym]
        
        if inv['qty'] == 0:
            inv['side'] = side
            inv['cost'] = price
            inv['entry_time'] = t
            inv['qty'] += qty
        else:
            if inv['side'] == side:
                # Weighted average entry
                inv['cost'] = ((inv['cost'] * inv['qty']) + (price * qty)) / (inv['qty'] + qty)
                inv['qty'] += qty
            else:
                closed_qty = min(inv['qty'], qty)
                inv['qty'] -= closed_qty
                if closed_qty > 0:
                    entry_p = inv['cost']
                    exit_p = price
                    if inv['side'] == 'buy':
                        pnl_pct = ((exit_p - entry_p) / entry_p) * 100.0
                        trade_type = 'Long'
                    else:
                        pnl_pct = ((entry_p - exit_p) / entry_p) * 100.0
                        trade_type = 'Short'
                        
                    trades.append({
                        'Ticker': sym,
                        'Type': trade_type,
                        'Entry_Time': inv['entry_time'],
                        'Exit_Time': t,
                        'Entry_Price': entry_p,
                        'Exit_Price': exit_p,
                        'PnL (%)': pnl_pct,
                        'Result': 'Win' if pnl_pct > 0 else 'Loss',
                        'MAE (%)': -abs(pnl_pct) if pnl_pct < 0 else 0.0,
                        'MFE (%)': pnl_pct if pnl_pct > 0 else 0.0,
                        'Slippage (%)': 0.0
                    })
                if inv['qty'] == 0:
                    inv['side'] = None

    if not trades:
        return pd.DataFrame()
        
    return pd.DataFrame(trades)

@st.cache_data(ttl=3600)
def get_correlation_matrix(tickers):
    if not tickers or len(tickers) < 2: return None
    try:
        df = yf.download(tickers, period="1mo", interval="1d", progress=False, threads=False)['Close']
        if isinstance(df, pd.Series): return None
        return df.corr()
    except: return None

def get_system_telemetry():
    try:
        start = time.time()
        requests.get("https://api.alpaca.markets/v2/clock", timeout=2)
        latency = int((time.time() - start) * 1000)
    except: latency = 999
    return 0.0, 0.0, latency

def calculate_drawdown(df):
    df = df.copy()
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
    df['is_high'] = df['equity'] >= df['peak']
    df['underwater_days'] = df.groupby(df['is_high'].cumsum()).cumcount()
    return df

def calculate_seasonality(df):
    s_df = df.copy()
    if s_df['timestamp'].dt.tz is None: s_df['timestamp'] = s_df['timestamp'].dt.tz_localize('UTC')
    s_df['timestamp'] = s_df['timestamp'] - pd.Timedelta(hours=12)
    s_df['timestamp'] = s_df['timestamp'].dt.tz_convert('America/New_York')

    s_df['daily_return'] = s_df['equity'].pct_change() * 100
    s_df['Day'] = s_df['timestamp'].dt.day_name()
    s_df['Month'] = s_df['timestamp'].dt.strftime('%b')
    s_df['Month_Num'] = s_df['timestamp'].dt.month
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    day_stats = s_df.groupby('Day')['daily_return'].agg(
        Avg_Return='mean', Win_Rate=lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ).reindex(day_order)
    
    monthly_stats = s_df.groupby(['Month_Num', 'Month'])['daily_return'].agg(
        Avg_Return='mean', Win_Rate=lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ).reset_index().sort_values('Month_Num').set_index('Month')
    return day_stats, monthly_stats

def calculate_advanced_metrics(hist_df):
    if hist_df.empty: return {}
    df = hist_df.copy()
    df['daily_return'] = df['equity'].pct_change()
    returns = df['daily_return'].dropna()
    
    start_date, current_date = df['timestamp'].min(), df['timestamp'].max()
    if current_date.tz is None: current_date = current_date.tz_localize('UTC')
    if start_date.tz is None: start_date = start_date.tz_localize('UTC')
    
    days_active = max((current_date - start_date).days, 1)
    years_active = days_active / 365.25
    
    start_equity, end_equity = float(df['equity'].iloc[0]), float(df['equity'].iloc[-1])
    cagr = (end_equity / start_equity) ** (1 / years_active) - 1 if pd.notna(start_equity) and start_equity > 0 and years_active > 0 else 0.0
    
    df['peak'] = df['equity'].cummax()
    max_dd = ((df['equity'] - df['peak']) / df['peak']).min()
    mar = (cagr / abs(max_dd)) if max_dd != 0 else 0

    volatility = returns.std() * (252 ** 0.5)
    sharpe = (cagr - 0.04) / volatility if volatility > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * (252 ** 0.5) if not downside_returns.empty else 0
    sortino = (cagr - 0.04) / downside_vol if downside_vol > 0 else 0

    positive_sum = returns[returns > 0].sum()
    negative_sum = abs(returns[returns < 0].sum())
    profit_factor = (positive_sum / negative_sum) if negative_sum > 0 else float('inf')

    df_with_dd = calculate_drawdown(df)
    max_underwater_days = int(df_with_dd['underwater_days'].max()) if 'underwater_days' in df_with_dd.columns else 0
    ulcer_index = ((df_with_dd['drawdown'] * 100) ** 2).mean() ** 0.5 if 'drawdown' in df_with_dd.columns else 0.0

    if 'benchmark_return' in df.columns:
        active_return = returns - df['benchmark_return']
        tracking_error = active_return.std()
        if tracking_error > 1e-9:
            information_ratio = (active_return.mean() * 252) / (tracking_error * (252 ** 0.5))
        else: information_ratio = 0.0
    else:
        tracking_error = returns.std()
        if tracking_error > 1e-9: information_ratio = (returns.mean() * 252) / (tracking_error * (252 ** 0.5))
        else: information_ratio = 0.0

    wins = len(returns[returns > 0])
    total_active = len(returns[returns != 0])
    win_rate = (wins / total_active) if total_active > 0 else 0
    
    avg_win = returns[returns > 0].mean() if pd.notna(returns[returns > 0].mean()) else 0.0
    avg_loss = abs(returns[returns < 0].mean()) if pd.notna(returns[returns < 0].mean()) else 0.0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    sqn = (total_active ** 0.5) * (expectancy / returns.std()) if returns.std() > 0 else 0
    omega_ratio = (positive_sum / negative_sum) if negative_sum > 0 else float('inf')
    
    skewness_val = skew(returns) if len(returns) > 2 else 0
    kurt = kurtosis(returns) if len(returns) > 2 else 0
    cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100 if len(returns) > 20 else 0
    
    gain_to_pain = omega_ratio
    exposure_pct = total_active / len(returns) if len(returns) > 0 else 1.0
    exposure_efficiency = cagr / exposure_pct if exposure_pct > 0 else 0

    return {
        "CAGR": cagr, "Max Drawdown": max_dd, "Recovery Time": max_underwater_days, "Ulcer Index": ulcer_index,
        "Sharpe Ratio": sharpe, "Sortino Ratio": sortino, "Information Ratio": information_ratio, "MAR Ratio": mar,
        "Profit Factor": profit_factor, "Win Rate (Daily)": win_rate, "Expectancy": expectancy, "SQN": sqn,
        "Omega Ratio": omega_ratio, "Skewness": skewness_val, "Kurtosis": kurt, "CVaR (95%)": cvar_95,
        "Gain-to-Pain": gain_to_pain, "Exposure Efficiency": exposure_efficiency
    }

def create_scorecard_df(metrics, hit_rate, trade_count):
    data = [
        {"METRIC": "CAGR (Account)", "YOURS": f"{metrics.get('CAGR', 0):.1%}", "BENCHMARK": "> 20%", "VERDICT": "🏆 Elite" if metrics.get('CAGR', 0) > 0.2 else "😐 Std"},
        {"METRIC": "MAR Ratio", "YOURS": f"{metrics.get('MAR Ratio', 0):.2f}", "BENCHMARK": "> 1.0", "VERDICT": "🚀 Elite" if metrics.get('MAR Ratio', 0) > 1.0 else "😐 Std"},
        {"METRIC": "Max Drawdown", "YOURS": f"{metrics.get('Max Drawdown', 0):.1%}", "BENCHMARK": "< 15%", "VERDICT": "🛡️ Safe" if abs(metrics.get('Max Drawdown', 0)) < 0.15 else "⚠️ High Risk"},
        {"METRIC": "Recovery Time", "YOURS": f"{metrics.get('Recovery Time', 0)} Days", "BENCHMARK": "< 30 Days", "VERDICT": "⚡ Fast" if metrics.get('Recovery Time', 0) < 30 else "🐢 Slow"},
        {"METRIC": "Sharpe Ratio", "YOURS": f"{metrics.get('Sharpe Ratio', 0):.2f}", "BENCHMARK": "> 1.5", "VERDICT": "🔥 Good" if metrics.get('Sharpe Ratio', 0) > 1.5 else "😐 Std"},
        {"METRIC": "Sortino Ratio", "YOURS": f"{metrics.get('Sortino Ratio', 0):.2f}", "BENCHMARK": "> 2.0", "VERDICT": "💎 Strong" if metrics.get('Sortino Ratio', 0) > 2.0 else "😐 Std"},
        {"METRIC": "Profit Factor", "YOURS": f"{metrics.get('Profit Factor', 0):.2f}", "BENCHMARK": "> 1.5", "VERDICT": "💰 Rich" if metrics.get('Profit Factor', 0) > 1.5 else "😐 Std"},
        {"METRIC": "Daily Reliability", "YOURS": f"{metrics.get('Win Rate (Daily)', 0):.0%}", "BENCHMARK": "50-55%", "VERDICT": "✅ Stable" if metrics.get('Win Rate (Daily)', 0) > 0.5 else "🔻 Low"},
        {"METRIC": "Trade Hit Rate", "YOURS": f"{hit_rate:.0%} ({trade_count} Trades)", "BENCHMARK": "40-50%", "VERDICT": "🎯 Sniper" if hit_rate >= 0.45 else "😐 Std"},
    ]
    return pd.DataFrame(data)

def calculate_institutional_score(metrics):
    score = 0
    score += min(30, (metrics.get('Sharpe Ratio', 0) / 2.0) * 30)
    score += min(25, (metrics.get('MAR Ratio', 0) / 1.0) * 25)
    dd = abs(metrics.get('Max Drawdown', 0))
    if dd < 0.10: score += 25
    elif dd < 0.20: score += 15
    elif dd < 0.30: score += 5
    score += min(20, (metrics.get('Sortino Ratio', 0) / 3.0) * 20)
    return min(100, score)

def calculate_future_projections(start_date, starting_equity, target_cagr, weekly_deposits=[0, 70, 140], inflation_rate=0.03):
    start_date = pd.to_datetime(start_date).tz_localize(None).normalize()
    today = pd.Timestamp.now().normalize()
    
    target_dates = [start_date]
    for i in range(1, 37): target_dates.append(start_date + pd.DateOffset(months=i))
    for i in range(4, 21): target_dates.append(start_date + pd.DateOffset(years=i))
    target_dates.append(today)
    target_dates = sorted(list(set(target_dates)))
    
    weekly_rate = ((1 + target_cagr) ** (1 / 52.1429)) - 1
    
    projections = []
    for date in target_dates:
        years_from_start = (date - start_date).days / 365.25
        weeks_from_start = (date - start_date).days / 7
        if years_from_start < 0: continue
        
        base_fv = starting_equity * ((1 + target_cagr) ** years_from_start)
        base_inflated = base_fv * ((1 + inflation_rate) ** years_from_start)
        
        row = {"Date": date, "Base (No Deposits)": base_fv, "Base (+3% Inflation)": base_inflated}
        for dep in weekly_deposits:
            if dep == 0: continue
            deposit_fv = dep * (((1 + weekly_rate) ** weeks_from_start - 1) / weekly_rate) if weekly_rate > 0 else dep * weeks_from_start
            total_fv = base_fv + deposit_fv
            total_inflated = total_fv * ((1 + inflation_rate) ** years_from_start)
            row[f"+${dep}/wk"] = total_fv
            row[f"+${dep}/wk (+3% Inflation)"] = total_inflated
        projections.append(row)
    return pd.DataFrame(projections)

@st.cache_data(ttl=86400)
def get_historical_spy(start_date_str):
    try:
        spy = yf.download("SPY", start=start_date_str, progress=False, threads=False)
        close_series = spy['Close'].iloc[:, 0] if isinstance(spy.columns, pd.MultiIndex) else spy['Close']
        df = pd.DataFrame({'spy_close': close_series})
        df.index = pd.to_datetime(df.index).tz_localize(None).floor('D')
        df['spy_return'] = df['spy_close'].pct_change()
        return df[['spy_return']].dropna()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=3600)
def run_monte_carlo_simulation(historical_returns, starting_equity, weekly_deposit=140, years=20, paths=500):
    if len(historical_returns) < 10: return pd.DataFrame()
    days = int(years * 252)
    daily_dep = weekly_deposit / 5.0 
    sim_returns = np.random.choice(historical_returns, size=(paths, days))
    equity_paths = np.zeros((paths, days + 1))
    equity_paths[:, 0] = starting_equity
    for t in range(1, days + 1): equity_paths[:, t] = equity_paths[:, t-1] * (1 + sim_returns[:, t-1]) + daily_dep
    p10, p50, p90 = np.percentile(equity_paths, 10, axis=0), np.percentile(equity_paths, 50, axis=0), np.percentile(equity_paths, 90, axis=0)
    start_date = pd.Timestamp.today().normalize()
    dates = [start_date + pd.Timedelta(days=int((i/252)*365.25)) for i in range(days + 1)]
    return pd.DataFrame({'Date': dates, '10th Percentile (Pessimistic)': p10, '50th Percentile (Expected)': p50, '90th Percentile (Optimistic)': p90})

def calculate_3d_physics(df):
    phys_df = df.copy()
    phys_df['velocity'] = phys_df['equity'].pct_change() * 100
    phys_df['acceleration'] = phys_df['velocity'].diff()
    phys_df['jerk'] = phys_df['acceleration'].diff()
    phys_df['vel_smooth'] = phys_df['velocity'].ewm(span=3, adjust=False).mean()
    phys_df['acc_smooth'] = phys_df['acceleration'].rolling(3).mean()
    phys_df['jerk_smooth'] = phys_df['jerk'].rolling(3).mean()
    phys_df['dfe'] = np.sqrt(phys_df['vel_smooth']**2 + phys_df['acc_smooth']**2 + phys_df['jerk_smooth']**2)
    return phys_df.dropna()

@st.cache_data(ttl=600) 
def generate_proxied_ppo_landscape(phys_df, log_state, conviction_data, grid_size=50):
    if phys_df.empty: return None, None, None, None, "NO DATA"
    recent_data = phys_df.tail(20).copy()
    x_dim, y_dim = recent_data['vel_smooth'], recent_data['jerk_smooth']
    v_mean, v_std = x_dim.mean() if not x_dim.empty else 0.0, x_dim.std() if len(x_dim) > 1 else 0.1
    j_mean, j_std = y_dim.mean() if not y_dim.empty else 0.0, y_dim.std() if len(y_dim) > 1 else 0.1
    if v_std == 0 or np.isnan(v_std): v_std = 0.1
    if j_std == 0 or np.isnan(j_std): j_std = 0.1
    
    X, Y = np.meshgrid(np.linspace(v_mean - 3*v_std, v_mean + 3*v_std, grid_size), np.linspace(j_mean - 3*j_std, j_mean + 3*j_std, grid_size))
    Z_base = np.sin(np.sqrt(X**2 + Y**2)) / (np.sqrt(X**2 + Y**2) + 1)
    latest_jerk, latest_vel = abs(y_dim.iloc[-1]) if len(y_dim) > 0 else 0, abs(x_dim.iloc[-1]) if len(x_dim) > 0 else 0
    is_stalled = latest_vel < 0.05 and len(conviction_data) == 0
    
    if is_stalled:
        Z = np.full((grid_size, grid_size), 0.5) + np.random.normal(0, 0.005, (grid_size, grid_size))
        status_label = "🔴 MODE COLLAPSE"
    elif latest_jerk > 0.5:
        Z_static = gaussian_filter(np.random.normal(0, 0.4, (grid_size, grid_size)), sigma=0.8)
        Z = np.clip((((Z_base * np.exp(-0.2 * X**2)) + Z_static) + 1) / 2, 0.0, 1.0) 
        status_label = "⚡ HIGH CHAOS"
    else:
        Z = np.clip((((np.cos(X) * np.sin(Y)) + np.random.normal(0, 0.02, (grid_size, grid_size))) + 1) / 2, 0.1, 0.9)
        status_label = "🟢 HEALTHY EDGE"

    avg_real_confidence = sum(d["Confidence"] for d in conviction_data.values()) / len(conviction_data) / 100.0 if conviction_data else 0.5
    x_np, y_np = x_dim.to_numpy(), y_dim.to_numpy()

    if is_stalled: z_traj_np = np.full(len(x_np), 0.52) 
    elif latest_jerk > 0.5: z_traj_np = np.clip((((np.sin(np.sqrt(x_np**2 + y_np**2)) / (np.sqrt(x_np**2 + y_np**2) + 1)) * np.exp(-0.2 * x_np**2)) + 1) / 2, 0.0, 1.0) + 0.05
    else: z_traj_np = np.clip(((np.cos(x_np) * np.sin(y_np)) + 1) / 2, 0.1, 0.9) + 0.02

    z_traj_np[-1] = avg_real_confidence + 0.05
    return X, Y, Z, pd.Series(z_traj_np, index=recent_data.index), status_label

@st.cache_data(ttl=600)
def generate_stgnn_pca_landscape(bot_state, grid_size=50):
    json_signals = bot_state.get("tickers", bot_state.get("signals", {}))
    raw_tickers, raw_features, raw_confidences = [], [], []
    
    for t, data in json_signals.items():
        if isinstance(data, dict):
            state_tensor = data.get("entry_state")
            if state_tensor and isinstance(state_tensor, list) and len(state_tensor) >= 2:
                feature_vec = [float(x) for x in state_tensor]
            else:
                act_raw = data.get("action", 0)
                feature_vec = [float(data.get("drift_status", 0.0)), float(data.get("confidence_score", data.get("confidence", 0.0))), float(data.get("execution_latency_ms", 0.0)), float(act_raw) if str(act_raw).replace('.','',1).isdigit() else 0.0]
                
            if feature_vec and len(feature_vec) >= 2:
                raw_tickers.append(t)
                raw_features.append(feature_vec)
                raw_confidences.append(float(data.get("confidence_score", data.get("confidence", 0.0))))

    if len(raw_tickers) < 3: return None, None, None, None, "🔴 INSUFFICIENT DATA"
    target_dim = max([len(vec) for vec in raw_features])
    
    tickers, features, confidences = [], [], []
    for t, vec, conf in zip(raw_tickers, raw_features, raw_confidences):
        if len(vec) == target_dim:
            tickers.append(t); features.append(vec); confidences.append(conf)

    if len(tickers) < 3: return None, None, None, None, f"🔴 DIMENSION MISMATCH"
    features_np = np.array(features, dtype=float)
    features_scaled = (features_np - np.mean(features_np, axis=0)) / (np.std(features_np, axis=0) + 1e-9)

    pca = PCA(n_components=2)
    components = pca.fit_transform(features_scaled)
    pca1, pca2 = components[:, 0], components[:, 1]
    X, Y = np.meshgrid(np.linspace(pca1.min() - 1.5, pca1.max() + 1.5, grid_size), np.linspace(pca2.min() - 1.5, pca2.max() + 1.5, grid_size))
    Z = np.zeros_like(X)
    
    for i in range(len(tickers)): Z += confidences[i] * np.exp(-((X - pca1[i])**2 + (Y - pca2[i])**2) / (2 * 1.0**2))
    if Z.max() > 0: Z = np.clip(Z / Z.max(), 0.1, 0.95)

    return X, Y, Z, {'tickers': tickers, 'x': pca1, 'y': pca2, 'z': confidences}, f"🟢 PCA ALIGNED (Var Explained: {sum(pca.explained_variance_ratio_) * 100:.1f}%)"

@st.cache_data(ttl=600)
def generate_phase_portrait(phys_df, grid_size=20):
    if phys_df.empty: return None
    v, a = phys_df['vel_smooth'], phys_df['acc_smooth']
    v_min, v_max, a_min, a_max = v.min(), v.max(), a.min(), a.max()
    v_grid, a_grid = np.linspace(v_min, v_max, grid_size), np.linspace(a_min, a_max, grid_size)
    V, A = np.meshgrid(v_grid, a_grid)
    U, V_dir = A, -V 
    norm = np.sqrt(U**2 + V_dir**2)
    norm[norm == 0] = 1 
    U_norm, V_norm = U / norm, V_dir / norm
    Energy = 0.5 * (V**2 + A**2)
    
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=v_max, y1=a_max, fillcolor="rgba(0, 255, 65, 0.1)", line_width=0, layer="below") 
    fig.add_shape(type="rect", x0=v_min, y0=0, x1=0, y1=a_max, fillcolor="rgba(86, 156, 214, 0.1)", line_width=0, layer="below") 
    fig.add_shape(type="rect", x0=v_min, y0=a_min, x1=0, y1=0, fillcolor="rgba(255, 75, 75, 0.1)", line_width=0, layer="below") 
    fig.add_shape(type="rect", x0=0, y0=a_min, x1=v_max, y1=0, fillcolor="rgba(255, 176, 0, 0.1)", line_width=0, layer="below") 
    
    fig.add_annotation(x=v_max*0.5, y=a_max*0.9, text="TREND<br>(Expanding Bull)", showarrow=False, font=dict(color="#00ff41", size=12))
    fig.add_annotation(x=v_min*0.5, y=a_max*0.9, text="ACCUMULATION<br>(Slowing Bear)", showarrow=False, font=dict(color="#569cd6", size=12))
    fig.add_annotation(x=v_min*0.5, y=a_min*0.9, text="PANIC / SHOCK<br>(Expanding Bear)", showarrow=False, font=dict(color="#ff4b4b", size=12))
    fig.add_annotation(x=v_max*0.5, y=a_min*0.9, text="DISTRIBUTION<br>(Slowing Bull)", showarrow=False, font=dict(color="#ffb000", size=12))

    fig.add_trace(go.Contour(x=v_grid, y=a_grid, z=Energy, colorscale='Greys_r', opacity=0.3, showscale=False, contours=dict(showlines=True, coloring='none'), hoverinfo='skip'))

    step = 2
    for i in range(0, grid_size, step):
        for j in range(0, grid_size, step):
            fig.add_annotation(x=V[i, j], y=A[i, j], ax=V[i, j] - (U_norm[i, j] * 0.1), ay=A[i, j] - (V_norm[i, j] * 0.1), xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='rgba(255,255,255,0.2)')

    recent = phys_df.tail(50)
    fig.add_trace(go.Scatter(
        x=recent['vel_smooth'], y=recent['acc_smooth'], mode='lines+markers',
        marker=dict(size=abs(recent['jerk']) * 5 + 4, color=recent['jerk'], colorscale='Turbo', showscale=True, colorbar=dict(title="Jerk (Color)", len=0.5, y=0.5, x=1.05, tickfont={'color': "#cccccc"})),
        line=dict(color='white', width=2), name="Trajectory", customdata=recent['timestamp'].dt.strftime('%Y-%m-%d %H:%M'), hovertemplate='<b>Date</b>: %{customdata}<br><b>Vel</b>: %{x:.2f}%<br><b>Acc</b>: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(x=[recent['vel_smooth'].iloc[-1]], y=[recent['acc_smooth'].iloc[-1]], mode='markers+text', text=["📍 LIVE"], textposition="top center", marker=dict(size=12, color='#00ff41', symbol='diamond', line=dict(color='white', width=2)), textfont=dict(color="white", size=14, family="Arial Black"), name="Live Position", hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', marker=dict(size=35, color='rgba(255, 255, 255, 0)', symbol='circle-cross-open', line=dict(color='rgba(255, 255, 255, 0.3)', width=2)), text=["Equilibrium (0,0)"], textposition="bottom right", textfont=dict(color="rgba(255, 255, 255, 0.5)", size=11), name="Equilibrium", hoverinfo='skip'))

    fig.update_layout(xaxis_title='Velocity (Returns %)', yaxis_title='Acceleration (Change in Returns)', xaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=2, showgrid=False), yaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=2, showgrid=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#cccccc"), margin=dict(l=0, r=0, t=30, b=0), height=500, showlegend=False)
    return fig

def calculate_rolling_edge(df, window=30):
    r_df = df.copy()
    r_df['daily_return'] = r_df['equity'].pct_change()
    r_df['rolling_return'] = r_df['equity'].pct_change(periods=window) * 100
    roll_mean, roll_std = r_df['daily_return'].rolling(window).mean(), r_df['daily_return'].rolling(window).std()
    r_df['rolling_sharpe'] = (roll_mean / roll_std) * (252 ** 0.5)
    
    rolling_peak = r_df['equity'].rolling(window=window, min_periods=1).max()
    r_df['rolling_dd_raw'] = (r_df['equity'] - rolling_peak) / rolling_peak
    r_df['rolling_dd'] = r_df['rolling_dd_raw'] * 100
    r_df['rolling_dd_sq'] = r_df['rolling_dd_raw'] ** 2
    r_df['rolling_ulcer'] = (r_df['rolling_dd_sq'].rolling(window).mean()) ** 0.5 * 100

    downside_returns = r_df['daily_return'].copy()
    downside_returns[downside_returns > 0] = 0
    roll_downside_std = downside_returns.rolling(window).std()
    r_df['rolling_sortino'] = r_df.apply(lambda row: 0.0 if roll_downside_std.loc[row.name] == 0 else (roll_mean.loc[row.name] / roll_downside_std.loc[row.name]) * (252 ** 0.5), axis=1)
    
    r_df['is_win'] = (r_df['daily_return'] > 0).astype(int)
    r_df['rolling_win_rate'] = r_df['is_win'].rolling(window=window).mean() * 100
    r_df['rolling_vol'] = roll_std * (252 ** 0.5) * 100
    r_df['rolling_active_days'] = (r_df['daily_return'] != 0).rolling(window).sum()
    r_df['rolling_sqn'] = (r_df['rolling_active_days'] ** 0.5) * (roll_mean / roll_std)
    return r_df.dropna(subset=['rolling_return', 'rolling_sharpe', 'rolling_dd'])

def generate_tactical_alerts(roll_df, global_metrics, margin_util, phys_df):
    alerts = []
    if roll_df.empty or len(roll_df) < 5: return alerts

    latest_sharpe, latest_ulcer, latest_win_rate = roll_df['rolling_sharpe'].iloc[-1], roll_df['rolling_ulcer'].iloc[-1], roll_df['rolling_win_rate'].iloc[-1]

    if pd.notna(latest_sharpe):
        if latest_sharpe < 0.5: alerts.append({"level": "error", "icon": "📉", "title": f"Regime Shift: Rolling Sharpe is weak ({latest_sharpe:.2f})", "action": "POSITION SIZING HALVED. The risk-adjusted edge is decaying. Base lot sizes reduced by 50% until Sharpe recovers > 1.0."})
        elif latest_sharpe > 1.5: alerts.append({"level": "success", "icon": "🟢", "title": f"Elite Edge: Sharpe is surging ({latest_sharpe:.2f})", "action": "BASE SIZING RESTORED. The regime is highly favorable. System is deploying full-lot sizes."})

    if pd.notna(latest_ulcer):
        if latest_ulcer > 4.0: alerts.append({"level": "warning", "icon": "🛡️", "title": f"Pain Threshold Reached: Ulcer Index elevated ({latest_ulcer:.2f})", "action": "DEFENSIVE MONITORING ENGAGED. Drawdowns are elevated. Agent continues to rely on baseline 2x ATR stops."})
        elif latest_ulcer < 1.5: alerts.append({"level": "success", "icon": "🕊️", "title": f"Smooth Sailing: Low Ulcer Index ({latest_ulcer:.2f})", "action": "EDGE CONFIRMED. Drawdowns are minimal. Trades are operating cleanly within standard ATR boundaries."})

    if pd.notna(latest_win_rate):
        if latest_win_rate < 45.0: alerts.append({"level": "info", "icon": "✂️", "title": f"Choppy Execution: Win rate dropping ({latest_win_rate:.1f}%)", "action": "MARKET LACKS FOLLOW-THROUGH. Execution probabilities are skewed negatively in this environment."})
        elif latest_win_rate > 55.0: alerts.append({"level": "success", "icon": "🏃‍♂️", "title": f"High Hit Rate: Win rate is strong ({latest_win_rate:.1f}%)", "action": "MOMENTUM CONFIRMED. Market is respecting mathematical targets efficiently."})

    if margin_util > 75.0: alerts.append({"level": "error", "icon": "🚨", "title": f"Leverage Warning: Margin at {margin_util:.1f}%", "action": "BUYING FROZEN. Leverage limits reached. No new capital will be deployed."})

    if not phys_df.empty:
        latest_vel, latest_acc, latest_dfe = phys_df['vel_smooth'].iloc[-1], phys_df['acc_smooth'].iloc[-1], phys_df['dfe'].iloc[-1]
        if latest_vel <= 0 and latest_acc < 0: alerts.append({"level": "error", "icon": "🛡️", "title": "Regime Drift: PANIC / SHOCK", "action": f"Vector field confirms downward acceleration. Expected Shortfall (CVaR) is {global_metrics.get('CVaR (95%)', 0):.2f}%. Trading Agent active regime flag synced."})
        elif latest_dfe > 2.5: alerts.append({"level": "warning", "icon": "⚠️", "title": f"Extreme Phase Stretch (DFE: {latest_dfe:.2f})", "action": "System is highly extended from equilibrium. Mean-reversion shock probability is elevated."})

    return alerts

def transmit_manual_directives(is_halted, manual_sizing_multiplier):
    payload_str = json.dumps({"global_directives": {"active_regime": "EMERGENCY_HALT" if is_halted else "STABLE", "sizing_multiplier": float(manual_sizing_multiplier)}}, indent=4)
    if 'last_transmitted_payload' in st.session_state and st.session_state['last_transmitted_payload'] == payload_str: return 
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open("Angel_Bot_Logs")
        try: worksheet = sh.worksheet("Overrides")
        except gspread.exceptions.WorksheetNotFound: worksheet = sh.add_worksheet(title="Overrides", rows="10", cols="5")
        worksheet.update(range_name='A1', values=[[payload_str]])
        st.session_state['last_transmitted_payload'] = payload_str
    except Exception as e: st.error(f"Agent Comms Failure: Could not write to Google Sheets. {e}")

def format_log_line(line):
    clean_line = line.replace("<", "&lt;").replace(">", "&gt;")
    clean_line = re.sub(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})', r'<span class="log-ts">\1</span>', clean_line)
    clean_line = clean_line.replace("[INFO]", '<span class="log-info">[INFO]</span>').replace("[WARNING]", '<span class="log-warn">[WARNING]</span>').replace("[ERROR]", '<span class="log-err">[ERROR]</span>')
    clean_line = re.sub(r'(?<!\w)\[([A-Z]{2,5})\](?!\w)', r'<span class="log-ticker">[\1]</span>', clean_line)
    clean_line = clean_line.replace("[Neo4j]", '<span class="log-neo4j">[Neo4j]</span>').replace("✨", '<span class="log-stgnn">✨</span>')
    return f'<div class="log-line">{clean_line}</div>'

# === SIDEBAR CONFIG ===
with st.sidebar:
    st.header("🦅 AQI Mission Control")
    auto_refresh = st.toggle("Enable Auto-Refresh (60s)", value=True)
    st.divider()
    st.subheader("🚨 Emergency Overrides")
    st.caption("Manual intervention only. Overrides Trading Agent autonomy.")
    sys_halt = st.checkbox("🛑 HALT ALL NEW ENTRIES")
    sys_sizing = st.slider("Global Sizing Multiplier", 0.0, 2.0, 1.0, 0.1)
    transmit_manual_directives(sys_halt, sys_sizing)
    st.divider()
    st.subheader("🔮 Projection Tuning")
    use_manual_cagr = st.checkbox("Manual CAGR Override")
    manual_cagr = st.slider("Target CAGR %", 0, 100, 25) / 100
    if st.button("Force Refresh Now", type="primary"): st.cache_data.clear(); st.rerun()

# === DASHBOARD LOGIC ===
api = init_alpaca()
if not api: st.stop()

account, positions, orders = get_account_data(api)

# --- REPLACE ALPACA EXCURSIONS WITH TIMESCALEDB ---
df_ex_db = fetch_timescaledb_telemetry()
if not df_ex_db.empty:
    df_ex = df_ex_db
else:
    df_ex = get_trade_excursions(api, orders)

# === STATE INITIALIZATION FIX ===
# Prevents NameErrors if Alpaca rate-limits or returns None for 'account'
logs = []
trading_state, inference_state = {}, {}
model_health = st.session_state.get('saved_model_health', {})
conviction_data = st.session_state.get('saved_conviction', {})
parsed_signals = st.session_state.get('saved_signals', {})
watchlist_data = st.session_state.get('saved_watchlist', [])
last_run_str = "API Offline/Connecting..."
last_run_dt = None
neo4j_status = "Unknown"
# ================================

if account:
    col1, col2, col_alpha, col3, col_var, col4 = st.columns(6)

    equity, last_equity, buying_power = float(account['equity']), float(account['last_equity']), float(account['buying_power'])
    daily_pl_pct, daily_pl_abs = (equity - last_equity) / last_equity * 100, equity - last_equity
    spy_return = get_market_benchmark()
    daily_alpha = daily_pl_pct - spy_return

    trading_state, inference_state = get_cloud_telemetry() 
    json_signals = inference_state.get("tickers", inference_state.get("signals", {}))

    total_var = 0.0
    if positions:
        for p in positions:
            sym = p['symbol']
            mv = abs(float(p['market_value']))
            current_atr = json_signals.get(sym, {}).get("atr_norm", 0.03) if sym in json_signals and isinstance(json_signals[sym], dict) else 0.03
            total_var += mv * max(current_atr * 1.5, 0.02)

    var_pct = (total_var / equity) * 100 if equity > 0 else 0.0

    col1.metric("Net Liquidity", f"${equity:,.2f}", f"{daily_pl_pct:.2f}%")
    col2.metric("Day P/L", f"${daily_pl_abs:,.2f}")
    col_alpha.metric("Daily Alpha (vs SPY)", f"{daily_alpha:+.2f}%", f"SPY: {spy_return:+.2f}%", delta_color="normal")
    col3.metric("Buying Power", f"${buying_power:,.2f}")
    col_var.metric("Open Risk (VaR)", f"${total_var:,.2f}", f"-{var_pct:.2f}% Eq", delta_color="inverse")

    logs = read_bot_logs()
    last_run_str, last_run_dt, parsed_signals, watchlist_data, conviction_data, model_health, neo4j_status = parse_latest_run_logic(logs, inference_state, df_ex)

    if conviction_data and len(conviction_data) > 0:
        st.session_state['saved_conviction'] = conviction_data
        st.session_state['saved_signals'] = parsed_signals
        st.session_state['saved_watchlist'] = watchlist_data
    else:
        conviction_data = st.session_state.get('saved_conviction', {})
        parsed_signals = st.session_state.get('saved_signals', {})
        watchlist_data = st.session_state.get('saved_watchlist', [])

    if model_health and len(model_health) > 0: st.session_state['saved_model_health'] = model_health
    else: model_health = st.session_state.get('saved_model_health', {})

    status_val = "Unknown"
    if last_run_dt:
        seconds_ago = int((datetime.now() - last_run_dt).total_seconds())
        minutes_ago = int(seconds_ago / 60)
        if minutes_ago < 10: status_val = "🟢 Active"
        elif minutes_ago < 60: status_val = f"🟡 Idle ({minutes_ago}m)"
        else: status_val = f"🔴 Stale ({int(minutes_ago/60)}h)"
    
    col4.metric("Bot Status", status_val, delta=f"Last Log: {last_run_str}", delta_color="off")
    if status_val == "🟢 Active" and seconds_ago < 300:
        st.progress(int(max(0, min(100, (max(0, seconds_ago) / 300.0) * 100))), text=f"⏳ Next Market Scan in ~{max(0, 300 - max(0, seconds_ago))}s")

st.divider()

hist_df_raw = get_portfolio_history(api)
hist_df_adj = hist_df_raw.copy()
roll_df, phys_df = pd.DataFrame(), pd.DataFrame() 

if not hist_df_raw.empty and account:
    if hist_df_raw['timestamp'].dt.tz is None: hist_df_raw['timestamp'] = hist_df_raw['timestamp'].dt.tz_localize('UTC')
    hist_df_raw = pd.concat([hist_df_raw, pd.DataFrame([{'timestamp': pd.Timestamp.now(tz='UTC'), 'equity': float(account['equity'])}])], ignore_index=True)
    hist_df_adj = apply_twr_adjustments(hist_df_raw.copy())

    spy_df = get_historical_spy(hist_df_adj['timestamp'].min().strftime('%Y-%m-%d'))
    if not spy_df.empty:
        hist_df_adj['date_only'] = hist_df_adj['timestamp'].dt.tz_localize(None).dt.floor('D')
        spy_df['date_only'] = spy_df.index
        hist_df_adj = pd.merge(hist_df_adj, spy_df, on='date_only', how='left')
        hist_df_adj['benchmark_return'] = hist_df_adj['spy_return'].fillna(0.0)
        hist_df_adj.drop(columns=['date_only', 'spy_return'], inplace=True)

    st.session_state['global_metrics'] = calculate_advanced_metrics(hist_df_adj)
    roll_df = calculate_rolling_edge(hist_df_adj, window=30)
    phys_df = calculate_3d_physics(hist_df_adj)

maint_margin = float(account.get('maintenance_margin', 0)) if account else 0.0
equity_val = float(account['equity']) if account else 0.0
margin_util = (maint_margin / equity_val * 100) if equity_val > 0 else 0.0

tab1, tab2, tab3, tab5, tab6 = st.tabs(["🧠 Bot Logic & Positions", "📜 Raw Logs", "📈 Real Performance", "🌌 Phase Space", "🧬 Model Lifecycle"])

with tab1:
    avg_market_move = sum([float(p['unrealized_plpc']) for p in positions]) * 100 if positions else 0.0
    sentiment_score = max(0.0, min(1.0, 0.5 + (avg_market_move / 5))) if positions else 0.5

    st.markdown("### 🌡️ Market Pulse")
    s_col1, s_col2, s_col3 = st.columns([3, 1, 6])
    with s_col1: st.progress(int(max(0, min(100, sentiment_score * 100))))
    with s_col2:
        if avg_market_move > 0.5: st.success("BULLISH")
        elif avg_market_move < -0.5: st.error("BEARISH")
        else: st.warning("NEUTRAL")

    st.markdown("#### 📅 Tactical Macro Briefing & System Posture")
    
    import xml.etree.ElementTree as ET
    @st.cache_data(ttl=3600)
    def fetch_macro_calendar_dashboard():
        try:
            response = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            response.raise_for_status()
            events_list = []
            for event in ET.fromstring(response.content).findall('event'):
                country = event.find('country').text if event.find('country') is not None else ''
                impact = event.find('impact').text if event.find('impact') is not None else ''
                title = event.find('title').text if event.find('title') is not None else ''
                date_str = event.find('date').text if event.find('date') is not None else ''
                time_str = event.find('time').text if event.find('time') is not None else ''
                if country in ['USD', 'JPY'] and time_str and time_str.lower() != 'all day':
                    if impact == 'High' or any(keyword in title.lower() for keyword in ['cpi', 'fomc', 'fed', 'payroll', 'nfp', 'inflation', 'retail sales', 'wage price', 'interest rate', 'rate decision']):
                        try:
                            event_dt_est = pytz.timezone('US/Eastern').localize(datetime.strptime(f"{date_str} {time_str.replace('am', 'AM').replace('pm', 'PM')}", "%m-%d-%Y %I:%M%p"))
                            hours_until = (event_dt_est - datetime.now(pytz.timezone('US/Eastern'))).total_seconds() / 3600.0
                            if hours_until > -1.0: 
                                is_critical = any(kw in title.lower() for kw in ['cpi', 'inflation', 'fomc', 'fed ', 'interest rate', 'rate decision'])
                                events_list.append({
                                    "Event": f"{country}: {title}",
                                    "Severity": '🔴 High' if impact == 'High' else ('🟡 Medium' if impact == 'Medium' else ('🟢 Low' if impact == 'Low' else '⚪ None')),
                                    "Time (AEST)": event_dt_est.astimezone(pytz.timezone('Australia/Brisbane')).strftime('%a %I:%M %p'),
                                    "Hours Until": hours_until,
                                    "Critical": is_critical
                                })
                        except Exception: continue
            return events_list
        except Exception as e:
            st.cache_data.clear(); st.warning(f"Macro feed disconnected: {e}"); return None

    upcoming_macro = fetch_macro_calendar_dashboard()
    
    if upcoming_macro:
        active_events = [e for e in upcoming_macro if e["Hours Until"] >= -0.5]
        if active_events:
            active_events.sort(key=lambda x: x["Hours Until"])
            next_event = active_events[0]
            hrs = next_event["Hours Until"]
            c_mac1, c_mac2 = st.columns([1, 1])
            with c_mac1:
                if next_event["Critical"]: st.error(f"**Next Event:** {next_event['Event']} ({next_event['Time (AEST)']})") 
                else: st.warning(f"**Next Event:** {next_event['Event']} ({next_event['Time (AEST)']})") 
            with c_mac2:
                if hrs <= (1.0 if next_event["Critical"] else 0.5) and hrs >= -0.5:
                    if next_event["Critical"]: st.error("🚨 **System Posture:** THE STRADDLE ENGAGED (Entries Frozen)")
                    else: st.warning("⚠️ **System Posture:** SAFETY LOCK ENGAGED (Entries Frozen)")
                else: st.success(f"🟢 **System Posture:** STANDARD TRAIL & TARGETS (T-{hrs:.1f} hours to lock)")
            with st.expander("View Full Weekly Calendar"): st.dataframe(pd.DataFrame(upcoming_macro).drop(columns=['Critical']), width='stretch', hide_index=True)
        else:
            st.success("🟢 **System Posture:** STANDARD TRAIL & TARGETS"); st.info("No immediate Tier-1 Macro Events pending.")
    else:
        st.success("🟢 **System Posture:** STANDARD TRAIL & TARGETS"); st.info("No critical Tier-1 Macro Events scheduled for USD/JPY for the remainder of the week.")

    st.divider()

    alerts = generate_tactical_alerts(roll_df, st.session_state.get('global_metrics', {}), margin_util, phys_df)
    if alerts:
        st.markdown("### ⚡ Active System Overrides")
        for alert in alerts:
            msg = f"**{alert['title']}** — {alert['action']}"
            if alert['level'] == "error": st.error(f"{alert['icon']} {msg}")
            elif alert['level'] == "warning": st.warning(f"{alert['icon']} {msg}")
            elif alert['level'] == "success": st.success(f"{alert['icon']} {msg}") 
            else: st.info(f"{alert['icon']} {msg}")
        st.divider()

    if isinstance(orders, list):
        for po in [o for o in orders if isinstance(o, dict) and o.get('status') in ['new', 'accepted', 'partially_filled', 'pending_new']]:
            if po.get('created_at'):
                try:
                    seconds_open = max(0, (pd.Timestamp.now(tz='UTC') - pd.to_datetime(po.get('created_at')).tz_convert('UTC')).total_seconds())
                    if seconds_open > 60: st.error(f"⚠️ **Execution Alert:** {po.get('side', 'UNKNOWN').upper()} order for {po.get('qty', '?')} {po.get('symbol', '?')} has been pending for {int(seconds_open)}s! High slippage risk.")
                    else: st.info(f"🔄 **Transmitting:** {po.get('side', 'UNKNOWN').upper()} {po.get('qty', '?')} {po.get('symbol', '?')} (Routing to market: {int(seconds_open)}s ago)")
                except Exception: pass
        
    if model_health:
        valid_models = [m for m in model_health.values() if 'Live IR' in m and 'Base IR' in m]
        if valid_models:
            avg_base_ir = sum(float(m['Base IR']) for m in valid_models) / len(valid_models)
            avg_live_ir = sum(float(m['Live IR']) for m in valid_models) / len(valid_models)
            ir_div = avg_live_ir - avg_base_ir
        else: avg_base_ir, avg_live_ir, ir_div = 0.0, 0.0, 0.0
    else: avg_base_ir, avg_live_ir, ir_div = 0.0, 0.0, 0.0
        
    current_ulcer = st.session_state.get('global_metrics', {}).get('Ulcer Index', 0.0)
    
    gl1, gl2, gl3 = st.columns(3)
    gl1.metric("Swarm Benchmark (Base IR)", f"{avg_base_ir:.2f}")
    gl2.metric("Swarm Reality (Live IR)", f"{avg_live_ir:.2f}", f"{ir_div:+.2f} Divergence", delta_color="inverse" if ir_div < 0 else "normal")
    gl3.metric("System Pain (Ulcer Index)", f"{current_ulcer:.2f}", "Threshold: > 4.0", delta_color="inverse" if current_ulcer > 4.0 else "normal")

    st.divider()

    st.markdown("#### 🔋 Capital Deployment Status")
    active_capital = sum([abs(float(p['market_value'])) for p in positions]) if positions else 0.0
    cash_capital = equity_val - active_capital 
    total_capital = active_capital + cash_capital
    active_pct = (active_capital / total_capital * 100) if total_capital > 0 else 0
    cash_pct = (cash_capital / total_capital * 100) if total_capital > 0 else 100
    
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("💼 Active Capital", f"${active_capital:,.2f}", f"{active_pct:.1f}% Deployed", delta_color="off")
    sc2.metric("💵 Dry Powder", f"${cash_capital:,.2f}", f"{cash_pct:.1f}% Cash", delta_color="off")
    
    asset_index_map = load_global_config().get("asset_index_map", {})
    monitored_tickers = list(asset_index_map.keys()) if asset_index_map else []
    sc3.metric("🤖 Active Agents", f"{len(positions)} / {len(monitored_tickers) if monitored_tickers else '?'}")

    if conviction_data:
        long_count = sum(1 for d in conviction_data.values() if d.get("Action") == "LONG")
        short_count = sum(1 for d in conviction_data.values() if d.get("Action") == "SHORT")
        hold_count = len(conviction_data) - long_count - short_count
        st.markdown("#### ⚖️ Bot Macro Bias (Neural Skew)")
        st.progress(int(max(0, min(100, ((long_count + (hold_count * 0.5)) / len(conviction_data) if len(conviction_data) > 0 else 0.5) * 100))))
        b1, b2, b3 = st.columns(3)
        b1.caption(f"🟢 Long Bias: {long_count}")
        b2.caption(f"⚪ Neutral/Hold: {hold_count}")
        b3.caption(f"🔴 Short Bias: {short_count}")

    st.divider()

    st.subheader("🧠 Neural Conviction Levels")
    if conviction_data:
        df_conv = pd.DataFrame([{"Ticker": t, "Confidence": d["Confidence"], "Action": d["Action"]} for t, d in conviction_data.items()]).sort_values(by='Confidence', ascending=False)
        df_conv['Chart_Text'] = df_conv.apply(lambda row: f"{row['Action']}<br>{row['Confidence']:.1f}%" if row['Action'] else f"{row['Confidence']:.1f}%", axis=1)

        fig_conf = px.bar(df_conv, x='Ticker', y='Confidence', color='Confidence', color_continuous_scale=['#4a1c1c', '#ffb000', '#00ff41'], range_y=[0, 100], text='Chart_Text')
        fig_conf.update_traces(textposition='inside', textfont_size=14, textfont_color='white')
        fig_conf.update_layout(height=150, margin=dict(l=0, r=0, t=10, b=10), xaxis_title=None, yaxis_title="Confidence %", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font={'color': '#cccccc'}, yaxis=dict(showgrid=True, gridcolor='#333'), xaxis=dict(showgrid=False, categoryorder='total descending'))
        st.plotly_chart(fig_conf, width='stretch')
    else: st.info("Waiting for first model run to populate conviction data...")

    st.divider()

    c1, c2 = st.columns([3, 4])
    with c1:
        inner_nav = st.radio("Navigation", ["🔭 Watchlist", "📝 Decisions", "🖥️ Risk & Telemetry", "🔪 Execution & Edge"], horizontal=True, label_visibility="collapsed")
        st.divider()
        
        if inner_nav == "🔭 Watchlist":
            if watchlist_data: st.dataframe(pd.DataFrame(watchlist_data), width='stretch', hide_index=True)
            else: st.caption("No high-confidence setups detected yet.")

        elif inner_nav == "📝 Decisions":
            if parsed_signals: st.dataframe(pd.DataFrame(list(parsed_signals.items()), columns=["Ticker", "Decision"]), width='stretch', hide_index=True)
            else: st.info("No signals parsed from recent logs.")
                
        elif inner_nav == "🖥️ Risk & Telemetry":
            st.markdown("#### Server & API Telemetry")
            cpu, ram, ping = get_system_telemetry()
            t1, t2, t3 = st.columns(3)
            t1.metric("CPU Load", f"{cpu}%", delta="High" if cpu > 80 else "Normal", delta_color="inverse")
            t2.metric("RAM Util", f"{ram}%", delta="High" if ram > 85 else "Normal", delta_color="inverse")
            t3.metric("API Latency", f"{ping}ms", delta="Lag" if ping > 300 else "Fast", delta_color="inverse")

            st.divider()
            st.markdown("#### Margin Distance")
            st.progress(int(max(0, min(100, margin_util))), text=f"Margin Capacity Used: {margin_util:.1f}%")
            if margin_util > 80: st.error("⚠️ CRITICAL: Approaching Maintenance Margin Call!")

            st.divider()
            st.markdown("#### Active Position Correlation")
            if positions and len(positions) > 1:
                corr_matrix = get_correlation_matrix([p['symbol'] for p in positions])
                if corr_matrix is not None:
                    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                    fig_corr.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', font={'color': '#cccccc'})
                    st.plotly_chart(fig_corr, width='stretch')
            else: st.caption("Need at least 2 active positions to plot correlation.")

        elif inner_nav == "🔪 Execution & Edge":
            st.markdown("#### ⚖️ Edge Quality")
            e1, e2 = st.columns(2)
            st.markdown("#### 🎯 Excursion Analysis (MAE vs MFE)")
            st.caption("Scatter plot of recent closed trades. Identifies if stops are too tight or winners are choked.")
            
            if not df_ex.empty:
                fig_ex = px.scatter(
                    df_ex, x="MAE (%)", y="MFE (%)", color="Result",
                    marginal_x="histogram", marginal_y="histogram", 
                    hover_data=["Ticker", "PnL (%)", "Type"],
                    color_discrete_map={"Win": "#00ff41", "Loss": "#ff4b4b"}
                )
                fig_ex.add_vline(x=-2.0, line_dash="dash", line_color="red", annotation_text="Absolute Risk Floor (-2%)", annotation_position="top right")
                fig_ex.add_hline(y=4.0, line_dash="dash", line_color="green", annotation_text="Min Target Floor (+4%)", annotation_position="bottom right")
                fig_ex.update_traces(selector=dict(type='scatter'), marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
                fig_ex.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#cccccc'},
                    xaxis=dict(title="Max Adverse Excursion (Pain %)", showgrid=True, gridcolor='#333', zerolinecolor='white'),
                    yaxis=dict(title="Max Favorable (Gain %)", showgrid=True, gridcolor='#333', zerolinecolor='white')
                )
                st.plotly_chart(fig_ex, width='stretch')
            else: st.info("Gathering excursion data. Close more trades to populate scatter plot.")

    with c2:
        st.subheader("💼 Capital & Active Portfolio")
        
        allocation_data = [{"Asset": "CASH", "Value": cash_capital}]
        for p in positions: allocation_data.append({"Asset": p['symbol'], "Value": abs(float(p['market_value']))})
        
        if allocation_data:
            fig_alloc = px.pie(pd.DataFrame(allocation_data), values='Value', names='Asset', hole=0.65, color_discrete_sequence=['#2d2d2d'] + px.colors.qualitative.Pastel)
            fig_alloc.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=220, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#cccccc'}, showlegend=True, legend=dict(orientation="v", yanchor="auto", y=0.5, xanchor="left", x=1.0))
            fig_alloc.add_annotation(text=f"Total Eq<br>${equity_val:,.0f}", x=0.5, y=0.5, font_size=14, showarrow=False)
            st.plotly_chart(fig_alloc, width='stretch')
            
            st.caption(f"🤖 **Bot Pre-Auth:** Estimated next trade size is **~${(equity_val / len(monitored_tickers) if len(monitored_tickers) > 0 else 0.0):,.2f}** per signal.")
            
            sector_data = {}
            for p in positions:
                sec = asset_index_map.get(p['symbol'], 'Other')
                sector_data[sec] = sector_data.get(sec, 0) + abs(float(p['market_value']))
            
            if sector_data:
                df_sec = pd.DataFrame(list(sector_data.items()), columns=['Index', 'Exposure']).sort_values('Exposure', ascending=True)
                fig_sec = px.bar(df_sec, x='Exposure', y='Index', orientation='h', text_auto='$.0f')
                fig_sec.update_traces(marker_color='#569cd6', textposition='inside')
                fig_sec.update_layout(height=120 + (len(df_sec) * 20), margin=dict(l=0, r=0, t=25, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#cccccc'}, xaxis_visible=False, title=dict(text="Risk by Mapped Index", font=dict(size=14)))
                st.plotly_chart(fig_sec, width='stretch')

        if positions:
            pos_data = []
            for p in positions:
                sym = p['symbol']
                side = p['side'].lower()
                entry = float(p['avg_entry_price'])
                current = float(p['current_price'])
                qty = abs(float(p['qty']))
                
                current_atr = conviction_data[sym].get("ATR", 0.03) if conviction_data and sym in conviction_data else 0.03
                active_sl, active_tp = max(current_atr * 1.5, 0.02), max(current_atr * 3.0, 0.04)
                
                if side == 'long':
                    sl, tp = entry * (1 - active_sl), entry * (1 + active_tp)
                    progress = max(0.0, min(1.0, (current - sl) / (tp - sl)))
                else:
                    sl, tp = entry * (1 + active_sl), entry * (1 - active_tp)
                    progress = max(0.0, min(1.0, (sl - current) / (sl - tp)))

                days_held = 0
                if isinstance(orders, list):
                    for o in orders:
                        if isinstance(o, dict) and o.get('symbol') == sym and o.get('status') == 'filled':
                            filled_at = o.get('filled_at')
                            if filled_at:
                                try: days_held = max(0, (pd.Timestamp.now(tz='UTC') - pd.to_datetime(filled_at).tz_convert('UTC')).days)
                                except Exception: pass
                            break

                pos_data.append({
                    "Ticker": sym, "Side": side.upper(), "Invested": entry * qty, "Qty": qty,
                    "P/L (%)": float(p['unrealized_plpc']) * 100, "Journey": progress, "Days Held": f"{days_held}/5"
                })
            
            st.dataframe(pd.DataFrame(pos_data), width='stretch', column_config={"Invested": st.column_config.NumberColumn("Invested", format="$%.2f"), "P/L (%)": st.column_config.NumberColumn("P/L (%)", format="%.2f%%"), "Journey": st.column_config.ProgressColumn("Journey to TP", help="Green bar moving right towards Take Profit.", min_value=0.0, max_value=1.0, format="%.2f")}, hide_index=True)

            st.markdown("##### 🎯 Immediate Flashpoints (True R-Multiple)")
            closest_tp, closest_sl = None, None
            max_r, min_r = -999.0, 999.0

            for p_data in pos_data:
                current_atr = conviction_data[p_data["Ticker"]].get("ATR", 0.03) if conviction_data and p_data["Ticker"] in conviction_data else 0.03
                true_r = p_data["P/L (%)"] / (max(current_atr * 1.5, 0.02) * 100)
                
                if true_r > max_r: max_r, closest_tp = true_r, p_data["Ticker"]
                if true_r < min_r: min_r, closest_sl = true_r, p_data["Ticker"]

            f1, f2 = st.columns(2)
            if closest_tp and max_r > 0: f1.success(f"🟢 **Highest R:** {closest_tp} (Floating: +{max_r:.2f}R)")
            if closest_sl and min_r < 0: f2.error(f"🔴 **Lowest R:** {closest_sl} (Floating: {min_r:.2f}R)")
                
        else: st.caption("No active positions currently held.")

        st.divider()

        c_ord1, c_ord2 = st.columns([3, 1])
        c_ord1.subheader("📜 Recent Fills & Execution Quality")
        
        # --- REPLACED: RECENT FILLS READS FROM TIMESCALEDB IF AVAILABLE ---
        if not df_ex_db.empty:
            trades_today = len(df_ex_db[pd.to_datetime(df_ex_db['Exit_Time']).dt.date == pd.Timestamp.now(tz='UTC').date()])
            if trades_today > 4: c_ord2.error(f"⚠️ Trades Today: {trades_today}")
            else: c_ord2.info(f"⚡ Trades Today: {trades_today}")
            
            order_data = []
            recent_db = df_ex_db.tail(5).iloc[::-1]
            for _, row in recent_db.iterrows():
                order_data.append({
                    "Time": row['Exit_Time'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['Exit_Time']) else "N/A",
                    "Ticker": row['Ticker'],
                    "Side": str(row['Type']).upper(),
                    "Qty": "DB (Exited)", 
                    "Fill Price": f"${row['Exit_Price']:.2f}",
                    "Slippage": f"{row['Slippage (%)']:+.2f}%"
                })
        else:
            # Fallback to Alpaca
            trades_today = sum(1 for o in orders if isinstance(o, dict) and o.get('status') == 'filled' and pd.to_datetime(o.get('filled_at')).tz_convert('UTC').date() == pd.Timestamp.now(tz='UTC').date()) if isinstance(orders, list) else 0
            if trades_today > 4: c_ord2.error(f"⚠️ Trades Today: {trades_today}")
            else: c_ord2.info(f"⚡ Trades Today: {trades_today}")
            
            order_data = []
            if isinstance(orders, list):
                for o in orders[:5]: 
                    if isinstance(o, dict) and o.get('status') == 'filled':
                        t = o.get('filled_at', '')
                        t_fmt = t[5:16].replace('T', ' ') if len(t) >= 16 else t
                        limit_price, fill_price = float(o.get('limit_price', 0)) if o.get('limit_price') else 0.0, float(o.get('filled_avg_price', 0)) if o.get('filled_avg_price') else 0.0
                        slippage = (((fill_price - limit_price) / limit_price) * 100) if o.get('side') == 'buy' else (((limit_price - fill_price) / limit_price) * 100) if limit_price > 0 and fill_price > 0 else 0.0
                        order_data.append({"Time": t_fmt, "Ticker": o.get('symbol', 'N/A'), "Side": o.get('side', 'N/A').upper(), "Qty": o.get('filled_qty', '0'), "Fill Price": f"${fill_price:.2f}", "Slippage": f"{slippage:+.2f}%" if limit_price > 0 else "N/A (MKT)"})

        if order_data:
            df_orders = pd.DataFrame(order_data)
            def highlight_slippage(val):
                if isinstance(val, str) and "%" in val:
                    num = float(val.replace("%", "").replace("+", ""))
                    if num > 0: return 'color: #ff4b4b' 
                    if num < 0: return 'color: #00ff41' 
                return ''
            def highlight_side(val):
                if val == 'BUY' or val == 'LONG': return 'color: #00ff41; font-weight: bold;'
                if val == 'SELL' or val == 'SHORT': return 'color: #ff4b4b; font-weight: bold;'
                return ''

            styled_df = df_orders.style.map(highlight_slippage, subset=['Slippage']).map(highlight_side, subset=['Side'])
            st.dataframe(styled_df, width="stretch", hide_index=True)
        else: st.caption("No recent filled orders found.")

with tab2:
    st.markdown("### Terminal Output (Last 3000 Lines)")
    if logs:
        log_html = "".join([format_log_line(line) for line in logs[-3000:]])
        st.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
    else: st.write("No logs found.")

with tab3:
    if not hist_df_raw.empty and account:
        current_equity_raw = float(account['equity'])
        metrics = st.session_state.get('global_metrics', {})
        
        # --- FIX: UNIFIED HIT RATE CALCULATION (DB & ALPACA FALLBACK) ---
        if not df_ex.empty:
            hit_rate = len(df_ex[df_ex['Result'] == 'Win']) / len(df_ex) if len(df_ex) > 0 else 0.0
            trade_count = len(df_ex)
        else:
            # Fallback if there are absolutely 0 trades across all databases/brokers
            hit_rate, trade_count = 0.0, 0
        # -----------------------------------------------------------------
        
        scorecard_df = create_scorecard_df(metrics, hit_rate, trade_count)
        inst_score = calculate_institutional_score(metrics)
        valid_cagr = metrics.get("CAGR", 0.0)
        
        dd_df = calculate_drawdown(hist_df_adj) 
        day_stats, monthly_stats = calculate_seasonality(hist_df_adj)
        projection_rate = manual_cagr if use_manual_cagr else valid_cagr
        
        inception_dt, starting_principal = hist_df_raw['timestamp'].min(), hist_df_raw['equity'].iloc[0]
        proj_df = calculate_future_projections(inception_dt, starting_principal, projection_rate)

        col_gauge, col_scorecard = st.columns([1, 2.5])
        
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = inst_score, domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Strategy Grade", 'font': {'size': 20, 'color': '#e0e0e0'}},
                number = {'suffix': "/100", 'font': {'color': '#e0e0e0'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#333"},
                    'bar': {'color': "#00ff41" if inst_score > 80 else "#ffb000"},
                    'bgcolor': "#1e1e1e", 'borderwidth': 2, 'bordercolor': "#333",
                    'steps': [{'range': [0, 50], 'color': 'rgba(255, 75, 75, 0.3)'}, {'range': [50, 80], 'color': 'rgba(255, 176, 0, 0.3)'}, {'range': [80, 100], 'color': 'rgba(0, 255, 65, 0.3)'}],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': inst_score}
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, width='stretch')
            
            if inst_score > 80: st.markdown("<div style='text-align: center; color: #00ff41; font-weight: bold;'>🚀 INSTITUTIONAL GRADE</div>", unsafe_allow_html=True)
            elif inst_score > 50: st.markdown("<div style='text-align: center; color: #ffb000; font-weight: bold;'>⚡ PROFESSIONAL RETAIL</div>", unsafe_allow_html=True)
            else: st.markdown("<div style='text-align: center; color: #ff4b4b; font-weight: bold;'>🎲 DEGEN / RETAIL</div>", unsafe_allow_html=True)

        with col_scorecard:
            st.markdown("### 📊 Metrics Breakdown (Adj. for Deposits)")
            st.dataframe(scorecard_df, width="stretch", hide_index=True, column_config={"METRIC": st.column_config.TextColumn("Metric", width="medium"), "YOURS": st.column_config.TextColumn("Your Bot", width="small"), "BENCHMARK": st.column_config.TextColumn("Target", width="small"), "VERDICT": st.column_config.TextColumn("Verdict", width="small")}, height=280)

        st.divider()

        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            st.markdown(f"### 📈 Real Equity Curve (${current_equity_raw:,.2f})")
            fig_eq = px.area(hist_df_raw, x='timestamp', y='equity')
            fig_eq.update_traces(line_color='#00ff41', fillcolor='rgba(0, 255, 65, 0.1)')
            fig_eq.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None, showlegend=False, height=300, yaxis=dict(range=[hist_df_raw['equity'].min() * 0.95, hist_df_raw['equity'].max() * 1.02], rangemode="normal"))
            st.plotly_chart(fig_eq, width='stretch')

        with col_perf2:
            st.markdown("### 📉 Real Risk (Drawdown)")
            fig_dd = px.area(dd_df, x='timestamp', y='drawdown')
            fig_dd.update_traces(line_color='#ff4b4b', fillcolor='rgba(255, 75, 75, 0.2)')
            fig_dd.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None, showlegend=False, height=300, yaxis=dict(tickformat=".1%"))
            st.plotly_chart(fig_dd, width='stretch')

        st.divider()
        st.subheader("⚔️ Long vs. Short Attribution")
        
        if isinstance(orders, list) and len(orders) > 0:
            long_wins, long_losses, short_wins, short_losses = 0, 0, 0, 0
            for pos in positions:
                if pos['side'] == 'long':
                    if float(pos['unrealized_pl']) > 0: long_wins += 1
                    else: long_losses += 1
                elif pos['side'] == 'short':
                    if float(pos['unrealized_pl']) > 0: short_wins += 1
                    else: short_losses += 1

            total_longs, total_shorts = long_wins + long_losses, short_wins + short_losses
            long_wr = (long_wins / total_longs * 100) if total_longs > 0 else 0
            short_wr = (short_wins / total_shorts * 100) if total_shorts > 0 else 0
            
            c_ls1, c_ls2, _spacer = st.columns([1, 1, 4])
            c_ls1.metric("🟢 Long Win Rate (Active)", f"{long_wr:.1f}%", f"{total_longs} positions", delta_color="off")
            c_ls2.metric("🔴 Short Win Rate (Active)", f"{short_wr:.1f}%", f"{total_shorts} positions", delta_color="off")
            st.caption("*Note: Displays active state. Full historical attribution requires database integration.*")

        st.divider()
        st.markdown("### 🔄 30-Day Rolling Edge (Momentum, Defense & Regime)")
        
        if not roll_df.empty:
            c_roll1, c_roll2 = st.columns(2)
            c_roll3, c_roll4 = st.columns(2)
            c_roll5, c_roll6 = st.columns(2)
            c_roll7, c_roll8 = st.columns(2)

            with c_roll1:
                st.caption("30-Day Rolling Return (%)")
                fig_roll_ret = px.area(roll_df, x='timestamp', y='rolling_return')
                fig_roll_ret.update_traces(line_color='#569cd6', fillcolor='rgba(86, 156, 214, 0.2)')
                fig_roll_ret.add_hline(y=2.0, line_dash="dot", line_color="#00ff41", annotation_text="Pro Target")
                fig_roll_ret.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Breakeven")
                fig_roll_ret.add_hline(y=-2.0, line_dash="dot", line_color="#ff4b4b", annotation_text="Pain Threshold")
                fig_roll_ret.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_roll_ret, width='stretch')

            with c_roll2:
                st.caption("30-Day Rolling Sharpe Ratio")
                fig_roll_shp = px.line(roll_df, x='timestamp', y='rolling_sharpe')
                fig_roll_shp.update_traces(line_color='#c586c0')
                fig_roll_shp.add_hline(y=1.5, line_dash="dot", line_color="#00ff41", annotation_text="Pro Target")
                fig_roll_shp.add_hline(y=0.0, line_dash="dot", line_color="#ff4b4b", annotation_text="Stress Warning")
                fig_roll_shp.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_roll_shp, width='stretch')

            with c_roll3:
                st.caption("30-Day Rolling Max Drawdown (%)")
                fig_roll_dd = px.area(roll_df, x='timestamp', y='rolling_dd')
                fig_roll_dd.update_traces(line_color='#ff4b4b', fillcolor='rgba(255, 75, 75, 0.2)')
                fig_roll_dd.add_hline(y=-2.0, line_dash="dot", line_color="#00ff41", annotation_text="Pro Limit")
                fig_roll_dd.add_hline(y=-5.0, line_dash="dot", line_color="#ffb000", annotation_text="Pain Threshold")
                fig_roll_dd.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_roll_dd, width='stretch')

            with c_roll4:
                st.caption("30-Day Rolling Sortino Ratio")
                fig_roll_srt = px.line(roll_df, x='timestamp', y='rolling_sortino')
                fig_roll_srt.update_traces(line_color='#cca700') 
                fig_roll_srt.add_hline(y=2.0, line_dash="dot", line_color="#00ff41", annotation_text="Pro Target")
                fig_roll_srt.add_hline(y=0.0, line_dash="dot", line_color="#ff4b4b", annotation_text="Stress Warning")
                fig_roll_srt.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_roll_srt, width='stretch')
                
            with c_roll5:
                st.caption("30-Day Rolling Daily Reliability (%)")
                fig_roll_win = px.bar(roll_df, x='timestamp', y='rolling_win_rate')
                fig_roll_win.update_traces(marker_color='#4CAF50', opacity=0.7)
                fig_roll_win.add_hline(y=60.0, line_dash="dot", line_color="#00ff41", annotation_text="Pro Target")
                fig_roll_win.add_hline(y=50.0, line_dash="dash", line_color="white", annotation_text="Breakeven")
                fig_roll_win.add_hline(y=45.0, line_dash="dot", line_color="#ff4b4b", annotation_text="Pain Threshold")
                fig_roll_win.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None, yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig_roll_win, width='stretch')

            with c_roll6:
                st.caption("30-Day Rolling Volatility (Annualized %)")
                fig_roll_vol = px.line(roll_df, x='timestamp', y='rolling_vol')
                fig_roll_vol.update_traces(line_color='#ff9800')
                fig_roll_vol.add_hline(y=15.0, line_dash="dot", line_color="#00ff41", annotation_text="Pro Target")
                fig_roll_vol.add_hline(y=25.0, line_dash="dot", line_color="#ffb000", annotation_text="Stress Warning")
                fig_roll_vol.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_roll_vol, width='stretch')

            with c_roll7:
                st.caption("30-Day Rolling SQN (System Quality)")
                fig_roll_sqn = px.line(roll_df, x='timestamp', y='rolling_sqn')
                fig_roll_sqn.update_traces(line_color='#00ff41')
                fig_roll_sqn.add_hline(y=2.0, line_dash="dot", line_color="#00ff41", annotation_text="Pro Target")
                fig_roll_sqn.add_hline(y=1.0, line_dash="dot", line_color="#ff4b4b", annotation_text="Stress Warning")
                fig_roll_sqn.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_roll_sqn, width='stretch')

            with c_roll8:
                st.caption("30-Day Rolling Ulcer Index (Pain)")
                fig_roll_ulcer = px.area(roll_df, x='timestamp', y='rolling_ulcer')
                fig_roll_ulcer.update_traces(line_color='#e91e63', fillcolor='rgba(233, 30, 99, 0.2)')
                fig_roll_ulcer.add_hline(y=2.0, line_dash="dot", line_color="#00ff41", annotation_text="Pro Target") 
                fig_roll_ulcer.add_hline(y=5.0, line_dash="dot", line_color="#ffb000", annotation_text="Stress Warning") 
                fig_roll_ulcer.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_roll_ulcer, width='stretch')
        else: st.caption("Not enough data yet for 30-Day Rolling metrics.")

        st.divider()
        st.subheader("⚖️ Return Distribution & Temporal Heatmap")
        
        c_dist1, c_dist2 = st.columns(2)
        
        with c_dist1:
            st.markdown("**📊 Daily Return Distribution (Asymmetry Test)**")
            st.caption(f"Skewness: {metrics.get('Skewness', 0):.2f} | Kurtosis: {metrics.get('Kurtosis', 0):.2f} | CVaR: {metrics.get('CVaR (95%)', 0):.2f}%")
            
            active_returns = hist_df_adj[hist_df_adj['daily_return'] != 0]['daily_return'] * 100
            if not active_returns.empty:
                fig_hist = px.histogram(active_returns, nbins=50, color_discrete_sequence=['#569cd6'], marginal="box")
                fig_hist.add_vline(x=0, line_dash="dash", line_color="white")
                fig_hist.update_layout(showlegend=False, height=280, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#cccccc"), xaxis_title="Daily Return (%)", yaxis_title="Frequency")
                st.plotly_chart(fig_hist, width='stretch')

        with c_dist2:
            st.markdown("**🗓️ Seasonality Heatmap (Win Rate %)**")
            st.caption("Darker green indicates high-probability temporal windows.")
            if not hist_df_adj.empty:
                heat_df = hist_df_adj.copy()
                heat_df['timestamp'] = heat_df['timestamp'].dt.tz_convert('America/New_York')
                day_of_week = heat_df['timestamp'].dt.dayofweek
                heat_df.loc[day_of_week == 5, 'timestamp'] -= pd.Timedelta(days=1) 
                heat_df.loc[day_of_week == 6, 'timestamp'] += pd.Timedelta(days=1) 
                
                heat_df['Day'] = heat_df['timestamp'].dt.day_name()
                heat_df['Month'] = heat_df['timestamp'].dt.strftime('%b') 
                heat_df['is_win'] = (heat_df['daily_return'] > 0).astype(int)
                heat_df['is_trade'] = (heat_df['daily_return'] != 0).astype(int)
                
                pivot = heat_df.groupby(['Day', 'Month'])[['is_win', 'is_trade']].sum().reset_index()
                pivot['Win Rate'] = (pivot['is_win'] / pivot['is_trade'] * 100).fillna(0)
                
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                matrix = pivot.pivot(index='Day', columns='Month', values='Win Rate').reindex(index=days_order, columns=months_order)
                
                fig_heat = px.imshow(matrix, text_auto=".0f", color_continuous_scale="Greens", aspect="auto", zmin=0, zmax=100)
                fig_heat.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#cccccc"), coloraxis_showscale=False)
                st.plotly_chart(fig_heat, width='stretch')

        st.divider()
        st.markdown("### ⚖️ Macro Alignment (Rolling Beta & Correlation to SPY)")
        st.caption("Isolates true Alpha. Beta tracks volatility relative to the S&P 500 (1.0 = moves identical to market). Correlation tracks directional grouping.")
        
        start_date_str = hist_df_adj['timestamp'].min().strftime('%Y-%m-%d')
        spy_df = get_historical_spy(start_date_str)
        
        if not spy_df.empty:
            merge_df = hist_df_adj[['timestamp', 'daily_return']].copy()
            merge_df['date_only'] = merge_df['timestamp'].dt.tz_localize(None).dt.floor('D')
            spy_df['date_only'] = spy_df.index
            macro_df = pd.merge(merge_df, spy_df, on='date_only', how='left').fillna(0)
            
            rolling_cov = macro_df['daily_return'].rolling(30).cov(macro_df['spy_return'])
            rolling_spy_var = macro_df['spy_return'].rolling(30).var()
            
            macro_df['rolling_beta'] = (rolling_cov / rolling_spy_var).replace([np.inf, -np.inf], 0).fillna(0)
            macro_df['rolling_corr'] = macro_df['daily_return'].rolling(30).corr(macro_df['spy_return']).fillna(0)
            
            c_mac1, c_mac2 = st.columns(2)
            with c_mac1:
                fig_beta = px.line(macro_df, x='timestamp', y='rolling_beta')
                fig_beta.update_traces(line_color='#c586c0')
                fig_beta.add_hline(y=1.0, line_dash="dot", line_color="#ff4b4b", annotation_text="Market Benchmark (1.0)")
                fig_beta.add_hline(y=0.0, line_dash="dash", line_color="white", annotation_text="Market Neutral (0.0)")
                fig_beta.update_layout(title="30-Day Rolling Beta", margin=dict(l=0, r=0, t=30, b=0), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title="Beta")
                st.plotly_chart(fig_beta, width='stretch')
                
            with c_mac2:
                fig_corr = px.area(macro_df, x='timestamp', y='rolling_corr')
                fig_corr.update_traces(line_color='#569cd6', fillcolor='rgba(86, 156, 214, 0.2)')
                fig_corr.add_hline(y=0.0, line_dash="dash", line_color="white")
                fig_corr.update_layout(title="30-Day Rolling Correlation (R)", margin=dict(l=0, r=0, t=30, b=0), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title="Correlation")
                st.plotly_chart(fig_corr, width='stretch')
                
            latest_beta = macro_df['rolling_beta'].iloc[-1]
            if abs(latest_beta) < 0.3: st.success(f"**Current Posture:** Highly Decoupled (Beta: {latest_beta:.2f}). The system is generating pure uncorrelated Alpha.")
            elif latest_beta >= 0.8: st.warning(f"**Current Posture:** Highly Correlated (Beta: {latest_beta:.2f}). The system is acting essentially as a leveraged SPY ETF.")
            else: st.info(f"**Current Posture:** Moderately Correlated (Beta: {latest_beta:.2f}).")
        else: st.caption("Waiting for SPY historical data to populate macro charts...")

        st.divider()
        
        projection_rate = manual_cagr if use_manual_cagr else valid_cagr
        proj_label = "Manual" if use_manual_cagr else "Adj."
        
        st.markdown(f"### 🔮 Actuals vs. Projections (Based on {proj_label} CAGR: {projection_rate:.1%})")
        st.caption("Tracking live execution against the mathematical baseline to eliminate emotional bias during drawdown cycles.")
        
        if not proj_df.empty:
            c_p1, c_p2 = st.columns([2, 1])
            with c_p1:
                melted_proj = proj_df.melt(id_vars=['Date'], value_vars=['Base (No Deposits)', '+$70/wk', '+$140/wk'], var_name='Scenario', value_name='Projected Value')
                fig_proj = go.Figure()
                color_map = {"Base (No Deposits)": "rgba(86, 156, 214, 0.5)", "+$70/wk": "rgba(197, 134, 192, 0.5)", "+$140/wk": "rgba(0, 255, 65, 0.5)"}
                
                for scenario in ['Base (No Deposits)', '+$70/wk', '+$140/wk']:
                    scenario_data = melted_proj[melted_proj['Scenario'] == scenario]
                    fig_proj.add_trace(go.Scatter(x=scenario_data['Date'], y=scenario_data['Projected Value'], mode='lines', name=f"Proj: {scenario}", line=dict(color=color_map[scenario], width=2, dash='dot')))
                
                actuals_df = hist_df_raw[['timestamp', 'equity']].copy()
                actuals_df['timestamp'] = actuals_df['timestamp'].dt.tz_localize(None)
                
                fig_proj.add_trace(go.Scatter(x=actuals_df['timestamp'], y=actuals_df['equity'], mode='lines', name='Live Equity (Actual)', line=dict(color='#ff9800', width=4)))
                fig_proj.update_layout(margin=dict(l=0, r=0, t=30, b=0), xaxis_title=None, yaxis_title=None, height=400, template="plotly_dark", legend=dict(orientation="h", y=1.1, x=0, title=None), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_proj, width='stretch')
                
            with c_p2:
                final_base, final_base_inflated = proj_df['Base (No Deposits)'].iloc[-1], proj_df['Base (+3% Inflation)'].iloc[-1]
                final_140_nom, final_140_inflated = proj_df['+$140/wk'].iloc[-1], proj_df['+$140/wk (+3% Inflation)'].iloc[-1]
                
                mb1, mb2 = st.columns(2)
                mb1.metric("20-Yr Base Target", f"${final_base:,.0f}", f"{projection_rate:.1%} Rate")
                mb2.metric("20-Yr Base (+3% Infl)", f"${final_base_inflated:,.0f}", "+3.0% Yearly Inflation")
                
                m1, m2 = st.columns(2)
                m1.metric("Max Account (+$140/wk)", f"${final_140_nom:,.0f}")
                m2.metric("Max Account (+3% Infl)", f"${final_140_inflated:,.0f}", "+3.0% Yearly Inflation")

                today_norm = pd.Timestamp.now().normalize()
                display_df = proj_df[proj_df['Date'] >= today_norm].copy()
                
                st.dataframe(display_df, width="stretch", hide_index=True, column_config={"Date": st.column_config.DatetimeColumn(format="YYYY-MM"), "Base (No Deposits)": st.column_config.NumberColumn("Base Target", format="$%.0f"), "Base (+3% Inflation)": st.column_config.NumberColumn("Base (+3% Infl)", format="$%.0f"), "+$70/wk": st.column_config.NumberColumn("+$70/wk", format="$%.0f"), "+$70/wk (+3% Inflation)": None, "+$140/wk": st.column_config.NumberColumn("+$140/wk Target", format="$%.0f"), "+$140/wk (+3% Inflation)": st.column_config.NumberColumn("+$140/wk (+3% Infl)", format="$%.0f")}, height=220)

        st.divider()
        st.markdown("### 🎲 Monte Carlo Risk Simulation (Sequence of Returns)")
        st.caption("Bootstraps your actual historical daily returns to project 500 possible 20-year futures. This simulates 'Sequence of Returns Risk' (what happens if your losses cluster early vs. late). Visualized for the +$140/wk scenario.")
        
        mc_returns = hist_df_adj['daily_return'].dropna().values
        mc_df = run_monte_carlo_simulation(mc_returns, current_equity_raw, weekly_deposit=140, years=20, paths=500)
        
        if not mc_df.empty:
            mc_df_yearly = mc_df.set_index('Date').resample('YE').last().reset_index()
            fig_mc = go.Figure()
            
            fig_mc.add_trace(go.Scatter(x=mc_df_yearly['Date'], y=mc_df_yearly['90th Percentile (Optimistic)'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_mc.add_trace(go.Scatter(x=mc_df_yearly['Date'], y=mc_df_yearly['10th Percentile (Pessimistic)'], mode='lines', fill='tonexty', fillcolor='rgba(0, 255, 65, 0.1)', line=dict(width=0), name='80% Probability Range'))
            fig_mc.add_trace(go.Scatter(x=mc_df_yearly['Date'], y=mc_df_yearly['50th Percentile (Expected)'], mode='lines+markers', line=dict(color='#00ff41', width=3), name='Median Expected Path'))

            fig_mc.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#cccccc"), legend=dict(orientation="h", y=1.1, x=0), yaxis=dict(title="Portfolio Value ($)", gridcolor="#333", zerolinecolor='white'), xaxis=dict(gridcolor="#333"))
            st.plotly_chart(fig_mc, width='stretch')

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("90th Percentile (Optimistic)", f"${mc_df_yearly['90th Percentile (Optimistic)'].iloc[-1]:,.0f}")
            mc2.metric("50th Percentile (Median)", f"${mc_df_yearly['50th Percentile (Expected)'].iloc[-1]:,.0f}")
            mc3.metric("10th Percentile (Pessimistic)", f"${mc_df_yearly['10th Percentile (Pessimistic)'].iloc[-1]:,.0f}")
    else: st.write("No history data available yet.")

with tab5:
    st.markdown("### 🧭 Dynamic Phase Portrait & Vector Flow")
    st.caption("A 2D representation of the market's state machine. Removes time to show cycle structure, momentum flow, and regime probability.")
    
    if not phys_df.empty:
        col_text1, col_plot1 = st.columns([1, 3])
        
        latest_vel, latest_acc, latest_jerk_abs, latest_dfe = phys_df['vel_smooth'].iloc[-1], phys_df['acc_smooth'].iloc[-1], abs(phys_df['jerk_smooth'].iloc[-1]), phys_df['dfe'].iloc[-1]
        
        with col_text1:
            st.markdown("#### 📊 System State")
            if latest_dfe > 3.0:
                st.error(f"**Distance from Eq:** Extreme ({latest_dfe:.2f})")
                st.write("System highly overextended. Mean-reverting vectors suggest a violent snapback.")
            elif latest_dfe > 1.5:
                st.warning(f"**Distance from Eq:** Elevated ({latest_dfe:.2f})")
                st.write("System riding high energy contours. Vulnerable to shocks.")
            else:
                st.success(f"**Distance from Eq:** Stable ({latest_dfe:.2f})")
                st.write("System is near equilibrium. Momentum is compressed.")
                
            st.divider()
            st.markdown("#### 🗺️ Regime Classification")
            if latest_vel > 0 and latest_acc > 0: st.success("**TREND (Top Right)**\n\nExpanding bull market.")
            elif latest_vel > 0 and latest_acc <= 0: st.warning("**DISTRIBUTION (Bottom Right)**\n\nSlowing bull market.")
            elif latest_vel <= 0 and latest_acc < 0: st.error("**PANIC / SHOCK (Bottom Left)**\n\nExpanding bear market.")
            elif latest_vel <= 0 and latest_acc >= 0: st.info("**ACCUMULATION (Top Left)**\n\nSlowing bear market.")
            else: st.write("Transitioning across zero-bound.")
                
        with col_plot1:
            fig_phase = generate_phase_portrait(phys_df)
            if fig_phase: st.plotly_chart(fig_phase, width='stretch')
    else: st.info("Not enough data points for Phase Portrait analysis.")

    st.divider()
    st.markdown("### 🎢 Portfolio Physics & Trajectory Surface")
    X_phys, Y_phys, Z_phys, z_traj, phys_status = generate_proxied_ppo_landscape(phys_df, inference_state, conviction_data)

    if not phys_df.empty and X_phys is not None:
        col_text2, col_plot2 = st.columns([1, 3])
        with col_text2:
            st.markdown("#### 🌊 Market Turbulence Topology")
            st.caption("Maps the trajectory of the portfolio through velocity and volatility.")
            if "HEALTHY" in phys_status:
                st.success(f"**Topology:** {phys_status}")
                st.write("Market movements are fluid and predictable. The agent navigates clean structural waves.")
            elif "CHAOS" in phys_status:
                st.warning(f"**Topology:** {phys_status}")
                st.write("High jerk/volatility detected. The surface is rugged, indicating choppy execution.")
            else:
                st.error(f"**Topology:** {phys_status}")
                st.write("Mode collapse detected. The market has flatlined or data is stalled.")
            st.divider()
            st.markdown("#### 🛰️ The Journey (Orange Line)")
            st.write("The path represents the last 20 periods of portfolio acceleration and velocity.")

        with col_plot2:
            fig_phys = go.Figure()
            fig_phys.add_trace(go.Surface(x=X_phys, y=Y_phys, z=Z_phys, colorscale='YlGnBu_r', opacity=0.8, showscale=False, lighting=dict(ambient=0.4, diffuse=0.9, roughness=0.1, specular=0.2), hoverinfo='none', cmin=0, cmax=1, contours_z=dict(show=True, usecolormap=True, highlightcolor="#fff", project_z=True)))
            recent_phys = phys_df.tail(20) 
            fig_phys.add_trace(go.Scatter3d(x=recent_phys['vel_smooth'], y=recent_phys['jerk_smooth'], z=z_traj, mode='lines+markers', name='Historical Path', customdata=recent_phys['timestamp'].dt.strftime('%Y-%m-%d %H:%M'), marker=dict(size=abs(recent_phys['jerk']) * 8 + 4, color=recent_phys['acceleration'], colorscale='Viridis', opacity=1.0, line=dict(color='white', width=1), colorbar=dict(title="Accel", len=0.5, y=0.2, x=0.9, tickfont={'color': "#cccccc"})), line=dict(color='#ff9800', width=5), hovertemplate='<b>Date</b>: %{customdata}<br><b>Vel Proxy</b>: %{x:.2f}%<br><b>Jerk Proxy</b>: %{y:.2f}%<extra></extra>'))
            fig_phys.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5), xaxis_title='Returns (Velocity)', yaxis_title='Jerk (Volatility)', zaxis_title='Base Conviction', xaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True, zerolinecolor='white'), yaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True, zerolinecolor='white'), zaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True, tickvals=[0, 0.5, 1.0], zerolinecolor='white')), paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#cccccc"), margin=dict(l=0, r=0, t=10, b=0), height=500, showlegend=False, annotations=[dict(showarrow=False, text=f"PHYSICS TOPOLOGY: {phys_status}", xref="paper", yref="paper", x=0.02, y=0.95, xanchor="left", yanchor="top", font=dict(size=14, color="#e91e63", weight="bold"), bgcolor="#1e1e1e")])
            st.plotly_chart(fig_phys, width='stretch')

    st.divider()
    st.markdown("### 🌌 AI Policy Landscape & Feature Space (PCA)")
    X_pca, Y_pca, Z_pca, swarm_data, pca_status = generate_stgnn_pca_landscape(inference_state)

    if X_pca is not None:
        col_text3, col_plot3 = st.columns([1, 3])
        with col_text3:
            st.markdown("#### 🧠 Agent Brain State")
            st.caption("Translating the 30-D STGNN mathematical terrain into actionable logic.")
            if "ALIGNED" in pca_status:
                st.success(f"**Topology:** {pca_status}")
                st.write("Peaks represent regions of the 30-D feature space where the model has high conviction.")
            else:
                st.error(f"**Topology:** {pca_status}")
                st.write("Insufficient tensor data to map the feature space.")
            st.divider()
            st.markdown("#### 🛸 Neural Clustering")
            st.write("Each glowing orb represents an asset mapped by PCA:")
            st.write("- **PCA 1 (X-Axis):** Dominant Macro/Market drift.")
            st.write("- **PCA 2 (Y-Axis):** Asset-specific divergence.")
            st.write("- **Clusters:** Tickers grouped tightly are exhibiting identical structural setups to the neural network.")

        with col_plot3:
            fig_pca = go.Figure()
            fig_pca.add_trace(go.Surface(x=X_pca, y=Y_pca, z=Z_pca, colorscale='YlGnBu_r', opacity=0.8, showscale=False, lighting=dict(ambient=0.4, diffuse=0.9, roughness=0.1, specular=0.2), hoverinfo='none', cmin=0, cmax=1, contours_z=dict(show=True, usecolormap=True, highlightcolor="#fff", project_z=True)))
            if swarm_data:
                for i, ticker in enumerate(swarm_data['tickers']):
                    x_pos, y_pos, z_pos = swarm_data['x'][i], swarm_data['y'][i], swarm_data['z'][i]
                    fig_pca.add_trace(go.Scatter3d(x=[x_pos, x_pos], y=[y_pos, y_pos], z=[0, z_pos], mode='lines', line=dict(color='rgba(255, 255, 255, 0.4)', width=2, dash='dot'), showlegend=False, hoverinfo='skip'))
                    fig_pca.add_trace(go.Scatter3d(x=[x_pos], y=[y_pos], z=[z_pos], mode='markers+text', name=ticker, text=[ticker], textposition="top center", textfont=dict(color="white", size=11, family="Arial Black"), marker=dict(size=10, color=z_pos, colorscale='YlGnBu_r', cmin=0, cmax=1, line=dict(color='white', width=2)), hovertemplate=f'<b>{ticker}</b><br>Conviction: {z_pos:.1%}<br>PCA1: {x_pos:.2f}<br>PCA2: {y_pos:.2f}<extra></extra>'))

            fig_pca.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.6), xaxis_title='PCA 1 (Macro Factor)', yaxis_title='PCA 2 (Asset Factor)', zaxis_title='True Conviction', xaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True, zerolinecolor='white'), yaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True, zerolinecolor='white'), zaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True, tickvals=[0, 0.5, 1.0], zerolinecolor='white')), paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#cccccc"), margin=dict(l=0, r=0, t=10, b=0), height=500, showlegend=False, annotations=[dict(showarrow=False, text=f"STGNN TOPOLOGY: {pca_status}", xref="paper", yref="paper", x=0.02, y=0.95, xanchor="left", yanchor="top", font=dict(size=14, color="#00ff41", weight="bold"), bgcolor="#1e1e1e")])
            st.plotly_chart(fig_pca, width='stretch')

            st.divider()
            st.markdown("#### 🗺️ 2D PCA Topographical View (Distortion-Free Cluster Map)")
            fig_contour = go.Figure()
            fig_contour.add_trace(go.Contour(x=X_pca[0], y=Y_pca[:, 0], z=Z_pca, colorscale='YlGnBu_r', opacity=0.8, contours=dict(showlines=True, coloring='heatmap'), hoverinfo='skip'))
            if swarm_data:
                c_x, c_y, c_z = swarm_data['x'], swarm_data['y'], swarm_data['z']
                c_text = [f"{t}<br>{z:.1%}" for t, z in zip(swarm_data['tickers'], c_z)]
                fig_contour.add_trace(go.Scatter(x=c_x, y=c_y, mode='markers+text', text=c_text, textposition="top center", textfont=dict(color="white", size=10, family="Arial Black"), marker=dict(size=12, color=c_z, colorscale='YlGnBu_r', cmin=0, cmax=1, line=dict(color='white', width=1.5)), name="Live Feature Space", hoverinfo='skip'))
                
            fig_contour.update_layout(xaxis_title='PCA 1 (Macro Factor)', yaxis_title='PCA 2 (Asset Factor)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#cccccc"), margin=dict(l=0, r=0, t=10, b=0), height=450)
            st.plotly_chart(fig_contour, width='stretch')
    else: st.info("Gathering historical Policy Landscape data. Waiting for Daily Inference Agent payload...")

with tab6:
    st.subheader("🧠 Quantum Alpha Model Lifecycle Monitor")
    st.markdown("Real-time alignment tracking between weekend optimization blueprints and live out-of-sample market execution.")

    if model_health:
        if isinstance(model_health, str):
            try: model_health = json.loads(model_health)
            except ValueError: model_health = {}
        sorted_health = sorted(model_health.items(), key=lambda x: 0 if 'DEGRADED' in x[1].get('Status', '') else 1) if isinstance(model_health, dict) else []

        html_output = ""
        for ticker, profile in sorted_health:
            status, base_ir, live_ir, decay, mdd = profile['Status'], float(profile['Base IR']), float(profile['Live IR']), float(profile['Decay']), int(profile['MDD'])
            base_wr, live_wr, base_mdd, live_trades = float(profile.get('Base WR', 0.0)), float(profile.get('Live WR', 0.0)), int(profile.get('Base MDD', 0)), int(profile.get('Trades', 0))

            statusColor = '#00ff41' if 'OPTIMAL' in status else ('#ffb000' if 'STABLE' in status else '#ff4b4b')
            ir_diff = live_ir - base_ir
            
            # --- FIX: Custom Override UI Text ---
            if 'ABORTED' in status: ir_text = f"experiencing a <strong style='color: #ff4b4b;'>Critical Early Failure</strong>. The warmup phase was terminated early due to extreme out-of-sample losses (Live IR: {live_ir:.2f})."
            elif 'QUARANTINED' in status: ir_text = "currently completely suspended."
            elif 'Empirical Override' in status: ir_text = f"an impressive <strong>Live Information Ratio of {live_ir:.2f}</strong>, <span style='color: #00ff41;'>overriding</span> its negative weekend benchmark ({base_ir:.2f})."
            elif live_trades < 5: ir_text = f"currently in a <strong>Warmup Phase ({live_trades}/5 trades)</strong>. Edge decay algorithms will engage once sufficient out-of-sample data is collected against the benchmark IR of {base_ir:.2f}."
            elif live_ir >= base_ir: ir_text = f"an impressive <strong>Live Information Ratio of {live_ir:.2f}</strong>, <span style='color: #00ff41;'>outperforming</span> its weekend benchmark ({base_ir:.2f}) by +{ir_diff:.2f}."
            elif live_ir >= 0: ir_text = f"a <strong>Live Information Ratio of {live_ir:.2f}</strong>. While generating positive alpha, it is <span style='color: #ffb000;'>underperforming</span> its weekend benchmark ({base_ir:.2f}) by {ir_diff:.2f}."
            else: ir_text = f"a negative <strong>Live Information Ratio of {live_ir:.2f}</strong>, <span style='color: #ff4b4b;'>failing</span> to meet its weekend benchmark ({base_ir:.2f}) by a margin of {ir_diff:.2f}."

            if 'Empirical Override' in status: decay_text = "The Empirical Override gate is active. Live reality has superseded the validation fold constraint, protecting empirical alpha."
            elif decay == 0.0 and base_ir <= 0.0: decay_text = "Model is quarantined due to a negative baseline edge. Trading must be disabled."
            elif decay == 1.0 and 'ABORTED' in status: decay_text = "Model execution halted to protect capital."
            elif decay >= 0.70: decay_text = f"The asset decay factor is excellent at <strong>{decay:.2f}</strong>, indicating strong structural alignment with the training blueprint."
            elif decay >= 0.40: decay_text = f"The asset decay factor sits at <strong style='color: #ffb000;'>{decay:.2f}</strong>, showing moderate edge erosion but remaining above the 0.40 throttle threshold."
            else: decay_text = f"Severe edge erosion detected with a decay factor of <strong style='color: #ff4b4b;'>{decay:.2f}</strong> (Critically below the 0.40 threshold), triggering autonomous risk throttling."

            if mdd <= 21: mdd_text = f"Drawdown duration is safely contained at <strong>{mdd} days</strong>."
            elif mdd <= 42: mdd_text = f"Drawdown duration is stretching to <strong style='color: #ffb000;'>{mdd} days</strong>, approaching pain thresholds."
            else: mdd_text = f"Drawdown duration has breached limits at <strong style='color: #ff4b4b;'>{mdd} days</strong>."

            lifecycle = profile.get('Lifecycle', 'Unknown') 

            html_output += f'<div style="margin-bottom: 12px; padding: 15px; border-left: 5px solid {statusColor}; background-color: #1e1e1e; border-radius: 6px;">'
            html_output += f'<strong style="font-size: 1.2em; color: #fff;">{ticker}</strong><span style="background-color: {statusColor}; color: #111; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; margin-left: 10px;">{status}</span>'
            html_output += f'<div style="margin-top: 8px; font-size: 0.9em; color: #aaa;"><strong>Lifecycle Phase:</strong> <span style="color: #fff;">{lifecycle}</span></div>'
            html_output += f'<div style="margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.85em; color: #aaa; background: #2a2a2a; padding: 10px; border-radius: 4px;">'
            html_output += f'<div><strong style="color: #fff;">🏗️ Training Blueprint</strong><br>Base IR: {base_ir:.2f} &nbsp;|&nbsp; Win Rate: {base_wr:.1f}% &nbsp;|&nbsp; MDD: {base_mdd}d</div>'
            html_output += f'<div><strong style="color: #fff;">⚡ Live Execution ({live_trades} Trades)</strong><br>Live IR: {live_ir:.2f} &nbsp;|&nbsp; Win Rate: {live_wr:.1f}% &nbsp;|&nbsp; Decay: {decay:.2f}</div>'
            html_output += f'</div><p style="margin: 10px 0 0 0; font-size: 0.95em; line-height: 1.6; color: #ccc;">The model is {ir_text}<br><br>'
            if live_trades >= 5 or 'Empirical' in status: html_output += f'{decay_text} {mdd_text}'
            html_output += f'</p></div>'

        st.markdown(html_output, unsafe_allow_html=True)
    else: st.info("Awaiting model performance data from the live execution log stream...")

if auto_refresh:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60000, key="mission_control_refresh")