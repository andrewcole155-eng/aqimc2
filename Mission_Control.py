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
import psutil
import requests
import tzdata
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter
import os
from sklearn.decomposition import PCA
import plotly.graph_objects as go

st.set_page_config(
    page_title="AQI Mission Control",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# OPTIMIZATION: Cache data for 60s to prevent Google API rate limits
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
def get_bot_state():
    """Reads the live bot state from the Google Sheets bridge."""
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open("Angel_Bot_Logs")
        
        worksheet = sh.worksheet("Trading_State")
        state_str = worksheet.acell('A1').value
        
        if state_str:
            return json.loads(state_str)
        return {}
    except Exception as e:
        # Fails gracefully if the tab doesn't exist yet or API rate limits hit
        return {}

@st.cache_data(ttl=30)
def get_account_data(_api):
    try:
        account = _api.get_account()._raw
        positions = [p._raw for p in _api.list_positions()]
        
        # FIX: Filter strictly for 'filled' status and expand the limit to 500
        # This gives the FIFO parser a massive backlog of clean execution data
        orders = [o._raw for o in _api.list_orders(status='filled', limit=500, direction='desc')]
        
        return account, positions, orders
    except:
        return None, [], []

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
        # Fetch ALL history first
        history = _api.get_portfolio_history(period='all', timeframe='1D')
        
        # Guard clause if Alpaca returns empty data
        if not history.timestamp: 
            return pd.DataFrame()
            
        df = pd.DataFrame({'timestamp': history.timestamp, 'equity': history.equity})
        
        # FIX: Force strict UTC timezone awareness immediately upon conversion
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        
        # FIX: Ensure the cutoff date is also strictly UTC aware to prevent comparison crashes
        start_date = pd.Timestamp("2025-05-24", tz='UTC')
        df = df[df['timestamp'] >= start_date].copy()
        
        # Sort to ensure calculations are correct
        df = df.sort_values('timestamp')
        
        return df
    except Exception as e:
        # Expose the error to the dashboard so it never fails silently again
        st.error(f"Portfolio History API Error: {e}") 
        return pd.DataFrame()

def parse_latest_run_logic(logs, bot_state=None):
    """
    Parses live bot state from the JSON payload first, falling back to logs for 
    unstructured system telemetry (Neo4j, legacy model health strings).
    """
    if bot_state is None:
        bot_state = {}

    signals = {}
    watchlist = [] 
    neural_conviction = {} 
    model_health = {} 
    last_run_timestamp = None
    last_run_str = "Unknown"
    neo4j_status = "Unknown" 
    
    # 1. PARSE STRUCTURED JSON FROM DAILY INFERENCE AGENT (Primary Truth)
    json_signals = bot_state.get("signals", {})
    action_map = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "CLOSE"}

    for ticker, data in json_signals.items():
        if isinstance(data, dict):
            # Extract structured logic
            raw_action = data.get("action", 0)
            conf_val = data.get("confidence", 0.0) * 100.0  
            sig_text = data.get("signal", "HOLD (Unknown)")
            
            mapped_action = action_map.get(raw_action, "HOLD")
            neural_conviction[ticker] = {"Confidence": conf_val, "Action": mapped_action}
            
            # Format signal string for dashboard
            if "Hold" in sig_text or "Suppressed" in sig_text:
                signals[ticker] = "⏸️ " + sig_text
            elif "Error" in sig_text:
                signals[ticker] = "❌ " + sig_text
            else:
                signals[ticker] = "✅ " + sig_text
                
            # Populate Watchlist based on threshold
            if conf_val > 40.0 and mapped_action != "HOLD":
                tag = "🔥 Screaming Setup" if conf_val > 80.0 else "⚡ High Conviction"
                watchlist.append({"Ticker": ticker, "Conf": f"{conf_val:.1f}%", "Status": tag})

            # --- ARCHITECTURAL FIX: PARSE MODEL HEALTH DIRECTLY FROM JSON ---
            if "base_ir" in data:
                status_clean = "🟢 OPTIMAL" if conf_val >= 60.0 else ("🟡 STABLE" if conf_val >= 50.0 else "🔴 DEGRADED")
                decay_val = data.get("decay", 0.0)
                mdd_val = data.get("mdd_days", 0)
                
                lifecycle_stage = "Unknown"
                if "OPTIMAL" in status_clean:
                    lifecycle_stage = "🟢 ACTIVE (Challenger)" if decay_val > 0.9 else "🟢 ACTIVE (Production)"
                elif "STABLE" in status_clean:
                    lifecycle_stage = "🟡 MATURE (Monitoring)"
                elif "DEGRADED" in status_clean:
                    lifecycle_stage = "🔴 DEPRECATED (Pending Rollback)" if mdd_val > 42 else "🟠 DRIFTING (Requires Retraining)"

                model_health[ticker] = {
                    "Status": status_clean,
                    "Lifecycle": lifecycle_stage,
                    "Base IR": data.get("base_ir", 0.0),
                    "Live IR": data.get("live_ir", 0.0),
                    "Decay": decay_val,
                    "MDD": mdd_val,
                    "Base MDD": data.get("base_mdd_days", 0),
                    "Base WR": data.get("base_wr", 0.0) * 100.0
                }

    # 2. PARSE UNSTRUCTURED LOGS (Only for Neo4j Status now)
    ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})')
    
    for line in reversed(logs):
        if "Successfully connected to Neo4j" in line:
            if neo4j_status == "Unknown": neo4j_status = "🟢 Connected"
        elif "Failed to connect to Neo4j" in line:
            if neo4j_status == "Unknown": neo4j_status = "🔴 Disconnected"
            
        # Retain Model Health parsing (as this is still logged as text by the training/inference agent)
        if "Profile:" in line and "Base IR:" in line:
            try:
                parts = line.split("|")
                ticker_match = re.search(r"(?:🧠)?\s*([A-Z]+)\s+Profile:\s+(.*?)\s*$", parts[0])
                if ticker_match:
                    t_name = ticker_match.group(1)
                    if t_name not in model_health:
                        status_clean = ticker_match.group(2).strip()
                        raw_base_ir = float(parts[1].split(":")[1].strip()) if "Base IR" in parts[1] else 0.0
                        live_ir = float(parts[2].split(":")[1].strip()) if "Live IR" in parts[2] else 0.0
                        decay_val = float(parts[3].split(":")[1].strip()) if "Decay" in parts[3] else 1.0
                        
                        mdd_match = re.search(r"(\d+)d", parts[4]) if len(parts) > 4 else None
                        mdd_val = int(mdd_match.group(1)) if mdd_match else 0
                        
                        lifecycle_stage = "Unknown"
                        if "OPTIMAL" in status_clean:
                            lifecycle_stage = "🟢 ACTIVE (Challenger)" if decay_val > 0.9 else "🟢 ACTIVE (Production)"
                        elif "STABLE" in status_clean:
                            lifecycle_stage = "🟡 MATURE (Monitoring)"
                        elif "DEGRADED" in status_clean:
                            lifecycle_stage = "🔴 DEPRECATED (Pending Rollback)" if mdd_val > 42 else "🟠 DRIFTING (Requires Retraining)"
                                
                        base_mdd_match = re.search(r"(\d+)d", parts[5]) if len(parts) > 5 else None
                        base_wr_match = re.search(r"([\d\.]+)%", parts[6]) if len(parts) > 6 else None
                        
                        model_health[t_name] = {
                            "Status": status_clean,
                            "Lifecycle": lifecycle_stage,
                            "Base IR": raw_base_ir,
                            "Live IR": live_ir,
                            "Decay": decay_val,
                            "MDD": mdd_val,
                            "Base MDD": int(base_mdd_match.group(1)) if base_mdd_match else 0,
                            "Base WR": float(base_wr_match.group(1)) if base_wr_match else 50.0
                        }
            except Exception:
                pass

        if last_run_str == "Unknown":
            match = ts_pattern.search(line)
            if match:
                last_run_str = match.group(1)
                try:
                    last_run_timestamp = datetime.strptime(last_run_str, '%Y-%m-%d %H:%M:%S')
                except:
                    pass

    if last_run_str == "Unknown" and len(logs) > 0:
        last_run_str = "Sheet Stream Live"
        last_run_timestamp = datetime.now()

    if not model_health and 'saved_model_health' in st.session_state:
        model_health = st.session_state['saved_model_health']

    # Final cleanup for explicit labels
    for ticker, data in neural_conviction.items():
        if data["Action"] == "":
            data["Action"] = "HOLD"

    final_conviction = {k: v for k, v in neural_conviction.items() if v["Confidence"] > 0}
    unique_watchlist = {v['Ticker']:v for v in watchlist}.values()
    
    return last_run_str, last_run_timestamp, signals, list(unique_watchlist), final_conviction, model_health, neo4j_status

@st.cache_data(ttl=300)
def get_market_benchmark():
    """Fetches SPY daily return for the Alpha calculation."""
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="2d")
        if len(hist) >= 2:
            return ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
        return 0.0
    except:
        return 0.0

@st.cache_data(ttl=3600)
def get_trade_excursions(_api, orders):
    """
    Parses recent filled orders to find closed 'round-trip' trades.
    Fetches historical data to calculate MAE (Max Adverse Excursion) 
    and MFE (Max Favorable Excursion) for stop-loss optimization.
    """
    if not orders: return pd.DataFrame()
    
    trades = []
    # Sort oldest to newest to build the trade timeline
    filled_orders = sorted([o for o in orders if isinstance(o, dict) and o.get('status') == 'filled'], 
                           key=lambda x: x.get('filled_at', ''))
    
    # Lightweight FIFO matcher
    inventory = {}
    for o in filled_orders:
        sym = o.get('symbol')
        side = o.get('side')
        qty = float(o.get('filled_qty', 0))
        price = float(o.get('filled_avg_price', 0))
        
        try:
            t = pd.to_datetime(o.get('filled_at')).tz_convert('UTC')
        except:
            continue
            
        if sym not in inventory:
            inventory[sym] = {'qty': 0, 'cost': 0, 'entry_time': t, 'side': None}
            
        inv = inventory[sym]
        
        # Open new position
        if inv['qty'] == 0:
            inv['side'] = side
            inv['cost'] = price
            inv['entry_time'] = t
            inv['qty'] += qty
        else:
            # Add to existing position
            if inv['side'] == side:
                inv['cost'] = ((inv['cost'] * inv['qty']) + (price * qty)) / (inv['qty'] + qty)
                inv['qty'] += qty
            # Close/Reduce position -> THIS IS A COMPLETED TRADE
            else:
                closed_qty = min(inv['qty'], qty)
                inv['qty'] -= closed_qty
                
                if closed_qty > 0:
                    trades.append({
                        'Ticker': sym,
                        'Type': 'Long' if inv['side'] == 'buy' else 'Short',
                        'Entry_Time': inv['entry_time'],
                        'Exit_Time': t,
                        'Entry_Price': inv['cost'],
                        'Exit_Price': price,
                    })
                if inv['qty'] == 0:
                    inv['side'] = None

    # Fetch highs/lows for the last 25 closed trades
    recent_trades = trades[-25:]
    excursion_data = []
    
    if not recent_trades: return pd.DataFrame()
    
    # --- ARCHITECTURAL FIX: BATCH DOWNLOAD ---
    start_date = min([t['Entry_Time'] for t in recent_trades]).strftime('%Y-%m-%d')
    end_date = (max([t['Exit_Time'] for t in recent_trades]) + timedelta(days=2)).strftime('%Y-%m-%d')
    tickers = list(set([t['Ticker'] for t in recent_trades]))
    
    try:
        yf_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False)
    except Exception as e:
        return pd.DataFrame() # Fail gracefully if Yahoo Finance blocks the connection

    for t in recent_trades:
        sym = t['Ticker']
        try:
            if len(tickers) == 1:
                df_sym = yf_data
            else:
                df_sym = yf_data[sym]
                
            # Filter to the exact trade window
            trade_mask = (df_sym.index >= t['Entry_Time'].strftime('%Y-%m-%d')) & (df_sym.index <= (t['Exit_Time'] + timedelta(days=1)).strftime('%Y-%m-%d'))
            df_trade = df_sym.loc[trade_mask]
            
            if not df_trade.empty:
                trade_high = float(df_trade['High'].max())
                trade_low = float(df_trade['Low'].min())
                entry = t['Entry_Price']
                exit_p = t['Exit_Price']
                
                if t['Type'] == 'Long':
                    mfe = (trade_high - entry) / entry * 100
                    mae = (entry - trade_low) / entry * 100 
                    pnl = (exit_p - entry) / entry * 100
                else:
                    mfe = (entry - trade_low) / entry * 100
                    mae = (trade_high - entry) / entry * 100
                    pnl = (entry - exit_p) / entry * 100
                    
                t['MFE (%)'] = mfe
                t['MAE (%)'] = -mae 
                t['PnL (%)'] = pnl
                t['Result'] = 'Win' if pnl > 0 else 'Loss'
                excursion_data.append(t)
        except Exception:
            continue
            
    return pd.DataFrame(excursion_data)

def calculate_trade_hit_rate(orders):
    """
    Lightweight FIFO parser to determine the true Trade Hit Rate 
    (Execution Accuracy) from raw Alpaca order history.
    """
    if not orders: return 0.0, 0
    
    trades = []
    # Sort oldest to newest to build the trade timeline accurately
    filled_orders = sorted([o for o in orders if isinstance(o, dict) and o.get('status') == 'filled'], 
                           key=lambda x: x.get('filled_at', ''))
    
    inventory = {}
    for o in filled_orders:
        sym = o.get('symbol')
        side = o.get('side')
        qty = float(o.get('filled_qty', 0))
        price = float(o.get('filled_avg_price', 0))
        
        if sym not in inventory:
            inventory[sym] = {'qty': 0, 'cost': 0, 'side': None}
            
        inv = inventory[sym]
        
        # Open new position
        if inv['qty'] == 0:
            inv['side'] = side
            inv['cost'] = price
            inv['qty'] += qty
        else:
            # Add to existing position (Averaging)
            if inv['side'] == side:
                inv['cost'] = ((inv['cost'] * inv['qty']) + (price * qty)) / (inv['qty'] + qty)
                inv['qty'] += qty
            # Close or reduce position
            else:
                closed_qty = min(inv['qty'], qty)
                inv['qty'] -= closed_qty
                
                if closed_qty > 0:
                    # Determine PnL
                    if inv['side'] == 'buy': # Long
                        pnl = price - inv['cost']
                    else: # Short
                        pnl = inv['cost'] - price
                        
                    trades.append('Win' if pnl > 0 else 'Loss')
                    
                if inv['qty'] == 0:
                    inv['side'] = None
                    
    if not trades: return 0.0, 0
    hit_rate = trades.count('Win') / len(trades)
    return hit_rate, len(trades)

@st.cache_data(ttl=3600)
def get_correlation_matrix(tickers):
    """Generates a 30-day correlation matrix for active positions."""
    if not tickers or len(tickers) < 2: return None
    try:
        df = yf.download(tickers, period="1mo", interval="1d", progress=False)['Close']
        if isinstance(df, pd.Series): return None
        return df.corr()
    except:
        return None

def get_system_telemetry():
    """Fetches local CPU, RAM, and API latency."""
    cpu_pct = psutil.cpu_percent(interval=0.1)
    ram_pct = psutil.virtual_memory().percent
    try:
        start = time.time()
        requests.get("https://api.alpaca.markets/v2/clock", timeout=2)
        latency = int((time.time() - start) * 1000)
    except:
        latency = 999
    return cpu_pct, ram_pct, latency


def calculate_drawdown(df):
    """Calculates Drawdown % and Time Underwater (Recovery Days)."""
    df = df.copy()
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
    
    # Calculate days spent below the high-water mark
    df['is_high'] = df['equity'] >= df['peak']
    # Groups consecutive underwater days and counts them
    df['underwater_days'] = df.groupby(df['is_high'].cumsum()).cumcount()
    
    return df

def calculate_daily_returns(df):
    """Calculates daily percentage change."""
    df = df.copy()
    df['daily_return'] = df['equity'].pct_change() * 100
    # Color logic: Green for positive, Red for negative
    df['color'] = df['daily_return'].apply(lambda x: '#00ff41' if x >= 0 else '#ff4b4b')
    return df

def calculate_seasonality(df):
    """
    Analyzes performance by Day of Week and Month of Year.
    Returns Avg Return and Win Rate for both.
    """
    s_df = df.copy()
    
    # === FIX: Normalize to US Market Time and Correct Midnight Drift ===
    if s_df['timestamp'].dt.tz is None:
        s_df['timestamp'] = s_df['timestamp'].dt.tz_localize('UTC')
        
    # Subtract 12 hours to pull EOD/Midnight timestamps back into the actual US trading session
    s_df['timestamp'] = s_df['timestamp'] - pd.Timedelta(hours=12)
    s_df['timestamp'] = s_df['timestamp'].dt.tz_convert('America/New_York')
    # ===================================================================

    s_df['daily_return'] = s_df['equity'].pct_change() * 100
    s_df['Day'] = s_df['timestamp'].dt.day_name()
    s_df['Month'] = s_df['timestamp'].dt.strftime('%b')
    s_df['Month_Num'] = s_df['timestamp'].dt.month
    
    # 1. Day of Week Stats (Standard Mon-Fri Market Week)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    day_stats = s_df.groupby('Day')['daily_return'].agg(
        Avg_Return='mean',
        Win_Rate=lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ).reindex(day_order)
    
    # 2. Monthly Stats
    monthly_stats = s_df.groupby(['Month_Num', 'Month'])['daily_return'].agg(
        Avg_Return='mean',
        Win_Rate=lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ).reset_index().sort_values('Month_Num').set_index('Month')
    
    return day_stats, monthly_stats

def calculate_advanced_metrics(hist_df):
    """Calculates strict Portfolio Metrics including advanced Institutional metrics."""
    if hist_df.empty: return {}
    
    df = hist_df.copy()
    df['daily_return'] = df['equity'].pct_change()
    returns = df['daily_return'].dropna()
    
    # --- 1. FIXED: Sync Return & Time Elapsed ---
    # Dynamically grab the true start and end dates from the data
    start_date = df['timestamp'].min() 
    current_date = df['timestamp'].max()
    
    # Ensure timezone awareness matches to prevent subtraction errors
    if current_date.tz is None:
        current_date = current_date.tz_localize('UTC')
    if start_date.tz is None:
        start_date = start_date.tz_localize('UTC')
    
    days_active = (current_date - start_date).days
    if days_active < 1: days_active = 1
    years_active = days_active / 365.25
    
    start_equity = df['equity'].iloc[0]
    end_equity = df['equity'].iloc[-1]
    
    cagr = (end_equity / start_equity) ** (1 / years_active) - 1 if years_active > 0 else 0
    
    df['peak'] = df['equity'].cummax()
    max_dd = ((df['equity'] - df['peak']) / df['peak']).min()
    mar = (cagr / abs(max_dd)) if max_dd != 0 else 0

    # --- 2. FIXED: Explicit 4% Risk-Free Rate applied to Sharpe & Sortino ---
    volatility = returns.std() * (252 ** 0.5)
    sharpe = (cagr - 0.04) / volatility if volatility > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * (252 ** 0.5) if not downside_returns.empty else 0
    sortino = (cagr - 0.04) / downside_vol if downside_vol > 0 else 0

    # --- 3. FIXED: Profit Factor mapped to TWR return sums ---
    positive_sum = returns[returns > 0].sum()
    negative_sum = abs(returns[returns < 0].sum())
    profit_factor = (positive_sum / negative_sum) if negative_sum > 0 else float('inf')

    # --- 4. DASHBOARD-SPECIFIC METRICS ---
    df_with_dd = calculate_drawdown(df)
    max_underwater_days = int(df_with_dd['underwater_days'].max()) if 'underwater_days' in df_with_dd.columns else 0
    
    # FIX: Scale the Ulcer Index correctly by multiplying the raw drawdown by 100 before squaring
    ulcer_index = ((df_with_dd['drawdown'] * 100) ** 2).mean() ** 0.5 if 'drawdown' in df_with_dd.columns else 0.0

    if 'benchmark_return' in df.columns:
        active_return = returns - df['benchmark_return']
        tracking_error = active_return.std()
        
        # FIX: Ensure robust scaling for the Information Ratio
        if tracking_error > 1e-9:
            # Annualize the mean active return and tracking error properly
            annualized_active_return = active_return.mean() * 252
            annualized_tracking_error = tracking_error * (252 ** 0.5)
            information_ratio = annualized_active_return / annualized_tracking_error
        else:
            information_ratio = 0.0
    else:
        # If no benchmark is provided, fallback to a standard Sharpe calculation as a proxy for IR
        tracking_error = returns.std()
        if tracking_error > 1e-9:
             annualized_return = returns.mean() * 252
             annualized_tracking_error = tracking_error * (252 ** 0.5)
             information_ratio = annualized_return / annualized_tracking_error
        else:
             information_ratio = 0.0

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
        "CAGR": cagr,
        "Max Drawdown": max_dd,
        "Recovery Time": max_underwater_days,
        "Ulcer Index": ulcer_index,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Information Ratio": information_ratio,
        "MAR Ratio": mar,
        "Profit Factor": profit_factor,
        "Win Rate (Daily)": win_rate,
        "Expectancy": expectancy,
        "SQN": sqn,
        "Omega Ratio": omega_ratio,
        "Skewness": skewness_val,
        "Kurtosis": kurt,
        "CVaR (95%)": cvar_95,
        "Gain-to-Pain": gain_to_pain,
        "Exposure Efficiency": exposure_efficiency
    }

def create_scorecard_df(metrics, hit_rate, trade_count):
    """Formats the simplified Strategy Scorecard."""
    
    data = [
        # --- RETURN ---
        {"METRIC": "CAGR (Account)", "YOURS": f"{metrics.get('CAGR', 0):.1%}", "BENCHMARK": "> 20%", "VERDICT": "🏆 Elite" if metrics.get('CAGR', 0) > 0.2 else "😐 Std"},
        {"METRIC": "MAR Ratio", "YOURS": f"{metrics.get('MAR Ratio', 0):.2f}", "BENCHMARK": "> 1.0", "VERDICT": "🚀 Elite" if metrics.get('MAR Ratio', 0) > 1.0 else "😐 Std"},
        
        # --- RISK ---
        {"METRIC": "Max Drawdown", "YOURS": f"{metrics.get('Max Drawdown', 0):.1%}", "BENCHMARK": "< 15%", "VERDICT": "🛡️ Safe" if abs(metrics.get('Max Drawdown', 0)) < 0.15 else "⚠️ High Risk"},
        {"METRIC": "Recovery Time", "YOURS": f"{metrics.get('Recovery Time', 0)} Days", "BENCHMARK": "< 30 Days", "VERDICT": "⚡ Fast" if metrics.get('Recovery Time', 0) < 30 else "🐢 Slow"},
        {"METRIC": "Sharpe Ratio", "YOURS": f"{metrics.get('Sharpe Ratio', 0):.2f}", "BENCHMARK": "> 1.5", "VERDICT": "🔥 Good" if metrics.get('Sharpe Ratio', 0) > 1.5 else "😐 Std"},
        {"METRIC": "Sortino Ratio", "YOURS": f"{metrics.get('Sortino Ratio', 0):.2f}", "BENCHMARK": "> 2.0", "VERDICT": "💎 Strong" if metrics.get('Sortino Ratio', 0) > 2.0 else "😐 Std"},

        # --- CONSISTENCY ---
        {"METRIC": "Profit Factor", "YOURS": f"{metrics.get('Profit Factor', 0):.2f}", "BENCHMARK": "> 1.5", "VERDICT": "💰 Rich" if metrics.get('Profit Factor', 0) > 1.5 else "😐 Std"},
        {"METRIC": "Daily Reliability", "YOURS": f"{metrics.get('Win Rate (Daily)', 0):.0%}", "BENCHMARK": "50-55%", "VERDICT": "✅ Stable" if metrics.get('Win Rate (Daily)', 0) > 0.5 else "🔻 Low"},
        {"METRIC": "Trade Hit Rate", "YOURS": f"{hit_rate:.0%} ({trade_count} Trades)", "BENCHMARK": "40-50%", "VERDICT": "🎯 Sniper" if hit_rate >= 0.45 else "😐 Std"},
    ]
    return pd.DataFrame(data)

def calculate_institutional_score(metrics):
    """
    Calculates a weighted score (0-100) to rate the strategy's professionalism.
    Focuses on Risk-Adjusted Returns (Sharpe/MAR) over raw gains.
    """
    score = 0
    max_score = 0
    
    # 1. Sharpe Ratio (Weight: 30%) -> Insts love Sharpe > 2.0
    # Score 30 pts if Sharpe >= 2.0, scaled down if lower
    sharpe = metrics.get('Sharpe Ratio', 0)
    score += min(30, (sharpe / 2.0) * 30)
    max_score += 30
    
    # 2. MAR Ratio (Weight: 25%) -> Return / MaxDD > 1.0 is elite
    mar = metrics.get('MAR Ratio', 0)
    score += min(25, (mar / 1.0) * 25)
    max_score += 25
    
    # 3. Max Drawdown (Weight: 25%) -> Penalize heavy drawdowns
    # Full 25 pts if DD < 10%. 0 pts if DD > 30%
    dd = abs(metrics.get('Max Drawdown', 0))
    if dd < 0.10: score += 25
    elif dd < 0.20: score += 15
    elif dd < 0.30: score += 5
    max_score += 25
    
    # 4. Sortino (Weight: 20%) -> Penalize downside volatility
    sortino = metrics.get('Sortino Ratio', 0)
    score += min(20, (sortino / 3.0) * 20)
    max_score += 20
    
    return min(100, score)

def calculate_future_projections(current_equity, target_cagr, weekly_deposits=[0, 70, 140], inflation_rate=0.03):
    """
    Projects equity based on a provided CAGR, alongside scenarios for weekly injections.
    Includes Institutional Intelligence: Principal tracking, Inflation discounting, and Scale Drag.
    """
    today = pd.Timestamp.now().normalize()
    target_dates = []
    
    # A. Monthly: End of month for next 12 months
    for i in range(0, 13): 
        future_date = today + pd.tseries.offsets.MonthEnd(i)
        if future_date < today: 
            future_date = today + pd.tseries.offsets.MonthEnd(i+1)
        target_dates.append(future_date)
        
    # B. Yearly: End of [Current Month] for next 20 YEARS
    current_month_index = today.month 
    for i in range(2, 21): # <-- EXTENDED TO 20 YEARS
        future_year = today.year + i
        future_dt = pd.Timestamp(year=future_year, month=current_month_index, day=1) + pd.tseries.offsets.MonthEnd(0)
        target_dates.append(future_dt)

    target_dates = sorted(list(set(target_dates)))
    
    # Calculate exact weekly rates
    weekly_rate = ((1 + target_cagr) ** (1 / 52.1429)) - 1
    
    projections = []
    for date in target_dates:
        years_future = (date - today).days / 365.25
        weeks_future = (date - today).days / 7
        
        # Base Future Value
        base_fv = current_equity * ((1 + target_cagr) ** years_future)
        
        row = {
            "Date": date,
            "Timeline": "Next 12 Months" if years_future <= 1.05 else "20-Year Vision",
            "Base (No Deposits)": base_fv
        }
        
        for dep in weekly_deposits:
            if dep == 0: continue
            
            # FV of Annuity
            deposit_fv = dep * (((1 + weekly_rate) ** weeks_future - 1) / weekly_rate) if weekly_rate > 0 else dep * weeks_future
            total_fv = base_fv + deposit_fv
            
            # --- THE INTELLIGENCE ---
            # 1. Raw Principal Deposited
            total_principal = current_equity + (dep * weeks_future)
            # 2. Purchasing Power (Discounted by 3% inflation)
            real_value = total_fv / ((1 + inflation_rate) ** years_future)
            
            row[f"+${dep}/wk"] = total_fv
            row[f"+${dep}/wk (Principal)"] = total_principal
            row[f"+${dep}/wk (Real Value)"] = real_value
            
        projections.append(row)
        
    return pd.DataFrame(projections)

@st.cache_data(ttl=86400) # Cache for 24 hours to avoid rate limits
def get_historical_spy(start_date_str):
    """Fetches historical SPY returns to calculate Beta and Correlation."""
    try:
        # Suppress output and fetch from start_date
        spy = yf.download("SPY", start=start_date_str, progress=False)
        
        # Safely extract 'Close' depending on yfinance multi-index vs single-index versions
        if isinstance(spy.columns, pd.MultiIndex):
            close_series = spy['Close']['SPY']
        else:
            close_series = spy['Close']
            
        df = pd.DataFrame({'spy_close': close_series})
        # Strip timezones for robust date-matching later
        df.index = pd.to_datetime(df.index).tz_localize(None).floor('D')
        df['spy_return'] = df['spy_close'].pct_change()
        
        return df[['spy_return']].dropna()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def run_monte_carlo_simulation(historical_returns, starting_equity, weekly_deposit=140, years=20, paths=500):
    """
    Runs a vectorized Bootstrap Monte Carlo simulation.
    Randomly samples actual historical daily returns to build probability cones.
    """
    if len(historical_returns) < 10: return pd.DataFrame()
    
    # 252 trading days in a year
    days = int(years * 252)
    daily_dep = weekly_deposit / 5.0 # Spread the weekly injection across trading days
    
    # Randomly sample historical returns (with replacement) to create the simulation grid
    sim_returns = np.random.choice(historical_returns, size=(paths, days))
    
    # Initialize the equity tracking array
    equity_paths = np.zeros((paths, days + 1))
    equity_paths[:, 0] = starting_equity
    
    # Vectorized walk forward
    for t in range(1, days + 1):
        equity_paths[:, t] = equity_paths[:, t-1] * (1 + sim_returns[:, t-1]) + daily_dep
        
    # Extract the percentiles
    p10 = np.percentile(equity_paths, 10, axis=0)
    p50 = np.percentile(equity_paths, 50, axis=0)
    p90 = np.percentile(equity_paths, 90, axis=0)
    
    # Map back to future dates
    start_date = pd.Timestamp.today().normalize()
    # Approx calendar mapping: (trading day / 252) * 365.25
    dates = [start_date + pd.Timedelta(days=int((i/252)*365.25)) for i in range(days + 1)]
    
    mc_df = pd.DataFrame({
        'Date': dates, 
        '10th Percentile (Pessimistic)': p10, 
        '50th Percentile (Expected)': p50, 
        '90th Percentile (Optimistic)': p90
    })
    return mc_df

def calculate_3d_physics(df):
    """
    Calculates Velocity, Acceleration, and Jerk (The 3rd Derivative).
    Includes Distance-from-Equilibrium (DFE) metric.
    """
    phys_df = df.copy()
    
    # 1. Velocity (Daily Return %)
    phys_df['velocity'] = phys_df['equity'].pct_change() * 100
    
    # 2. Acceleration (Change in Velocity)
    phys_df['acceleration'] = phys_df['velocity'].diff()
    
    # 3. Jerk (Change in Acceleration - The "Whiplash" factor)
    phys_df['jerk'] = phys_df['acceleration'].diff()

    # Smooth slightly to reduce noise
    phys_df['vel_smooth'] = phys_df['velocity'].ewm(span=3, adjust=False).mean()
    phys_df['acc_smooth'] = phys_df['acceleration'].rolling(3).mean()
    phys_df['jerk_smooth'] = phys_df['jerk'].rolling(3).mean()
    
    # --- NEW: Distance from Equilibrium (Euclidean Distance from 0,0,0) ---
    phys_df['dfe'] = np.sqrt(phys_df['vel_smooth']**2 + phys_df['acc_smooth']**2 + phys_df['jerk_smooth']**2)
    
    return phys_df.dropna()

@st.cache_data(ttl=600) 
def generate_proxied_ppo_landscape(phys_df, log_state, conviction_data, grid_size=50):
    """
    Simulates a mathematical PPO Policy Landscape (BUY Probability surface).
    X-axis = Environmental Vector 1 (Proxied by Smooth Velocity)
    Y-axis = Environmental Vector 2 (Proxied by Jerk / Shock)
    Z-axis = Simulated Agent Conviction (modulated by market physics and logs).
    """
    if phys_df.empty: return None, None, None, None, "NO DATA"
    
    # 1. Define Phase Space Bounds based on historical reality
    recent_data = phys_df.tail(20).copy()
    
    x_dim = recent_data['vel_smooth']
    y_dim = recent_data['jerk_smooth']

    v_mean = x_dim.mean() if not x_dim.empty else 0.0
    v_std = x_dim.std() if len(x_dim) > 1 else 0.1
    j_mean = y_dim.mean() if not y_dim.empty else 0.0
    j_std = y_dim.std() if len(y_dim) > 1 else 0.1
    
    # Safety net: prevent grid collapse on zero variance (e.g., brand new bot start)
    if v_std == 0 or np.isnan(v_std): v_std = 0.1
    if j_std == 0 or np.isnan(j_std): j_std = 0.1
    
    # Create the grid points
    x = np.linspace(v_mean - 3*v_std, v_mean + 3*v_std, grid_size)
    y = np.linspace(j_mean - 3*j_std, j_mean + 3*j_std, grid_size)
    X, Y = np.meshgrid(x, y)

    # 2. Mathematical topological shape modulated by current Log State
    Z_base = np.sin(np.sqrt(X**2 + Y**2)) / (np.sqrt(X**2 + Y**2) + 1)
    
    latest_jerk = abs(y_dim.iloc[-1]) if len(y_dim) > 0 else 0
    latest_vel = abs(x_dim.iloc[-1]) if len(x_dim) > 0 else 0
    
    # FIX: Mode collapse should only trigger if the bot is actually dead (no conviction data)
    is_stalled = latest_vel < 0.05 and len(conviction_data) == 0
    
    if is_stalled:
        # MODE COLLAPSE: Flatten surface to 0.5 (total uncertainty)
        Z_topology = np.full((grid_size, grid_size), 0.5)
        Z_static = np.random.normal(0, 0.005, (grid_size, grid_size))
        Z = Z_topology + Z_static
        status_label = "🔴 MODE COLLAPSE"
    elif latest_jerk > 0.5:
        # HIGH CHAOS: Add severe jaggedness, but maintain topological continuity 
        # Generate larger raw noise, then smooth it to create rugged ridges instead of static
        raw_noise = np.random.normal(0, 0.4, (grid_size, grid_size))
        Z_static = gaussian_filter(raw_noise, sigma=0.8)
        
        Z_healthy = Z_base * np.exp(-0.2 * X**2) 
        Z = np.clip(((Z_healthy + Z_static) + 1) / 2, 0.0, 1.0) 
        status_label = "⚡ HIGH CHAOS"
    else:
        # HEALTHY EDGE: Smooth mountains/valleys
        Z_static = np.random.normal(0, 0.02, (grid_size, grid_size)) 
        Z_healthy = np.cos(X) * np.sin(Y) 
        Z = np.clip(((Z_healthy + Z_static) + 1) / 2, 0.1, 0.9)
        status_label = "🟢 HEALTHY EDGE"

    # --- 3. TRUE NEURAL CONVICTION MAPPING (TRAJECTORY) ---
    if conviction_data:
        avg_real_confidence = sum(d["Confidence"] for d in conviction_data.values()) / len(conviction_data)
        avg_real_confidence = avg_real_confidence / 100.0
    else:
        avg_real_confidence = 0.5 # Default if no data is passed

    # Calculate Z-trajectory so it snaps perfectly to the terrain
    x_np = x_dim.to_numpy()
    y_np = y_dim.to_numpy()

    if is_stalled:
        z_traj_np = np.full(len(x_np), 0.52) 
    elif latest_jerk > 0.5:
        Z_base_traj = np.sin(np.sqrt(x_np**2 + y_np**2)) / (np.sqrt(x_np**2 + y_np**2) + 1)
        z_traj_np = np.clip(((Z_base_traj * np.exp(-0.2 * x_np**2)) + 1) / 2, 0.0, 1.0) + 0.05
    else:
        z_traj_np = np.clip(((np.cos(x_np) * np.sin(y_np)) + 1) / 2, 0.1, 0.9) + 0.02

    # 📍 OVERRIDE the very last point with the TRUE bot confidence!
    z_traj_np[-1] = avg_real_confidence + 0.05
    
    # Convert back to pandas Series to keep index alignment for Plotly
    z_traj = pd.Series(z_traj_np, index=recent_data.index)

    return X, Y, Z, z_traj, status_label

@st.cache_data(ttl=600)
def generate_stgnn_pca_landscape(bot_state, grid_size=50):
    """
    Simulates the true mathematical PPO Policy Landscape using Principal Component Analysis.
    X-axis = PCA Component 1 (Primary Feature Variance - usually Macro/Market trend)
    Y-axis = PCA Component 2 (Secondary Variance - usually Asset-specific divergence)
    Z-axis = True Neural Conviction.
    """
    json_signals = bot_state.get("signals", {})
    tickers = []
    features = []
    confidences = []
    
    # 1. Extract the 27-D STGNN tensors from the JSON payload
    for t, data in json_signals.items():
        if isinstance(data, dict):
            state_tensor = data.get("entry_state")
            if state_tensor and len(state_tensor) >= 27:
                tickers.append(t)
                features.append(state_tensor[:27])
                confidences.append(data.get("confidence", 0.0))

    if len(tickers) < 3:
        return None, None, None, None, "🔴 INSUFFICIENT DATA (Requires >= 3 Assets with Tensors)"

    # 2. Standardize the Feature Matrix
    features_np = np.array(features)
    features_scaled = (features_np - np.mean(features_np, axis=0)) / (np.std(features_np, axis=0) + 1e-9)

    # 3. Collapse 27 Dimensions into 2 (PCA)
    pca = PCA(n_components=2)
    components = pca.fit_transform(features_scaled)
    pca1 = components[:, 0]
    pca2 = components[:, 1]
    
    variance_explained = sum(pca.explained_variance_ratio_) * 100

    # 4. Generate the Phase Space Grid
    x_min, x_max = pca1.min() - 1.5, pca1.max() + 1.5
    y_min, y_max = pca2.min() - 1.5, pca2.max() + 1.5
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # 5. Build the Conviction Terrain (Gaussian RBF Surface)
    Z = np.zeros_like(X)
    sigma = 1.0  
    
    for i in range(len(tickers)):
        c = confidences[i]
        Z += c * np.exp(-((X - pca1[i])**2 + (Y - pca2[i])**2) / (2 * sigma**2))

    # Normalize terrain to prevent clipping
    if Z.max() > 0:
        Z = Z / Z.max()
        Z = np.clip(Z, 0.1, 0.95)

    swarm_data = {
        'tickers': tickers,
        'x': pca1,
        'y': pca2,
        'z': confidences
    }

    status_label = f"🟢 PCA ALIGNED (Var Explained: {variance_explained:.1f}%)"

    return X, Y, Z, swarm_data, status_label

@st.cache_data(ttl=600)
def generate_phase_portrait(phys_df, grid_size=20):
    """
    Generates a 2D Phase Portrait (Velocity vs Acceleration) with Vector Flow Field, 
    Energy Contours, and Regime Classification.
    """
    if phys_df.empty: return None

    # Use smoothed data for stability
    v = phys_df['vel_smooth']
    a = phys_df['acc_smooth']
    
    # Grid boundaries
    v_min, v_max = v.min(), v.max()
    a_min, a_max = a.min(), a.max()
    
    # Create Grid for Vector Field
    v_grid = np.linspace(v_min, v_max, grid_size)
    a_grid = np.linspace(a_min, a_max, grid_size)
    V, A = np.meshgrid(v_grid, a_grid)
    
    # --- 1. Vector Flow Field (Expected Motion) ---
    # In a simple harmonic oscillator, dv/dt = a, and da/dt = -k*v. 
    # Market mean-reversion mimics this.
    # U = expected change in Velocity (which is exactly Acceleration)
    # V_dir = expected change in Acceleration (Jerk). We proxy Jerk as mean-reverting (-Velocity).
    U = A 
    V_dir = -V 
    
    # Normalize vectors for plotting
    norm = np.sqrt(U**2 + V_dir**2)
    norm[norm == 0] = 1 # Avoid division by zero
    U_norm = U / norm
    V_norm = V_dir / norm

    # --- 2. Energy Landscape (Potential Field) ---
    # System Energy E = Kinetic (0.5 * v^2) + Potential (0.5 * k * x^2, proxied by a^2)
    Energy = 0.5 * (V**2 + A**2)
    
    # --- 3. Regime Classification Map ---
    # Quadrant 1: +V, +A -> Trend (Expanding Bull)
    # Quadrant 2: -V, +A -> Accumulation (Bearish but slowing down/reversing)
    # Quadrant 3: -V, -A -> Panic / Shock (Expanding Bear)
    # Quadrant 4: +V, -A -> Distribution (Bullish but slowing down/reversing)
    
    # Create the Plotly Figure
    fig = go.Figure()
    
    # A. Add Regime Shading (Background Rectangles)
    fig.add_shape(type="rect", x0=0, y0=0, x1=v_max, y1=a_max, fillcolor="rgba(0, 255, 65, 0.1)", line_width=0, layer="below") # Q1: Trend
    fig.add_shape(type="rect", x0=v_min, y0=0, x1=0, y1=a_max, fillcolor="rgba(86, 156, 214, 0.1)", line_width=0, layer="below") # Q2: Accumulation
    fig.add_shape(type="rect", x0=v_min, y0=a_min, x1=0, y1=0, fillcolor="rgba(255, 75, 75, 0.1)", line_width=0, layer="below") # Q3: Panic
    fig.add_shape(type="rect", x0=0, y0=a_min, x1=v_max, y1=0, fillcolor="rgba(255, 176, 0, 0.1)", line_width=0, layer="below") # Q4: Distribution
    
    # Add Regime Text Labels
    fig.add_annotation(x=v_max*0.5, y=a_max*0.9, text="TREND<br>(Expanding Bull)", showarrow=False, font=dict(color="#00ff41", size=12))
    fig.add_annotation(x=v_min*0.5, y=a_max*0.9, text="ACCUMULATION<br>(Slowing Bear)", showarrow=False, font=dict(color="#569cd6", size=12))
    fig.add_annotation(x=v_min*0.5, y=a_min*0.9, text="PANIC / SHOCK<br>(Expanding Bear)", showarrow=False, font=dict(color="#ff4b4b", size=12))
    fig.add_annotation(x=v_max*0.5, y=a_min*0.9, text="DISTRIBUTION<br>(Slowing Bull)", showarrow=False, font=dict(color="#ffb000", size=12))

    # B. Add Energy Contours
    fig.add_trace(go.Contour(
        x=v_grid, y=a_grid, z=Energy,
        colorscale='Greys_r', opacity=0.3, showscale=False,
        contours=dict(showlines=True, coloring='none'),
        hoverinfo='skip'
    ))

    # C. Add Vector Flow Field (Quiver Plot via annotations)
    # Subsample grid for arrows to prevent clutter
    step = 2
    for i in range(0, grid_size, step):
        for j in range(0, grid_size, step):
            fig.add_annotation(
                x=V[i, j], y=A[i, j],
                ax=V[i, j] - (U_norm[i, j] * 0.1), ay=A[i, j] - (V_norm[i, j] * 0.1),
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
                arrowcolor='rgba(255,255,255,0.2)'
            )

    # D. Add Actual Trajectory (Last 50 periods)
    recent = phys_df.tail(50)
    fig.add_trace(go.Scatter(
        x=recent['vel_smooth'], y=recent['acc_smooth'],
        mode='lines+markers',
        marker=dict(
            size=abs(recent['jerk']) * 5 + 4, 
            color=recent['jerk'], colorscale='Turbo', showscale=True,
            colorbar=dict(title="Jerk (Color)", len=0.5, y=0.5, x=1.05, tickfont={'color': "#cccccc"})
        ),
        line=dict(color='white', width=2),
        name="Trajectory",
        customdata=recent['timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
        hovertemplate='<b>Date</b>: %{customdata}<br><b>Vel</b>: %{x:.2f}%<br><b>Acc</b>: %{y:.2f}%<extra></extra>'
    ))
    
    # E. Highlight Current Position
    fig.add_trace(go.Scatter(
        x=[recent['vel_smooth'].iloc[-1]], y=[recent['acc_smooth'].iloc[-1]],
        mode='markers+text', text=["📍 LIVE"], textposition="top center",
        marker=dict(size=12, color='#00ff41', symbol='diamond', line=dict(color='white', width=2)),
        textfont=dict(color="white", size=14, family="Arial Black"),
        name="Live Position", hoverinfo='skip'
    ))

    # F. Add Equilibrium "Gravity Well" Marker
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(
            size=35, 
            color='rgba(255, 255, 255, 0)', 
            symbol='circle-cross-open', # <-- Changed to a proper crosshair target
            line=dict(color='rgba(255, 255, 255, 0.3)', width=2) # <-- Removed the invalid 'dash' property
        ),
        text=["Equilibrium (0,0)"],
        textposition="bottom right",
        textfont=dict(color="rgba(255, 255, 255, 0.5)", size=11),
        name="Equilibrium",
        hoverinfo='skip'
    ))

    fig.update_layout(
        xaxis_title='Velocity (Returns %)', yaxis_title='Acceleration (Change in Returns)',
        xaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=2, showgrid=False),
        yaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=2, showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#cccccc"),
        margin=dict(l=0, r=0, t=30, b=0), height=500, showlegend=False
    )
    return fig

def calculate_rolling_edge(df, window=30):
    r_df = df.copy()
    r_df['daily_return'] = r_df['equity'].pct_change()
    
    # --- OFFENSIVE METRICS ---
    # Rolling Return
    r_df['rolling_return'] = r_df['equity'].pct_change(periods=window) * 100
    
    # Rolling Sharpe
    roll_mean = r_df['daily_return'].rolling(window).mean()
    roll_std = r_df['daily_return'].rolling(window).std()
    r_df['rolling_sharpe'] = (roll_mean / roll_std) * (252 ** 0.5)
    
    # --- DEFENSIVE METRICS ---
    # Rolling Drawdown
    rolling_peak = r_df['equity'].rolling(window=window, min_periods=1).max()
    r_df['rolling_dd_raw'] = (r_df['equity'] - rolling_peak) / rolling_peak
    r_df['rolling_dd'] = r_df['rolling_dd_raw'] * 100

    # NEW: Rolling Ulcer Index (Quadratic Mean of Drawdowns)
    r_df['rolling_dd_sq'] = r_df['rolling_dd_raw'] ** 2
    r_df['rolling_ulcer'] = (r_df['rolling_dd_sq'].rolling(window).mean()) ** 0.5 * 100

    # Rolling Sortino
    downside_returns = r_df['daily_return'].copy()
    downside_returns[downside_returns > 0] = 0
    roll_downside_std = downside_returns.rolling(window).std()
    
    r_df['rolling_sortino'] = r_df.apply(
        lambda row: 0.0 if roll_downside_std.loc[row.name] == 0 
        else (roll_mean.loc[row.name] / roll_downside_std.loc[row.name]) * (252 ** 0.5), axis=1
    )
    
    # --- CONSISTENCY & REGIME METRICS ---
    # Rolling Win Rate (%)
    r_df['is_win'] = (r_df['daily_return'] > 0).astype(int)
    r_df['rolling_win_rate'] = r_df['is_win'].rolling(window=window).mean() * 100
    
    # Rolling Volatility
    r_df['rolling_vol'] = roll_std * (252 ** 0.5) * 100
    
    # NEW: Rolling SQN (Using active days as a proxy for trade count)
    r_df['rolling_active_days'] = (r_df['daily_return'] != 0).rolling(window).sum()
    r_df['rolling_sqn'] = (r_df['rolling_active_days'] ** 0.5) * (roll_mean / roll_std)
    
    return r_df.dropna(subset=['rolling_return', 'rolling_sharpe', 'rolling_dd'])

def generate_tactical_alerts(roll_df, global_metrics, margin_util, phys_df):
    """Evaluates rolling metrics and reports active autonomous system adjustments aligned with ATR bounds."""
    alerts = []
    
    if roll_df.empty or len(roll_df) < 5:
        return alerts

    latest_sharpe = roll_df['rolling_sharpe'].iloc[-1]
    latest_ulcer = roll_df['rolling_ulcer'].iloc[-1]
    latest_win_rate = roll_df['rolling_win_rate'].iloc[-1]

    # --- 1. SHARPE (Position Sizing) ---
    if pd.notna(latest_sharpe):
        if latest_sharpe < 0.5:
            alerts.append({"level": "error", "icon": "📉", "title": f"Regime Shift: Rolling Sharpe is weak ({latest_sharpe:.2f})", "action": "POSITION SIZING HALVED. The risk-adjusted edge is decaying. Base lot sizes reduced by 50% until Sharpe recovers > 1.0."})
        elif latest_sharpe > 1.5:
            alerts.append({"level": "success", "icon": "🟢", "title": f"Elite Edge: Sharpe is surging ({latest_sharpe:.2f})", "action": "BASE SIZING RESTORED. The regime is highly favorable. System is deploying full-lot sizes."})

    # --- 2. ULCER INDEX (Defense & Monitoring) ---
    if pd.notna(latest_ulcer):
        if latest_ulcer > 4.0:
            alerts.append({"level": "warning", "icon": "🛡️", "title": f"Pain Threshold Reached: Ulcer Index elevated ({latest_ulcer:.2f})", "action": "DEFENSIVE MONITORING ENGAGED. Drawdowns are elevated. Agent continues to rely on baseline 2x ATR stops."})
        elif latest_ulcer < 1.5:
            alerts.append({"level": "success", "icon": "🕊️", "title": f"Smooth Sailing: Low Ulcer Index ({latest_ulcer:.2f})", "action": "EDGE CONFIRMED. Drawdowns are minimal. Trades are operating cleanly within standard ATR boundaries."})

    # --- 3. WIN RATE (Regime Context) ---
    if pd.notna(latest_win_rate):
        if latest_win_rate < 45.0:
            alerts.append({"level": "info", "icon": "✂️", "title": f"Choppy Execution: Win rate dropping ({latest_win_rate:.1f}%)", "action": "MARKET LACKS FOLLOW-THROUGH. Execution probabilities are skewed negatively in this environment."})
        elif latest_win_rate > 55.0:
            alerts.append({"level": "success", "icon": "🏃‍♂️", "title": f"High Hit Rate: Win rate is strong ({latest_win_rate:.1f}%)", "action": "MOMENTUM CONFIRMED. Market is respecting mathematical targets efficiently."})

    # --- 4. MARGIN (Leverage) ---
    if margin_util > 75.0:
        alerts.append({"level": "error", "icon": "🚨", "title": f"Leverage Warning: Margin at {margin_util:.1f}%", "action": "BUYING FROZEN. Leverage limits reached. No new capital will be deployed."})

    # --- 5. MARKET PHYSICS (Tail Risk) ---
    if not phys_df.empty:
        latest_vel = phys_df['vel_smooth'].iloc[-1]
        latest_acc = phys_df['acc_smooth'].iloc[-1]
        latest_dfe = phys_df['dfe'].iloc[-1]
        cvar = global_metrics.get("CVaR (95%)", 0)
        
        # Panic Regime Detected
        if latest_vel <= 0 and latest_acc < 0:
            alerts.append({"level": "error", "icon": "🛡️", "title": "Regime Drift: PANIC / SHOCK", "action": f"Vector field confirms downward acceleration. Expected Shortfall (CVaR) is {cvar:.2f}%. Trading Agent active regime flag synced."})
        # Extreme Stretching
        elif latest_dfe > 2.5:
            alerts.append({"level": "warning", "icon": "⚠️", "title": f"Extreme Phase Stretch (DFE: {latest_dfe:.2f})", "action": "System is highly extended from equilibrium. Mean-reversion shock probability is elevated."})

    return alerts

def transmit_manual_directives(is_halted, manual_sizing_multiplier):
    """
    Translates Human-in-the-Loop Streamlit overrides into a JSON payload.
    Mission Control must NOT autonomously change these values.
    """
    directives = {
        "active_regime": "EMERGENCY_HALT" if is_halted else "STABLE",
        "sizing_multiplier": float(manual_sizing_multiplier)
    }
    
    payload = {"global_directives": directives}
    payload_str = json.dumps(payload, indent=4)
    
    # RATE LIMIT PROTECTION
    if 'last_transmitted_payload' in st.session_state:
        if st.session_state['last_transmitted_payload'] == payload_str:
            return 
            
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open("Angel_Bot_Logs")
        
        try:
            worksheet = sh.worksheet("Overrides")
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sh.add_worksheet(title="Overrides", rows="10", cols="5")
            
        worksheet.update(range_name='A1', values=[[payload_str]])
        st.session_state['last_transmitted_payload'] = payload_str
        
    except Exception as e:
        st.error(f"Agent Comms Failure: Could not write to Google Sheets. {e}")

def format_log_line(line):
    """Formats a single log line to look like VS Code syntax highlighting."""
    # 1. Safety escape for HTML
    clean_line = line.replace("<", "&lt;").replace(">", "&gt;")
    
    # 2. Colorize Timestamps (e.g., 2026-02-07 09:11:52)
    clean_line = re.sub(
        r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})', 
        r'<span class="log-ts">\1</span>', 
        clean_line
    )
    
    # 3. Colorize Tags ([INFO], [ERROR], etc.)
    clean_line = clean_line.replace("[INFO]", '<span class="log-info">[INFO]</span>')
    clean_line = clean_line.replace("[WARNING]", '<span class="log-warn">[WARNING]</span>')
    clean_line = clean_line.replace("[ERROR]", '<span class="log-err">[ERROR]</span>')
    
    # 4. Colorize Tickers (e.g., [AAPL])
    clean_line = re.sub(
        r'\[([A-Z]{2,5})\]', 
        r'<span class="log-ticker">[\1]</span>', 
        clean_line
    )

    # 5. Colorize Neo4j and STGNN/Quantum Features
    clean_line = clean_line.replace("[Neo4j]", '<span class="log-neo4j">[Neo4j]</span>')
    clean_line = clean_line.replace("✨", '<span class="log-stgnn">✨</span>')

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
    
    # Push human directives to bridge
    transmit_manual_directives(sys_halt, sys_sizing)
    
    st.divider()
    st.subheader("🔮 Projection Tuning")
    use_manual_cagr = st.checkbox("Manual CAGR Override")
    manual_cagr = st.slider("Target CAGR %", 0, 100, 25) / 100
    
    if st.button("Force Refresh Now", type="primary"):
        st.cache_data.clear() 
        st.rerun()

# === DASHBOARD LOGIC ===
api = init_alpaca()
if not api: st.stop()

# 1. ACCOUNT OVERVIEW
account, positions, orders = get_account_data(api)

if account:
    # Expanded to 6 columns to fit the Alpha Gauge
    col1, col2, col_alpha, col3, col_var, col4 = st.columns(6) 
    
    equity = float(account['equity'])
    last_equity = float(account['last_equity'])
    buying_power = float(account['buying_power'])
    
    daily_pl_pct = (equity - last_equity) / last_equity * 100
    daily_pl_abs = equity - last_equity
    
    # --- NEW: Alpha Calculation ---
    spy_return = get_market_benchmark()
    daily_alpha = daily_pl_pct - spy_return
    
    # Value at Risk Calculation (Your existing code)
    total_var = sum([abs(float(p['market_value'])) * 0.03 for p in positions]) if positions else 0.0
    var_pct = (total_var / equity) * 100 if equity > 0 else 0.0
    
    col1.metric("Net Liquidity", f"${equity:,.2f}", f"{daily_pl_pct:.2f}%")
    col2.metric("Day P/L", f"${daily_pl_abs:,.2f}")
    
    # --- NEW: Alpha Metric ---
    col_alpha.metric("Daily Alpha (vs SPY)", f"{daily_alpha:+.2f}%", f"SPY: {spy_return:+.2f}%", delta_color="normal")
    
    col3.metric("Buying Power", f"${buying_power:,.2f}")
    col_var.metric("Open Risk (VaR)", f"${total_var:,.2f}", f"-{var_pct:.2f}% Eq", delta_color="inverse")
    
    # Process Logs & JSON State
    logs = read_bot_logs()
    bot_json_state = get_bot_state() # <-- Grab the JSON
    
    # Pass bot_state to the updated parser
    last_run_str, last_run_dt, parsed_signals, watchlist_data, conviction_data, model_health, neo4j_status = parse_latest_run_logic(logs, bot_json_state)

    # --- NEW: WEEKEND PERSISTENCE MEMORY ---
    if conviction_data and len(conviction_data) > 0:
        st.session_state['saved_conviction'] = conviction_data
        st.session_state['saved_signals'] = parsed_signals
        st.session_state['saved_watchlist'] = watchlist_data
    else:
        conviction_data = st.session_state.get('saved_conviction', {})
        parsed_signals = st.session_state.get('saved_signals', {})
        watchlist_data = st.session_state.get('saved_watchlist', [])

    # Cache Model Health separately since it updates via telemetry, not just trade signals
    if model_health and len(model_health) > 0:
        st.session_state['saved_model_health'] = model_health
    else:
        model_health = st.session_state.get('saved_model_health', {})

    # Calculate "Time Since Last Run"
    status_label = "Bot Status"
    status_val = "Unknown"

    if last_run_dt:
        # Streamlit server time vs Log time safety alignment
        diff = datetime.now() - last_run_dt 
        seconds_ago = int(diff.total_seconds())
        minutes_ago = int(seconds_ago / 60)
        
        if minutes_ago < 10:
            status_val = "🟢 Active"
        elif minutes_ago < 60:
            status_val = f"🟡 Idle ({minutes_ago}m)"
        else:
            status_val = f"🔴 Stale ({int(minutes_ago/60)}h)"
    
    col4.metric(status_label, status_val, delta=f"Last Log: {last_run_str}", delta_color="off")
    
    # --- ADDED: BOT HEARTBEAT COUNTDOWN ---
    if status_val == "🟢 Active" and seconds_ago < 300:
        safe_seconds_ago = max(0, seconds_ago) 
        seconds_left = max(0, 300 - safe_seconds_ago)
        progress_val = int(max(0, min(100, (safe_seconds_ago / 300.0) * 100)))
        st.progress(progress_val, text=f"⏳ Next Market Scan in ~{seconds_left}s")

st.divider()

# =====================================================================
# --- GLOBAL DATA PREP FOR ACTION CENTER & TABS ---
# =====================================================================
hist_df_raw = get_portfolio_history(api)
hist_df_adj = hist_df_raw.copy()
roll_df = pd.DataFrame()
phys_df = pd.DataFrame() # <--- ADD THIS LINE HERE

if not hist_df_raw.empty and account:
    # Ensure UTC 
    if hist_df_raw['timestamp'].dt.tz is None:
        hist_df_raw['timestamp'] = hist_df_raw['timestamp'].dt.tz_localize('UTC')
    
    current_equity_raw = float(account['equity'])
    now_ts = pd.Timestamp.now(tz='UTC') 
    live_row = pd.DataFrame([{'timestamp': now_ts, 'equity': current_equity_raw}])
    hist_df_raw = pd.concat([hist_df_raw, live_row], ignore_index=True)
    hist_df_adj = hist_df_raw.copy()

    # =====================================================================
    # --- TIME-WEIGHTED RETURN (TWR) ADJUSTMENT ---
    # Apply deposits using daily return neutralization to prevent base distortion
    # =====================================================================
    deposit_dates = [
        "2026-01-24", "2026-02-12", "2026-02-16", "2026-02-26", 
        "2026-03-04", "2026-03-13", "2026-03-21", "2026-04-09", 
        "2026-04-15", "2026-04-23", "2026-04-29", "2026-05-06",
        "2026-05-14", "2026-05-21", "2026-07-02"
    ]
        
    # 1. Calculate raw returns first
    hist_df_adj['daily_return'] = hist_df_adj['equity'].pct_change()
    hist_df_adj.loc[hist_df_adj.index[0], 'daily_return'] = 0.0

    # 2. Safely neutralize deposit shocks by finding the NEXT valid market day
    for d_date in deposit_dates:
        ts = pd.Timestamp(d_date, tz='UTC')
        
        # Find all recorded market days on or after the deposit date
        future_market_days = hist_df_adj[hist_df_adj['timestamp'] >= ts]
        
        if not future_market_days.empty:
            idx = future_market_days.index[0]
            
            # THE FIX: Calculate the 5-day mean using the days strictly BEFORE the deposit.
            # This stops the massive deposit spike from inflating its own replacement value.
            prev_returns = hist_df_adj.loc[:idx-1, 'daily_return'].dropna().tail(5)
            clean_proxy_return = prev_returns.mean() if not prev_returns.empty else 0.0
            
            # Apply the clean proxy return to the day of the shock
            hist_df_adj.loc[idx, 'daily_return'] = clean_proxy_return

    # 3. Rebuild the normalized equity curve
    # Anchor to your true base to ensure CAGR matches Excel
    true_starting_principal = 3711.11
    hist_df_adj.loc[hist_df_adj.index[0], 'equity'] = true_starting_principal
    hist_df_adj['equity'] = true_starting_principal * (1 + hist_df_adj['daily_return']).cumprod()
    hist_df_adj['equity'] = hist_df_adj['equity'].fillna(true_starting_principal)

    # =====================================================================
    # --- MERGE SPY BENCHMARK FOR INFORMATION RATIO ---
    # =====================================================================
    start_date_str = hist_df_adj['timestamp'].min().strftime('%Y-%m-%d')
    spy_df = get_historical_spy(start_date_str)
    
    if not spy_df.empty:
        # Create a timezone-naive date key for reliable merging
        hist_df_adj['date_only'] = hist_df_adj['timestamp'].dt.tz_localize(None).dt.floor('D')
        spy_df['date_only'] = spy_df.index
        
        hist_df_adj = pd.merge(hist_df_adj, spy_df, on='date_only', how='left')
        hist_df_adj['benchmark_return'] = hist_df_adj['spy_return'].fillna(0.0)
        hist_df_adj.drop(columns=['date_only', 'spy_return'], inplace=True)

    # =====================================================================
    # --- PRE-CALCULATE METRICS ---
    # =====================================================================
    st.session_state['global_metrics'] = calculate_advanced_metrics(hist_df_adj)
    roll_df = calculate_rolling_edge(hist_df_adj, window=30)
    phys_df = calculate_3d_physics(hist_df_adj)

# =====================================================================
# --- TACTICAL ACTION CENTER (SELF-HEALING ARCHITECTURE) ---
# =====================================================================
maint_margin = float(account.get('maintenance_margin', 0)) if account else 0.0
equity_val = float(account['equity']) if account else 0.0
margin_util = (maint_margin / equity_val * 100) if equity_val > 0 else 0.0

# =====================================================================
# 2. MAIN TABS
# =====================================================================
tab1, tab2, tab3, tab5, tab6 = st.tabs([
    "🧠 Bot Logic & Positions", 
    "📜 Raw Logs", 
    "📈 Real Performance", 
    "🌌 Phase Space",
    "🧬 Model Lifecycle"
])

with tab1:
    # --- 1. MARKET PULSE ---
    avg_market_move = 0.0
    if positions:
        avg_market_move = sum([float(p['unrealized_plpc']) for p in positions]) * 100
        sentiment_score = max(0.0, min(1.0, 0.5 + (avg_market_move / 5)))
    else:
        sentiment_score = 0.5

    st.markdown("### 🌡️ Market Pulse")
    s_col1, s_col2, s_col3 = st.columns([3, 1, 6])
    with s_col1:
        st.progress(int(max(0, min(100, sentiment_score * 100))))
    with s_col2:
        if avg_market_move > 0.5: st.success("BULLISH")
        elif avg_market_move < -0.5: st.error("BEARISH")
        else: st.warning("NEUTRAL")

    # =====================================================================
    # --- MACRO CALENDAR & SYSTEM POSTURE ---
    # =====================================================================
    st.markdown("#### 📅 Tactical Macro Briefing & System Posture")
    
    import xml.etree.ElementTree as ET
    
    @st.cache_data(ttl=3600)
    def fetch_macro_calendar_dashboard():
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        headers = {'User-Agent': 'Mozilla/5.0'}
        events_list = []
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            for event in root.findall('event'):
                country = event.find('country').text if event.find('country') is not None else ''
                impact = event.find('impact').text if event.find('impact') is not None else ''
                title = event.find('title').text if event.find('title') is not None else ''
                date_str = event.find('date').text if event.find('date') is not None else ''
                time_str = event.find('time').text if event.find('time') is not None else ''
                
                if country in ['USD', 'AUD'] and time_str and time_str.lower() != 'all day':
                    title_lower = title.lower()
                    is_high_impact = impact == 'High'
                    has_keyword = any(keyword in title_lower for keyword in ['cpi', 'fomc', 'fed', 'payroll', 'nfp', 'inflation', 'rba', 'retail sales', 'wage price', 'interest rate', 'rate decision'])
                    
                    if is_high_impact or has_keyword:
                        try:
                            event_dt_str = f"{date_str} {time_str.replace('am', 'AM').replace('pm', 'PM')}"
                            event_dt_est = datetime.strptime(event_dt_str, "%m-%d-%Y %I:%M%p")
                            event_dt_est = pytz.timezone('US/Eastern').localize(event_dt_est)
                            
                            brisbane_tz = pytz.timezone('Australia/Brisbane')
                            event_dt_bne = event_dt_est.astimezone(brisbane_tz)
                            
                            now_est = datetime.now(pytz.timezone('US/Eastern'))
                            hours_until = (event_dt_est - now_est).total_seconds() / 3600.0
                            
                            if hours_until > -1.0: 
                                is_critical = any(kw in title_lower for kw in ['cpi', 'inflation', 'fomc', 'fed ', 'rba', 'interest rate', 'rate decision'])
                                
                                if impact == 'High': severity_label = '🔴 High'
                                elif impact == 'Medium': severity_label = '🟡 Medium'
                                elif impact == 'Low': severity_label = '🟢 Low'
                                else: severity_label = '⚪ None'
                                
                                events_list.append({
                                    "Event": f"{country}: {title}",
                                    "Severity": severity_label,
                                    "Time (AEST)": event_dt_bne.strftime('%a %I:%M %p'),
                                    "Hours Until": hours_until,
                                    "Critical": is_critical
                                })
                        except Exception:
                            continue
            return events_list
        except Exception as e:
            st.cache_data.clear()
            st.warning(f"Macro feed disconnected: {e}")
            return None

    upcoming_macro = fetch_macro_calendar_dashboard()
    macro_frozen = False
    
    if upcoming_macro:
        active_events = [e for e in upcoming_macro if e["Hours Until"] >= -0.5]
        
        if active_events:
            active_events.sort(key=lambda x: x["Hours Until"])
            next_event = active_events[0]
            
            hrs = next_event["Hours Until"]
            pre_buffer = 1.0 if next_event["Critical"] else 0.5
            
            c_mac1, c_mac2 = st.columns([1, 1])
            
            with c_mac1:
                if next_event["Critical"]:
                    st.error(f"**Next Event:** {next_event['Event']} ({next_event['Time (AEST)']})") 
                else:
                    st.warning(f"**Next Event:** {next_event['Event']} ({next_event['Time (AEST)']})") 
                    
            with c_mac2:
                if hrs <= pre_buffer and hrs >= -0.5:
                    macro_frozen = True
                    if next_event["Critical"]:
                        st.error("🚨 **System Posture:** THE STRADDLE ENGAGED (Entries Frozen)")
                    else:
                        st.warning("⚠️ **System Posture:** SAFETY LOCK ENGAGED (Entries Frozen)")
                else:
                    st.success(f"🟢 **System Posture:** STANDARD TRAIL & TARGETS (T-{hrs:.1f} hours to lock)")
                    
            with st.expander("View Full Weekly Calendar"):
                st.dataframe(
                    pd.DataFrame(upcoming_macro).drop(columns=['Critical']), 
                    width='stretch', 
                    hide_index=True
                )
        else:
            st.success("🟢 **System Posture:** STANDARD TRAIL & TARGETS")
            st.info("No immediate Tier-1 Macro Events pending.")
    else:
        st.success("🟢 **System Posture:** STANDARD TRAIL & TARGETS")
        st.info("No critical Tier-1 Macro Events scheduled for USD/AUD for the remainder of the week.")

    st.divider()

    # Evaluate alerts
    alerts = generate_tactical_alerts(roll_df, st.session_state.get('global_metrics', {}), margin_util, phys_df)

    if alerts:
        st.markdown("### ⚡ Active System Overrides")
        for alert in alerts:
            msg = f"**{alert['title']}** — {alert['action']}"
            if alert['level'] == "error":
                st.error(f"{alert['icon']} {msg}")
            elif alert['level'] == "warning":
                st.warning(f"{alert['icon']} {msg}")
            elif alert['level'] == "success": 
                st.success(f"{alert['icon']} {msg}") 
            else:
                st.info(f"{alert['icon']} {msg}")
    st.divider()

# --- PENDING / STUCK ORDER ALERTS ---
if isinstance(orders, list):
    pending_orders = [o for o in orders if isinstance(o, dict) and o.get('status') in ['new', 'accepted', 'partially_filled', 'pending_new']]
    for po in pending_orders:
        created_at = po.get('created_at')
        if created_at:
            try:
                created_dt = pd.to_datetime(created_at).tz_convert('UTC')
                now_dt = pd.Timestamp.now(tz='UTC')
                seconds_open = max(0, (now_dt - created_dt).total_seconds())
                
                side_str = po.get('side', 'UNKNOWN').upper()
                qty_str = po.get('qty', '?')
                sym_str = po.get('symbol', '?')
                
                if seconds_open > 60:
                    st.error(f"⚠️ **Execution Alert:** {side_str} order for {qty_str} {sym_str} has been pending for {int(seconds_open)}s! High slippage risk.")
                else:
                    st.info(f"🔄 **Transmitting:** {side_str} {qty_str} {sym_str} (Routing to market: {int(seconds_open)}s ago)")
            except Exception:
                pass
        
        # Calculate Average IRs across the Swarm
        if model_health:
            valid_models = [m for m in model_health.values() if 'Live IR' in m and 'Base IR' in m]
            if valid_models:
                avg_base_ir = sum(float(m['Base IR']) for m in valid_models) / len(valid_models)
                avg_live_ir = sum(float(m['Live IR']) for m in valid_models) / len(valid_models)
                ir_div = avg_live_ir - avg_base_ir
            else:
                avg_base_ir, avg_live_ir, ir_div = 0.0, 0.0, 0.0
        else:
            avg_base_ir, avg_live_ir, ir_div = 0.0, 0.0, 0.0
            
        # Extract Ulcer Index from cached global metrics
        current_ulcer = st.session_state.get('global_metrics', {}).get('Ulcer Index', 0.0)
        
        gl1, gl2, gl3 = st.columns(3)
        gl1.metric("Swarm Benchmark (Base IR)", f"{avg_base_ir:.2f}")
        gl2.metric("Swarm Reality (Live IR)", f"{avg_live_ir:.2f}", f"{ir_div:+.2f} Divergence", delta_color="inverse" if ir_div < 0 else "normal")
        gl3.metric("System Pain (Ulcer Index)", f"{current_ulcer:.2f}", "Threshold: > 4.0", delta_color="inverse" if current_ulcer > 4.0 else "normal")
    # ----------------------------------------------------

    # --- UPGRADED: CAPITAL DEPLOYMENT STATES ---
    st.markdown("#### 🔋 Capital Deployment Status")
    
    # Calculate exactly how much cash is locked in positions
    active_capital = sum([abs(float(p['market_value'])) for p in positions]) if positions else 0.0
    cash_capital = equity_val - active_capital  # <-- FIX: Standardized to safe equity_val
    total_capital = active_capital + cash_capital
    
    # Calculate percentages
    active_pct = (active_capital / total_capital * 100) if total_capital > 0 else 0
    cash_pct = (cash_capital / total_capital * 100) if total_capital > 0 else 100
    
    bot_states = extract_bot_states(logs)
    
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("💼 Active Capital", f"${active_capital:,.2f}", f"{active_pct:.1f}% Deployed", delta_color="off")
    sc2.metric("💵 Dry Powder", f"${cash_capital:,.2f}", f"{cash_pct:.1f}% Cash", delta_color="off")
    
    # FIX: Hardcoded the ticker list so it doesn't look for the missing 'config' variable
    monitored_tickers = ['IONQ', 'KO', 'OXY', 'BAC', 'GM', 'PFE', 'PYPL','FCX','SOFI','T','F','CCL']
    sc3.metric("🤖 Active Agents", f"{len(positions)} / {len(monitored_tickers)}")

    # --- ADDED: NEURAL SKEW / MACRO BIAS ---
    if conviction_data:
        long_count = sum(1 for d in conviction_data.values() if d.get("Action") == "LONG")
        short_count = sum(1 for d in conviction_data.values() if d.get("Action") == "SHORT")
        hold_count = len(conviction_data) - long_count - short_count
        
        st.markdown("#### ⚖️ Bot Macro Bias (Neural Skew)")
        # Normalize for progress bar (0.0 to 1.0)
        total_signals = len(conviction_data)
        skew_val = (long_count + (hold_count * 0.5)) / total_signals if total_signals > 0 else 0.5
        
        st.progress(int(max(0, min(100, skew_val * 100))))
        b1, b2, b3 = st.columns(3)
        b1.caption(f"🟢 Long Bias: {long_count}")
        b2.caption(f"⚪ Neutral/Hold: {hold_count}")
        b3.caption(f"🔴 Short Bias: {short_count}")

    st.divider()

    # --- 2. NEURAL CONVICTION RADAR ---
    st.subheader("🧠 Neural Conviction Levels")
    if conviction_data:
        # Convert nested dictionary to flat DataFrame
        flat_data = [
            {"Ticker": t, "Confidence": d["Confidence"], "Action": d["Action"]} 
            for t, d in conviction_data.items()
        ]
        df_conv = pd.DataFrame(flat_data)
        df_conv = df_conv.sort_values(by='Confidence', ascending=False)
        
        # Create Chart text combining Action and Confidence
        df_conv['Chart_Text'] = df_conv.apply(lambda row: f"{row['Action']}<br>{row['Confidence']:.1f}%" if row['Action'] else f"{row['Confidence']:.1f}%", axis=1)

        fig_conf = px.bar(
            df_conv, 
            x='Ticker', 
            y='Confidence', 
            color='Confidence',
            color_continuous_scale=['#4a1c1c', '#ffb000', '#00ff41'], 
            range_y=[0, 100],
            text='Chart_Text' # <--- This puts the Action State on the bar
        )
        
        fig_conf.update_traces(textposition='inside', textfont_size=14, textfont_color='white')
        
        fig_conf.update_layout(
            height=150, 
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis_title=None, 
            yaxis_title="Confidence %",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#cccccc'},
            yaxis=dict(showgrid=True, gridcolor='#333'),
            xaxis=dict(showgrid=False, categoryorder='total descending') 
        )
        st.plotly_chart(fig_conf, width='stretch')
    else:
        st.info("Waiting for first model run to populate conviction data...")

    st.divider()

    # --- 3. MAIN COLUMNS ---
    c1, c2 = st.columns([3, 4])
    
    with c1:
        # --- UPGRADED: 4 Tabs ---
        tab_wl, tab_log, tab_health, tab_edge = st.tabs(["🔭 Watchlist", "📝 Decisions", "🖥️ Risk & Telemetry", "🔪 Execution & Edge"])
        
        with tab_wl:
            if watchlist_data:
                wl_df = pd.DataFrame(watchlist_data)
                st.dataframe(wl_df, width='stretch', hide_index=True)
            else:
                st.caption("No high-confidence setups detected yet.")

        with tab_log:
            if parsed_signals:
                sig_df = pd.DataFrame(list(parsed_signals.items()), columns=["Ticker", "Decision"])
                st.dataframe(sig_df, width='stretch', hide_index=True)
            else:
                st.info("No signals parsed from recent logs.")
                
        with tab_health:
            st.markdown("#### Server & API Telemetry")
            cpu, ram, ping = get_system_telemetry()
            
            t1, t2, t3 = st.columns(3)
            t1.metric("CPU Load", f"{cpu}%", delta="High" if cpu > 80 else "Normal", delta_color="inverse")
            t2.metric("RAM Util", f"{ram}%", delta="High" if ram > 85 else "Normal", delta_color="inverse")
            t3.metric("API Latency", f"{ping}ms", delta="Lag" if ping > 300 else "Fast", delta_color="inverse")

            st.divider()
            
            st.divider()
            
            st.markdown("#### Margin Distance")
            maint_margin = float(account.get('maintenance_margin', 0)) if account else 0.0
            
            # FIX: Standardized to safe equity_val to prevent NameError
            margin_util = (maint_margin / equity_val * 100) if equity_val > 0 else 0.0
            
            st.progress(int(max(0, min(100, margin_util))), text=f"Margin Capacity Used: {margin_util:.1f}%")
            if margin_util > 80:
                st.error("⚠️ CRITICAL: Approaching Maintenance Margin Call!")

            st.divider()

            st.markdown("#### Active Position Correlation")
            if positions and len(positions) > 1:
                active_tickers = [p['symbol'] for p in positions]
                corr_matrix = get_correlation_matrix(active_tickers)
                
                if corr_matrix is not None:
                    fig_corr = px.imshow(
                        corr_matrix, 
                        text_auto=".2f", 
                        color_continuous_scale="RdBu_r", 
                        zmin=-1, zmax=1
                    )
                    fig_corr.update_layout(
                        height=280, 
                        margin=dict(l=0, r=0, t=10, b=0), 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        font={'color': '#cccccc'}
                    )
                    st.plotly_chart(fig_corr, width='stretch')
            else:
                st.caption("Need at least 2 active positions to plot correlation.")

        # --- NEW TAB: EXECUTION & EDGE ---
        with tab_edge:
            st.markdown("#### ⚖️ Edge Quality")
            
            # Use session_state to prevent NameError on first load
            global_metrics = st.session_state.get('global_metrics', {})
            sqn_val = global_metrics.get('SQN', 0)
            ulcer_val = global_metrics.get('Ulcer Index', 0)
            
            e1, e2 = st.columns(2)
            st.markdown("#### 🎯 Excursion Analysis (MAE vs MFE)")
            st.caption("Scatter plot of recent closed trades. Identifies if stops are too tight or winners are choked.")
            
            df_ex = get_trade_excursions(api, orders)
            
            if not df_ex.empty:
                # 1. CREATE the figure first (with marginal histograms)
                fig_ex = px.scatter(
                    df_ex, x="MAE (%)", y="MFE (%)", color="Result",
                    marginal_x="histogram", marginal_y="histogram", 
                    hover_data=["Ticker", "PnL (%)", "Type"],
                    color_discrete_map={"Win": "#00ff41", "Loss": "#ff4b4b"}
                )
                
                # 2. Add your crosshairs for standard Stop Loss / Take Profit boundaries
                fig_ex.add_vline(x=-2.0, line_dash="dash", line_color="red", annotation_text="Hard Stop (-2%)", annotation_position="top right")
                fig_ex.add_hline(y=4.0, line_dash="dash", line_color="green", annotation_text="Standard TP (+4%)", annotation_position="bottom right")
                
                # 3. UPDATE the traces (targeting ONLY the scatter points)
                fig_ex.update_traces(
                    selector=dict(type='scatter'), 
                    marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey'))
                )
                
                # 4. Apply your layout styling
                fig_ex.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#cccccc'},
                    xaxis=dict(title="Max Adverse Excursion (Pain %)", showgrid=True, gridcolor='#333', zerolinecolor='white'),
                    yaxis=dict(title="Max Favorable (Gain %)", showgrid=True, gridcolor='#333', zerolinecolor='white')
                )
                st.plotly_chart(fig_ex, width='stretch')
            else:
                st.info("Gathering excursion data. Close more trades to populate scatter plot.")

    with c2:
        st.subheader("💼 Capital & Active Portfolio")
        
        # --- UPGRADED: CAPITAL ALLOCATION DONUT CHART ---
        allocation_data = [{"Asset": "CASH", "Value": cash_capital}]
        for p in positions:
            allocation_data.append({"Asset": p['symbol'], "Value": abs(float(p['market_value']))})
        
        if allocation_data:
            fig_alloc = px.pie(
                pd.DataFrame(allocation_data), values='Value', names='Asset', hole=0.65,
                color_discrete_sequence=['#2d2d2d'] + px.colors.qualitative.Pastel
            )
            fig_alloc.update_layout(
                margin=dict(l=0, r=0, t=10, b=10), height=220,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#cccccc'}, showlegend=True,
                legend=dict(orientation="v", yanchor="auto", y=0.5, xanchor="left", x=1.0)
            )
            
            # FIX: Standardized to safe equity_val
            fig_alloc.add_annotation(text=f"Total Eq<br>${equity_val:,.0f}", x=0.5, y=0.5, font_size=14, showarrow=False)
            st.plotly_chart(fig_alloc, width='stretch')
            
            # --- ADDED: NEXT SLOT DEPLOYMENT ESTIMATE ---
            monitored_tickers = ['IONQ', 'KO', 'OXY', 'BAC', 'GM', 'PFE', 'PYPL', 'FCX','SOFI','T','F','CCL']
            
            # FIX: Standardized to safe equity_val
            est_slot_size = equity_val / len(monitored_tickers) if len(monitored_tickers) > 0 else 0.0
            
            st.caption(f"🤖 **Bot Pre-Auth:** Estimated next trade size is **~${est_slot_size:,.2f}** per signal.")
            
            # --- ADDED: SECTOR / INDEX EXPOSURE ---
            ASSET_INDEX_MAP = {
                'IONQ': 'Tech/Quantum', 'KO': 'Consumer Defensive', 'OXY': 'Energy',
                'BAC': 'Financials', 'GM': 'Consumer Cyclical', 'PFE': 'Healthcare',
                'PYPL': 'Financials', 'FCX': 'Basic Materials', 'SOFI': 'Financials',
                'T': 'Communication Services', 'F': 'Consumer Cyclical', 'CCL': 'Consumer Cyclical'
            }
            sector_data = {}

            for p in positions:
                sec = ASSET_INDEX_MAP.get(p['symbol'], 'Other')
                sector_data[sec] = sector_data.get(sec, 0) + abs(float(p['market_value']))
            
            if sector_data:
                df_sec = pd.DataFrame(list(sector_data.items()), columns=['Index', 'Exposure']).sort_values('Exposure', ascending=True)
                fig_sec = px.bar(df_sec, x='Exposure', y='Index', orientation='h', text_auto='$.0f')
                fig_sec.update_traces(marker_color='#569cd6', textposition='inside')
                fig_sec.update_layout(
                    height=120 + (len(df_sec) * 20), margin=dict(l=0, r=0, t=25, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#cccccc'}, xaxis_visible=False,
                    title=dict(text="Risk by Mapped Index", font=dict(size=14))
                )
                st.plotly_chart(fig_sec, width='stretch')

        # --- UPGRADED PORTFOLIO TABLE ---
        if positions:
            pos_data = []
            for p in positions:
                sym = p['symbol']
                side = p['side'].lower()
                entry = float(p['avg_entry_price'])
                current = float(p['current_price'])
                qty = abs(float(p['qty']))
                
                # Calculate Invested Amount
                invested_amt = entry * qty
                
                # Calculate Journey to TP (0.0 to 1.0) - Exact match to Trading Agent ATR logic
                estimated_atr = 0.03 # Fallback aligning with trading agent
                dynamic_sl = estimated_atr * 2.0
                dynamic_tp = estimated_atr * 3.0
                
                conf_sl = 0.02
                conf_tp = 0.04
                
                active_sl = min(dynamic_sl, conf_sl * 1.5) 
                active_tp = max(dynamic_tp, conf_tp)
                
                if side == 'long':
                    sl, tp = entry * (1 - active_sl), entry * (1 + active_tp)
                    progress = max(0.0, min(1.0, (current - sl) / (tp - sl)))
                else:
                    sl, tp = entry * (1 + active_sl), entry * (1 - active_tp)
                    progress = max(0.0, min(1.0, (sl - current) / (sl - tp)))

                # Calculate Days Held (Max 5)
                days_held = 0
                if isinstance(orders, list):
                    for o in orders:
                        if isinstance(o, dict) and o.get('symbol') == sym and o.get('status') == 'filled':
                            filled_at = o.get('filled_at')
                            if filled_at:
                                try:
                                    filled_dt = pd.to_datetime(filled_at).tz_convert('UTC')
                                    now_dt = pd.Timestamp.now(tz='UTC')
                                    days_held = max(0, (now_dt - filled_dt).days)
                                except Exception:
                                    pass
                            break

                pos_data.append({
                    "Ticker": sym, 
                    "Side": side.upper(),
                    "Invested": invested_amt,
                    "Qty": qty,
                    "P/L (%)": float(p['unrealized_plpc']) * 100,
                    "Journey": progress,
                    "Days Held": f"{days_held}/5"
                })
            
            st.dataframe(
                pd.DataFrame(pos_data),
                width='stretch',
                column_config={
                    "Invested": st.column_config.NumberColumn("Invested", format="$%.2f"),
                    "P/L (%)": st.column_config.NumberColumn("P/L (%)", format="%.2f%%"),
                    "Journey": st.column_config.ProgressColumn(
                        "Journey to TP", help="Green bar moving right towards Take Profit.",
                        min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                },
                hide_index=True
            )

            # --- UPGRADED: THE FLASHPOINT ALERT (TRUE R-MULTIPLES) ---
            st.markdown("##### 🎯 Immediate Flashpoints (True R-Multiple)")
            closest_tp, closest_sl = None, None
            max_r, min_r = -999.0, 999.0

            estimated_atr = 0.03 # Fallback proxy to match Trading Agent
            dynamic_sl = estimated_atr * 2.0
            proxy_sl_pct = min(dynamic_sl, 0.02 * 1.5) * 100 # Exact match: capped at 3.0%

            for p_data in pos_data:
                # Calculate True R: (Current PnL %) / (Stop Loss %)
                true_r = p_data["P/L (%)"] / proxy_sl_pct
                
                # Find the trade closest to Take Profit (Highest +R)
                if true_r > max_r:
                    max_r = true_r
                    closest_tp = p_data["Ticker"]
                    
                # Find the trade closest to Stop Loss (Lowest -R)
                if true_r < min_r:
                    min_r = true_r
                    closest_sl = p_data["Ticker"]

            f1, f2 = st.columns(2)
            if closest_tp and max_r > 0: 
                f1.success(f"🟢 **Highest R:** {closest_tp} (Floating: +{max_r:.2f}R)")
            if closest_sl and min_r < 0: 
                f2.error(f"🔴 **Lowest R:** {closest_sl} (Floating: {min_r:.2f}R)")
                
        else:
            st.caption("No active positions currently held.")

        # --- UPGRADED: RECENT ORDERS & SLIPPAGE ---
        st.divider()

        today_utc = pd.Timestamp.now(tz='UTC').date()
        if isinstance(orders, list):
            trades_today = sum(1 for o in orders if isinstance(o, dict) and o.get('status') == 'filled' and pd.to_datetime(o.get('filled_at')).tz_convert('UTC').date() == today_utc)
        else:
            trades_today = 0

        c_ord1, c_ord2 = st.columns([3, 1])
        c_ord1.subheader("📜 Recent Fills & Execution Quality")
        if trades_today > 4:
            c_ord2.error(f"⚠️ Trades Today: {trades_today}")
        else:
            c_ord2.info(f"⚡ Trades Today: {trades_today}")

        if orders and isinstance(orders, list):
            order_data = []
            for o in orders[:5]: 
                if isinstance(o, dict) and o.get('status') == 'filled':
                    t = o.get('filled_at', '')
                    t_fmt = t[5:16].replace('T', ' ') if len(t) >= 16 else t
                    
                    # Calculate Slippage if it was a Limit order that filled
                    limit_price = float(o.get('limit_price', 0)) if o.get('limit_price') else 0.0
                    fill_price = float(o.get('filled_avg_price', 0)) if o.get('filled_avg_price') else 0.0
                    
                    slippage = 0.0
                    if limit_price > 0 and fill_price > 0:
                        if o.get('side') == 'buy':
                            slippage = ((fill_price - limit_price) / limit_price) * 100
                        else:
                            slippage = ((limit_price - fill_price) / limit_price) * 100

                    order_data.append({
                        "Time": t_fmt,
                        "Ticker": o.get('symbol', 'N/A'),
                        "Side": o.get('side', 'N/A').upper(),
                        "Qty": o.get('filled_qty', '0'),
                        "Fill Price": f"${fill_price:.2f}",
                        "Slippage": f"{slippage:+.2f}%" if limit_price > 0 else "N/A (MKT)"
                    })

            if order_data:
                df_orders = pd.DataFrame(order_data)
                
                # Apply conditional formatting to the slippage column
                def highlight_slippage(val):
                    if isinstance(val, str) and "%" in val:
                        num = float(val.replace("%", "").replace("+", ""))
                        if num > 0: return 'color: #ff4b4b' # Red for bad slippage
                        if num < 0: return 'color: #00ff41' # Green for price improvement
                    return ''
                
                # NEW: Color code the BUY/SELL side
                def highlight_side(val):
                    if val == 'BUY': return 'color: #00ff41; font-weight: bold;'
                    if val == 'SELL': return 'color: #ff4b4b; font-weight: bold;'
                    return ''

                # Chain the mappings together
                styled_df = (df_orders.style
                             .map(highlight_slippage, subset=['Slippage'])
                             .map(highlight_side, subset=['Side']))

                st.dataframe(styled_df, width="stretch", hide_index=True)
            else:
                st.caption("No recent filled orders found.")
        else:
            st.caption("No recent filled orders found.")

with tab2:
    st.markdown("### Terminal Output (Last 3000 Lines)")
    
    if logs:
        recent_logs = logs[-3000:] 
        formatted_logs = [format_log_line(line) for line in recent_logs]
        log_html = "".join(formatted_logs)
        st.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.write("No logs found.")

with tab3:
    if not hist_df_raw.empty and account:
        current_equity_raw = float(account['equity'])
        
        # --- CALCULATIONS ---
        # A. METRICS: Read from the pre-calculated global state
        metrics = st.session_state.get('global_metrics', {})
        
        # Calculate the new True Hit Rate
        hit_rate, trade_count = calculate_trade_hit_rate(orders)
        
        # Pass it to the scorecard
        scorecard_df = create_scorecard_df(metrics, hit_rate, trade_count)
        inst_score = calculate_institutional_score(metrics)
        
        # Capture the honest CAGR
        valid_cagr = metrics.get("CAGR", 0.0)
        
        # B. VISUALS: Use Adjusted Data (Immune to Capital Injections)
        dd_df = calculate_drawdown(hist_df_adj) 
        day_stats, monthly_stats = calculate_seasonality(hist_df_adj)
        
        # C. PROJECTIONS: Use valid_cagr (or manual) applied to Real Money
        projection_rate = manual_cagr if use_manual_cagr else valid_cagr
        
        # Calculate projection
        proj_df = calculate_future_projections(current_equity_raw, projection_rate)

        # --- SECTION 1: THE INSTITUTIONAL GAUGE ---
        col_gauge, col_scorecard = st.columns([1, 2.5])
        
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = inst_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Strategy Grade", 'font': {'size': 20, 'color': '#e0e0e0'}},
                number = {'suffix': "/100", 'font': {'color': '#e0e0e0'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#333"},
                    'bar': {'color': "#00ff41" if inst_score > 80 else "#ffb000"},
                    'bgcolor': "#1e1e1e",
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255, 75, 75, 0.3)'},
                        {'range': [50, 80], 'color': 'rgba(255, 176, 0, 0.3)'},
                        {'range': [80, 100], 'color': 'rgba(0, 255, 65, 0.3)'}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': inst_score}
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, width='stretch')
            
            if inst_score > 80:
                st.markdown("<div style='text-align: center; color: #00ff41; font-weight: bold;'>🚀 INSTITUTIONAL GRADE</div>", unsafe_allow_html=True)
            elif inst_score > 50:
                st.markdown("<div style='text-align: center; color: #ffb000; font-weight: bold;'>⚡ PROFESSIONAL RETAIL</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align: center; color: #ff4b4b; font-weight: bold;'>🎲 DEGEN / RETAIL</div>", unsafe_allow_html=True)

        with col_scorecard:
            st.markdown("### 📊 Metrics Breakdown (Adj. for Deposits)")
            st.dataframe(
                scorecard_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "METRIC": st.column_config.TextColumn("Metric", width="medium"),
                    "YOURS": st.column_config.TextColumn("Your Bot", width="small"),
                    "BENCHMARK": st.column_config.TextColumn("Target", width="small"),
                    "VERDICT": st.column_config.TextColumn("Verdict", width="small"),
                },
                height=280
            )

        st.divider()

        # --- SECTION 2: CHARTS (USING RAW DATA) ---
        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            st.markdown(f"### 📈 Real Equity Curve (${current_equity_raw:,.2f})")
            
            # Using RAW DF for the chart
            max_equity = hist_df_raw['equity'].max()
            fig_eq = px.area(hist_df_raw, x='timestamp', y='equity')
            fig_eq.update_traces(line_color='#00ff41', fillcolor='rgba(0, 255, 65, 0.1)')
            fig_eq.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None,
                yaxis_title=None,
                showlegend=False,
                height=300,
                yaxis=dict(range=[hist_df_raw['equity'].min() * 0.95, max_equity * 1.02], rangemode="normal")
            )
            st.plotly_chart(fig_eq, width='stretch')

        with col_perf2:
            st.markdown("### 📉 Real Risk (Drawdown)")
            # Using RAW DF (Drawdowns will look smaller relative to new higher peaks)
            fig_dd = px.area(dd_df, x='timestamp', y='drawdown')
            fig_dd.update_traces(line_color='#ff4b4b', fillcolor='rgba(255, 75, 75, 0.2)')
            fig_dd.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None, showlegend=False, height=300, yaxis=dict(tickformat=".1%"))
            st.plotly_chart(fig_dd, width='stretch')

        # --- NEW SECTION: LONG VS SHORT ATTRIBUTION ---
        st.divider()
        st.subheader("⚔️ Long vs. Short Attribution")
        
        if isinstance(orders, list) and len(orders) > 0:
            long_wins, long_losses = 0, 0
            short_wins, short_losses = 0, 0
            
            # Simple heuristic: Look at realized PnL of closed legs
            for pos in positions:
                # Currently active positions (unrealized)
                if pos['side'] == 'long':
                    if float(pos['unrealized_pl']) > 0: long_wins += 1
                    else: long_losses += 1
                elif pos['side'] == 'short':
                    if float(pos['unrealized_pl']) > 0: short_wins += 1
                    else: short_losses += 1

            total_longs = long_wins + long_losses
            total_shorts = short_wins + short_losses
            
            long_wr = (long_wins / total_longs * 100) if total_longs > 0 else 0
            short_wr = (short_wins / total_shorts * 100) if total_shorts > 0 else 0
            
            c_ls1, c_ls2, _spacer = st.columns([1, 1, 4])
            c_ls1.metric("🟢 Long Win Rate (Active)", f"{long_wr:.1f}%", f"{total_longs} positions", delta_color="off")
            c_ls2.metric("🔴 Short Win Rate (Active)", f"{short_wr:.1f}%", f"{total_shorts} positions", delta_color="off")
            st.caption("*Note: Displays active state. Full historical attribution requires database integration.*")

        # --- NEW SECTION: ROLLING EDGE ---
        st.divider()
        st.markdown("### 🔄 30-Day Rolling Edge (Momentum, Defense & Regime)")
        
        roll_df = calculate_rolling_edge(hist_df_adj, window=30) # <--- ENSURE THIS IS HERE
        
        if not roll_df.empty:
            # Create a 3x2 grid (Updated to 4x2 based on your code structure)
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
                st.caption("30-Day Rolling Daily Reliability (%)") # <--- NEW LABEL
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
                
        else:
            st.caption("Not enough data yet for 30-Day Rolling metrics.")

        # --- SECTION 3: INSTITUTIONAL DISTRIBUTIONS & SEASONALITY ---
        st.divider()
        st.subheader("⚖️ Return Distribution & Temporal Heatmap")
        
        c_dist1, c_dist2 = st.columns(2)
        
        with c_dist1:
            st.markdown("**📊 Daily Return Distribution (Asymmetry Test)**")
            st.caption(f"Skewness: {metrics.get('Skewness', 0):.2f} | Kurtosis: {metrics.get('Kurtosis', 0):.2f} | CVaR: {metrics.get('CVaR (95%)', 0):.2f}%")
            
            # Filter out 0% days to see actual trading volatility
            active_returns = hist_df_adj[hist_df_adj['daily_return'] != 0]['daily_return'] * 100
            
            if not active_returns.empty:
                fig_hist = px.histogram(
                    active_returns, nbins=50, 
                    color_discrete_sequence=['#569cd6'],
                    marginal="box" # Adds a box plot above to show outliers
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="white")
                fig_hist.update_layout(
                    showlegend=False, height=280, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#cccccc"), xaxis_title="Daily Return (%)", yaxis_title="Frequency"
                )
                st.plotly_chart(fig_hist, width='stretch')

        with c_dist2:
            st.markdown("**🗓️ Seasonality Heatmap (Win Rate %)**")
            st.caption("Darker green indicates high-probability temporal windows.")
            
            # Create a pivot table for the Heatmap: Day of Week vs Month
            if not hist_df_adj.empty:
                heat_df = hist_df_adj.copy()
                
                # === FIX: US Wall Street Time Normalization ===
                # Alpaca often stamps EOD equity in UTC, which pushes US Monday into UTC Tuesday.
                # Converting to New York time forces the dates back into their correct US slots.
                heat_df['timestamp'] = heat_df['timestamp'].dt.tz_convert('America/New_York')
                
                # Snap any live weekend ticks (from AEST live executions) back to Friday
                day_of_week = heat_df['timestamp'].dt.dayofweek
                heat_df.loc[day_of_week == 5, 'timestamp'] -= pd.Timedelta(days=1) # Sat -> Fri
                heat_df.loc[day_of_week == 6, 'timestamp'] += pd.Timedelta(days=1) # Sun -> Mon
                # ==============================================
                
                heat_df['Day'] = heat_df['timestamp'].dt.day_name()
                heat_df['Month'] = heat_df['timestamp'].dt.strftime('%b') 
                
                # Calculate Win Rate per Day/Month intersection
                heat_df['is_win'] = (heat_df['daily_return'] > 0).astype(int)
                heat_df['is_trade'] = (heat_df['daily_return'] != 0).astype(int)
                
                pivot = heat_df.groupby(['Day', 'Month'])[['is_win', 'is_trade']].sum().reset_index()
                pivot['Win Rate'] = (pivot['is_win'] / pivot['is_trade'] * 100).fillna(0)
                
                # Structure the matrix
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                matrix = pivot.pivot(index='Day', columns='Month', values='Win Rate').reindex(index=days_order, columns=months_order)
                
                fig_heat = px.imshow(
                    matrix, text_auto=".0f", color_continuous_scale="Greens",
                    aspect="auto", zmin=0, zmax=100
                )
                fig_heat.update_layout(
                    height=280, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#cccccc"), coloraxis_showscale=False
                )
                st.plotly_chart(fig_heat, width='stretch')

        # =====================================================================
        # --- SECTION 4: ROLLING MACRO CORRELATION (BETA) ---
        # =====================================================================
        st.divider()
        st.markdown("### ⚖️ Macro Alignment (Rolling Beta & Correlation to SPY)")
        st.caption("Isolates true Alpha. Beta tracks volatility relative to the S&P 500 (1.0 = moves identical to market). Correlation tracks directional grouping.")
        
        start_date_str = hist_df_adj['timestamp'].min().strftime('%Y-%m-%d')
        spy_df = get_historical_spy(start_date_str)
        
        if not spy_df.empty:
            # Safely merge bot history with SPY history
            merge_df = hist_df_adj[['timestamp', 'daily_return']].copy()
            merge_df['date_only'] = merge_df['timestamp'].dt.tz_localize(None).dt.floor('D')
            spy_df['date_only'] = spy_df.index
            
            macro_df = pd.merge(merge_df, spy_df, on='date_only', how='left').fillna(0)
            
            # Calculate 30-Day Rolling Metrics
            # Covariance / SPY Variance = Beta
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
                fig_beta.update_layout(
                    title="30-Day Rolling Beta", margin=dict(l=0, r=0, t=30, b=0), height=250,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title="Beta"
                )
                st.plotly_chart(fig_beta, width='stretch')
                
            with c_mac2:
                fig_corr = px.area(macro_df, x='timestamp', y='rolling_corr')
                fig_corr.update_traces(line_color='#569cd6', fillcolor='rgba(86, 156, 214, 0.2)')
                fig_corr.add_hline(y=0.0, line_dash="dash", line_color="white")
                fig_corr.update_layout(
                    title="30-Day Rolling Correlation (R)", margin=dict(l=0, r=0, t=30, b=0), height=250,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title="Correlation"
                )
                st.plotly_chart(fig_corr, width='stretch')
                
            latest_beta = macro_df['rolling_beta'].iloc[-1]
            if abs(latest_beta) < 0.3:
                st.success(f"**Current Posture:** Highly Decoupled (Beta: {latest_beta:.2f}). The system is generating pure uncorrelated Alpha.")
            elif latest_beta >= 0.8:
                st.warning(f"**Current Posture:** Highly Correlated (Beta: {latest_beta:.2f}). The system is acting essentially as a leveraged SPY ETF.")
            else:
                st.info(f"**Current Posture:** Moderately Correlated (Beta: {latest_beta:.2f}).")
        else:
            st.caption("Waiting for SPY historical data to populate macro charts...")

        # --- SECTION 5: FUTURE PROJECTIONS ---
        st.divider()
        
        projection_rate = manual_cagr if use_manual_cagr else valid_cagr
        proj_label = "Manual" if use_manual_cagr else "Adj."
        
        # Generates projection out to 20 years
        proj_df = calculate_future_projections(current_equity_raw, projection_rate, weekly_deposits=[0, 70, 140])
        
        st.markdown(f"### 🔮 20-Year Projections (Based on {proj_label} CAGR: {projection_rate:.1%})")
        st.caption("Includes institutional reality checks: Base compounding, cumulative principal tracking, and 3% inflation discounting for actual future purchasing power.")
        
        if not proj_df.empty:
            c_p1, c_p2 = st.columns([2, 1])
            with c_p1:
                # Melt for the chart (Keeping only the Nominal totals for a clean visual)
                melted_proj = proj_df.melt(
                    id_vars=['Date', 'Timeline'], 
                    value_vars=['Base (No Deposits)', '+$70/wk', '+$140/wk'],
                    var_name='Scenario', 
                    value_name='Projected Value'
                )

                fig_proj = px.line(
                    melted_proj, x='Date', y='Projected Value', color='Scenario', markers=True,
                    color_discrete_map={
                        "Base (No Deposits)": "#569cd6", 
                        "+$70/wk": "#c586c0",            
                        "+$140/wk": "#00ff41"            
                    }
                )
                
                fig_proj.update_traces(line_width=3)
                fig_proj.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0), 
                    xaxis_title=None, 
                    yaxis_title=None, 
                    height=400, 
                    template="plotly_dark",
                    legend=dict(orientation="h", y=1.1, x=0, title=None),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_proj, width='stretch')
                
            with c_p2:
                # Grab the 20-Year terminal values for the top display
                final_base = proj_df['Base (No Deposits)'].iloc[-1]
                final_140_nom = proj_df['+$140/wk'].iloc[-1]
                final_140_real = proj_df['+$140/wk (Real Value)'].iloc[-1]

                st.metric("20-Year Target (Base)", f"${final_base:,.0f}", f"{projection_rate:.1%} Rate")
                
                m1, m2 = st.columns(2)
                m1.metric("Max Nom (+$140/wk)", f"${final_140_nom:,.0f}")
                m2.metric("Real Power (Adj. Infl)", f"${final_140_real:,.0f}", "-3.0% Yearly Drag", delta_color="inverse")
                
                # Format the table to show the intelligent breakdown
                display_df = proj_df.drop(columns=['Timeline'])
                
                st.dataframe(
                    display_df, 
                    width="stretch", 
                    hide_index=True,
                    column_config={
                        "Date": st.column_config.DatetimeColumn(format="YYYY"),
                        "Base (No Deposits)": st.column_config.NumberColumn(format="$%.0f"),
                        "+$70/wk": st.column_config.NumberColumn(format="$%.0f"),
                        "+$70/wk (Principal)": st.column_config.NumberColumn("Prin. 70", format="$%.0f"),
                        "+$70/wk (Real Value)": None, # Hide intermediate real values to save horizontal space
                        "+$140/wk": st.column_config.NumberColumn(format="$%.0f"),
                        "+$140/wk (Principal)": st.column_config.NumberColumn("Prin. 140", format="$%.0f"),
                        "+$140/wk (Real Value)": st.column_config.NumberColumn("Real $140", format="$%.0f"),
                    },
                    height=280
                )

        # =====================================================================
        # --- SECTION 6: MONTE CARLO PROBABILITY CONE ---
        # =====================================================================
        st.divider()
        st.markdown("### 🎲 Monte Carlo Risk Simulation (Sequence of Returns)")
        st.caption("Bootstraps your actual historical daily returns to project 500 possible 20-year futures. This simulates 'Sequence of Returns Risk' (what happens if your losses cluster early vs. late). Visualized for the +$140/wk scenario.")
        
        # Run the simulation using the scrubbed, real historical returns
        mc_returns = hist_df_adj['daily_return'].dropna().values
        mc_df = run_monte_carlo_simulation(mc_returns, current_equity_raw, weekly_deposit=140, years=20, paths=500)
        
        if not mc_df.empty:
            # Resample to end of year to make the chart rendering extremely fast and clean
            mc_df_yearly = mc_df.set_index('Date').resample('YE').last().reset_index()
            
            fig_mc = go.Figure()
            
            # 90th Percentile (Top band)
            fig_mc.add_trace(go.Scatter(
                x=mc_df_yearly['Date'], y=mc_df_yearly['90th Percentile (Optimistic)'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            # 10th Percentile (Bottom band - shaded between 90th)
            fig_mc.add_trace(go.Scatter(
                x=mc_df_yearly['Date'], y=mc_df_yearly['10th Percentile (Pessimistic)'],
                mode='lines', fill='tonexty', fillcolor='rgba(0, 255, 65, 0.1)', line=dict(width=0), 
                name='80% Probability Range'
            ))
            
            # 50th Percentile (The Median path)
            fig_mc.add_trace(go.Scatter(
                x=mc_df_yearly['Date'], y=mc_df_yearly['50th Percentile (Expected)'],
                mode='lines+markers', line=dict(color='#00ff41', width=3),
                name='Median Expected Path'
            ))

            fig_mc.update_layout(
                margin=dict(l=0, r=0, t=10, b=0), height=350,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#cccccc"),
                legend=dict(orientation="h", y=1.1, x=0),
                yaxis=dict(title="Portfolio Value ($)", gridcolor="#333", zerolinecolor='white'),
                xaxis=dict(gridcolor="#333")
            )
            st.plotly_chart(fig_mc, width='stretch')

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("90th Percentile (Optimistic)", f"${mc_df_yearly['90th Percentile (Optimistic)'].iloc[-1]:,.0f}")
            mc2.metric("50th Percentile (Median)", f"${mc_df_yearly['50th Percentile (Expected)'].iloc[-1]:,.0f}")
            mc3.metric("10th Percentile (Pessimistic)", f"${mc_df_yearly['10th Percentile (Pessimistic)'].iloc[-1]:,.0f}")

    else:
        st.write("No history data available yet.")

with tab5:
    # Calculate required data for charts upfront
    bot_states = extract_bot_states(logs)

    # =====================================================================
    # --- CHART 1: MARKET PHYSICS PHASE PORTRAIT (2D) ---
    # =====================================================================
    st.markdown("### 🧭 Dynamic Phase Portrait & Vector Flow")
    st.caption("A 2D representation of the market's state machine. Removes time to show cycle structure, momentum flow, and regime probability.")
    
    if not phys_df.empty:
        col_text1, col_plot1 = st.columns([1, 3])
        
        latest_vel = phys_df['vel_smooth'].iloc[-1]
        latest_acc = phys_df['acc_smooth'].iloc[-1]
        latest_jerk_abs = abs(phys_df['jerk_smooth'].iloc[-1])
        latest_dfe = phys_df['dfe'].iloc[-1]
        
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
            if latest_vel > 0 and latest_acc > 0:
                st.success("**TREND (Top Right)**\n\nExpanding bull market.")
            elif latest_vel > 0 and latest_acc <= 0:
                st.warning("**DISTRIBUTION (Bottom Right)**\n\nSlowing bull market.")
            elif latest_vel <= 0 and latest_acc < 0:
                st.error("**PANIC / SHOCK (Bottom Left)**\n\nExpanding bear market.")
            elif latest_vel <= 0 and latest_acc >= 0:
                st.info("**ACCUMULATION (Top Left)**\n\nSlowing bear market.")
            else:
                st.write("Transitioning across zero-bound.")
                
        with col_plot1:
            fig_phase = generate_phase_portrait(phys_df)
            if fig_phase:
                st.plotly_chart(fig_phase, width='stretch')

    else:
        st.info("Not enough data points for Phase Portrait analysis.")

    st.divider()

    # =====================================================================
    # --- CHART 2: PORTFOLIO PHYSICS & CONVICTION TRAJECTORY (3D) ---
    # =====================================================================
    st.markdown("### 🎢 Portfolio Physics & Trajectory Surface")
    
    # Generate Physics Landscape
    X_phys, Y_phys, Z_phys, z_traj, phys_status = generate_proxied_ppo_landscape(phys_df, bot_states, conviction_data)

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
            
            fig_phys.add_trace(go.Surface(
                x=X_phys, y=Y_phys, z=Z_phys,
                colorscale='YlGnBu_r', opacity=0.8, showscale=False,
                lighting=dict(ambient=0.4, diffuse=0.9, roughness=0.1, specular=0.2),
                hoverinfo='none', cmin=0, cmax=1,
                contours_z=dict(show=True, usecolormap=True, highlightcolor="#fff", project_z=True),
            ))
            
            recent_phys = phys_df.tail(20) 
            x_traj = recent_phys['vel_smooth']
            y_traj = recent_phys['jerk_smooth']
            hover_dates = recent_phys['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

            fig_phys.add_trace(go.Scatter3d(
                x=x_traj, y=y_traj, z=z_traj,  
                mode='lines+markers', name='Historical Path', customdata=hover_dates,
                marker=dict(
                    size=abs(recent_phys['jerk']) * 8 + 4, color=recent_phys['acceleration'],              
                    colorscale='Viridis', opacity=1.0, line=dict(color='white', width=1),
                    colorbar=dict(title="Accel", len=0.5, y=0.2, x=0.9, tickfont={'color': "#cccccc"})
                ),
                line=dict(color='#ff9800', width=5),
                hovertemplate='<b>Date</b>: %{customdata}<br><b>Vel Proxy</b>: %{x:.2f}%<br><b>Jerk Proxy</b>: %{y:.2f}%<extra></extra>'
            ))

            fig_phys.update_layout(
                scene=dict(
                    aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5), 
                    xaxis_title='Returns (Velocity)', yaxis_title='Jerk (Volatility)', zaxis_title='Base Conviction',
                    xaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True, zerolinecolor='white'),
                    yaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True, zerolinecolor='white'),
                    zaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True, tickvals=[0, 0.5, 1.0], zerolinecolor='white'),
                ),
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#cccccc"),
                margin=dict(l=0, r=0, t=10, b=0), height=500, showlegend=False,
                annotations=[dict(
                    showarrow=False, text=f"PHYSICS TOPOLOGY: {phys_status}",
                    xref="paper", yref="paper", x=0.02, y=0.95,
                    xanchor="left", yanchor="top",
                    font=dict(size=14, color="#e91e63", weight="bold"), bgcolor="#1e1e1e"
                )]
            )
            st.plotly_chart(fig_phys, width='stretch')

    st.divider()

    # =====================================================================
    # --- CHART 3: TRUE STGNN POLICY LANDSCAPE (PCA) ---
    # =====================================================================
    st.markdown("### 🌌 AI Policy Landscape & Feature Space (PCA)")
    
    # Generate the PCA landscape variables
    X_pca, Y_pca, Z_pca, swarm_data, pca_status = generate_stgnn_pca_landscape(bot_json_state)

    if X_pca is not None:
        col_text3, col_plot3 = st.columns([1, 3])
        
        with col_text3:
            st.markdown("#### 🧠 Agent Brain State")
            st.caption("Translating the 27-D STGNN mathematical terrain into actionable logic.")
            
            if "ALIGNED" in pca_status:
                st.success(f"**Topology:** {pca_status}")
                st.write("Peaks represent regions of the 27-D feature space where the model has high conviction.")
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
            
            # TRACE 1: THE POLICY LANDSCAPE (Gaussian PCA Surface)
            fig_pca.add_trace(go.Surface(
                x=X_pca, y=Y_pca, z=Z_pca,
                colorscale='YlGnBu_r', opacity=0.8, showscale=False,
                lighting=dict(ambient=0.4, diffuse=0.9, roughness=0.1, specular=0.2),
                hoverinfo='none', cmin=0, cmax=1,
                contours_z=dict(show=True, usecolormap=True, highlightcolor="#fff", project_z=True),
            ))

            # TRACE 2: THE "SWARM" (Exact PCA Coordinates)
            if swarm_data:
                for i, ticker in enumerate(swarm_data['tickers']):
                    x_pos = swarm_data['x'][i]
                    y_pos = swarm_data['y'][i]
                    z_pos = swarm_data['z'][i]
                    
                    fig_pca.add_trace(go.Scatter3d(
                        x=[x_pos, x_pos], y=[y_pos, y_pos], z=[0, z_pos], 
                        mode='lines', line=dict(color='rgba(255, 255, 255, 0.4)', width=2, dash='dot'),
                        showlegend=False, hoverinfo='skip'
                    ))
                    
                    fig_pca.add_trace(go.Scatter3d(
                        x=[x_pos], y=[y_pos], z=[z_pos], 
                        mode='markers+text', name=ticker, text=[ticker],
                        textposition="top center",
                        textfont=dict(color="white", size=11, family="Arial Black"),
                        marker=dict(
                            size=10, color=z_pos, colorscale='YlGnBu_r', 
                            cmin=0, cmax=1, line=dict(color='white', width=2)
                        ),
                        hovertemplate=f'<b>{ticker}</b><br>Conviction: {z_pos:.1%}<br>PCA1: {x_pos:.2f}<br>PCA2: {y_pos:.2f}<extra></extra>'
                    ))

            fig_pca.update_layout(
                scene=dict(
                    aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.6), 
                    xaxis_title='PCA 1 (Macro Factor)', yaxis_title='PCA 2 (Asset Factor)', zaxis_title='True Conviction',
                    xaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True, zerolinecolor='white'),
                    yaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True, zerolinecolor='white'),
                    zaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True, tickvals=[0, 0.5, 1.0], zerolinecolor='white'),
                ),
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#cccccc"),
                margin=dict(l=0, r=0, t=10, b=0), height=500, showlegend=False,
                annotations=[dict(
                    showarrow=False, text=f"STGNN TOPOLOGY: {pca_status}",
                    xref="paper", yref="paper", x=0.02, y=0.95,
                    xanchor="left", yanchor="top",
                    font=dict(size=14, color="#00ff41", weight="bold"), bgcolor="#1e1e1e"
                )]
            )
            st.plotly_chart(fig_pca, width='stretch')

            # --- POINT 3: 2D TOPOGRAPHICAL CONVICTION MAP ---
            st.divider()
            st.markdown("#### 🗺️ 2D PCA Topographical View (Distortion-Free Cluster Map)")
            
            fig_contour = go.Figure()
            
            fig_contour.add_trace(go.Contour(
                x=X_pca[0], y=Y_pca[:, 0], z=Z_pca,
                colorscale='YlGnBu_r', opacity=0.8,
                contours=dict(showlines=True, coloring='heatmap'),
                hoverinfo='skip'
            ))
            
            if swarm_data:
                c_x = swarm_data['x']
                c_y = swarm_data['y']
                c_z = swarm_data['z']
                c_text = [f"{t}<br>{z:.1%}" for t, z in zip(swarm_data['tickers'], c_z)]
                    
                fig_contour.add_trace(go.Scatter(
                    x=c_x, y=c_y, mode='markers+text', text=c_text,
                    textposition="top center",
                    textfont=dict(color="white", size=10, family="Arial Black"),
                    marker=dict(size=12, color=c_z, colorscale='YlGnBu_r', cmin=0, cmax=1, line=dict(color='white', width=1.5)),
                    name="Live Feature Space", hoverinfo='skip'
                ))
                
            fig_contour.update_layout(
                xaxis_title='PCA 1 (Macro Factor)', yaxis_title='PCA 2 (Asset Factor)',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#cccccc"), margin=dict(l=0, r=0, t=10, b=0), height=450
            )

            st.plotly_chart(fig_contour, width='stretch')

    else:
        st.info("Gathering historical Policy Landscape data. Waiting for Daily Inference Agent payload...")

with tab6:
    # --- QUANTUM ALPHA MODEL LIFECYCLE MONITOR ---
    st.subheader("🧠 Quantum Alpha Model Lifecycle Monitor")
    st.markdown("Real-time alignment tracking between weekend optimization blueprints and live out-of-sample market execution.")
        
    #st.write(model_health) 
    # --------------------------------------

    if model_health:
       # Safety Check: If it came through as a string, try to parse it as JSON
        if isinstance(model_health, str):
            try:
               import json
               model_health = json.loads(model_health)
            except ValueError:
               model_health = {} # Fallback to emptyc if it's an unparseable string

       # Only run the dictionary sort if we actually have a dictionary
        if isinstance(model_health, dict):
            sorted_health = sorted(
               model_health.items(),
               key=lambda x: 0 if 'DEGRADED' in x[1].get('Status', '') else 1
            )
        else:
            sorted_health = [] # Fallback to prevent crash

        html_output = ""
        for ticker, profile in sorted_health:
            status = profile['Status']
            base_ir = float(profile['Base IR'])
            live_ir = float(profile['Live IR'])
            decay = float(profile['Decay'])
            mdd = int(profile['MDD'])
            base_mdd = int(profile.get('Base MDD', 0))
            base_wr = float(profile.get('Base WR', 0.0))
            
            # 1. Main Card Color
            statusColor = '#444' 
            if 'OPTIMAL' in status: statusColor = '#00ff41'
            elif 'STABLE' in status: statusColor = '#ffb000'
            elif 'DEGRADED' in status: statusColor = '#ff4b4b'
            
            # 2. Information Ratio Intelligence
            ir_diff = live_ir - base_ir
            if live_ir >= base_ir:
                ir_text = f"an impressive <strong>Live Information Ratio of {live_ir:.2f}</strong>, <span style='color: #00ff41;'>outperforming</span> its weekend benchmark ({base_ir:.2f}) by +{ir_diff:.2f}."
            elif live_ir >= 0:
                ir_text = f"a <strong>Live Information Ratio of {live_ir:.2f}</strong>. While still generating positive alpha, it is <span style='color: #ffb000;'>underperforming</span> its weekend benchmark ({base_ir:.2f}) by {ir_diff:.2f}."
            else:
                ir_text = f"a negative <strong>Live Information Ratio of {live_ir:.2f}</strong>, <span style='color: #ff4b4b;'>failing</span> to meet its weekend benchmark ({base_ir:.2f}) by a margin of {ir_diff:.2f}."

            # 3. Decay Intelligence
            if decay >= 0.70:
                decay_text = f"The asset decay factor is excellent at <strong>{decay:.2f}</strong> (Target: 1.0), indicating strong structural alignment with the training blueprint."
            elif decay >= 0.40:
                decay_text = f"The asset decay factor sits at <strong style='color: #ffb000;'>{decay:.2f}</strong>, showing moderate edge erosion but remaining above the 0.40 throttle threshold."
            else:
                decay_text = f"Severe edge erosion detected with a decay factor of <strong style='color: #ff4b4b;'>{decay:.2f}</strong> (Critically below the 0.40 threshold), triggering autonomous risk throttling."

            # 4. Drawdown Intelligence 
            if mdd <= 21:
                mdd_text = f"Drawdown duration is safely contained at <strong>{mdd} days</strong> (Optimal: < 21 days)."
            elif mdd <= 42:
                mdd_text = f"Drawdown duration is stretching to <strong style='color: #ffb000;'>{mdd} days</strong>, approaching structural pain thresholds."
            else:
                mdd_text = f"Drawdown duration has breached limits at <strong style='color: #ff4b4b;'>{mdd} days</strong> (Danger: > 42 days)."

            lifecycle = profile.get('Lifecycle', 'Unknown') # <--- Retrieve the new value

            # 5. Build the HTML Block with Side-By-Side Comparison Grid
            html_output += f'<div style="margin-bottom: 12px; padding: 15px; border-left: 5px solid {statusColor}; background-color: #1e1e1e; border-radius: 6px;">'
            html_output += f'<strong style="font-size: 1.2em; color: #fff;">{ticker}</strong>'
            html_output += f'<span style="background-color: {statusColor}; color: #111; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; margin-left: 10px;">{status}</span>'
            
            # --- RENDER LIFECYCLE ---
            html_output += f'<div style="margin-top: 8px; font-size: 0.9em; color: #aaa;">'
            html_output += f'<strong>Lifecycle Phase:</strong> <span style="color: #fff;">{lifecycle}</span>'
            html_output += f'</div>'
            # ------------------------

            # --- THE NEW SIDE-BY-SIDE HUD ---
            html_output += f'<div style="margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.85em; color: #aaa; background: #2a2a2a; padding: 10px; border-radius: 4px;">'
            html_output += f'<div><strong style="color: #fff;">🏗️ Training Blueprint</strong><br>Base IR: {base_ir:.2f} &nbsp;|&nbsp; Win Rate: {base_wr:.1f}% &nbsp;|&nbsp; MDD: {base_mdd}d</div>'
            html_output += f'<div><strong style="color: #fff;">⚡ Live Execution</strong><br>Live IR: {live_ir:.2f} &nbsp;|&nbsp; Decay: {decay:.2f} &nbsp;|&nbsp; MDD: {mdd}d</div>'
            html_output += f'</div>'
            # --------------------------------
            
            html_output += f'<p style="margin: 10px 0 0 0; font-size: 0.95em; line-height: 1.6; color: #ccc;">'
            html_output += f'The model is currently displaying {ir_text}<br><br>'
            html_output += f'{decay_text} {mdd_text}'
            html_output += f'</p></div>'
            
        st.markdown(html_output, unsafe_allow_html=True)
        
    else:
        st.info("Awaiting model performance data from the live execution log stream...")
    # --------------------------------------------------

# Complete Replacement for the AUTO REFRESH LOOP at the end of the file
# === AUTO REFRESH LOOP ===
if auto_refresh:
    import streamlit.components.v1 as components
    # Offload the wait time to the client's browser, freeing the server thread immediately.
    components.html(
        """
        <script>
        setTimeout(function() {
            window.parent.location.reload();
        }, 60000);
        </script>
        """,
        height=0
    )