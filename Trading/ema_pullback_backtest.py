"""
EMA Pullback Momentum Strategy Backtest
Strategy Rules:
- Entry: Price pulls back to 9 EMA after being above it, then closes above 9 EMA again
- Filters: RSI(14) between 55-70, Volume > 1M shares, Nifty not broken
- Exit: 3% target OR 2% stop loss
- Position Sizing: Risk 2% of capital per trade
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

# Backtest Period
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

# Capital and Risk
INITIAL_CAPITAL = 100000  # ₹1 lakh
RISK_PER_TRADE = 0.02  # 2% of capital per trade
MAX_POSITIONS = 3  # Maximum concurrent positions

# Strategy Parameters
EMA_PERIOD = 9
RSI_PERIOD = 14
RSI_MIN = 55
RSI_MAX = 70
MIN_VOLUME = 1000000  # 1M shares

# Exit Rules
TARGET_PERCENT = 0.03  # 3%
STOP_PERCENT = 0.02   # 2%

# Costs
SLIPPAGE = 0.001  # 0.1% slippage
BROKERAGE = 0.0003  # 0.03% per trade (entry + exit = 0.06% total)

# Stock Universe (Nifty 50 + some liquid midcaps)
STOCK_UNIVERSE = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS',
    'AXISBANK.NS', 'ITC.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
    'WIPRO.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'HCLTECH.NS', 'BAJFINANCE.NS',
    'TECHM.NS', 'SUNPHARMA.NS', 'INDUSINDBK.NS', 'TATAMOTORS.NS', 'ADANIENT.NS'
]

# ==================== TECHNICAL INDICATORS ====================

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period):
    """Calculate RSI"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_nifty_trend(date):
    """Check if Nifty is in uptrend (simplified: above 50 EMA)"""
    # Download Nifty data once and cache
    if not hasattr(get_nifty_trend, 'nifty_data'):
        nifty = yf.download('^NSEI', start=START_DATE, end=END_DATE, progress=False)
        nifty['EMA50'] = nifty['Close'].ewm(span=50, adjust=False).mean()
        get_nifty_trend.nifty_data = nifty
    
    try:
        nifty_close = get_nifty_trend.nifty_data.loc[date, 'Close']
        nifty_ema = get_nifty_trend.nifty_data.loc[date, 'EMA50']
        return nifty_close > nifty_ema
    except:
        return True  # If date not found, assume OK

# ==================== STRATEGY LOGIC ====================

def prepare_data(ticker):
    """Download and prepare stock data with indicators"""
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        
        if len(data) < 50:
            return None
        
        # Calculate indicators
        data['EMA9'] = calculate_ema(data, EMA_PERIOD)
        data['RSI'] = calculate_rsi(data, RSI_PERIOD)
        
        # Price position relative to EMA
        data['Above_EMA'] = data['Close'] > data['EMA9']
        data['Prev_Above_EMA'] = data['Above_EMA'].shift(1)
        
        # Pullback detection: was above, went to/below, now back above
        data['Pullback_Signal'] = (
            (data['Prev_Above_EMA'] == False) &  # Was at/below EMA yesterday
            (data['Above_EMA'] == True) &         # Above EMA today
            (data['Close'].shift(2) > data['EMA9'].shift(2))  # Was above 2 days ago
        )
        
        data['Ticker'] = ticker
        return data
        
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

def check_entry_conditions(row, date):
    """Check if all entry conditions are met"""
    conditions = {
        'pullback': row['Pullback_Signal'],
        'rsi': RSI_MIN <= row['RSI'] <= RSI_MAX,
        'volume': row['Volume'] > MIN_VOLUME,
        'nifty': get_nifty_trend(date)
    }
    return all(conditions.values()), conditions

def calculate_position_size(capital, entry_price, stop_percent):
    """Calculate position size based on risk"""
    risk_amount = capital * RISK_PER_TRADE
    risk_per_share = entry_price * stop_percent
    shares = int(risk_amount / risk_per_share)
    return max(shares, 1)  # At least 1 share

# ==================== BACKTESTING ENGINE ====================

class Trade:
    def __init__(self, ticker, entry_date, entry_price, shares, stop_loss, target):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.stop_loss = stop_loss
        self.target = target
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0
        self.return_pct = 0

def run_backtest():
    """Main backtesting engine"""
    
    print("=" * 60)
    print("EMA PULLBACK MOMENTUM STRATEGY BACKTEST")
    print("=" * 60)
    print(f"\nPeriod: {START_DATE} to {END_DATE}")
    print(f"Initial Capital: ₹{INITIAL_CAPITAL:,.0f}")
    print(f"Risk per Trade: {RISK_PER_TRADE*100}%")
    print(f"\nDownloading data for {len(STOCK_UNIVERSE)} stocks...")
    
    # Download all data
    stock_data = {}
    for ticker in STOCK_UNIVERSE:
        data = prepare_data(ticker)
        if data is not None:
            stock_data[ticker] = data
    
    print(f"Successfully loaded {len(stock_data)} stocks\n")
    
    # Initialize tracking
    capital = INITIAL_CAPITAL
    open_positions = []
    closed_trades = []
    daily_equity = []
    
    # Get all trading dates
    sample_data = list(stock_data.values())[0]
    trading_dates = sample_data.index
    
    # Iterate through each trading day
    for date in trading_dates:
        # Check exits for open positions
        for position in open_positions[:]:  # Copy list to modify during iteration
            ticker = position.ticker
            if date not in stock_data[ticker].index:
                continue
                
            row = stock_data[ticker].loc[date]
            high = row['High']
            low = row['Low']
            close = row['Close']
            
            # Check if stop loss or target hit
            if low <= position.stop_loss:
                # Stop loss hit
                exit_price = position.stop_loss * (1 - SLIPPAGE)  # Slippage on stop
                position.exit_date = date
                position.exit_price = exit_price
                position.exit_reason = 'Stop Loss'
                
                # Calculate P&L
                entry_cost = position.entry_price * position.shares * (1 + SLIPPAGE + BROKERAGE)
                exit_value = exit_price * position.shares * (1 - BROKERAGE)
                position.pnl = exit_value - entry_cost
                position.return_pct = (exit_price / position.entry_price - 1) * 100
                
                capital += exit_value
                closed_trades.append(position)
                open_positions.remove(position)
                
            elif high >= position.target:
                # Target hit
                exit_price = position.target * (1 - SLIPPAGE)  # Slippage on target
                position.exit_date = date
                position.exit_price = exit_price
                position.exit_reason = 'Target'
                
                # Calculate P&L
                entry_cost = position.entry_price * position.shares * (1 + SLIPPAGE + BROKERAGE)
                exit_value = exit_price * position.shares * (1 - BROKERAGE)
                position.pnl = exit_value - entry_cost
                position.return_pct = (exit_price / position.entry_price - 1) * 100
                
                capital += exit_value
                closed_trades.append(position)
                open_positions.remove(position)
        
        # Look for new entries (if we have room)
        if len(open_positions) < MAX_POSITIONS:
            for ticker in stock_data:
                if date not in stock_data[ticker].index:
                    continue
                
                # Skip if already in position
                if any(p.ticker == ticker for p in open_positions):
                    continue
                
                row = stock_data[ticker].loc[date]
                
                # Check entry conditions
                entry_ok, conditions = check_entry_conditions(row, date)
                
                if entry_ok:
                    # Calculate position
                    entry_price = row['Close'] * (1 + SLIPPAGE)  # Buy at close with slippage
                    stop_loss = entry_price * (1 - STOP_PERCENT)
                    target = entry_price * (1 + TARGET_PERCENT)
                    
                    shares = calculate_position_size(capital, entry_price, STOP_PERCENT)
                    position_value = entry_price * shares * (1 + SLIPPAGE + BROKERAGE)
                    
                    # Check if we have enough capital
                    if position_value <= capital:
                        trade = Trade(ticker, date, entry_price, shares, stop_loss, target)
                        open_positions.append(trade)
                        capital -= position_value
                        
                        if len(open_positions) >= MAX_POSITIONS:
                            break
        
        # Track daily equity
        open_value = sum([
            stock_data[p.ticker].loc[date, 'Close'] * p.shares 
            for p in open_positions 
            if date in stock_data[p.ticker].index
        ])
        total_equity = capital + open_value
        daily_equity.append({'Date': date, 'Equity': total_equity})
    
    # Close any remaining positions at end
    last_date = trading_dates[-1]
    for position in open_positions:
        if last_date in stock_data[position.ticker].index:
            exit_price = stock_data[position.ticker].loc[last_date, 'Close'] * (1 - SLIPPAGE)
            position.exit_date = last_date
            position.exit_price = exit_price
            position.exit_reason = 'End of Period'
            
            entry_cost = position.entry_price * position.shares * (1 + SLIPPAGE + BROKERAGE)
            exit_value = exit_price * position.shares * (1 - BROKERAGE)
            position.pnl = exit_value - entry_cost
            position.return_pct = (exit_price / position.entry_price - 1) * 100
            
            closed_trades.append(position)
    
    return closed_trades, daily_equity

# ==================== PERFORMANCE ANALYSIS ====================

def analyze_performance(trades, daily_equity):
    """Generate detailed performance metrics"""
    
    if len(trades) == 0:
        print("\nNo trades were executed. Check your filters.")
        return
    
    # Basic stats
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    total_pnl = sum(t.pnl for t in trades)
    avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
    
    # Calculate returns
    equity_curve = pd.DataFrame(daily_equity).set_index('Date')
    final_capital = equity_curve['Equity'].iloc[-1]
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    # Calculate max drawdown
    equity_curve['Peak'] = equity_curve['Equity'].cummax()
    equity_curve['Drawdown'] = (equity_curve['Equity'] / equity_curve['Peak'] - 1) * 100
    max_drawdown = equity_curve['Drawdown'].min()
    
    # Calculate Sharpe (simplified)
    equity_curve['Returns'] = equity_curve['Equity'].pct_change()
    sharpe = (equity_curve['Returns'].mean() / equity_curve['Returns'].std() * np.sqrt(252)) if equity_curve['Returns'].std() > 0 else 0
    
    # Time analysis
    holding_periods = [(t.exit_date - t.entry_date).days for t in trades]
    avg_holding = np.mean(holding_periods)
    
    # Print results
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 OVERALL PERFORMANCE")
    print(f"Initial Capital:      ₹{INITIAL_CAPITAL:,.0f}")
    print(f"Final Capital:        ₹{final_capital:,.0f}")
    print(f"Total Return:         {total_return:+.2f}%")
    print(f"Total P&L:            ₹{total_pnl:+,.0f}")
    print(f"Max Drawdown:         {max_drawdown:.2f}%")
    print(f"Sharpe Ratio:         {sharpe:.2f}")
    
    print(f"\n📈 TRADE STATISTICS")
    print(f"Total Trades:         {total_trades}")
    print(f"Winning Trades:       {len(winning_trades)} ({win_rate:.1f}%)")
    print(f"Losing Trades:        {len(losing_trades)} ({100-win_rate:.1f}%)")
    print(f"Average Win:          ₹{avg_win:,.0f}")
    print(f"Average Loss:         ₹{avg_loss:,.0f}")
    print(f"Win/Loss Ratio:       {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")
    print(f"Avg Holding Period:   {avg_holding:.1f} days")
    
    # Exit reasons
    print(f"\n🎯 EXIT BREAKDOWN")
    target_exits = len([t for t in trades if t.exit_reason == 'Target'])
    stop_exits = len([t for t in trades if t.exit_reason == 'Stop Loss'])
    other_exits = total_trades - target_exits - stop_exits
    
    print(f"Target Hit:           {target_exits} ({target_exits/total_trades*100:.1f}%)")
    print(f"Stop Loss Hit:        {stop_exits} ({stop_exits/total_trades*100:.1f}%)")
    print(f"Other:                {other_exits} ({other_exits/total_trades*100:.1f}%)")
    
    # Best and worst trades
    best_trade = max(trades, key=lambda t: t.return_pct)
    worst_trade = min(trades, key=lambda t: t.return_pct)
    
    print(f"\n🏆 BEST/WORST TRADES")
    print(f"Best:  {best_trade.ticker:12} {best_trade.return_pct:+.2f}% on {best_trade.entry_date.date()}")
    print(f"Worst: {worst_trade.ticker:12} {worst_trade.return_pct:+.2f}% on {worst_trade.entry_date.date()}")
    
    # Reality check
    print(f"\n⚠️  REALITY CHECK")
    weeks_in_period = (equity_curve.index[-1] - equity_curve.index[0]).days / 7
    avg_weekly_return = total_return / weeks_in_period
    print(f"Average Weekly Return: {avg_weekly_return:.2f}%")
    
    if avg_weekly_return >= 3:
        print(f"✓ Target achieved! But verify with forward testing.")
    else:
        print(f"✗ Below 3%/week target. This is more realistic.")
    
    print(f"\nNote: Past performance ≠ future results")
    print(f"Real trading will have: more slippage, emotion, timing delays")
    
    # Export trades to CSV
    trades_df = pd.DataFrame([{
        'Ticker': t.ticker,
        'Entry_Date': t.entry_date,
        'Entry_Price': t.entry_price,
        'Exit_Date': t.exit_date,
        'Exit_Price': t.exit_price,
        'Shares': t.shares,
        'Exit_Reason': t.exit_reason,
        'P&L': t.pnl,
        'Return_%': t.return_pct
    } for t in trades])
    
    trades_df.to_csv('/home/claude/backtest_trades.csv', index=False)
    print(f"\n💾 Detailed trades exported to: backtest_trades.csv")
    
    return equity_curve

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    trades, daily_equity = run_backtest()
    equity_curve = analyze_performance(trades, daily_equity)
    
    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)
