import numpy as np
import pandas as pd
import logging
import re
from typing import Dict

class Backtester:
    def __init__(self):
        self.rule_weights = {
            "sma": 0.8, "ema": 0.9, "rsi": 1.0, "macd": 1.1,
            "volume": 0.7, "liquidity": 1.2, "order block": 1.3,
            "fair value gap": 1.1, "market structure": 1.2,
            "support": 0.9, "resistance": 0.9, "123": 1.4,
            "reversal": 1.3, "supply": 1.0, "demand": 1.0
        }
    
    def test_strategy(self, strategy: Dict, market_data: pd.DataFrame, symbol: str) -> Dict:
        try:
            df = market_data.copy()
            signals = pd.Series(0, index=df.index)
            
            # Generate signals
            signals = self._generate_signals(strategy, df).astype(int)
            
            if signals.sum() == 0:
                return {"win_rate": 0, "profit": 0, "score": 0}
            
            pip_multiplier = self._get_pip_multiplier(symbol)
            
            # Entry/exit logic
            entry_prices = df['open'].shift(-1).ffill()
            exit_prices = df['close'].shift(-1).ffill()
            
            # Calculate P&L based on direction
            direction = 1 if strategy.get("direction", "buy") == "buy" else -1
            price_diff = (exit_prices - entry_prices) / (0.0001 if "GBPJPYm" not in symbol else 0.01) * direction
            strategy_profit = signals * price_diff
            
            # Apply spread cost
            spread_cost = 3 if "GBPJPYm" in symbol else 5
            strategy_profit = strategy_profit - np.abs(signals) * spread_cost
            
            # Calculate metrics
            winning_trades = strategy_profit > 0
            total_trades = signals.sum()
            win_rate = (winning_trades.sum() / total_trades) * 100 if total_trades > 0 else 0
            total_profit = strategy_profit.sum()
            score = (win_rate * 0.6) + (total_profit * 0.4)
            
            return {
                "win_rate": win_rate,
                "profit": total_profit,
                "score": score
            }
        except Exception as e:
            logging.error(f"Backtesting error: {str(e)}")
            return {"win_rate": 0, "profit": 0, "score": 0}

    def _generate_signals(self, strategy: Dict, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        
        for rule in strategy.get("rules", []):
            rule_lower = rule.lower()
            weight = self._get_rule_weight(rule_lower)
            
            try:
                # Moving Average rules
                if "sma" in rule_lower or "ema" in rule_lower:
                    period = self._extract_period(rule_lower)
                    if not period:
                        continue
                        
                    ma_type = "sma" if "sma" in rule_lower else "ema"
                    ma_col = f"{ma_type}_{period}"
                    
                    if ma_col not in data.columns:
                        if ma_type == "sma":
                            data[ma_col] = data['close'].rolling(period).mean()
                        else:
                            data[ma_col] = data['close'].ewm(span=period, adjust=False).mean()
                    
                    if ">" in rule_lower:
                        signals += (data['close'] > data[ma_col]).astype(int) * weight
                    elif "<" in rule_lower:
                        signals += (data['close'] < data[ma_col]).astype(int) * weight
                
                # RSI rules
                elif "rsi" in rule_lower:
                    period = self._extract_period(rule_lower) or 14
                    rsi_col = f"rsi_{period}"
                    
                    if rsi_col not in data.columns:
                        data[rsi_col] = self.calculate_rsi(data['close'], period)
                    
                    threshold = self._extract_number(rule_lower)
                    if not threshold:
                        continue
                    
                    if ">" in rule_lower:
                        signals += (data[rsi_col] > threshold).astype(int) * weight
                    elif "<" in rule_lower:
                        signals += (data[rsi_col] < threshold).astype(int) * weight
            
                # MACD rules
                elif "macd" in rule_lower:
                    if "macd_line" not in data.columns:
                        data["macd_line"], data["macd_signal"] = self.calculate_macd(data['close'])
                    
                    if "crossover" in rule_lower:
                        signals += ((data["macd_line"] > data["macd_signal"]) & 
                                   (data["macd_line"].shift(1) <= data["macd_signal"].shift(1))).astype(int) * weight
                    elif "crossunder" in rule_lower:
                        signals += ((data["macd_line"] < data["macd_signal"]) & 
                                   (data["macd_line"].shift(1) >= data["macd_signal"].shift(1))).astype(int) * weight
            
                # Price action rules
                elif "liquidity sweep" in rule_lower:
                    lookback = 5
                    high = data['high'].rolling(lookback).max()
                    low = data['low'].rolling(lookback).min()
                    sweep_up = (data['high'] > high.shift(1)) & (data['close'] < data['open'])
                    sweep_down = (data['low'] < low.shift(1)) & (data['close'] > data['open'])
                    signals += (sweep_up | sweep_down).asype(int) * weight
            
                elif "order block" in rule_lower or "fair value gap" in rule_lower:
                    if "atr" not in data.columns:
                        data['atr'] = self.calculate_atr(data, 14)
                    large_move = (data['high'] - data['low']) > (data['atr'] * 1.5)
                    signals += large_move.astype(int) * weight
            
                elif "market structure" in rule_lower:
                    shifted = data['close'].shift(2)
                    mss_up = (data['close'] > data['close'].shift(1)) & (data['close'].shift(1) < shifted)
                    mss_down = (data['close'] < data['close'].shift(1)) & (data['close'].shift(1) > shifted)
                    signals += (mss_up | mss_down).astype(int) * weight
            
                # Support/resistance rules
                elif "support" in rule_lower or "resistance" in rule_lower:
                    swing_low = (data['low'] < data['low'].shift(1)) & (data['low'] < data['low'].shift(-1))
                    swing_high = (data['high'] > data['high'].shift(1)) & (data['high'] > data['high'].shift(-1))
                    signals += (swing_low | swing_high).astype(int) * weight
            
                elif "123 setup" in rule_lower or "reversal" in rule_lower:
                    # Bullish reversal pattern
                    cond1 = data['close'].shift(2) > data['close'].shift(1)
                    cond2 = data['close'].shift(1) < data['close']
                    cond3 = data['close'] > data['close'].shift(1)
                    bullish_pattern = cond1 & cond2 & cond3
                    
                    # Bearish reversal pattern
                    cond4 = data['close'].shift(2) < data['close'].shift(1)
                    cond5 = data['close'].shift(1) > data['close']
                    cond6 = data['close'] < data['close'].shift(1)
                    bearish_pattern = cond4 & cond5 & cond6
                    
                    pattern = bullish_pattern | bearish_pattern
                    signals += pattern.astype(int) * weight * 1.5
            
                # Golden ratio (Fibonacci)
                elif "golden ratio" in rule_lower:
                    high = data['high'].rolling(50).max()
                    low = data['low'].rolling(50).min()
                    price_range = high - low
                    golden_level = high - price_range * 0.618
                    
                    if "atr" not in data.columns:
                        data['atr'] = self.calculate_atr(data, 14)
                        
                    near_golden = np.abs(data['close'] - golden_level) < (data['atr'] * 0.1)
                    signals += near_golden.astype(int) * weight * 1.3
                
                # Default rule handling
                else:
                    signals += 0.5 * weight
            
            except Exception as e:
                logging.warning(f"Rule processing failed: {rule} - {str(e)}")
                continue

        # Normalize signals
        signals = (signals > 0.5).astype(int)
        return signals

    def _get_pip_multiplier(self, symbol: str) -> float:
        if "USDJPYm" in symbol:
            return 100  # 1 pip = 0.01 for JPY pairs
        return 10000   # 1 pip = 0.0001 for other pairs

    def _get_rule_weight(self, rule: str) -> float:
        for key, weight in self.rule_weights.items():
            if key in rule:
                return weight
        return 0.8

    def _extract_period(self, text: str) -> int:
        match = re.search(r'\d+', text)
        return int(match.group()) if match else None

    def _extract_number(self, text: str) -> float:
        match = re.search(r'\d+\.?\d*', text)
        return float(match.group()) if match else None

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        return atr

    def calculate_macd(self, close, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, macd_signal 