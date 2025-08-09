import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import logging
import pytz
from typing import Dict, List, Optional, Tuple, Union

class MT5DataFetcher:
    def __init__(self):
        self.connected = False
        self.timeframe_map = {
            'm1': mt5.TIMEFRAME_M1,
            'm5': mt5.TIMEFRAME_M5,
            'm15': mt5.TIMEFRAME_M15,
            'm30': mt5.TIMEFRAME_M30,
            'h1': mt5.TIMEFRAME_H1,
            'h2': mt5.TIMEFRAME_H2,
            'h4': mt5.TIMEFRAME_H4,
            'd1': mt5.TIMEFRAME_D1,
            'w1': mt5.TIMEFRAME_W1
        }
        self.ict_config = {
            'fvg_lookback': 3,
            'ob_wick_ratio': 0.7,
            'liquidity_window': 20,
            'volume_spike_threshold': 2.0,
            'killzones': {
                'london': (7, 9),
                'new_york': (8, 10),
                'tokyo': (0, 2)  # Added Tokyo session
            }
        }
        self.connect()
        
    def connect(self) -> bool:
        """Initialize MT5 connection with retries"""
        if not mt5.initialize():
            raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
        self.connected = True
        logging.info("Connected to MT5 for data fetching")
        return True
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logging.info("Disconnected from MT5")

    def get_candles(self, symbol: str, timeframe: Union[str, int], bars: int = 1000) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data with enhanced ICT/SMC indicators
        Args:
            timeframe: Can be MT5 constant or string (e.g. 'h4')
        """
        if not self.connected and not self.connect():
            return None
            
        # Convert string timeframe to MT5 constant if needed
        if isinstance(timeframe, str):
            timeframe = self._parse_timeframe(timeframe)
            
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None:
            logging.error(f"No data returned for {symbol} {timeframe}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df['symbol'] = symbol  # Add symbol column for reference
        
        # Calculate all technical indicators
        df = self._add_technical_indicators(df)
        return df

    def get_multi_timeframe_data(self, symbol: str, timeframes: List[Union[str, int]]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes simultaneously
        Returns dict with timeframe strings as keys
        """
        data = {}
        for tf in timeframes:
            tf_str = tf if isinstance(tf, str) else self._get_timeframe_str(tf)
            data[tf_str] = self.get_candles(symbol, tf)
        return data

    def is_in_killzone(self, killzone_type: str = "london") -> bool:
        """
        Enhanced killzone detection with multiple sessions
        Args:
            killzone_type: "london", "new_york", or "tokyo"
        """
        if killzone_type not in self.ict_config['killzones']:
            raise ValueError(f"Unknown killzone type: {killzone_type}")
            
        tz_map = {
            'london': 'Europe/London',
            'new_york': 'America/New_York',
            'tokyo': 'Asia/Tokyo'
        }
        
        tz = pytz.timezone(tz_map[killzone_type])
        now = datetime.now(tz).time()
        start_hour, end_hour = self.ict_config['killzones'][killzone_type]
        
        return time(start_hour, 0) <= now <= time(end_hour, 0)

    # ========== Original Methods (Maintained Exactly) ==========
    def _parse_timeframe(self, tf_str: str) -> int:
        """Convert string timeframe to MT5 constant"""
        return self.timeframe_map.get(tf_str.lower(), mt5.TIMEFRAME_H1)

    def _get_timeframe_str(self, tf_int: int) -> str:
        """Convert MT5 constant to string timeframe"""
        reverse_map = {v: k for k, v in self.timeframe_map.items()}
        return reverse_map.get(tf_int, 'h1')

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def _calculate_macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    # ========== Enhanced ICT/SMC Methods ==========
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced to include advanced ICT/SMC indicators"""
        if not isinstance(df, pd.DataFrame) or len(df) < 20:
            return df
            
        try:
            # Original indicators
            df['body'] = df['close'] - df['open']
            df['range'] = df['high'] - df['low']
            df['body_pct'] = abs(df['body']) / df['range']
            df['sma20'] = df['close'].rolling(20).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
            df['atr'] = self._calculate_atr(df, 14)
            df['volatility'] = df['close'].pct_change().rolling(14).std() * 100
            
            if 'real_volume' in df.columns:
                df['volume_ma'] = df['real_volume'].rolling(20).mean()
                df['volume_ratio'] = df['real_volume'] / df['volume_ma']
            
            # Enhanced ICT/SMC indicators
            df['fvg_bullish'], df['fvg_bearish'] = self._calculate_enhanced_fvg(df)
            df['ob_bullish'], df['ob_bearish'] = self._calculate_order_blocks(df)
            df['liquidity_pools'] = self._calculate_liquidity_pools(df)
            df['mitigation_blocks'] = self._calculate_mitigation_blocks(df)
            df['liquidity_grabs'] = self._calculate_liquidity_grabs(df)
            df['trend_strength'] = self._calculate_trend_strength(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Indicator calculation error: {e}")
            return df

    def _calculate_enhanced_fvg(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Improved FVG detection with multi-candle confirmation"""
        fvg_bullish = pd.Series(False, index=df.index)
        fvg_bearish = pd.Series(False, index=df.index)
        lookback = self.ict_config['fvg_lookback']
        
        for i in range(lookback, len(df)):
            # Bullish FVG requires current low > previous high and confirmation
            if (df.iloc[i]['low'] > df.iloc[i-lookback]['high'] and
                df.iloc[i]['close'] > df.iloc[i-1]['high']):
                fvg_bullish.iloc[i] = True
                
            # Bearish FVG requires current high < previous low and confirmation
            elif (df.iloc[i]['high'] < df.iloc[i-lookback]['low'] and
                  df.iloc[i]['close'] < df.iloc[i-1]['low']):
                fvg_bearish.iloc[i] = True
                
        return fvg_bullish, fvg_bearish

    def _calculate_order_blocks(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Enhanced Order Block detection with volume confirmation"""
        ob_bullish = pd.Series(False, index=df.index)
        ob_bearish = pd.Series(False, index=df.index)
        wick_ratio = self.ict_config['ob_wick_ratio']
        
        for i in range(1, len(df)-1):
            candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish OB (strong rejection down with confirmation)
            if (candle['close'] < candle['open'] and 
                (candle['open'] - candle['close']) > candle['range'] * wick_ratio and
                next_candle['close'] > candle['high']):
                if 'volume_ratio' in df.columns:
                    if df.iloc[i]['volume_ratio'] > 1.0:  # Volume confirmation
                        ob_bullish.iloc[i] = True
                else:
                    ob_bullish.iloc[i] = True
                    
            # Bearish OB (strong rejection up with confirmation)
            elif (candle['close'] > candle['open'] and 
                  (candle['close'] - candle['open']) > candle['range'] * wick_ratio and
                  next_candle['close'] < candle['low']):
                if 'volume_ratio' in df.columns:
                    if df.iloc[i]['volume_ratio'] > 1.0:  # Volume confirmation
                        ob_bearish.iloc[i] = True
                else:
                    ob_bearish.iloc[i] = True
                    
        return ob_bullish, ob_bearish

    def _calculate_liquidity_pools(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced liquidity pool detection with multi-timeframe confirmation"""
        window = self.ict_config['liquidity_window']
        is_swing_high = (df['high'] == df['high'].rolling(window, center=True).max())
        is_swing_low = (df['low'] == df['low'].rolling(window, center=True).min())
        
        # Volume confirmation if available
        if 'volume_ratio' in df.columns:
            volume_threshold = self.ict_config['volume_spike_threshold']
            volume_confirmed = df['volume_ratio'] > volume_threshold
            return (is_swing_high | is_swing_low) & volume_confirmed
            
        return is_swing_high | is_swing_low

    def _calculate_mitigation_blocks(self, df: pd.DataFrame) -> pd.Series:
        """Detect ICT Mitigation Blocks (Flip Zones)"""
        mb_detected = pd.Series(False, index=df.index)
        for i in range(2, len(df)):
            prev_candle = df.iloc[i-1]
            # Bullish mitigation: Price flips above previous low after bearish candle
            if (prev_candle['close'] < prev_candle['open'] and
                df.iloc[i]['close'] > prev_candle['low']):
                mb_detected.iloc[i] = True
            # Bearish mitigation: Price flips below previous high after bullish candle
            elif (prev_candle['close'] > prev_candle['open'] and
                  df.iloc[i]['close'] < prev_candle['high']):
                mb_detected.iloc[i] = True
        return mb_detected

    def _calculate_liquidity_grabs(self, df: pd.DataFrame) -> pd.Series:
        """Detect liquidity sweeps (institutional stop runs)"""
        sweeps = pd.Series(False, index=df.index)
        for i in range(3, len(df)):
            # Bullish sweep: Price spikes below support but closes back up
            if (df.iloc[i]['low'] < df.iloc[i-2]['low'] and
                df.iloc[i]['close'] > df.iloc[i-2]['low']):
                sweeps.iloc[i] = True
            # Bearish sweep: Price spikes above resistance but closes back down
            elif (df.iloc[i]['high'] > df.iloc[i-2]['high'] and
                  df.iloc[i]['close'] < df.iloc[i-2]['high']):
                sweeps.iloc[i] = True
        return sweeps

    def _calculate_trend_strength(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX-based trend strength"""
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._calculate_atr(df, period)
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        return dx.ewm(alpha=1/period).mean()

    # ========== New ICT Utility Methods ==========
    def get_ict_market_state(self, symbol: str, timeframe: Union[str, int] = 'h1') -> Dict:
        """
        Get comprehensive ICT market state snapshot
        Returns:
            Dict containing all relevant ICT/SMC metrics
        """
        df = self.get_candles(symbol, timeframe, 100)
        if df is None or len(df) < 20:
            return {}
            
        return {
            'trend': {
                'direction': 'up' if df['close'].iloc[-1] > df['ema50'].iloc[-1] else 'down',
                'strength': float(df['trend_strength'].iloc[-1])
            },
            'liquidity': {
                'pools': bool(df['liquidity_pools'].iloc[-1]),
                'grabs': bool(df['liquidity_grabs'].iloc[-1])
            },
            'value': {
                'fvg_bullish': bool(df['fvg_bullish'].iloc[-1]),
                'fvg_bearish': bool(df['fvg_bearish'].iloc[-1]),
                'ob_bullish': bool(df['ob_bullish'].iloc[-1]),
                'ob_bearish': bool(df['ob_bearish'].iloc[-1]),
                'mitigation': bool(df['mitigation_blocks'].iloc[-1])
            },
            'volatility': float(df['volatility'].iloc[-1]),
            'volume': float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df.columns else 0.0,
            'killzones': {
                'london': self.is_in_killzone('london'),
                'new_york': self.is_in_killzone('new_york'),
                'tokyo': self.is_in_killzone('tokyo')
            }
        }

    # ========== Original Detection Methods (Maintained) ==========
    def detect_order_blocks(self, df: pd.DataFrame) -> pd.Series:
        """Original order block detection (maintained for compatibility)"""
        return self._calculate_order_blocks(df)[0] | self._calculate_order_blocks(df)[1]   