import time
import logging
from datetime import datetime 
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Custom modules
from strategies import StrategyManager
from learner import StrategyLearner
from evaluator import StrategyEvaluator
from data_fetcher import MT5DataFetcher
from executor import MT5Executor

# Configuration
MT5_LOGIN = 210194168
MT5_PASSWORD = "7$Hepang"
MT5_SERVER = "Exness-MT5Trial9"
SYMBOLS = ["EURJPYm", "US30m", "USTECm"]
LOT_SIZE = 0.02
TIMEFRAME = mt5.TIMEFRAME_H1
BARS = 1050
SLEEP_INTERVAL = 15 * 60  # 5 minutes
MAX_CONNECTION_ATTEMPTS = 10
CONNECTION_RETRY_DELAY = 5
MAX_POSITIONS_PER_SYMBOL = 1
MIN_WIN_RATE = 50.0
TRAILING_START = 10000 # Pips profit to activate trailing stop
TRAILING_STEP = 5000   # Pips for trailing step
BREAKEVEN_AT = 7000    # Pips profit to move to breakeven
DYNAMIC_RISK_ENABLED = False  # Enable dynamic lot sizing
MAX_LOT_SIZE = 0.03     # Maximum lot size for dynamic risk
MIN_LOT_SIZE = 0.01   # Minimum lot size
TRADE_ANALYSIS_LOOKBACK = 50  # Number of trades to analyze for patterns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_ea.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ReversalDetector:
    def __init__(self):
        self.patterns = {
            'pin_bar': self._detect_pin_bar,
            'engulfing': self._detect_engulfing,
            'rsi_divergence': self._detect_rsi_divergence,
            'macd_cross': self._detect_macd_cross
        }
        self.pattern_weights = {
            'pin_bar': 1.0,
            'engulfing': 1.2,
            'rsi_divergence': 1.5,
            'macd_cross': 1.3
        }
    
    def detect(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, Optional[str], float]:
        """
        Detect potential reversal signals with confidence score.
        Returns: (bool, str, float) - reversal detected, direction, and confidence score
        """
        signals = []
        confidence = 0.0
        
        # Check for RSI divergence
        rsi_div = self._detect_rsi_divergence(df)
        if rsi_div:
            signals.append(rsi_div)
            confidence += self.pattern_weights['rsi_divergence']
        
        # Check for MACD cross
        macd_cross = self._detect_macd_cross(df)
        if macd_cross:
            signals.append(macd_cross)
            confidence += self.pattern_weights['macd_cross']
        
        # Check candlestick patterns
        pin_bar = self._detect_pin_bar(df)
        if pin_bar:
            signals.append(pin_bar)
            confidence += self.pattern_weights['pin_bar']
        
        engulfing = self._detect_engulfing(df)
        if engulfing:
            signals.append(engulfing)
            confidence += self.pattern_weights['engulfing']
        
        # Analyze signals
        if not signals:
            return False, None, 0.0
        
        # Count bullish and bearish signals
        bullish_count = sum(1 for s in signals if s == 'bullish')
        bearish_count = sum(1 for s in signals if s == 'bearish')
        
        # Calculate confidence score (normalized to 0-1 range)
        max_possible = sum(self.pattern_weights.values())
        confidence = min(confidence / max_possible, 1.0)
        
        # Require at least 2 signals in the same direction
        if bullish_count >= 2:
            logger.info(f"Bullish reversal detected on {symbol} (confidence: {confidence:.2f})")
            return True, 'bullish', confidence
        elif bearish_count >= 2:
            logger.info(f"Bearish reversal detected on {symbol} (confidence: {confidence:.2f})")
            return True, 'bearish', confidence
        
        return False, None, 0.0
    
    def _detect_rsi_divergence(self, df, period=14, lookback=20):
        """Detect bullish/bearish RSI divergence"""
        if 'rsi' not in df.columns or len(df) < lookback + period:
            return None
            
        prices = df['close'].tail(lookback).values
        rsis = df['rsi'].tail(lookback).values
        
        # Find peaks and troughs
        price_peaks = np.argsort(prices)[-2:]
        price_troughs = np.argsort(prices)[:2]
        rsi_peaks = np.argsort(rsis)[-2:]
        rsi_troughs = np.argsort(rsis)[:2]
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if price_peaks[-1] > price_peaks[-2] and rsi_peaks[-1] < rsi_peaks[-2]:
                return 'bearish'
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            if price_troughs[0] < price_troughs[1] and rsi_troughs[0] > rsi_troughs[1]:
                return 'bullish'
        
        return None

    def _detect_macd_cross(self, df):
        """Detect MACD crossover/crossunder"""
        if 'macd_line' not in df.columns or 'macd_signal' not in df.columns or len(df) < 3:
            return None
            
        macd_line = df['macd_line'].tail(2).values
        macd_signal = df['macd_signal'].tail(2).values
        
        # Bullish cross: MACD crosses above signal line
        if macd_line[-2] < macd_signal[-2] and macd_line[-1] > macd_signal[-1]:
            return 'bullish'
        
        # Bearish cross: MACD crosses below signal line
        if macd_line[-2] > macd_signal[-2] and macd_line[-1] < macd_signal[-1]:
            return 'bearish'
        
        return None

    def _detect_pin_bar(self, df):
        """Detect bullish/bearish pin bar patterns"""
        if len(df) < 1:
            return None
            
        candle = df.iloc[-1]
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return None
            
        body_ratio = body_size / total_range
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        # Bullish pin bar: long lower wick
        if body_ratio < 0.3 and lower_wick > 2 * upper_wick and lower_wick > 2 * body_size:
            return 'bullish'
        
        # Bearish pin bar: long upper wick
        if body_ratio < 0.3 and upper_wick > 2 * lower_wick and upper_wick > 2 * body_size:
            return 'bearish'
        
        return None

    def _detect_engulfing(self, df):
        """Detect bullish/bearish engulfing patterns"""
        if len(df) < 2:
            return None
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish engulfing: current green candle completely engulfs previous red candle
        if (current['close'] > current['open'] and 
            prev['close'] < prev['open'] and
            current['open'] < prev['close'] and 
            current['close'] > prev['open']):
            return 'bullish'
        
        # Bearish engulfing: current red candle completely engulfs previous green candle
        if (current['close'] < current['open'] and 
            prev['close'] > prev['open'] and
            current['open'] > prev['close'] and 
            current['close'] < prev['open']):
            return 'bearish'
        
        return None


class TradeAnalyzer:
    def __init__(self):
        self.trade_history = []
        self.pattern_cache = defaultdict(list)
        self.volatility_data = defaultdict(list)
    
    def add_trade(self, trade_data: Dict, market_data: Optional[pd.DataFrame] = None):
        """Record a completed trade with market context"""
        if market_data is not None:
            # Calculate volatility metrics if market data is provided
            trade_data['volatility'] = market_data['volatility'].iloc[-1] if 'volatility' in market_data.columns else 0
            trade_data['avg_volatility'] = market_data['volatility'].mean() if 'volatility' in market_data.columns else 0
            trade_data['trend_strength'] = market_data['trend_strength'].iloc[-1] if 'trend_strength' in market_data.columns else 0
            
            # Store additional market context
            trade_data['market_context'] = {
                'rsi': market_data['rsi'].iloc[-1] if 'rsi' in market_data.columns else None,
                'atr': market_data['atr'].iloc[-1] if 'atr' in market_data.columns else None,
                'volume': market_data['volume'].iloc[-1] if 'volume' in market_data.columns else None
            }
            
            # Update volatility history for this symbol
            self.volatility_data[trade_data['symbol']].append(trade_data['volatility'])
        
        self.trade_history.append(trade_data)
        if len(self.trade_history) > TRADE_ANALYSIS_LOOKBACK:
            self.trade_history.pop(0)
    
    def analyze_losing_patterns(self) -> Dict:
        """Identify common patterns in losing trades with fallback values"""
        if not self.trade_history:
            return {}
            
        losing_trades = [t for t in self.trade_history if t.get('profit', 0) <= 0]
        if not losing_trades:
            return {}
        
        analysis = {
            'time_of_day': self._analyze_time_patterns(losing_trades),
            'market_conditions': self._analyze_market_conditions(losing_trades),
            'strategy_patterns': self._analyze_strategy_patterns(losing_trades),
            'symbol_distribution': self._analyze_symbol_distribution(losing_trades)
        }
        
        return analysis
    
    def _analyze_time_patterns(self, trades: List[Dict]) -> Dict:
        """Find time-based patterns in losing trades"""
        time_counts = defaultdict(int)
        for trade in trades:
            if 'time' in trade and isinstance(trade['time'], datetime):
                hour = trade['time'].hour
                time_counts[hour] += 1
        
        total_losing = len(trades) or 1  # Prevent division by zero
        return {hour: count/total_losing for hour, count in time_counts.items()}
    
    def _analyze_market_conditions(self, trades: List[Dict]) -> Dict:
        """Find market condition patterns with safe defaults"""
        condition_counts = {
            'high_volatility': 0,
            'low_volatility': 0,
            'strong_trend': 0,
            'ranging': 0,
            'unknown_conditions': 0
        }
        
        for trade in trades:
            # Get volatility with fallback values
            volatility = trade.get('volatility', 0)
            avg_volatility = trade.get('avg_volatility', 1)  # Default to 1 to avoid division by zero
            trend_strength = trade.get('trend_strength', 0)
            
            # Classify market conditions
            if volatility > avg_volatility * 1.5:
                condition_counts['high_volatility'] += 1
            elif volatility < avg_volatility * 0.5:
                condition_counts['low_volatility'] += 1
            else:
                condition_counts['unknown_conditions'] += 1
                
            if trend_strength > 0.7:
                condition_counts['strong_trend'] += 1
            elif trend_strength < 0.3:
                condition_counts['ranging'] += 1
            else:
                condition_counts['unknown_conditions'] += 1
        
        total_losing = len(trades) or 1  # Prevent division by zero
        return {k: v/total_losing for k, v in condition_counts.items()}
    
    def _analyze_strategy_patterns(self, trades: List[Dict]) -> Dict:
        """Find patterns in losing strategies with fallback"""
        strategy_counts = defaultdict(int)
        for trade in trades:
            strategy_name = trade.get('strategy', 'unknown')
            strategy_counts[strategy_name] += 1
        
        total_losing = len(trades) or 1
        return {k: v/total_losing for k, v in strategy_counts.items()}
    
    def _analyze_symbol_distribution(self, trades: List[Dict]) -> Dict:
        """Analyze which symbols have most losing trades"""
        symbol_counts = defaultdict(int)
        for trade in trades:
            symbol = trade.get('symbol', 'unknown')
            symbol_counts[symbol] += 1
        
        total_losing = len(trades) or 1
        return {k: v/total_losing for k, v in symbol_counts.items()}
    
    def get_volatility_profile(self, symbol: str) -> Dict:
        """Get volatility statistics for a specific symbol"""
        volatilities = self.volatility_data.get(symbol, [])
        if not volatilities:
            return {}
            
        return {
            'current': volatilities[-1] if volatilities else 0,
            'average': sum(volatilities)/len(volatilities) if volatilities else 0,
            'max': max(volatilities) if volatilities else 0,
            'min': min(volatilities) if volatilities else 0,
            'count': len(volatilities)
        }

def initialize_mt5() -> bool:
    """Initialize MT5 connection with retries"""
    attempts = 0
    while attempts < MAX_CONNECTION_ATTEMPTS:
        try:
            if not mt5.initialize():
                raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
            
            if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
                raise ConnectionError(f"MT5 login failed: {mt5.last_error()}")
            
            logger.info("Successfully connected to MT5")
            return True
        except Exception as e:
            attempts += 1
            logger.error(f"Connection attempt {attempts} failed: {str(e)}")
            time.sleep(CONNECTION_RETRY_DELAY)
    
    logger.critical("Failed to connect to MT5 after multiple attempts")
    return False

def calculate_dynamic_lot_size(win_rate: float, confidence: float) -> float:
    """
    Calculate lot size based on strategy performance and current confidence.
    Returns: Adjusted lot size between MIN_LOT_SIZE and MAX_LOT_SIZE
    """
    if not DYNAMIC_RISK_ENABLED:
        return LOT_SIZE
    
    # Base multiplier from win rate (0.5 for 45% WR, 1.5 for 65%+ WR)
    win_rate_mult = min(max((win_rate - MIN_WIN_RATE) / 20, 0.5), 1.0)
    
    # Confidence multiplier
    confidence_mult = 0.5 + (confidence * 1.0)  # 0.5-2.0 range
    
    # Combined adjustment
    adjusted_lot = LOT_SIZE * win_rate_mult * confidence_mult
    
    # Apply bounds
    return min(max(adjusted_lot, MIN_LOT_SIZE), MAX_LOT_SIZE)

def run_cycle(cycle_count: int, trade_analyzer: TradeAnalyzer):
    logger.info(f"Cycle #{cycle_count} start: {datetime.now()}")
    
    if not initialize_mt5():
        return

    try:
        # Initialize components
        strategy_manager = StrategyManager()
        strategy_learner = StrategyLearner()
        strategy_evaluator = StrategyEvaluator()
        fetcher = MT5DataFetcher()
        executor = MT5Executor(
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            server=MT5_SERVER
        )
        detector = ReversalDetector()

        # Load strategies with recent performance data
        all_strategies = strategy_manager.get_active_strategies()
        logger.info(f"Loaded {len(all_strategies)} active strategies")
        
        # Get recent trade history for learning
        recent_trades = executor.get_trade_history(days=7)
        for trade in recent_trades:
            trade_analyzer.add_trade(trade)
        
        # Analyze losing trade patterns
        loss_patterns = trade_analyzer.analyze_losing_patterns()
        if loss_patterns:
            logger.info(f"Loss patterns detected: {loss_patterns}")
            strategy_manager.update_strategy_weights(loss_patterns)
        
        # Process each symbol
        for symbol in SYMBOLS:
            logger.info(f"Processing symbol: {symbol}")
            
            # Fetch fresh market data with additional indicators
            try:
                market_data = fetcher.get_candles(symbol, TIMEFRAME, BARS)
                if market_data is None or len(market_data) < 100:
                    logger.error(f"Insufficient data for {symbol} ({len(market_data) if market_data else 0} bars)")
                    continue
                
                # Add volatility and trend indicators
                market_data = strategy_evaluator.add_market_indicators(market_data)
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                continue

            # Detect potential reversals with confidence
            reversal_detected, reversal_direction, reversal_confidence = detector.detect(market_data, symbol)
            
            # Manage existing positions with enhanced logic
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                positions = []
                
            for position in positions:
                try:
                    # Get current price
                    tick = mt5.symbol_info_tick(symbol)
                    current_price = tick.ask if position.type == mt5.ORDER_TYPE_BUY else tick.bid
                    
                    # Calculate profit in pips
                    pip_multiplier = executor.get_pip_multiplier(symbol)
                    profit = (current_price - position.price_open) * (1 if position.type == mt5.ORDER_TYPE_BUY else -1)
                    profit_pips = profit / pip_multiplier
                    
                    # Enhanced trailing stop with volatility adjustment
                    if profit_pips > TRAILING_START:
                        current_volatility = market_data['volatility'].iloc[-1]
                        trailing_step = TRAILING_STEP * (1 + current_volatility)  # Adjust for volatility
                        
                        new_sl = position.price_open + (trailing_step * pip_multiplier * 
                                                       (1 if position.type == mt5.ORDER_TYPE_BUY else -1))
                        
                        # Ensure SL is better than current
                        if ((position.type == mt5.ORDER_TYPE_BUY and new_sl > position.sl) or
                            (position.type == mt5.ORDER_TYPE_SELL and new_sl < position.sl)):
                            executor.update_sl(position.ticket, new_sl)
                            logger.info(f"Updated trailing SL for {symbol} to {new_sl:.5f} (volatility adjusted)")
                    
                    # Dynamic breakeven based on confidence
                    elif profit_pips > BREAKEVEN_AT * (1 - reversal_confidence):
                        if ((position.type == mt5.ORDER_TYPE_BUY and position.sl < position.price_open) or
                            (position.type == mt5.ORDER_TYPE_SELL and position.sl > position.price_open)):
                            executor.update_sl(position.ticket, position.price_open)
                            logger.info(f"Moved to breakeven for {symbol} (confidence adjusted)")
                    
                    # Close profitable trades before reversal turns them to loss
                    if reversal_detected and profit_pips > 0:
                        if ((reversal_direction == 'bearish' and position.type == mt5.ORDER_TYPE_BUY) or
                            (reversal_direction == 'bullish' and position.type == mt5.ORDER_TYPE_SELL)):
                            
                            logger.info(f"Closing profitable trade before reversal: {symbol} {position.ticket}")
                            executor.close_position(position.ticket)
                            
                            # Record early closure for analysis
                            trade_analyzer.add_trade({
                                'symbol': symbol,
                                'profit': profit_pips,
                                'strategy': 'reversal_avoidance',
                                'time': datetime.now(),
                                'volatility': market_data['volatility'].iloc[-1],
                                'avg_volatility': market_data['volatility'].mean(),
                                'trend_strength': market_data['trend_strength'].iloc[-1]
                            })
                
                except Exception as e:
                    logger.error(f"Error managing position {position.ticket}: {str(e)}")

            # Evaluate strategies with market context
            results = []
            market_context = {
                'volatility': market_data['volatility'].iloc[-1],
                'trend_strength': market_data['trend_strength'].iloc[-1],
                'reversal_confidence': reversal_confidence
            }
            
            for strategy in all_strategies:
                try:
                    result = strategy_evaluator.evaluate_strategy(
                        strategy, 
                        market_data, 
                        symbol,
                        market_context
                    )
                    result["strategy"] = strategy
                    results.append(result)
                    logger.info(f"Evaluated {strategy['name']} on {symbol}: "
                              f"Win Rate={result['win_rate']:.2f}%, Profit={result['profit']:.2f} pips")
                except Exception as e:
                    logger.error(f"Evaluation failed for {strategy['name']} on {symbol}: {str(e)}")

            # Find and execute best strategy with dynamic risk
            if results:
                # Sort strategies by score and select top 3
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
                top_strategies = results[:3]
                
                for best_strategy in top_strategies:
                    # Enhanced execution criteria with dynamic lot sizing
                    if best_strategy["win_rate"] > MIN_WIN_RATE and best_strategy["profit"] > 0:
                        try:
                            positions = mt5.positions_get(symbol=symbol) or []
                            positions_count = len(positions)
                            
                            if positions_count < MAX_POSITIONS_PER_SYMBOL:
                                # Calculate dynamic lot size
                                lot_size = calculate_dynamic_lot_size(
                                    best_strategy["win_rate"],
                                    best_strategy.get("confidence", 0.5)
                                )
                                
                                # Execute trade with dynamic parameters
                                executor.place_order(
                                    symbol=symbol,
                                    lot=lot_size,
                                    order_type=best_strategy['strategy']['direction'],
                                    confidence=best_strategy.get("confidence", 0.5)
                                )
                                logger.info(f"Executed {best_strategy['strategy']['direction']} order using "
                                          f"{best_strategy['strategy']['name']} (lot: {lot_size:.3f})")
                                
                                # Record trade intent for learning
                                trade_analyzer.add_trade({
                                    'symbol': symbol,
                                    'strategy': best_strategy['strategy']['name'],
                                    'direction': best_strategy['strategy']['direction'],
                                    'time': datetime.now(),
                                    'volatility': market_context['volatility'],
                                    'avg_volatility': market_data['volatility'].mean(),
                                    'trend_strength': market_context['trend_strength'],
                                    'confidence': best_strategy.get("confidence", 0.5)
                                })
                                break  # Only execute one trade per symbol
                            else:
                                logger.info(f"Max positions ({MAX_POSITIONS_PER_SYMBOL}) reached for {symbol}, skipping")
                        except Exception as e:
                            logger.error(f"Execution failed for {symbol}: {str(e)}")
                    else:
                        logger.info(f"No trade: Strategy criteria not met for {symbol} with {best_strategy['strategy']['name']}")
            else:
                logger.warning(f"No valid strategies for {symbol}")

        # Enhanced learning with trade feedback
        try:
            strategy_manager.log_performance(results)
            
            # Learn from both strategy evaluation and actual trades
            new_strategies = strategy_learner.learn_and_generate(
                all_strategies, 
                results,
                trade_analyzer.trade_history
            )
            
            strategy_manager.add_generated_strategies(new_strategies)
            logger.info(f"Added {len(new_strategies)} new strategies")
            
            # Adaptive strategy refresh based on performance
            if cycle_count % 5 == 0 or len(trade_analyzer.trade_history) >= TRADE_ANALYSIS_LOOKBACK:
                strategy_manager.refresh_strategies(results, trade_analyzer.trade_history)
                logger.info("Refreshed strategy pool based on performance and trade history")
                
        except Exception as e:
            logger.error(f"Learning/generation failed: {str(e)}")

    except Exception as e:
        logger.error(f"Unhandled error in cycle: {str(e)}", exc_info=True)
    finally:
        mt5.shutdown()
        logger.info("Disconnected from MT5")

    logger.info(f"Cycle #{cycle_count} complete")



def main():
    logger.info("AI EA running in live loop mode with enhanced learning")
    cycle_count = 0
    trade_analyzer = TradeAnalyzer()
    
    while True:
        cycle_count += 1
        run_cycle(cycle_count, trade_analyzer)
        
        # Adaptive sleep based on market conditions
        current_hour = datetime.now().hour
        if 0 <= current_hour < 5:  # Less frequent during low liquidity
            sleep_time = SLEEP_INTERVAL * 1.5
        else:
            sleep_time = SLEEP_INTERVAL
        
        logger.info(f"Sleeping {sleep_time//60} minutes")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()       