import logging
from Backtester import Backtester
import pandas as pd

class StrategyOptimizer:
    def __init__(self):
        self.backtester = Backtester()
    
    def optimize_parameters(self, strategy: Dict, market_data: pd.DataFrame) -> Dict:
        """
        Find optimal stop loss and take profit levels
        Returns best parameters: {"sl": best_sl, "tp": best_tp}
        """
        best_profit = float('-inf')
        best_params = {}
        
        # Test different parameter combinations
        for sl in [15, 20, 25]:
            for tp in [30, 40, 50]:
                # Create strategy variation
                strategy_var = strategy.copy()
                strategy_var["sl"] = sl
                strategy_var["tp"] = tp
                
                # Test performance
                result = self.backtester.test_strategy(strategy_var, market_data)
                
                # Track best performing parameters
                if result["profit"] > best_profit:
                    best_profit = result["profit"]
                    best_params = {"sl": sl, "tp": tp}
        
        logging.info(f"Optimized parameters: SL={best_params['sl']}, TP={best_params['tp']}")
        return best_params       