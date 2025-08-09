import json
import os
import logging
import math
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

class StrategyManager:
    def __init__(self, path: str = "data/strategies.json"):
        self.path = path
        self.weights_path = "data/strategy_weights.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({"user": [], "generated": []}, f)
        
        # Initialize weights file if it doesn't exist
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        if not os.path.exists(self.weights_path):
            with open(self.weights_path, "w") as f:
                json.dump({}, f)
    
    def load_strategies(self) -> Dict[str, List[Dict]]:
        """Load strategies from JSON file"""
        try:
            with open(self.path, "r") as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"Failed to load strategies: {str(e)}")
            return {"user": [], "generated": []}
    
    def load_weights(self) -> Dict[str, float]:
        """Load strategy weights from JSON file"""
        try:
            with open(self.weights_path, "r") as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"Failed to load strategy weights: {str(e)}")
            return {}
    
    def save_strategies(self, strategies: dict) -> None:
        """Save strategies to JSON file"""
        try:
            with open(self.path, "w") as file:
                json.dump(strategies, file, indent=2)
        except Exception as e:
            logging.error(f"Failed to save strategies: {str(e)}")
    
    def save_weights(self, weights: Dict[str, float]) -> None:
        """Save strategy weights to JSON file"""
        try:
            with open(self.weights_path, "w") as file:
                json.dump(weights, file, indent=2)
        except Exception as e:
            logging.error(f"Failed to save strategy weights: {str(e)}")

    def get_all_strategies(self) -> List[Dict]:
        """Get all strategies (both user and generated)"""
        data = self.load_strategies()
        return data["user"] + data["generated"]

    def get_active_strategies(self) -> List[Dict]:
        """Get only active strategies with weights applied"""
        data = self.load_strategies()
        weights = self.load_weights()
        
        active_strategies = []
        for strategy in data["user"] + data["generated"]:
            if strategy.get("active", True):
                # Apply weight if it exists
                strategy_name = strategy.get("name", "")
                if strategy_name in weights:
                    strategy["weight"] = weights[strategy_name]
                active_strategies.append(strategy)
        
        return active_strategies

    def add_user_strategy(self, strategy: Dict) -> None:
        """Add a new user-defined strategy"""
        data = self.load_strategies()
        strategy["version"] = 1
        strategy["created"] = datetime.now().isoformat()
        strategy["active"] = True
        data["user"].append(strategy)
        self.save_strategies(data)

    def add_generated_strategies(self, new_strategies: List[Dict]) -> None:
        """Add newly generated strategies"""
        data = self.load_strategies()
        data["generated"].extend(new_strategies)
        self.save_strategies(data)
    
    def update_strategy_weights(self, loss_patterns: Dict) -> None:
        """
        Update strategy weights based on loss patterns analysis
        Args:
            loss_patterns: Dictionary containing analysis of losing trades
                          Expected keys: 'time_of_day', 'market_conditions', 
                          'strategy_patterns', 'symbol_distribution'
        """
        weights = self.load_weights()
        strategies = self.get_all_strategies()
        
        # Initialize weights for all strategies if they don't exist
        for strategy in strategies:
            strategy_name = strategy.get("name", "")
            if strategy_name and strategy_name not in weights:
                weights[strategy_name] = 1.0  # Default weight
        
        # Adjust weights based on loss patterns
        if 'strategy_patterns' in loss_patterns:
            for strategy_name, loss_ratio in loss_patterns['strategy_patterns'].items():
                if strategy_name in weights:
                    # Reduce weight for strategies with high loss ratio
                    weights[strategy_name] *= max(0.5, 1 - loss_ratio)  # Reduce by up to 50%
        
        if 'time_of_day' in loss_patterns:
            # Get current hour
            current_hour = datetime.now().hour
            hour_loss_ratio = loss_patterns['time_of_day'].get(current_hour, 0)
            
            # Adjust all strategies active during high-loss hours
            if hour_loss_ratio > 0.2:  # Significant loss ratio for this hour
                for strategy in strategies:
                    if strategy.get("active", True):
                        strategy_name = strategy.get("name", "")
                        if strategy_name in weights:
                            weights[strategy_name] *= max(0.7, 1 - hour_loss_ratio)
        
        if 'market_conditions' in loss_patterns:
            market_conditions = loss_patterns['market_conditions']
            # Reduce weights for strategies that perform poorly in current market conditions
            if market_conditions.get('high_volatility', 0) > 0.3:
                for strategy in strategies:
                    if "high_volatility" in strategy.get("tags", []):
                        strategy_name = strategy.get("name", "")
                        if strategy_name in weights:
                            weights[strategy_name] *= 0.8
            
            if market_conditions.get('low_volatility', 0) > 0.3:
                for strategy in strategies:
                    if "low_volatility" in strategy.get("tags", []):
                        strategy_name = strategy.get("name", "")
                        if strategy_name in weights:
                            weights[strategy_name] *= 0.8
        
        # Ensure weights stay within reasonable bounds
        for strategy_name in weights:
            weights[strategy_name] = max(0.1, min(2.0, weights[strategy_name]))
        
        self.save_weights(weights)
        logging.info(f"Updated strategy weights based on loss patterns")
    
    def refresh_strategies(self, results: List[Dict], trade_history: Optional[List[Dict]] = None) -> None:
        """
        Refresh strategy pool based on performance results and trade history
        Args:
            results: List of strategy evaluation results
            trade_history: Optional list of recent trades for additional context
        """
        data = self.load_strategies()
        weights = self.load_weights()
        
        # Calculate performance metrics
        strategy_performance = {}
        for result in results:
            strategy_name = result.get("name", "")
            if strategy_name:
                score = result.get("score", 0)
                win_rate = result.get("win_rate", 0)
                profit = result.get("profit", 0)
                strategy_performance[strategy_name] = {
                    'score': score,
                    'win_rate': win_rate,
                    'profit': profit
                }
        
        # Add trade history context if available
        if trade_history:
            trade_counts = defaultdict(int)
            trade_profits = defaultdict(float)
            for trade in trade_history:
                strategy_name = trade.get("strategy", "")
                if strategy_name:
                    trade_counts[strategy_name] += 1
                    trade_profits[strategy_name] += trade.get("profit", 0)
            
            for strategy_name in trade_counts:
                if strategy_name in strategy_performance:
                    strategy_performance[strategy_name]['trade_count'] = trade_counts[strategy_name]
                    strategy_performance[strategy_name]['realized_profit'] = trade_profits[strategy_name]
        
        # Update strategy status and weights
        for strategy in data["generated"]:
            strategy_name = strategy.get("name", "")
            if strategy_name in strategy_performance:
                perf = strategy_performance[strategy_name]
                
                # Deactivate underperforming strategies
                if perf.get('win_rate', 0) < 40 or perf.get('profit', 0) < 0:
                    strategy["active"] = False
                    weights[strategy_name] = max(0.1, weights.get(strategy_name, 1.0) * 0.5)
                
                # Boost high performers
                elif perf.get('win_rate', 0) > 50 and perf.get('profit', 0) > 0:
                    strategy["active"] = True
                    weights[strategy_name] = min(2.0, weights.get(strategy_name, 1.0) * 1.2)
        
        # Keep top performers active regardless
        top_performers = sorted(strategy_performance.items(), 
                               key=lambda x: x[1].get('score', 0), 
                               reverse=True)[:5]
        for strategy_name, _ in top_performers:
            for strategy in data["generated"]:
                if strategy.get("name", "") == strategy_name:
                    strategy["active"] = True
                    weights[strategy_name] = min(2.0, weights.get(strategy_name, 1.0) * 1.5)
                    break
        
        self.save_strategies(data)
        self.save_weights(weights)
        logging.info("Refreshed strategy pool and weights based on performance")
    
    def log_performance(self, results: List[Dict]) -> None:
        """Log strategy performance metrics"""
        log_dir = "logs/performance"
        os.makedirs(log_dir, exist_ok=True)
        
        filename = f"performance_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = os.path.join(log_dir, filename)
        
        try:
            with open(filepath, "a") as f:
                for result in results:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "strategy": result["name"],
                        "symbol": result.get("symbol", "N/A"),
                        "win_rate": result["win_rate"],
                        "profit": result["profit"],
                        "direction": result["direction"],
                        "score": result.get("score", 0),
                        "weight": self.load_weights().get(result["name"], 1.0)
                    }
                    f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logging.error(f"Failed to log performance: {str(e)}")