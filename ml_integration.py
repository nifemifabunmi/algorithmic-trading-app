# ml_integration.py
"""
Machine Learning Integration for Trading Strategy Enhancement
Advanced ML models for signal prediction, risk assessment, and portfolio optimization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging
from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from main import moving_avg_strategy
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FeatureEngineering:
    """
    Advanced feature engineering for financial time series data.
    """
    
    def __init__(self):
        self.feature_names = []
        self.scaler = None
        
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical analysis features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical features
        """
        features_df = data.copy()
        
        # Price-based features
        features_df['returns'] = features_df['Close'].pct_change()
        features_df['log_returns'] = np.log(features_df['Close'] / features_df['Close'].shift(1))
        
        # Moving averages of different periods
        for period in [5, 10, 20, 50, 100, 200]:
            features_df[f'ma_{period}'] = features_df['Close'].rolling(window=period).mean()
            features_df[f'price_ma_{period}_ratio'] = features_df['Close'] / features_df[f'ma_{period}']
            features_df[f'ma_{period}_slope'] = features_df[f'ma_{period}'].diff(5)
        
        # Exponential moving averages
        for period in [12, 26, 9]:
            features_df[f'ema_{period}'] = features_df['Close'].ewm(span=period).mean()
        
        # MACD
        features_df['macd'] = features_df['ema_12'] - features_df['ema_26']
        features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # RSI
        features_df['rsi'] = self._calculate_rsi(features_df['Close'])
        features_df['rsi_sma'] = features_df['rsi'].rolling(window=14).mean()
        
        # Bollinger Bands
        features_df['bb_upper'], features_df['bb_lower'], features_df['bb_middle'] = self._calculate_bollinger_bands(features_df['Close'])
        features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / features_df['bb_middle']
        features_df['bb_position'] = (features_df['Close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # Volatility features
        for period in [10, 20, 30]:
            features_df[f'volatility_{period}'] = features_df['returns'].rolling(window=period).std()
            features_df[f'volatility_ratio_{period}'] = features_df[f'volatility_{period}'] / features_df[f'volatility_{period}'].rolling(window=60).mean()
        
        # Volume features
        if 'Volume' in features_df.columns:
            features_df['volume_ma'] = features_df['Volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_ma']
            features_df['price_volume'] = features_df['Close'] * features_df['Volume']
        
        # Price momentum features
        for period in [1, 3, 5, 10, 20]:
            features_df[f'momentum_{period}'] = features_df['Close'] / features_df['Close'].shift(period) - 1
            features_df[f'price_change_{period}'] = features_df['Close'].diff(period)
        
        # Statistical features
        for window in [10, 20, 30]:
            features_df[f'price_std_{window}'] = features_df['Close'].rolling(window=window).std()
            features_df[f'price_skew_{window}'] = features_df['returns'].rolling(window=window).skew()
            features_df[f'price_kurt_{window}'] = features_df['returns'].rolling(window=window).kurt()
        
        # Support and Resistance levels
        features_df['high_20'] = features_df['High'].rolling(window=20).max()
        features_df['low_20'] = features_df['Low'].rolling(window=20).min()
        features_df['support_resistance_ratio'] = (features_df['Close'] - features_df['low_20']) / (features_df['high_20'] - features_df['low_20'])
        
        # Fibonacci retracement levels
        features_df = self._add_fibonacci_features(features_df)
        
        # Market regime features
        features_df = self._add_regime_features(features_df)
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
            features_df[f'rsi_lag_{lag}'] = features_df['rsi'].shift(lag)
            features_df[f'volume_lag_{lag}'] = features_df['Volume'].shift(lag) if 'Volume' in features_df.columns else 0
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_mult: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_mult)
        lower = middle - (std * std_mult)
        return upper, lower, middle
    
    def _add_fibonacci_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fibonacci retracement features."""
        # Calculate Fibonacci levels based on recent high/low
        window = 50
        df['fib_high'] = df['High'].rolling(window=window).max()
        df['fib_low'] = df['Low'].rolling(window=window).min()
        df['fib_range'] = df['fib_high'] - df['fib_low']
        
        # Fibonacci levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for level in fib_levels:
            df[f'fib_{int(level*1000)}'] = df['fib_high'] - (df['fib_range'] * level)
            df[f'price_fib_{int(level*1000)}_ratio'] = df['Close'] / df[f'fib_{int(level*1000)}']
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features."""
        # Trend strength
        df['trend_strength'] = np.abs(df['ma_50'].diff(10))
        
        # Market state indicators
        df['bull_market'] = (df['ma_20'] > df['ma_50']).astype(int)
        df['bear_market'] = (df['ma_20'] < df['ma_50']).astype(int)
        
        # Volatility regime
        vol_20 = df['returns'].rolling(window=20).std()
        vol_percentile = vol_20.rolling(window=252).quantile(0.8)
        df['high_vol_regime'] = (vol_20 > vol_percentile).astype(int)
        
        return df
    
    def create_target_variables(self, data: pd.DataFrame, prediction_horizon: int = 5) -> pd.DataFrame:
        """
        Create target variables for different ML tasks.
        
        Args:
            data: DataFrame with price data
            prediction_horizon: Number of periods ahead to predict
            
        Returns:
            DataFrame with target variables
        """
        targets_df = data.copy()
        
        # Price direction (classification)
        future_returns = data['Close'].shift(-prediction_horizon) / data['Close'] - 1
        targets_df['direction'] = np.where(future_returns > 0.01, 1, 
                                          np.where(future_returns < -0.01, -1, 0))
        
        # Future returns (regression)
        targets_df['future_returns'] = future_returns
        
        # Volatility prediction
        targets_df['future_volatility'] = data['returns'].shift(-prediction_horizon).rolling(window=prediction_horizon).std()
        
        # Drawdown prediction
        future_prices = data['Close'].shift(-prediction_horizon)
        future_max = data['Close'].rolling(window=prediction_horizon).max().shift(-prediction_horizon)
        targets_df['future_drawdown'] = (future_prices - future_max) / future_max
        
        return targets_df

class trading_strategy:
    """
    Machine Learning enhanced trading strategy.
    """
    
    def __init__(self, base_strategy: MovingAverageCrossoverStrategy):
        self.base_strategy = base_strategy
        self.feature_engineer = FeatureEngineering()
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.model_performance = {}
        
        # ML Configuration
        self.prediction_horizon = 5
        self.min_samples_for_training = 500
        self.retrain_frequency = 50  # Retrain every N days
        self.ensemble_weights = {}
        
    def prepare_ml_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """
        Prepare data for ML model training and prediction.
        
        Args:
            symbol: Stock symbol
            period: Data period
            
        Returns:
            DataFrame with features and targets
        """
        # Download base data
        self.base_strategy.download_data(symbol, period)
        self.base_strategy.calculate_technical_indicators()
        
        # Create comprehensive features
        features_df = self.feature_engineer.create_technical_features(self.base_strategy.data)
        
        # Create target variables
        targets_df = self.feature_engineer.create_target_variables(features_df, self.prediction_horizon)
        
        # Combine features and targets
        ml_data = pd.concat([features_df, targets_df[['direction', 'future_returns', 'future_volatility']]], axis=1)
        
        # Remove rows with missing targets
        ml_data = ml_data.dropna(subset=['direction'])
        
        logger.info(f"Prepared ML dataset with {len(ml_data)} samples and {ml_data.shape[1]} features for {symbol}")
        
        return ml_data
    
    def train_ensemble_models(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Train an ensemble of ML models for signal prediction.
        
        Args:
            data: Prepared ML dataset
            symbol: Stock symbol
            
        Returns:
            Dictionary with trained models and performance metrics
        """
        # Prepare features and targets
        feature_columns = [col for col in data.columns if col not in 
                          ['direction', 'future_returns', 'future_volatility', 'Date']]
        
        X = data[feature_columns].fillna(0)
        y_direction = data['direction']
        y_returns = data['future_returns']
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(50, len(feature_columns)))
        X_selected = selector.fit_transform(X, y_direction)
        selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        # Train and evaluate models
        model_scores = {}
        trained_models = {}
        
        for name, model in models.items():
            try:
                scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y_direction.iloc[train_idx], y_direction.iloc[val_idx]
                    
                    # Skip if not enough samples
                    if len(X_train) < 100:
                        continue
                    
                    # Train model
                    if name == 'gradient_boosting':
                        # Use returns for regression model
                        y_train_reg = y_returns.iloc[train_idx]
                        y_val_reg = y_returns.iloc[val_idx]
                        model.fit(X_train, y_train_reg)
                        y_pred_reg = model.predict(X_val)
                        # Convert regression to classification
                        y_pred = np.where(y_pred_reg > 0.01, 1, np.where(y_pred_reg < -0.01, -1, 0))
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                    
                    score = accuracy_score(y_val, y_pred)
                    scores.append(score)
                
                if scores:
                    avg_score = np.mean(scores)
                    model_scores[name] = avg_score
                    
                    # Retrain on full dataset
                    if name == 'gradient_boosting':
                        model.fit(X_scaled, y_returns)
                    else:
                        model.fit(X_scaled, y_direction)
                    
                    trained_models[name] = model
                    
                    logger.info(f"{name}: Average CV Score = {avg_score:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {name}: {str(e)}")
        
        # Calculate ensemble weights based on performance
        if model_scores:
            total_score = sum(model_scores.values())
            ensemble_weights = {name: score/total_score for name, score in model_scores.items()}
        else:
            ensemble_weights = {}
        
        # Store models and metadata
        self.models[symbol] = trained_models
        self.scalers[symbol] = scaler
        self.feature_names[symbol] = selected_features
        self.model_performance[symbol] = model_scores
        self.ensemble_weights[symbol] = ensemble_weights
        
        logger.info(f"Trained {len(trained_models)} models for {symbol}")
        
        return {
            'models': trained_models,
            'performance': model_scores,
            'weights': ensemble_weights,
            'feature_names': selected_features
        }
    
    def predict_signals(self, current_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Generate ML-enhanced trading signals.
        
        Args:
            current_data: Current market data with features
            symbol: Stock symbol
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        if symbol not in self.models or not self.models[symbol]:
            logger.warning(f"No trained models available for {symbol}")
            return self._get_base_strategy_signal(current_data)
        
        try:
            # Prepare features
            feature_columns = self.feature_names[symbol]
            X = current_data[feature_columns].fillna(0).iloc[-1:] # Latest observation
            
            # Scale features
            X_scaled = self.scalers[symbol].transform(X)
            
            # Generate predictions from each model
            predictions = {}
            probabilities = {}
            
            for name, model in self.models[symbol].items():
                try:
                    if name == 'gradient_boosting':
                        # Regression model - convert to classification
                        pred_return = model.predict(X_scaled)[0]
                        pred = 1 if pred_return > 0.01 else (-1 if pred_return < -0.01 else 0)
                        prob = abs(pred_return)
                    else:
                        pred = model.predict(X_scaled)[0]
                        if hasattr(model, 'predict_proba'):
                            probs = model.predict_proba(X_scaled)[0]
                            prob = max(probs)
                        else:
                            prob = 0.5
                    
                    predictions[name] = pred
                    probabilities[name] = prob
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for {name}: {str(e)}")
            
            # Ensemble prediction
            if predictions:
                ensemble_signal = self._calculate_ensemble_prediction(predictions, symbol)
                ensemble_confidence = np.mean(list(probabilities.values()))
                
                # Combine with base strategy
                base_signal = self._get_base_strategy_signal(current_data)['signal']
                
                # Final signal logic
                if ensemble_confidence > 0.7:  # High confidence
                    final_signal = ensemble_signal
                elif ensemble_signal == base_signal:  # Agreement
                    final_signal = ensemble_signal
                else:  # Low confidence or disagreement - use base strategy
                    final_signal = base_signal
                
                return {
                    'signal': final_signal,
                    'confidence': ensemble_confidence,
                    'ensemble_signal': ensemble_signal,
                    'base_signal': base_signal,
                    'individual_predictions': predictions,
                    'probabilities': probabilities
                }
            
            else:
                return self._get_base_strategy_signal(current_data)
                
        except Exception as e:
            logger.error(f"ML prediction failed for {symbol}: {str(e)}")
            return self._get_base_strategy_signal(current_data)
    
    def _calculate_ensemble_prediction(self, predictions: Dict[str, int], symbol: str) -> int:
        """Calculate weighted ensemble prediction."""
        if not predictions:
            return 0
        
        weights = self.ensemble_weights.get(symbol, {})
        
        # If no weights, use equal weighting
        if not weights:
            return int(np.sign(np.mean(list(predictions.values()))))
        
        # Weighted average
        weighted_sum = sum(pred * weights.get(name, 0) for name, pred in predictions.items())
        
        return int(np.sign(weighted_sum)) if abs(weighted_sum) > 0.3 else 0
    
    def _get_base_strategy_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get signal from base moving average strategy."""
        if len(data) < self.base_strategy.long_window:
            return {'signal': 0, 'confidence': 0.0}
        
        short_ma = data['Close'].rolling(window=self.base_strategy.short_window).mean().iloc[-1]
        long_ma = data['Close'].rolling(window=self.base_strategy.long_window).mean().iloc[-1]
        
        if short_ma > long_ma:
            signal = 1
        elif short_ma < long_ma:
            signal = -1
        else:
            signal = 0
        
        return {'signal': signal, 'confidence': 0.6}
    
    def backtest_ml_strategy(self, symbol: str, period: str = "2y", 
                           initial_capital: float = 100000) -> pd.DataFrame:
        """
        Backtest the ML-enhanced strategy.
        
        Args:
            symbol: Stock symbol
            period: Backtesting period
            initial_capital: Starting capital
            
        Returns:
            DataFrame with backtest results
        """
        # Prepare data
        ml_data = self.prepare_ml_data(symbol, period)
        
        # Split data for training and testing
        split_point = int(len(ml_data) * 0.7)
        train_data = ml_data.iloc[:split_point]
        test_data = ml_data.iloc[split_point:]
        
        # Train models on training data
        self.train_ensemble_models(train_data, symbol)
        
        # Simulate trading on test data
        portfolio_value = initial_capital
        cash = initial_capital
        shares = 0
        portfolio_values = []
        signals = []
        ml_signals = []
        base_signals = []
        confidences = []
        
        for i in range(len(test_data)):
            current_data = ml_data.iloc[:split_point + i + 1]
            price = test_data['Close'].iloc[i]
            
            # Get ML prediction
            prediction = self.predict_signals(current_data, symbol)
            signal = prediction['signal']
            confidence = prediction['confidence']
            
            signals.append(signal)
            ml_signals.append(prediction.get('ensemble_signal', 0))
            base_signals.append(prediction.get('base_signal', 0))
            confidences.append(confidence)
            
            # Execute trades based on signals
            if signal == 1 and shares == 0:  # Buy signal
                shares = cash // price
                cash -= shares * price * 1.001  # Include transaction costs
            elif signal == -1 and shares > 0:  # Sell signal
                cash += shares * price * 0.999  # Include transaction costs
                shares = 0
            
            # Calculate portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
        
        # Create results DataFrame
        results_df = test_data.copy()
        results_df['Portfolio_Value'] = portfolio_values
        results_df['ML_Signal'] = signals
        results_df['ML_Ensemble_Signal'] = ml_signals
        results_df['Base_Signal'] = base_signals
        results_df['Confidence'] = confidences
        
        return results_df
    
    def calculate_ml_performance_metrics(self, backtest_results: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for ML strategy."""
        portfolio_values = backtest_results['Portfolio_Value']
        ml_signals = backtest_results['ML_Signal']
        base_signals = backtest_results['Base_Signal']
        
        # Basic performance
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        # Annualized metrics
        days = len(backtest_results)
        annualized_return = (1 + total_return) ** (252 / days) - 1
        
        daily_returns = portfolio_values.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # ML-specific metrics
        signal_accuracy = np.mean(ml_signals == base_signals)
        
        # Signal distribution
        signal_counts = pd.Series(ml_signals).value_counts()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'signal_accuracy': signal_accuracy,
            'buy_signals': signal_counts.get(1, 0),
            'sell_signals': signal_counts.get(-1, 0),
            'hold_signals': signal_counts.get(0, 0)
        }
        
        return metrics
    
    def plot_ml_analysis(self, backtest_results: pd.DataFrame, symbol: str):
        """Create comprehensive ML strategy analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'ML Trading Strategy Analysis - {symbol}', fontsize=16, fontweight='bold')
        
        # Plot 1: Portfolio performance comparison
        ax1 = axes[0, 0]
        ax1.plot(backtest_results.index, backtest_results['Portfolio_Value'], 
                label='ML Strategy', linewidth=2, color='blue')
        
        # Benchmark (buy and hold)
        initial_price = backtest_results['Close'].iloc[0]
        initial_value = backtest_results['Portfolio_Value'].iloc[0]
        benchmark = backtest_results['Close'] / initial_price * initial_value
        ax1.plot(backtest_results.index, benchmark, 
                label='Buy & Hold', linewidth=2, color='orange', alpha=0.7)
        
        ax1.set_title('Portfolio Performance Comparison')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Signal comparison
        ax2 = axes[0, 1]
        ax2.plot(backtest_results.index, backtest_results['ML_Signal'], 
                label='ML Signals', alpha=0.7, linewidth=2)
        ax2.plot(backtest_results.index, backtest_results['Base_Signal'], 
                label='Base Strategy', alpha=0.7, linewidth=2)
        ax2.set_title('Signal Comparison')
        ax2.set_ylabel('Signal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confidence over time
        ax3 = axes[1, 0]
        ax3.plot(backtest_results.index, backtest_results['Confidence'], 
                color='green', linewidth=1.5)
        ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='High Confidence Threshold')
        ax3.set_title('ML Prediction Confidence')
        ax3.set_ylabel('Confidence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Signal distribution
        ax4 = axes[1, 1]
        signal_counts = backtest_results['ML_Signal'].value_counts()
        signal_labels = ['Sell', 'Hold', 'Buy']
        colors = ['red', 'gray', 'green']
        ax4.bar(signal_labels, [signal_counts.get(-1, 0), signal_counts.get(0, 0), signal_counts.get(1, 0)], 
               color=colors, alpha=0.7)
        ax4.set_title('Signal Distribution')
        ax4.set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, symbol: str, filepath: str):
        """Save trained models to disk."""
        if symbol in self.models:
            model_data = {
                'models': self.models[symbol],
                'scalers': self.scalers[symbol],
                'feature_names': self.feature_names[symbol],
                'performance': self.model_performance[symbol],
                'ensemble_weights': self.ensemble_weights[symbol]
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
    
    def load_models(self, symbol: str, filepath: str):
        """Load trained models from disk."""
        try:
            model_data = joblib.load(filepath)
            self.models[symbol] = model_data['models']
            self.scalers[symbol] = model_data['scalers']
            self.feature_names[symbol] = model_data['feature_names']
            self.model_performance[symbol] = model_data['performance']
            self.ensemble_weights[symbol] = model_data['ensemble_weights']
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")

def demo_ml_strategy():
    """Demonstration of ML-enhanced trading strategy."""
    
    # Initialize base strategy
    base_strategy = MovingAverageCrossoverStrategy(short_window=20, long_window=50)
    
    # Initialize ML strategy
    ml_strategy = MLTradingStrategy(base_strategy)
    
    # Test symbol
    symbol = "AAPL"
    
    print(f"Running ML strategy demo for {symbol}...")
    
    # Prepare data and train models
    print("Preparing ML data...")
    ml_data = ml_strategy.prepare_ml_data(symbol, "2y")
    print(f"Dataset prepared: {len(ml_data)} samples, {ml_data.shape[1]} features")
    
    # Run backtest
    print("Running ML backtest...")
    results = ml_strategy.backtest_ml_strategy(symbol, "2y", 100000)
    
    # Calculate and display performance
    metrics = ml_strategy.calculate_ml_performance_metrics(results)
    
    print(f"\nML Strategy Performance for {symbol}:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Signal Accuracy vs Base: {metrics['signal_accuracy']:.1%}")
    print(f"Buy Signals: {metrics['buy_signals']}")
    print(f"Sell Signals: {metrics['sell_signals']}")
    print(f"Hold Signals: {metrics['hold_signals']}")
    
    # Show model performance
    if symbol in ml_strategy.model_performance:
        print(f"\nIndividual Model Performance:")
        for model_name, score in ml_strategy.model_performance[symbol].items():
            print(f"{model_name}: {score:.3f}")
    
    # Plot analysis
    ml_strategy.plot_ml_analysis(results, symbol)
    
    # Save models
    model_path = f"{symbol}_ml_models.joblib"
    ml_strategy.save_models(symbol, model_path)
    print(f"Models saved to {model_path}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    demo_ml_strategy()
        
