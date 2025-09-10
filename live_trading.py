# realtime_trading.py
"""
Real-Time Trading Implementation
Live data integration, demo trading, and market monitoring capabilities
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
import threading
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from queue import Queue
import warnings
from main import moving_avg_strategy
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Real-time market data structure."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None

@dataclass
class Order:
    """Trading order structure."""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    timestamp: datetime
    status: str = 'PENDING'  # 'PENDING', 'FILLED', 'CANCELLED'
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None

@dataclass
class Position:
    """Trading position structure."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)

class RealTimeDataProvider:
    """
    Real-time data provider using Yahoo Finance.
    In production, this would connect to professional data feeds.
    """
    
    def __init__(self, symbols: List[str], update_interval: int = 1):
        """
        Initialize real-time data provider.
        
        Args:
            symbols: List of symbols to monitor
            update_interval: Update frequency in seconds
        """
        self.symbols = symbols
        self.update_interval = update_interval
        self.is_running = False
        self.data_queue = Queue()
        self.current_data = {}
        self.subscribers = []
        self.thread = None
        
    def subscribe(self, callback: Callable[[MarketData], None]):
        """Subscribe to real-time data updates."""
        self.subscribers.append(callback)
        
    def start(self):
        """Start real-time data collection."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._data_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started real-time data provider for {len(self.symbols)} symbols")
        
    def stop(self):
        """Stop real-time data collection."""
        self.is_running = False
        if self.thread:
            self.thread.join()
        logger.info("Stopped real-time data provider")
        
    def _data_loop(self):
        """Main data collection loop."""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    market_data = self._fetch_current_data(symbol)
                    if market_data:
                        self.current_data[symbol] = market_data
                        self._notify_subscribers(market_data)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in data loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def _fetch_current_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch current market data for symbol."""
        try:
            # Get current data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            
            # Get current price and basic info
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if current_price is None:
                # Fallback to recent trading data
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                else:
                    return None
            
            # Create market data object
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=float(current_price),
                volume=info.get('regularMarketVolume', 0),
                bid=info.get('bid'),
                ask=info.get('ask'),
                high=info.get('dayHigh'),
                low=info.get('dayLow'),
                open=info.get('regularMarketOpen')
            )
            
            return market_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch data for {symbol}: {str(e)}")
            return None
    
    def _notify_subscribers(self, market_data: MarketData):
        """Notify all subscribers of new data."""
        for callback in self.subscribers:
            try:
                callback(market_data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {str(e)}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        data = self.current_data.get(symbol)
        return data.price if data else None

class DemoTradingEngine:
    """
    Demo trading engine for paper trading and strategy testing.
    Simulates realistic trading conditions with latency and slippage.
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 commission_rate: float = 0.001,
                 slippage_bps: float = 2.0):
        """
        Initialize demo trading engine.
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission as percentage of trade value
            slippage_bps: Slippage in basis points
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps / 10000  # Convert bps to decimal
        
        # Trading state
        self.positions = {}
        self.orders = {}
        self.order_counter = 0
        self.trade_history = []
        
        # Performance tracking
        self.equity_curve = []
        self.daily_pnl = []
        
        # Risk management
        self.max_position_size = 0.2  # 20% of capital per position
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        
        logger.info(f"Initialized demo trading engine with ${initial_capital:,.2f}")
    
    def place_order(self, symbol: str, side: str, quantity: int, 
                   price: Optional[float] = None, order_type: str = "MARKET") -> str:
        """
        Place a trading order.
        
        Args:
            symbol: Symbol to trade
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            price: Limit price (for limit orders)
            order_type: 'MARKET', 'LIMIT', or 'STOP'
            
        Returns:
            Order ID
        """
        # Generate order ID
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter:06d}"
        
        # Risk checks
        if not self._validate_order(symbol, side, quantity, price):
            logger.warning(f"Order validation failed for {order_id}")
            return None
        
        # Create order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price or 0.0,
            order_type=order_type,
            timestamp=datetime.now()
        )
        
        self.orders[order_id] = order
        
        # For demo purposes, execute market orders immediately
        if order_type == "MARKET":
            self._execute_order(order_id)
        
        logger.info(f"Placed {side} order for {quantity} {symbol}: {order_id}")
        return order_id
    
    def _validate_order(self, symbol: str, side: str, quantity: int, price: Optional[float]) -> bool:
        """Validate order against risk management rules."""
        
        # Check available capital for buy orders
        if side == "BUY":
            estimated_cost = quantity * (price or 100)  # Rough estimate
            if estimated_cost > self.current_capital:
                logger.warning("Insufficient capital for buy order")
                return False
        
        # Check position size limits
        current_position = self.positions.get(symbol, Position(symbol, 0, 0.0))
        new_quantity = current_position.quantity + (quantity if side == "BUY" else -quantity)
        
        position_value = abs(new_quantity) * (price or 100)
        if position_value > self.current_capital * self.max_position_size:
            logger.warning("Order exceeds maximum position size limit")
            return False
        
        # Check for sufficient shares to sell
        if side == "SELL" and current_position.quantity < quantity:
            logger.warning("Insufficient shares for sell order")
            return False
        
        return True
    
    def _execute_order(self, order_id: str):
        """Execute a pending order."""
        order = self.orders.get(order_id)
        if not order:
            return
        
        try:
            # Simulate market data lookup (in real implementation, use live data)
            ticker = yf.Ticker(order.symbol)
            current_info = ticker.info
            market_price = current_info.get('currentPrice') or current_info.get('regularMarketPrice', 100)
            
            # Apply slippage for market orders
            if order.order_type == "MARKET":
                if order.side == "BUY":
                    fill_price = market_price * (1 + self.slippage_bps)
                else:
                    fill_price = market_price * (1 - self.slippage_bps)
            else:
                fill_price = order.price
            
            # Calculate trade value and commission
            trade_value = order.quantity * fill_price
            commission = trade_value * self.commission_rate
            
            # Update positions
            self._update_position(order.symbol, order.side, order.quantity, fill_price)
            
            # Update capital
            if order.side == "BUY":
                self.current_capital -= (trade_value + commission)
            else:
                self.current_capital += (trade_value - commission)
            
            # Update order status
            order.status = "FILLED"
            order.fill_price = fill_price
            order.fill_time = datetime.now()
            
            # Record trade
            self.trade_history.append({
                'timestamp': order.fill_time,
                'order_id': order_id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'price': fill_price,
                'value': trade_value,
                'commission': commission
            })
            
            logger.info(f"Executed order {order_id}: {order.side} {order.quantity} {order.symbol} @ ${fill_price:.2f}")
            
        except Exception as e:
            order.status = "CANCELLED"
            logger.error(f"Failed to execute order {order_id}: {str(e)}")
    
    def _update_position(self, symbol: str, side: str, quantity: int, price: float):
        """Update position after trade execution."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, 0, 0.0)
        
        position = self.positions[symbol]
        
        if side == "BUY":
            # Calculate new average cost
            total_cost = (position.quantity * position.avg_cost) + (quantity * price)
            total_quantity = position.quantity + quantity
            position.avg_cost = total_cost / total_quantity if total_quantity > 0 else 0
            position.quantity = total_quantity
        else:  # SELL
            # Calculate realized P&L
            realized_pnl = quantity * (price - position.avg_cost)
            position.realized_pnl += realized_pnl
            position.quantity -= quantity
            
            # If position is closed, reset average cost
            if position.quantity == 0:
                position.avg_cost = 0.0
        
        position.last_update = datetime.now()
    
    def update_positions(self, market_data: Dict[str, float]):
        """Update position values with current market data."""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.current_price = market_data[symbol]
                if position.quantity != 0:
                    position.unrealized_pnl = position.quantity * (position.current_price - position.avg_cost)
    
    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        portfolio_value = self.current_capital
        
        for position in self.positions.values():
            if position.quantity != 0:
                portfolio_value += position.quantity * position.current_price
        
        return portfolio_value
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics."""
        current_value = self.get_portfolio_value()
        total_return = (current_value / self.initial_capital) - 1
        
        # Calculate other metrics based on trade history
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history 
                           if trade['side'] == 'SELL' and 
                           self._calculate_trade_pnl(trade) > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'current_value': current_value,
            'total_return': total_return,
            'cash': self.current_capital,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'positions': len([p for p in self.positions.values() if p.quantity != 0])
        }
    
    def _calculate_trade_pnl(self, trade: Dict) -> float:
        """Calculate P&L for a specific trade."""
        # Simplified P&L calculation - in reality would need to match buy/sell pairs
        return 0.0

class RealTimeTradingStrategy:
    """
    Real-time implementation of trading strategy with live data processing.
    """
    
    def __init__(self, strategy_config: Dict[str, Any], 
                 trading_engine: DemoTradingEngine,
                 data_provider: RealTimeDataProvider):
        """
        Initialize real-time trading strategy.
        
        Args:
            strategy_config: Configuration parameters for the strategy
            trading_engine: Trading engine for order execution
            data_provider: Real-time data provider
        """
        self.config = strategy_config
        self.trading_engine = trading_engine
        self.data_provider = data_provider
        
        # Strategy state
        self.symbols = strategy_config.get('symbols', ['AAPL'])
        self.short_window = strategy_config.get('short_window', 20)
        self.long_window = strategy_config.get('long_window', 50)
        
        # Data storage for indicator calculations
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.max_history_length = max(self.long_window * 2, 100)
        
        # Strategy state
        self.current_signals = {symbol: 0 for symbol in self.symbols}
        self.last_signals = {symbol: 0 for symbol in self.symbols}
        
        # Performance tracking
        self.signal_history = []
        self.trades = []
        
        # Subscribe to market data
        self.data_provider.subscribe(self._on_market_data)
        
        logger.info(f"Initialized real-time trading strategy for {len(self.symbols)} symbols")
    
    def start(self):
        """Start the real-time trading strategy."""
        logger.info("Starting real-time trading strategy...")
        self.data_provider.start()
        
    def stop(self):
        """Stop the real-time trading strategy."""
        logger.info("Stopping real-time trading strategy...")
        self.data_provider.stop()
    
    def _on_market_data(self, market_data: MarketData):
        """Handle incoming market data."""
        symbol = market_data.symbol
        price = market_data.price
        
        # Update price history
        self.price_history[symbol].append({
            'timestamp': market_data.timestamp,
            'price': price
        })
        
        # Maintain history length
        if len(self.price_history[symbol]) > self.max_history_length:
            self.price_history[symbol].pop(0)
        
        # Calculate indicators and check for signals
        if len(self.price_history[symbol]) >= self.long_window:
            self._check_signals(symbol)
        
        # Update trading engine with current prices
        self.trading_engine.update_positions({symbol: price})
    
    def _check_signals(self, symbol: str):
        """Check for trading signals based on current data."""
        prices = [entry['price'] for entry in self.price_history[symbol]]
        
        if len(prices) < self.long_window:
            return
        
        # Calculate moving averages
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        
        # Previous MAs for crossover detection
        if len(prices) > self.long_window:
            prev_short_ma = np.mean(prices[-self.short_window-1:-1])
            prev_long_ma = np.mean(prices[-self.long_window-1:-1])
        else:
            prev_short_ma = short_ma
            prev_long_ma = long_ma
        
        # Detect crossovers
        signal = 0
        if short_ma > long_ma and prev_short_ma <= prev_long_ma:
            signal = 1  # Buy signal
        elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
            signal = -1  # Sell signal
        
        # Update signals
        self.last_signals[symbol] = self.current_signals[symbol]
        self.current_signals[symbol] = signal
        
        # Execute trades on signal changes
        if signal != 0 and signal != self.last_signals[symbol]:
            self._execute_signal(symbol, signal)
        
        # Log signal information
        self.signal_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'price': prices[-1],
            'short_ma': short_ma,
            'long_ma': long_ma,
            'signal': signal
        })
    
    def _execute_signal(self, symbol: str, signal: int):
        """Execute trading signal."""
        try:
            current_price = self.data_provider.get_current_price(symbol)
            if current_price is None:
                logger.warning(f"No current price available for {symbol}")
                return
            
            # Calculate position size (simple fixed percentage for demo)
            portfolio_value = self.trading_engine.get_portfolio_value()
            position_value = portfolio_value * 0.1  # 10% of portfolio per position
            quantity = int(position_value / current_price)
            
            if quantity < 1:
                logger.warning(f"Position size too small for {symbol}")
                return
            
            # Place orders based on signal
            if signal == 1:  # Buy signal
                order_id = self.trading_engine.place_order(
                    symbol=symbol,
                    side="BUY",
                    quantity=quantity,
                    order_type="MARKET"
                )
                if order_id:
                    logger.info(f"BUY signal executed for {symbol}: {quantity} shares")
            
            elif signal == -1:  # Sell signal
                # Check if we have position to sell
                position = self.trading_engine.positions.get(symbol)
                if position and position.quantity > 0:
                    sell_quantity = min(quantity, position.quantity)
                    order_id = self.trading_engine.place_order(
                        symbol=symbol,
                        side="SELL",
                        quantity=sell_quantity,
                        order_type="MARKET"
                    )
                    if order_id:
                        logger.info(f"SELL signal executed for {symbol}: {sell_quantity} shares")
            
        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {str(e)}")
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status and performance."""
        performance = self.trading_engine.get_performance_metrics()
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'performance': performance,
            'current_signals': self.current_signals.copy(),
            'positions': {symbol: {
                'quantity': pos.quantity,
                'avg_cost': pos.avg_cost,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl
            } for symbol, pos in self.trading_engine.positions.items() if pos.quantity != 0},
            'recent_signals': self.signal_history[-10:] if len(self.signal_history) > 10 else self.signal_history
        }
        
        return status

class RealTimeDashboard:
    """
    Real-time monitoring dashboard for trading strategy.
    """
    
    def __init__(self, strategy: RealTimeTradingStrategy):
        self.strategy = strategy
        self.is_running = False
        self.update_thread = None
        
    def start_monitoring(self):
        """Start the real-time monitoring dashboard."""
        if self.is_running:
            return
            
        self.is_running = True
        self.update_thread = threading.Thread(target=self._monitoring_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Started real-time monitoring dashboard")
    
    def stop_monitoring(self):
        """Stop the monitoring dashboard."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                status = self.strategy.get_strategy_status()
                self._display_status(status)
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(10)
    
    def _display_status(self, status: Dict[str, Any]):
        """Display current strategy status."""
        # Clear screen (works in most terminals)
        print("\033[2J\033[H")
        
        print("=" * 80)
        print("REAL-TIME TRADING STRATEGY DASHBOARD")
        print("=" * 80)
        print(f"Last Update: {status['timestamp']}")
        print()
        
        # Performance metrics
        perf = status['performance']
        print(f"Portfolio Value: ${perf['current_value']:,.2f}")
        print(f"Total Return: {perf['total_return']:+.2%}")
        print(f"Available Cash: ${perf['cash']:,.2f}")
        print(f"Active Positions: {perf['positions']}")
        print(f"Total Trades: {perf['total_trades']}")
        print(f"Win Rate: {perf['win_rate']:.1%}")
        print()
        
        # Current signals
        print("CURRENT SIGNALS:")
        print("-" * 20)
        for symbol, signal in status['current_signals'].items():
            signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
            print(f"{symbol}: {signal_text}")
        print()
        
        # Current positions
        if status['positions']:
            print("CURRENT POSITIONS:")
            print("-" * 40)
            print(f"{'Symbol':<8} {'Qty':<8} {'Avg Cost':<10} {'Current':<10} {'P&L':<10}")
            print("-" * 40)
            for symbol, pos in status['positions'].items():
                print(f"{symbol:<8} {pos['quantity']:<8} ${pos['avg_cost']:<9.2f} "
                      f"${pos['current_price']:<9.2f} ${pos['unrealized_pnl']:+<9.2f}")
            print()
        
        # Recent signals
        if status['recent_signals']:
            print("RECENT SIGNALS (Last 5):")
            print("-" * 60)
            for signal_data in status['recent_signals'][-5:]:
                timestamp = signal_data['timestamp'].strftime('%H:%M:%S')
                symbol = signal_data['symbol']
                price = signal_data['price']
                signal = signal_data['signal']
                signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "---"
                print(f"{timestamp} {symbol} ${price:.2f} -> {signal_text}")

def demo_realtime_trading():
    """Demonstration of real-time trading system."""
    
    # Configuration
    config = {
        'symbols': ['AAPL', 'MSFT'],
        'short_window': 5,  # Shorter for demo
        'long_window': 20,
        'update_interval': 5  # 5 seconds for demo
    }
    
    # Initialize components
    print("Initializing real-time trading system...")
    
    # Data provider
    data_provider = RealTimeDataProvider(
        symbols=config['symbols'],
        update_interval=config['update_interval']
    )
    
    # Trading engine
    trading_engine = DemoTradingEngine(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_bps=2.0
    )
    
    # Strategy
    strategy = RealTimeTradingStrategy(
        strategy_config=config,
        trading_engine=trading_engine,
        data_provider=data_provider
    )
    
    # Dashboard
    dashboard = RealTimeDashboard(strategy)
    
    try:
        # Start the system
        print("Starting real-time trading system...")
        strategy.start()
        dashboard.start_monitoring()
        
        # Run for demo period
        print("Running demo for 60 seconds... Press Ctrl+C to stop early")
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        # Clean shutdown
        print("Shutting down real-time trading system...")
        dashboard.stop_monitoring()
        strategy.stop()
        
        # Final status
        final_status = strategy.get_strategy_status()
        print("\nFINAL PERFORMANCE:")
        print(f"Portfolio Value: ${final_status['performance']['current_value']:,.2f}")
        print(f"Total Return: {final_status['performance']['total_return']:+.2%}")
        print(f"Total Trades: {final_status['performance']['total_trades']}")


if __name__ == "__main__":
    # Set up logging for demo
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    demo_realtime_trading()
