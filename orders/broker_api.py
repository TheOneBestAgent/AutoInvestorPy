import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class BrokerAPI:
    """Broker API integration for trade execution"""
    
    def __init__(self, config):
        self.config = config
        self.paper_trading = config.broker.paper_trading
        self.api_key = config.broker.api_key
        self.secret_key = config.broker.secret_key
        self.base_url = config.broker.base_url
        
        # Initialize based on paper trading mode
        if self.paper_trading:
            self.client = MockBrokerClient(config)
            logger.info("Initialized with paper trading (mock) client")
        else:
            # In a real implementation, this would use the actual Alpaca API
            # self.client = AlpacaClient(config)
            self.client = MockBrokerClient(config)  # Using mock for now
            logger.warning("Live trading not implemented yet, using mock client")
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        return self.client.get_account()
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        return self.client.get_positions()
    
    def get_orders(self, status: str = None) -> List[Dict[str, Any]]:
        """Get orders (optionally filtered by status)"""
        return self.client.get_orders(status)
    
    def submit_order(self, symbol: str, quantity: int, side: str, 
                    order_type: str = "market", time_in_force: str = "day",
                    stop_price: Optional[float] = None) -> Dict[str, Any]:
        """Submit an order"""
        
        order_data = {
            'symbol': symbol,
            'quantity': quantity,
            'side': side.lower(),  # buy or sell
            'type': order_type.lower(),
            'time_in_force': time_in_force.lower(),
            'stop_price': stop_price
        }
        
        logger.info(f"Submitting order: {side} {quantity} {symbol}")
        return self.client.submit_order(order_data)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        return self.client.cancel_order(order_id)
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        return self.client.get_current_price(symbol)
    
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close a position (market sell)"""
        positions = self.get_positions()
        position = next((p for p in positions if p['symbol'] == symbol), None)
        
        if not position:
            raise ValueError(f"No position found for {symbol}")
        
        quantity = abs(int(position['qty']))
        side = 'sell' if int(position['qty']) > 0 else 'buy'
        
        return self.submit_order(symbol, quantity, side, order_type='market')
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        return self.client.is_market_open()

class MockBrokerClient:
    """Mock broker client for testing and paper trading"""
    
    def __init__(self, config):
        self.config = config
        self.account_balance = 1000.0  # Starting balance
        self.positions = {}
        self.orders = []
        self.order_id_counter = 1
        
        # Mock prices for testing
        self.mock_prices = {
            'SPY': 420.50,
            'QQQ': 350.25,
            'IWM': 185.75,
            'VTI': 220.80,
            'VOO': 385.90,
            'XLF': 35.60,
            'XLK': 140.30,
            'XLE': 85.40,
            'XLV': 125.70,
            'XLI': 110.20,
            'EFA': 75.80,
            'EEM': 42.30,
            'VWO': 52.40,
            'GLD': 185.90,
            'TLT': 95.60
        }
        
        logger.info("Mock broker client initialized")
    
    def get_account(self) -> Dict[str, Any]:
        """Get mock account information"""
        # Calculate total position value
        position_value = sum(
            pos['qty'] * self.mock_prices.get(pos['symbol'], 100.0) 
            for pos in self.positions.values()
        )
        
        equity = self.account_balance + position_value
        
        return {
            'account_number': 'MOCK123456',
            'status': 'ACTIVE',
            'currency': 'USD',
            'cash': self.account_balance,
            'portfolio_value': equity,
            'equity': equity,
            'buying_power': self.account_balance,
            'long_market_value': max(0, position_value),
            'short_market_value': min(0, position_value),
            'initial_margin': 0.0,
            'maintenance_margin': 0.0,
            'day_trade_count': 0,
            'pattern_day_trader': False
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get mock positions"""
        positions = []
        
        for symbol, position in self.positions.items():
            if position['qty'] != 0:
                current_price = self.mock_prices.get(symbol, position['avg_entry_price'])
                market_value = position['qty'] * current_price
                unrealized_pnl = market_value - (position['qty'] * position['avg_entry_price'])
                
                positions.append({
                    'symbol': symbol,
                    'qty': str(position['qty']),
                    'side': 'long' if position['qty'] > 0 else 'short',
                    'market_value': str(market_value),
                    'avg_entry_price': str(position['avg_entry_price']),
                    'current_price': str(current_price),
                    'unrealized_pnl': str(unrealized_pnl),
                    'unrealized_plpc': str((unrealized_pnl / abs(market_value)) * 100) if market_value != 0 else '0.0',
                    'asset_class': 'us_equity',
                    'exchange': 'NASDAQ'
                })
        
        return positions
    
    def get_orders(self, status: str = None) -> List[Dict[str, Any]]:
        """Get mock orders"""
        if status:
            return [order for order in self.orders if order['status'] == status]
        return self.orders.copy()
    
    def submit_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit mock order"""
        symbol = order_data['symbol']
        quantity = order_data['quantity']
        side = order_data['side']
        order_type = order_data['type']
        
        # Generate order ID
        order_id = f"MOCK{self.order_id_counter:06d}"
        self.order_id_counter += 1
        
        # Get current price
        current_price = self.mock_prices.get(symbol, 100.0)
        
        # Add some realistic price slippage for market orders
        if order_type == 'market':
            slippage = 0.02  # 2 cents
            fill_price = current_price + slippage if side == 'buy' else current_price - slippage
        else:
            fill_price = current_price
        
        # Calculate order value
        order_value = quantity * fill_price
        
        # Create order
        order = {
            'id': order_id,
            'symbol': symbol,
            'qty': str(quantity),
            'side': side,
            'order_type': order_type,
            'time_in_force': order_data.get('time_in_force', 'day'),
            'filled_qty': '0',
            'filled_avg_price': None,
            'status': 'new',
            'submitted_at': datetime.now().isoformat(),
            'filled_at': None,
            'cancelled_at': None,
            'extended_hours': False,
            'legs': None
        }
        
        # Simulate immediate fill for market orders
        if order_type == 'market':
            success = self._fill_order(order, fill_price)
            if success:
                order['status'] = 'filled'
                order['filled_qty'] = str(quantity)
                order['filled_avg_price'] = str(fill_price)
                order['filled_at'] = datetime.now().isoformat()
            else:
                order['status'] = 'rejected'
                order['rejected_reason'] = 'Insufficient buying power'
        
        self.orders.append(order)
        
        logger.info(f"Mock order submitted: {order_id} - {side} {quantity} {symbol} @ ${fill_price:.2f}")
        return order
    
    def _fill_order(self, order: Dict[str, Any], fill_price: float) -> bool:
        """Simulate order fill"""
        symbol = order['symbol']
        quantity = int(order['qty'])
        side = order['side']
        
        # Calculate required cash
        if side == 'buy':
            required_cash = quantity * fill_price
            if required_cash > self.account_balance:
                logger.warning(f"Insufficient funds for order: ${required_cash:.2f} > ${self.account_balance:.2f}")
                return False
            
            # Update account balance
            self.account_balance -= required_cash
            
            # Update position
            if symbol in self.positions:
                current_qty = self.positions[symbol]['qty']
                current_value = current_qty * self.positions[symbol]['avg_entry_price']
                new_qty = current_qty + quantity
                new_value = current_value + (quantity * fill_price)
                new_avg_price = new_value / new_qty if new_qty != 0 else 0
                
                self.positions[symbol] = {
                    'qty': new_qty,
                    'avg_entry_price': new_avg_price
                }
            else:
                self.positions[symbol] = {
                    'qty': quantity,
                    'avg_entry_price': fill_price
                }
        
        else:  # sell
            # Check if we have enough shares to sell
            current_qty = self.positions.get(symbol, {}).get('qty', 0)
            if current_qty < quantity:
                logger.warning(f"Insufficient shares to sell: {current_qty} < {quantity}")
                return False
            
            # Update position
            self.positions[symbol]['qty'] -= quantity
            
            # Update account balance
            self.account_balance += quantity * fill_price
            
            # Remove position if quantity is zero
            if self.positions[symbol]['qty'] == 0:
                del self.positions[symbol]
        
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel mock order"""
        for order in self.orders:
            if order['id'] == order_id and order['status'] == 'new':
                order['status'] = 'cancelled'
                order['cancelled_at'] = datetime.now().isoformat()
                logger.info(f"Mock order cancelled: {order_id}")
                return True
        
        logger.warning(f"Order not found or cannot be cancelled: {order_id}")
        return False
    
    def get_current_price(self, symbol: str) -> float:
        """Get mock current price"""
        # Add some random variation to simulate market movement
        import random
        base_price = self.mock_prices.get(symbol, 100.0)
        variation = random.uniform(-0.5, 0.5)  # ±0.5% variation
        return base_price * (1 + variation / 100)
    
    def is_market_open(self) -> bool:
        """Mock market open check"""
        # For testing, assume market is always open
        return True
    
    def update_mock_prices(self, price_updates: Dict[str, float]):
        """Update mock prices (for testing scenarios)"""
        self.mock_prices.update(price_updates)
        logger.info(f"Mock prices updated: {price_updates}")

# Real Alpaca API implementation would go here
# class AlpacaClient:
#     def __init__(self, config):
#         import alpaca_trade_api as tradeapi
#         self.api = tradeapi.REST(
#             config.broker.api_key,
#             config.broker.secret_key,
#             config.broker.base_url,
#             api_version='v2'
#         )
#     
#     # Implement all the same methods using the real Alpaca API