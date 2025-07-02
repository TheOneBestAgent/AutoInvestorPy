import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils.error_handling import TradingError, handle_data_errors
from utils.validators import validate_trade_signal, validate_position_data

logger = logging.getLogger(__name__)

class OrderManager:
    """Manage order execution and portfolio coordination"""
    
    def __init__(self, config, broker_api, risk_manager):
        self.config = config
        self.broker_api = broker_api
        self.risk_manager = risk_manager
        
    def execute_weekly_trades(self, ranked_etfs: List[Dict], current_positions: List[Dict], 
                            account_info: Dict) -> Dict[str, Any]:
        """Execute weekly trading decisions"""
        
        try:
            trade_results = {
                'executed_trades': [],
                'rejected_trades': [],
                'position_updates': [],
                'errors': []
            }
            
            account_balance = float(account_info.get('equity', 0))
            
            # Update existing positions with current prices
            updated_positions = self._update_position_prices(current_positions)
            trade_results['position_updates'] = updated_positions
            
            # Check for stop-loss triggers
            stop_loss_trades = self._check_stop_losses(updated_positions)
            for trade in stop_loss_trades:
                result = self._execute_trade(trade)
                if result['success']:
                    trade_results['executed_trades'].append(result)
                else:
                    trade_results['rejected_trades'].append(result)
            
            # Execute new signals
            new_trades = self._generate_new_trades(ranked_etfs, updated_positions, account_balance)
            for trade in new_trades:
                result = self._execute_trade(trade)
                if result['success']:
                    trade_results['executed_trades'].append(result)
                else:
                    trade_results['rejected_trades'].append(result)
            
            logger.info(f"Trading cycle complete: {len(trade_results['executed_trades'])} executed, "
                       f"{len(trade_results['rejected_trades'])} rejected")
            
            return trade_results
            
        except Exception as e:
            logger.error(f"Error in weekly trade execution: {e}")
            return {
                'executed_trades': [],
                'rejected_trades': [],
                'position_updates': [],
                'errors': [str(e)]
            }
    
    def _update_position_prices(self, current_positions: List[Dict]) -> List[Dict]:
        """Update current prices for all positions"""
        updated_positions = []
        
        for position in current_positions:
            try:
                symbol = position['symbol']
                current_price = self.broker_api.get_current_price(symbol)
                
                # Convert broker position to our format
                position_data = {
                    'symbol': symbol,
                    'quantity': abs(int(float(position['qty']))),
                    'entry_price': float(position['avg_entry_price']),
                    'current_price': current_price
                }
                
                # Update with risk manager
                updated_position = self.risk_manager.update_stop_loss(position_data, current_price)
                updated_positions.append(updated_position)
                
            except Exception as e:
                logger.error(f"Error updating position {position.get('symbol', 'unknown')}: {e}")
        
        return updated_positions
    
    def _check_stop_losses(self, positions: List[Dict]) -> List[Dict]:
        """Check for stop-loss triggers and generate sell orders"""
        stop_loss_trades = []
        
        for position in positions:
            try:
                symbol = position['symbol']
                current_price = position['current_price']
                
                if self.risk_manager.check_stop_loss_trigger(position, current_price):
                    # Create stop-loss sell order
                    trade = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': position['quantity'],
                        'order_type': 'market',
                        'reason': 'stop_loss',
                        'trigger_price': current_price,
                        'stop_loss_price': position.get('stop_loss_price', 0)
                    }
                    stop_loss_trades.append(trade)
                    
                    logger.warning(f"Stop-loss triggered for {symbol}: "
                                 f"${current_price:.2f} <= ${position.get('stop_loss_price', 0):.2f}")
            
            except Exception as e:
                logger.error(f"Error checking stop-loss for {position.get('symbol', 'unknown')}: {e}")
        
        return stop_loss_trades
    
    def _generate_new_trades(self, ranked_etfs: List[Dict], current_positions: List[Dict], 
                           account_balance: float) -> List[Dict]:
        """Generate new trades based on signals"""
        new_trades = []
        
        try:
            # Check if we already have maximum positions
            if len(current_positions) >= self.config.trading.max_positions:
                logger.info("Maximum positions reached, no new trades generated")
                return new_trades
            
            # Get symbols we already hold
            held_symbols = {pos['symbol'] for pos in current_positions}
            
            # Look for BUY signals in top ranked ETFs
            for etf in ranked_etfs[:self.config.trading.max_positions]:
                signal = etf['signal']
                symbol = etf['symbol']
                
                # Skip if we already hold this symbol
                if symbol in held_symbols:
                    continue
                
                # Only act on BUY signals
                if signal['action'] != 'BUY':
                    continue
                
                # Check signal quality
                if signal['confidence'] < 0.5:
                    logger.debug(f"Skipping {symbol}: low confidence ({signal['confidence']:.2f})")
                    continue
                
                if not signal.get('volatility_ok', True):
                    logger.debug(f"Skipping {symbol}: high volatility")
                    continue
                
                # Calculate position size
                current_price = signal['price']
                position_calc = self.risk_manager.calculate_position_size(
                    account_balance, symbol, current_price
                )
                
                if position_calc['shares'] > 0:
                    trade = {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': position_calc['shares'],
                        'order_type': 'market',
                        'reason': 'signal',
                        'signal_confidence': signal['confidence'],
                        'expected_price': current_price,
                        'stop_loss_price': position_calc['stop_loss_price'],
                        'max_loss': position_calc['max_loss'],
                        'investment_amount': position_calc['investment_amount']
                    }
                    new_trades.append(trade)
                    
                    # Update available balance for next calculation
                    account_balance -= position_calc['investment_amount']
                    
                    logger.info(f"Generated BUY order for {symbol}: "
                               f"{position_calc['shares']} shares @ ~${current_price:.2f}")
                else:
                    logger.debug(f"Insufficient capital for {symbol}")
        
        except Exception as e:
            logger.error(f"Error generating new trades: {e}")
        
        return new_trades
    
    def _execute_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single trade"""
        try:
            symbol = trade['symbol']
            action = trade['action']
            quantity = trade['quantity']
            order_type = trade['order_type']
            
            # Validate trade
            if quantity <= 0:
                return {
                    'success': False,
                    'trade': trade,
                    'error': 'Invalid quantity',
                    'timestamp': datetime.now()
                }
            
            # Get current price for validation
            current_price = self.broker_api.get_current_price(symbol)
            
            # Additional validation for buy orders
            if action == 'BUY':
                account_info = self.broker_api.get_account()
                validation = self.risk_manager.validate_trade_size(
                    quantity, current_price, float(account_info['equity'])
                )
                
                if not validation['valid']:
                    return {
                        'success': False,
                        'trade': trade,
                        'error': f"Trade validation failed: {validation['reason']}",
                        'validation': validation,
                        'timestamp': datetime.now()
                    }
            
            # Submit order to broker
            logger.info(f"Executing {action} order: {quantity} {symbol} @ ${current_price:.2f}")
            
            order_result = self.broker_api.submit_order(
                symbol=symbol,
                quantity=quantity,
                side=action.lower(),
                order_type=order_type
            )
            
            # Check if order was filled
            if order_result.get('status') == 'filled':
                fill_price = float(order_result.get('filled_avg_price', current_price))
                total_value = quantity * fill_price
                
                execution_result = {
                    'success': True,
                    'trade': trade,
                    'order_id': order_result['id'],
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': fill_price,
                    'total_value': total_value,
                    'timestamp': datetime.now(),
                    'reason': trade.get('reason', 'signal'),
                    'signal_confidence': trade.get('signal_confidence', 0)
                }
                
                logger.info(f"Trade executed successfully: {action} {quantity} {symbol} @ ${fill_price:.2f}")
                return execution_result
                
            else:
                return {
                    'success': False,
                    'trade': trade,
                    'order_result': order_result,
                    'error': f"Order not filled: {order_result.get('status', 'unknown')}",
                    'timestamp': datetime.now()
                }
        
        except Exception as e:
            logger.error(f"Error executing trade {trade}: {e}")
            return {
                'success': False,
                'trade': trade,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def close_all_positions(self, reason: str = "manual_close") -> List[Dict]:
        """Close all open positions"""
        results = []
        
        try:
            positions = self.broker_api.get_positions()
            
            for position in positions:
                try:
                    symbol = position['symbol']
                    quantity = abs(int(float(position['qty'])))
                    
                    if quantity > 0:
                        trade = {
                            'symbol': symbol,
                            'action': 'SELL',
                            'quantity': quantity,
                            'order_type': 'market',
                            'reason': reason
                        }
                        
                        result = self._execute_trade(trade)
                        results.append(result)
                        
                except Exception as e:
                    logger.error(f"Error closing position {position.get('symbol', 'unknown')}: {e}")
                    results.append({
                        'success': False,
                        'symbol': position.get('symbol', 'unknown'),
                        'error': str(e)
                    })
            
            logger.info(f"Attempted to close {len(positions)} positions")
            return results
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return [{'success': False, 'error': str(e)}]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        try:
            account_info = self.broker_api.get_account()
            positions = self.broker_api.get_positions()
            
            # Convert positions to our format
            position_data = []
            for pos in positions:
                current_price = float(pos['current_price'])
                entry_price = float(pos['avg_entry_price'])
                quantity = int(float(pos['qty']))
                
                position_data.append({
                    'symbol': pos['symbol'],
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'current_value': current_price * quantity,
                    'unrealized_pnl': float(pos['unrealized_pnl']),
                    'unrealized_pnl_pct': (current_price - entry_price) / entry_price * 100
                })
            
            summary = {
                'account_balance': float(account_info['equity']),
                'cash_balance': float(account_info['cash']),
                'portfolio_value': float(account_info['portfolio_value']),
                'total_positions': len(positions),
                'positions': position_data,
                'timestamp': datetime.now()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}

class TradeValidator:
    """Validate trades before execution"""
    
    def __init__(self, config):
        self.config = config
    
    def validate_trade_parameters(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade parameters"""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ['symbol', 'action', 'quantity']
        for field in required_fields:
            if field not in trade:
                errors.append(f"Missing required field: {field}")
        
        # Validate action
        if trade.get('action') not in ['BUY', 'SELL']:
            errors.append(f"Invalid action: {trade.get('action')}")
        
        # Validate quantity
        quantity = trade.get('quantity', 0)
        if not isinstance(quantity, int) or quantity <= 0:
            errors.append(f"Invalid quantity: {quantity}")
        
        # Validate symbol
        symbol = trade.get('symbol', '')
        if not isinstance(symbol, str) or len(symbol) < 1:
            errors.append(f"Invalid symbol: {symbol}")
        
        # Check if quantity is reasonable
        if quantity > 1000:
            warnings.append(f"Large quantity: {quantity} shares")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def validate_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Validate market conditions for trading"""
        # This could include checks for:
        # - Market hours
        # - Unusual volume
        # - Circuit breakers
        # - News events
        
        return {
            'valid': True,
            'market_open': True,
            'conditions': 'normal'
        }