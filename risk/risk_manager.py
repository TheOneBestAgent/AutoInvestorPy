import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

from utils.error_handling import handle_data_errors, DataError, TradingError
from utils.validators import validate_position_data, validate_numeric_range

logger = logging.getLogger(__name__)

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, config):
        self.config = config
        self.stop_loss_pct = config.trading.stop_loss_pct
        self.max_positions = config.trading.max_positions
        self.cooling_period_days = config.trading.cooling_period_days
        self.position_size_pct = config.trading.position_size_pct
        
    @handle_data_errors
    def calculate_position_size(self, account_balance: float, symbol: str, 
                              current_price: float, volatility_adjustment: bool = True) -> Dict[str, Any]:
        """Calculate appropriate position size with risk adjustments"""
        
        # Validate inputs
        validate_numeric_range(account_balance, 0.0, float('inf'), 'account_balance')
        validate_numeric_range(current_price, 0.01, float('inf'), 'current_price')
        
        # Calculate base position size
        base_capital = account_balance * self.position_size_pct
        
        # Volatility adjustment (optional)
        if volatility_adjustment:
            # This would normally use historical volatility, for now use a simple factor
            volatility_factor = 1.0  # Placeholder - could be enhanced with actual volatility calculation
            adjusted_capital = base_capital * volatility_factor
        else:
            adjusted_capital = base_capital
        
        # Calculate shares (must be whole shares)
        max_shares = int(adjusted_capital / current_price)
        
        # Ensure we don't exceed account balance
        if max_shares * current_price > account_balance:
            max_shares = int(account_balance / current_price)
        
        # Minimum position check
        if max_shares < 1:
            return {
                'symbol': symbol,
                'shares': 0,
                'investment_amount': 0.0,
                'available_capital': account_balance,
                'stop_loss_price': 0.0,
                'max_loss': 0.0,
                'max_loss_pct': 0.0,
                'reason': 'insufficient_capital'
            }
        
        # Calculate actual investment amount
        investment_amount = max_shares * current_price
        
        # Calculate stop-loss price
        stop_loss_price = current_price * (1 - self.stop_loss_pct)
        
        # Calculate maximum loss
        max_loss = (current_price - stop_loss_price) * max_shares
        max_loss_pct = (max_loss / investment_amount) * 100 if investment_amount > 0 else 0
        
        result = {
            'symbol': symbol,
            'shares': max_shares,
            'investment_amount': float(investment_amount),
            'available_capital': float(account_balance),
            'stop_loss_price': float(stop_loss_price),
            'max_loss': float(max_loss),
            'max_loss_pct': float(max_loss_pct),
            'entry_price': float(current_price),
            'position_size_pct': float((investment_amount / account_balance) * 100) if account_balance > 0 else 0
        }
        
        logger.info(f"Position sizing for {symbol}: {max_shares} shares @ ${current_price:.2f} "
                   f"(${investment_amount:.2f}, {max_loss_pct:.1f}% max loss)")
        
        return result
    
    @handle_data_errors
    def update_stop_loss(self, position: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Update trailing stop-loss price"""
        
        # Validate position data
        validate_position_data(position)
        validate_numeric_range(current_price, 0.01, float('inf'), 'current_price')
        
        updated_position = position.copy()
        
        entry_price = position['entry_price']
        current_stop = position.get('stop_loss_price', entry_price * (1 - self.stop_loss_pct))
        
        # Calculate new trailing stop (only for long positions)
        new_stop = current_price * (1 - self.stop_loss_pct)
        
        # Only update if new stop is higher (for long positions)
        if new_stop > current_stop:
            updated_position['stop_loss_price'] = float(new_stop)
            updated_position['stop_updated'] = datetime.now()
            updated_position['stop_updated_price'] = float(current_price)
            
            # Calculate new unrealized P&L
            quantity = position['quantity']
            unrealized_pnl = (current_price - entry_price) * quantity
            updated_position['unrealized_pnl'] = float(unrealized_pnl)
            updated_position['unrealized_pnl_pct'] = float((unrealized_pnl / (entry_price * quantity)) * 100)
            
            logger.info(f"Updated trailing stop for {position['symbol']}: ${new_stop:.2f} "
                       f"(price: ${current_price:.2f})")
        else:
            # Update current values without changing stop
            quantity = position['quantity']
            unrealized_pnl = (current_price - entry_price) * quantity
            updated_position['unrealized_pnl'] = float(unrealized_pnl)
            updated_position['unrealized_pnl_pct'] = float((unrealized_pnl / (entry_price * quantity)) * 100)
        
        updated_position['current_price'] = float(current_price)
        updated_position['current_value'] = float(current_price * position['quantity'])
        updated_position['last_updated'] = datetime.now()
        
        return updated_position
    
    def check_stop_loss_trigger(self, position: Dict[str, Any], current_price: float) -> bool:
        """Check if stop-loss should be triggered"""
        
        try:
            validate_position_data(position)
            validate_numeric_range(current_price, 0.01, float('inf'), 'current_price')
            
            stop_price = position.get('stop_loss_price')
            
            if stop_price is None or stop_price <= 0:
                logger.warning(f"Invalid stop price for {position.get('symbol', 'unknown')}: {stop_price}")
                return False
            
            # Trigger if current price is at or below stop-loss (for long positions)
            triggered = current_price <= stop_price
            
            if triggered:
                symbol = position.get('symbol', 'unknown')
                unrealized_loss = (position['entry_price'] - current_price) * position['quantity']
                logger.warning(f"Stop-loss triggered for {symbol}: ${current_price:.2f} <= ${stop_price:.2f} "
                             f"(loss: ${unrealized_loss:.2f})")
            
            return triggered
            
        except Exception as e:
            logger.error(f"Error checking stop-loss trigger: {e}")
            return False
    
    def assess_portfolio_risk(self, positions: List[Dict[str, Any]], 
                            account_balance: float) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        
        try:
            if not positions:
                return {
                    'total_positions': 0,
                    'total_value': 0.0,
                    'account_balance': float(account_balance),
                    'portfolio_value': float(account_balance),
                    'unrealized_pnl': 0.0,
                    'unrealized_pnl_pct': 0.0,
                    'risk_level': 'LOW',
                    'max_drawdown_risk': 0.0,
                    'position_concentration': 0.0,
                    'recommendations': []
                }
            
            # Calculate portfolio metrics
            total_value = sum(pos.get('current_value', 0) for pos in positions)
            total_unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)
            total_invested = sum(pos.get('entry_price', 0) * pos.get('quantity', 0) for pos in positions)
            
            portfolio_value = account_balance + total_unrealized_pnl
            
            # Calculate percentages
            unrealized_pnl_pct = (total_unrealized_pnl / total_invested) * 100 if total_invested > 0 else 0
            
            # Calculate maximum potential drawdown (if all stops hit)
            max_drawdown_risk = 0.0
            for pos in positions:
                entry_price = pos.get('entry_price', 0)
                stop_price = pos.get('stop_loss_price', 0)
                quantity = pos.get('quantity', 0)
                
                if entry_price > 0 and stop_price > 0:
                    position_risk = (entry_price - stop_price) * quantity
                    max_drawdown_risk += position_risk
            
            max_drawdown_pct = (max_drawdown_risk / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            # Calculate position concentration (largest position as % of portfolio)
            if positions:
                largest_position = max(pos.get('current_value', 0) for pos in positions)
                position_concentration = (largest_position / portfolio_value) * 100 if portfolio_value > 0 else 0
            else:
                position_concentration = 0.0
            
            # Determine risk level
            risk_level = self._calculate_risk_level(unrealized_pnl_pct, max_drawdown_pct, position_concentration)
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(
                len(positions), unrealized_pnl_pct, max_drawdown_pct, position_concentration
            )
            
            portfolio_risk = {
                'total_positions': len(positions),
                'total_value': float(total_value),
                'account_balance': float(account_balance),
                'portfolio_value': float(portfolio_value),
                'unrealized_pnl': float(total_unrealized_pnl),
                'unrealized_pnl_pct': float(unrealized_pnl_pct),
                'risk_level': risk_level,
                'max_drawdown_risk': float(max_drawdown_risk),
                'max_drawdown_pct': float(max_drawdown_pct),
                'position_concentration': float(position_concentration),
                'total_invested': float(total_invested),
                'cash_balance': float(account_balance - total_invested) if total_invested < account_balance else 0.0,
                'recommendations': recommendations
            }
            
            logger.info(f"Portfolio risk assessment: {risk_level} risk, "
                       f"{unrealized_pnl_pct:.1f}% unrealized P&L, "
                       f"{max_drawdown_pct:.1f}% max drawdown risk")
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return {
                'total_positions': len(positions) if positions else 0,
                'risk_level': 'UNKNOWN',
                'error': str(e)
            }
    
    def _calculate_risk_level(self, unrealized_pnl_pct: float, max_drawdown_pct: float, 
                             concentration_pct: float) -> str:
        """Calculate overall risk level"""
        
        risk_factors = 0
        
        # Check unrealized P&L risk
        if unrealized_pnl_pct < -15:  # More than 15% loss
            risk_factors += 2
        elif unrealized_pnl_pct < -10:  # More than 10% loss
            risk_factors += 1
        
        # Check maximum drawdown risk
        if max_drawdown_pct > 10:  # More than 10% potential loss
            risk_factors += 2
        elif max_drawdown_pct > 6:  # More than 6% potential loss
            risk_factors += 1
        
        # Check concentration risk
        if concentration_pct > 90:  # More than 90% in one position
            risk_factors += 2
        elif concentration_pct > 70:  # More than 70% in one position
            risk_factors += 1
        
        # Determine risk level
        if risk_factors >= 4:
            return 'CRITICAL'
        elif risk_factors >= 2:
            return 'HIGH'
        elif risk_factors >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_risk_recommendations(self, num_positions: int, unrealized_pnl_pct: float,
                                     max_drawdown_pct: float, concentration_pct: float) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        # Position count recommendations
        if num_positions == 0:
            recommendations.append("Consider establishing positions based on signal analysis")
        elif num_positions > self.max_positions:
            recommendations.append(f"Reduce positions to maximum of {self.max_positions}")
        
        # P&L recommendations
        if unrealized_pnl_pct < -15:
            recommendations.append("Critical losses detected - review stop-loss strategy")
        elif unrealized_pnl_pct < -10:
            recommendations.append("Significant losses - consider position review")
        elif unrealized_pnl_pct > 20:
            recommendations.append("Strong gains - consider taking partial profits")
        
        # Drawdown recommendations
        if max_drawdown_pct > 10:
            recommendations.append("High potential drawdown - consider tightening stops")
        elif max_drawdown_pct > 6:
            recommendations.append("Moderate drawdown risk - monitor closely")
        
        # Concentration recommendations
        if concentration_pct > 90:
            recommendations.append("Excessive concentration - diversify positions")
        elif concentration_pct > 70:
            recommendations.append("High concentration risk - consider diversification")
        
        return recommendations
    
    def check_cooling_period(self, last_stop_loss_date: Optional[datetime]) -> bool:
        """Check if we're still in cooling period after stop-loss"""
        
        if last_stop_loss_date is None:
            return False
        
        cooling_end = last_stop_loss_date + timedelta(days=self.cooling_period_days)
        still_cooling = datetime.now() < cooling_end
        
        if still_cooling:
            days_remaining = (cooling_end - datetime.now()).days + 1
            logger.info(f"Still in cooling period - {days_remaining} days remaining")
        
        return still_cooling
    
    def validate_trade_size(self, proposed_shares: int, current_price: float, 
                           account_balance: float) -> Dict[str, Any]:
        """Validate if a proposed trade size is within risk limits"""
        
        try:
            trade_value = proposed_shares * current_price
            
            # Check if we have enough capital
            if trade_value > account_balance:
                return {
                    'valid': False,
                    'reason': 'insufficient_capital',
                    'required_capital': trade_value,
                    'available_capital': account_balance
                }
            
            # Check if trade size exceeds position size limits
            position_pct = (trade_value / account_balance) * 100
            max_position_pct = self.position_size_pct * 100
            
            if position_pct > max_position_pct:
                return {
                    'valid': False,
                    'reason': 'exceeds_position_limit',
                    'proposed_pct': position_pct,
                    'max_allowed_pct': max_position_pct
                }
            
            # Calculate risk metrics
            stop_loss_price = current_price * (1 - self.stop_loss_pct)
            max_loss = (current_price - stop_loss_price) * proposed_shares
            max_loss_pct = (max_loss / account_balance) * 100
            
            return {
                'valid': True,
                'trade_value': trade_value,
                'position_pct': position_pct,
                'max_loss': max_loss,
                'max_loss_pct': max_loss_pct,
                'stop_loss_price': stop_loss_price
            }
            
        except Exception as e:
            logger.error(f"Error validating trade size: {e}")
            return {
                'valid': False,
                'reason': 'validation_error',
                'error': str(e)
            }

class RiskMonitor:
    """Real-time risk monitoring and alerts"""
    
    def __init__(self, config):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.alert_thresholds = {
            'max_portfolio_loss_pct': 15.0,
            'max_position_loss_pct': 8.0,
            'max_drawdown_pct': 12.0,
            'min_account_balance': 25.0  # Minimum balance in dollars
        }
    
    def check_risk_alerts(self, portfolio_assessment: Dict[str, Any], 
                         positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for risk alerts and return list of alerts"""
        
        alerts = []
        
        try:
            # Portfolio-level alerts
            unrealized_pnl_pct = portfolio_assessment.get('unrealized_pnl_pct', 0)
            if unrealized_pnl_pct < -self.alert_thresholds['max_portfolio_loss_pct']:
                alerts.append({
                    'type': 'PORTFOLIO_LOSS',
                    'severity': 'HIGH',
                    'message': f"Portfolio loss exceeds threshold: {unrealized_pnl_pct:.1f}%",
                    'value': unrealized_pnl_pct,
                    'threshold': -self.alert_thresholds['max_portfolio_loss_pct']
                })
            
            # Account balance alert
            account_balance = portfolio_assessment.get('account_balance', 0)
            if account_balance < self.alert_thresholds['min_account_balance']:
                alerts.append({
                    'type': 'LOW_BALANCE',
                    'severity': 'CRITICAL',
                    'message': f"Account balance critically low: ${account_balance:.2f}",
                    'value': account_balance,
                    'threshold': self.alert_thresholds['min_account_balance']
                })
            
            # Position-level alerts
            for position in positions:
                symbol = position.get('symbol', 'unknown')
                position_loss_pct = position.get('unrealized_pnl_pct', 0)
                
                if position_loss_pct < -self.alert_thresholds['max_position_loss_pct']:
                    alerts.append({
                        'type': 'POSITION_LOSS',
                        'severity': 'MEDIUM',
                        'message': f"{symbol} loss exceeds threshold: {position_loss_pct:.1f}%",
                        'symbol': symbol,
                        'value': position_loss_pct,
                        'threshold': -self.alert_thresholds['max_position_loss_pct']
                    })
            
            # Risk level alert
            risk_level = portfolio_assessment.get('risk_level', 'UNKNOWN')
            if risk_level in ['HIGH', 'CRITICAL']:
                alerts.append({
                    'type': 'RISK_LEVEL',
                    'severity': 'HIGH' if risk_level == 'HIGH' else 'CRITICAL',
                    'message': f"Portfolio risk level: {risk_level}",
                    'value': risk_level
                })
            
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
            alerts.append({
                'type': 'SYSTEM_ERROR',
                'severity': 'HIGH',
                'message': f"Risk monitoring error: {e}"
            })
        
        return alerts