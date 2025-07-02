import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from utils.error_handling import handle_data_errors, DataError
from utils.validators import validate_trade_signal

logger = logging.getLogger(__name__)

class TradingStrategy:
    """SMA crossover strategy with multiple confirmations"""
    
    def __init__(self, config):
        self.config = config
        self.sma_short = config.trading.sma_short
        self.sma_long = config.trading.sma_long
        self.max_atr_pct = config.trading.max_atr_pct
        
    @handle_data_errors
    def calculate_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        if 'Close' not in data.columns:
            raise DataError("Close price column not found in data")
        
        if len(data) < period:
            logger.warning(f"Insufficient data for SMA calculation: {len(data)} < {period}")
            return pd.Series(index=data.index, dtype=float)
        
        return data['Close'].rolling(window=period, min_periods=period).mean()
    
    @handle_data_errors
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range for volatility"""
        required_columns = ['High', 'Low', 'Close']
        
        for col in required_columns:
            if col not in data.columns:
                raise DataError(f"Required column {col} not found in data")
        
        if len(data) < period + 1:
            logger.warning(f"Insufficient data for ATR calculation: {len(data)} < {period + 1}")
            return pd.Series(index=data.index, dtype=float)
        
        # Calculate True Range components
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        # True Range is the maximum of the three
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # ATR is the moving average of True Range
        return true_range.rolling(window=period, min_periods=period).mean()
    
    @handle_data_errors
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        if 'Close' not in data.columns:
            raise DataError("Close price column not found in data")
        
        if len(data) < period + 1:
            return pd.Series(index=data.index, dtype=float)
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @handle_data_errors
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        if 'Close' not in data.columns:
            raise DataError("Close price column not found in data")
        
        if len(data) < slow + signal:
            empty_series = pd.Series(index=data.index, dtype=float)
            return {'macd': empty_series, 'signal': empty_series, 'histogram': empty_series}
        
        # Calculate EMAs
        ema_fast = data['Close'].ewm(span=fast).mean()
        ema_slow = data['Close'].ewm(span=slow).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @handle_data_errors
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Generate trading signals with confirmations"""
        try:
            if data.empty or len(data) < max(self.sma_long, 14):
                logger.warning(f"Insufficient data for signal generation: {symbol}")
                return self._create_hold_signal(symbol, "insufficient_data")
            
            # Calculate technical indicators
            data = data.copy()  # Avoid modifying original data
            data['sma_short'] = self.calculate_sma(data, self.sma_short)
            data['sma_long'] = self.calculate_sma(data, self.sma_long)
            data['atr'] = self.calculate_atr(data)
            data['rsi'] = self.calculate_rsi(data)
            
            # Calculate MACD
            macd_data = self.calculate_macd(data)
            data['macd'] = macd_data['macd']
            data['macd_signal'] = macd_data['signal']
            data['macd_histogram'] = macd_data['histogram']
            
            # Get latest values (need at least 2 data points for crossover detection)
            if len(data) < 2:
                return self._create_hold_signal(symbol, "insufficient_recent_data")
            
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            # Check for NaN values in critical indicators
            if pd.isna(latest['sma_short']) or pd.isna(latest['sma_long']):
                return self._create_hold_signal(symbol, "missing_sma_data")
            
            # Primary signal: SMA crossover
            current_signal = 'BUY' if latest['sma_short'] > latest['sma_long'] else 'SELL'
            prev_signal = 'BUY' if prev['sma_short'] > prev['sma_long'] else 'SELL'
            
            # Check for crossover
            crossover = current_signal != prev_signal
            
            # Volatility filter
            atr_pct = 0.0
            volatility_ok = True
            if not pd.isna(latest['atr']) and latest['Close'] > 0:
                atr_pct = (latest['atr'] / latest['Close']) * 100
                volatility_ok = atr_pct <= self.max_atr_pct * 100
            
            # Additional confirmation signals
            confirmations = self._calculate_confirmations(data)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(data, symbol, confirmations)
            
            # Determine final action
            action = self._determine_action(current_signal, crossover, volatility_ok, confirmations, confidence)
            
            signal = {
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'crossover': crossover,
                'volatility_ok': volatility_ok,
                'atr_pct': atr_pct,
                'timestamp': datetime.now(),
                'price': float(latest['Close']),
                'sma_short': float(latest['sma_short']),
                'sma_long': float(latest['sma_long']),
                'rsi': float(latest['rsi']) if not pd.isna(latest['rsi']) else None,
                'macd': float(latest['macd']) if not pd.isna(latest['macd']) else None,
                'confirmations': confirmations
            }
            
            # Validate signal
            validate_trade_signal(signal)
            
            logger.debug(f"Generated signal for {symbol}: {action} (confidence: {confidence:.2f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return self._create_error_signal(symbol, str(e))
    
    def _calculate_confirmations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate additional confirmation signals"""
        confirmations = {}
        
        try:
            latest = data.iloc[-1]
            
            # Volume confirmation
            if 'Volume' in data.columns and len(data) >= 20:
                avg_volume = data['Volume'].tail(20).mean()
                volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1.0
                confirmations['volume_confirmation'] = volume_ratio > 1.2  # 20% above average
                confirmations['volume_ratio'] = float(volume_ratio)
            else:
                confirmations['volume_confirmation'] = False
                confirmations['volume_ratio'] = 1.0
            
            # RSI confirmation (avoid overbought/oversold extremes)
            if not pd.isna(latest['rsi']):
                rsi_value = latest['rsi']
                confirmations['rsi_ok'] = 30 < rsi_value < 70  # Not in extreme zones
                confirmations['rsi_value'] = float(rsi_value)
            else:
                confirmations['rsi_ok'] = True
                confirmations['rsi_value'] = 50.0
            
            # MACD confirmation
            if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
                macd_bullish = latest['macd'] > latest['macd_signal']
                confirmations['macd_bullish'] = macd_bullish
                confirmations['macd_value'] = float(latest['macd'])
            else:
                confirmations['macd_bullish'] = False
                confirmations['macd_value'] = 0.0
            
            # Trend strength (based on recent price action)
            if len(data) >= 5:
                recent_closes = data['Close'].tail(5)
                trend_strength = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]
                confirmations['trend_strength'] = float(trend_strength)
                confirmations['strong_trend'] = abs(trend_strength) > 0.02  # 2% move over 5 days
            else:
                confirmations['trend_strength'] = 0.0
                confirmations['strong_trend'] = False
                
        except Exception as e:
            logger.warning(f"Error calculating confirmations: {e}")
            confirmations = {
                'volume_confirmation': False,
                'volume_ratio': 1.0,
                'rsi_ok': True,
                'rsi_value': 50.0,
                'macd_bullish': False,
                'macd_value': 0.0,
                'trend_strength': 0.0,
                'strong_trend': False
            }
        
        return confirmations
    
    def _calculate_confidence(self, data: pd.DataFrame, symbol: str, confirmations: Dict) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence_factors = []
        
        try:
            # Base confidence from trend consistency
            if len(data) >= 10:
                recent_data = data.tail(10)
                if not recent_data['sma_short'].empty and not recent_data['sma_long'].empty:
                    trend_consistency = (recent_data['sma_short'] > recent_data['sma_long']).sum() / 10
                    # Convert to 0-1 scale with higher values for more consistent trends
                    consistency_score = abs(trend_consistency - 0.5) * 2
                    confidence_factors.append(consistency_score)
            
            # Volume confirmation factor
            if confirmations.get('volume_confirmation', False):
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.4)
            
            # RSI confirmation factor
            if confirmations.get('rsi_ok', True):
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.3)
            
            # MACD confirmation factor
            if confirmations.get('macd_bullish', False):
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            
            # Trend strength factor
            trend_strength = abs(confirmations.get('trend_strength', 0.0))
            strength_factor = min(trend_strength * 10, 1.0)  # Scale to 0-1
            confidence_factors.append(strength_factor)
            
            # Calculate weighted average
            if confidence_factors:
                base_confidence = np.mean(confidence_factors)
            else:
                base_confidence = 0.5
            
            # Apply volatility penalty
            if 'atr_pct' in data.columns:
                latest_atr = data['atr_pct'].iloc[-1] if not data['atr_pct'].empty else 2.0
                if latest_atr > self.max_atr_pct * 100:
                    volatility_penalty = 0.5
                else:
                    volatility_penalty = 1.0
                base_confidence *= volatility_penalty
            
            # Ensure confidence is within bounds
            final_confidence = max(0.0, min(1.0, base_confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating confidence for {symbol}: {e}")
            final_confidence = 0.5
        
        return final_confidence
    
    def _determine_action(self, signal: str, crossover: bool, volatility_ok: bool, 
                         confirmations: Dict, confidence: float) -> str:
        """Determine final trading action based on all factors"""
        
        # Don't trade if volatility is too high
        if not volatility_ok:
            return 'HOLD'
        
        # Don't trade if confidence is too low
        if confidence < 0.3:
            return 'HOLD'
        
        # Only act on crossovers for this strategy
        if not crossover:
            return 'HOLD'
        
        # Additional confirmation for BUY signals
        if signal == 'BUY':
            # Require at least some confirmations for buy signals
            confirmation_count = sum([
                confirmations.get('volume_confirmation', False),
                confirmations.get('rsi_ok', True),
                confirmations.get('macd_bullish', False)
            ])
            
            if confirmation_count >= 2:
                return 'BUY'
            else:
                return 'HOLD'
        
        # For SELL signals, be more conservative
        elif signal == 'SELL':
            # Only sell if we have strong confirmation
            if (not confirmations.get('rsi_ok', True) or  # RSI in extreme zone
                confirmations.get('trend_strength', 0) < -0.05):  # Strong downtrend
                return 'SELL'
            else:
                return 'HOLD'
        
        return 'HOLD'
    
    def _create_hold_signal(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Create a HOLD signal with reason"""
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'confidence': 0.0,
            'crossover': False,
            'volatility_ok': True,
            'atr_pct': 0.0,
            'timestamp': datetime.now(),
            'price': 0.0,
            'reason': reason,
            'confirmations': {}
        }
    
    def _create_error_signal(self, symbol: str, error: str) -> Dict[str, Any]:
        """Create an error signal"""
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'confidence': 0.0,
            'crossover': False,
            'volatility_ok': True,
            'atr_pct': 0.0,
            'timestamp': datetime.now(),
            'price': 0.0,
            'error': error,
            'confirmations': {}
        }

class PortfolioAnalyzer:
    """Analyze portfolio performance and generate insights"""
    
    def __init__(self, config):
        self.config = config
        
    def calculate_relative_strength(self, market_data: Dict[str, pd.DataFrame], 
                                  lookback_days: int = 30) -> Dict[str, float]:
        """Calculate relative strength for each symbol"""
        rs_scores = {}
        
        for symbol, data in market_data.items():
            try:
                if data.empty or len(data) < lookback_days:
                    rs_scores[symbol] = 0.0
                    continue
                
                # Calculate return over lookback period
                recent_data = data.tail(lookback_days)
                if len(recent_data) >= 2:
                    start_price = recent_data['Close'].iloc[0]
                    end_price = recent_data['Close'].iloc[-1]
                    
                    if start_price > 0:
                        returns = (end_price - start_price) / start_price
                        rs_scores[symbol] = float(returns)
                    else:
                        rs_scores[symbol] = 0.0
                else:
                    rs_scores[symbol] = 0.0
                    
            except Exception as e:
                logger.warning(f"Error calculating relative strength for {symbol}: {e}")
                rs_scores[symbol] = 0.0
        
        return rs_scores
    
    def rank_etfs(self, signals: Dict[str, Dict], market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Rank ETFs based on multiple criteria"""
        
        # Calculate relative strength
        rs_scores = self.calculate_relative_strength(market_data)
        
        etf_rankings = []
        
        for symbol, signal in signals.items():
            try:
                # Base score from signal confidence
                score = signal.get('confidence', 0.0)
                
                # Add relative strength component (30% weight)
                rs_score = rs_scores.get(symbol, 0.0)
                score += rs_score * 0.3
                
                # Adjust for sentiment (if available)
                sentiment = signal.get('sentiment', {})
                sentiment_score = sentiment.get('sentiment_score', 0.0)
                sentiment_confidence = sentiment.get('confidence', 0.0)
                score += sentiment_score * sentiment_confidence * 0.2
                
                # Bonus for strong signals
                if signal.get('action') == 'BUY' and signal.get('crossover', False):
                    score *= 1.3
                elif signal.get('action') == 'SELL':
                    score *= 0.7  # Penalty for sell signals
                
                # Penalty for high volatility
                if not signal.get('volatility_ok', True):
                    score *= 0.5
                
                etf_rankings.append({
                    'symbol': symbol,
                    'score': score,
                    'signal': signal,
                    'relative_strength': rs_score,
                    'sentiment_score': sentiment_score
                })
                
            except Exception as e:
                logger.warning(f"Error ranking {symbol}: {e}")
                etf_rankings.append({
                    'symbol': symbol,
                    'score': 0.0,
                    'signal': signal,
                    'relative_strength': 0.0,
                    'sentiment_score': 0.0
                })
        
        # Sort by score descending
        etf_rankings.sort(key=lambda x: x['score'], reverse=True)
        
        return etf_rankings