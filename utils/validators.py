from typing import Union, List, Dict
import pandas as pd
import numpy as np

class ValidationError(Exception):
    """Custom validation exception"""
    pass

def validate_price_data(data: pd.DataFrame) -> bool:
    """Validate OHLCV price data"""
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check if DataFrame is empty
    if data.empty:
        raise ValidationError("Price data is empty")
    
    # Normalize column names (yfinance uses capitalized columns)
    data_columns = [col.lower() for col in data.columns]
    required_lower = [col.lower() for col in required_columns]
    
    # Check required columns
    if not all(col in data_columns for col in required_lower):
        missing = [col for col in required_lower if col not in data_columns]
        raise ValidationError(f"Missing required columns: {missing}")
    
    # Check for negative prices
    price_columns = ['Open', 'High', 'Low', 'Close']
    existing_price_cols = [col for col in price_columns if col in data.columns]
    
    if len(existing_price_cols) > 0:
        if (data[existing_price_cols] < 0).any().any():
            raise ValidationError("Negative prices detected")
        
        # Check high >= low logic
        if 'High' in data.columns and 'Low' in data.columns:
            if (data['High'] < data['Low']).any():
                raise ValidationError("High price below low price detected")
    
    # Check for missing data (NaN values)
    if data[existing_price_cols].isnull().any().any():
        raise ValidationError("Missing price data detected")
    
    return True

def validate_trade_signal(signal: dict) -> bool:
    """Validate trading signal structure"""
    required_fields = ['symbol', 'action', 'confidence', 'timestamp']
    
    if not isinstance(signal, dict):
        raise ValidationError("Signal must be a dictionary")
    
    if not all(field in signal for field in required_fields):
        missing = [field for field in required_fields if field not in signal]
        raise ValidationError(f"Missing signal fields: {missing}")
    
    if signal['action'] not in ['BUY', 'SELL', 'HOLD']:
        raise ValidationError(f"Invalid action: {signal['action']}")
    
    if not isinstance(signal['confidence'], (int, float)) or not 0 <= signal['confidence'] <= 1:
        raise ValidationError(f"Invalid confidence: {signal['confidence']}")
    
    if not isinstance(signal['symbol'], str) or len(signal['symbol']) == 0:
        raise ValidationError("Invalid symbol")
    
    return True

def validate_position_data(position: dict) -> bool:
    """Validate position data structure"""
    required_fields = ['symbol', 'quantity', 'entry_price', 'current_price']
    
    if not isinstance(position, dict):
        raise ValidationError("Position must be a dictionary")
    
    if not all(field in position for field in required_fields):
        missing = [field for field in required_fields if field not in position]
        raise ValidationError(f"Missing position fields: {missing}")
    
    if not isinstance(position['quantity'], (int, float)) or position['quantity'] < 0:
        raise ValidationError("Invalid quantity")
    
    if not isinstance(position['entry_price'], (int, float)) or position['entry_price'] <= 0:
        raise ValidationError("Invalid entry price")
    
    if not isinstance(position['current_price'], (int, float)) or position['current_price'] <= 0:
        raise ValidationError("Invalid current price")
    
    return True

def validate_config(config) -> bool:
    """Validate configuration object"""
    # Check trading config
    if not 0 < config.trading.stop_loss_pct < 1:
        raise ValidationError(f"Invalid stop_loss_pct: {config.trading.stop_loss_pct}")
    
    if not 0 < config.trading.max_atr_pct < 1:
        raise ValidationError(f"Invalid max_atr_pct: {config.trading.max_atr_pct}")
    
    if config.trading.sma_short >= config.trading.sma_long:
        raise ValidationError("SMA short period must be less than long period")
    
    if not 0 < config.trading.position_size_pct <= 1:
        raise ValidationError(f"Invalid position_size_pct: {config.trading.position_size_pct}")
    
    if config.trading.max_positions < 1:
        raise ValidationError(f"Invalid max_positions: {config.trading.max_positions}")
    
    # Check that ETF universe is not empty
    if not config.trading.etf_universe or len(config.trading.etf_universe) == 0:
        raise ValidationError("ETF universe cannot be empty")
    
    return True

def validate_market_data(data: Dict[str, pd.DataFrame]) -> bool:
    """Validate market data dictionary"""
    if not isinstance(data, dict):
        raise ValidationError("Market data must be a dictionary")
    
    if len(data) == 0:
        raise ValidationError("Market data dictionary is empty")
    
    for symbol, df in data.items():
        if not isinstance(symbol, str):
            raise ValidationError(f"Symbol must be string: {symbol}")
        
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(f"Data for {symbol} must be DataFrame")
        
        validate_price_data(df)
    
    return True

def validate_sentiment_data(sentiment: dict) -> bool:
    """Validate sentiment analysis data"""
    required_fields = ['sentiment_score', 'confidence']
    
    if not isinstance(sentiment, dict):
        raise ValidationError("Sentiment data must be a dictionary")
    
    if not all(field in sentiment for field in required_fields):
        missing = [field for field in required_fields if field not in sentiment]
        raise ValidationError(f"Missing sentiment fields: {missing}")
    
    if not isinstance(sentiment['sentiment_score'], (int, float)) or not -1 <= sentiment['sentiment_score'] <= 1:
        raise ValidationError(f"Invalid sentiment_score: {sentiment['sentiment_score']}")
    
    if not isinstance(sentiment['confidence'], (int, float)) or not 0 <= sentiment['confidence'] <= 1:
        raise ValidationError(f"Invalid sentiment confidence: {sentiment['confidence']}")
    
    return True

def sanitize_symbol(symbol: str) -> str:
    """Sanitize and validate trading symbol"""
    if not isinstance(symbol, str):
        raise ValidationError("Symbol must be a string")
    
    # Remove whitespace and convert to uppercase
    clean_symbol = symbol.strip().upper()
    
    # Check length (most symbols are 1-5 characters)
    if not 1 <= len(clean_symbol) <= 10:
        raise ValidationError(f"Invalid symbol length: {clean_symbol}")
    
    # Check for valid characters (letters, numbers, some special chars for ETFs)
    valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-')
    if not all(c in valid_chars for c in clean_symbol):
        raise ValidationError(f"Invalid characters in symbol: {clean_symbol}")
    
    return clean_symbol

def validate_numeric_range(value: Union[int, float], min_val: float, max_val: float, name: str) -> bool:
    """Validate that a numeric value is within a specified range"""
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric")
    
    if not min_val <= value <= max_val:
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")
    
    return True