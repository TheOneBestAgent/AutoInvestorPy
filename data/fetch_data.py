import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import time

from utils.error_handling import retry_with_backoff, handle_api_errors, DataError, APIError
from utils.validators import validate_price_data, sanitize_symbol

logger = logging.getLogger(__name__)

class DataFetcher:
    """Market data fetching with error handling and caching"""
    
    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.cache_timestamps = {}
        
    @retry_with_backoff(max_retries=3, exceptions=(APIError, DataError))
    @handle_api_errors
    def fetch_ohlcv(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch OHLCV data with validation"""
        try:
            # Sanitize symbol
            clean_symbol = sanitize_symbol(symbol)
            
            # Check cache first
            cache_key = f"{clean_symbol}_{period}"
            if self._is_cache_valid(cache_key):
                logger.debug(f"Using cached data for {clean_symbol}")
                return self.cache[cache_key].copy()
            
            logger.info(f"Fetching fresh data for {clean_symbol}")
            
            # Fetch data from yfinance
            ticker = yf.Ticker(clean_symbol)
            data = ticker.history(period=period, interval="1d")
            
            if data.empty:
                raise DataError(f"No data returned for {clean_symbol}")
            
            # Validate data quality
            validate_price_data(data)
            
            # Add symbol column
            data['Symbol'] = clean_symbol
            
            # Cache the data
            self.cache[cache_key] = data.copy()
            self.cache_timestamps[cache_key] = datetime.now()
            
            logger.info(f"Successfully fetched {len(data)} records for {clean_symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise
    
    def fetch_multiple_symbols(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols with parallel processing"""
        results = {}
        errors = []
        
        for symbol in symbols:
            try:
                data = self.fetch_ohlcv(symbol)
                results[symbol] = data
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                error_msg = f"{symbol}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                
        if errors:
            logger.warning(f"Errors fetching data for {len(errors)} symbols: {errors}")
            
        logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_etf_info(self, symbol: str) -> Dict:
        """Get ETF information and metadata"""
        try:
            clean_symbol = sanitize_symbol(symbol)
            ticker = yf.Ticker(clean_symbol)
            info = ticker.info
            
            # Extract relevant information
            etf_info = {
                'symbol': clean_symbol,
                'name': info.get('longName', clean_symbol),
                'sector': info.get('category', 'Unknown'),
                'expense_ratio': info.get('annualReportExpenseRatio', 0.0),
                'aum': info.get('totalAssets', 0.0),
                'inception_date': info.get('fundInceptionDate', None),
                'dividend_yield': info.get('yield', 0.0),
                'beta': info.get('beta', 1.0),
                'currency': info.get('currency', 'USD')
            }
            
            return etf_info
            
        except Exception as e:
            logger.warning(f"Could not fetch info for {symbol}: {e}")
            return {'symbol': symbol, 'name': symbol}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current/latest price for a symbol"""
        try:
            clean_symbol = sanitize_symbol(symbol)
            ticker = yf.Ticker(clean_symbol)
            
            # Try to get current price from info
            info = ticker.info
            current_price = info.get('regularMarketPrice')
            
            if current_price is None:
                # Fallback to latest close price
                data = self.fetch_ohlcv(symbol, period="5d")
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                else:
                    raise DataError(f"No price data available for {symbol}")
            
            return float(current_price)
            
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            raise DataError(f"Could not fetch current price for {symbol}: {e}")
    
    def get_volume_profile(self, symbol: str, days: int = 30) -> Dict:
        """Get volume analysis for a symbol"""
        try:
            data = self.fetch_ohlcv(symbol, period=f"{max(days, 30)}d")
            
            if data.empty:
                return {'avg_volume': 0, 'volume_trend': 'UNKNOWN'}
            
            recent_data = data.tail(days)
            
            volume_profile = {
                'avg_volume': float(recent_data['Volume'].mean()),
                'median_volume': float(recent_data['Volume'].median()),
                'volume_std': float(recent_data['Volume'].std()),
                'recent_volume': float(recent_data['Volume'].iloc[-1]),
                'volume_trend': self._analyze_volume_trend(recent_data['Volume'])
            }
            
            return volume_profile
            
        except Exception as e:
            logger.warning(f"Could not analyze volume for {symbol}: {e}")
            return {'avg_volume': 0, 'volume_trend': 'UNKNOWN'}
    
    def _analyze_volume_trend(self, volume_series: pd.Series) -> str:
        """Analyze volume trend over time"""
        if len(volume_series) < 5:
            return 'INSUFFICIENT_DATA'
        
        # Calculate trend using linear regression slope
        x = np.arange(len(volume_series))
        y = volume_series.values
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > volume_series.mean() * 0.05:  # 5% threshold
                return 'INCREASING'
            elif slope < -volume_series.mean() * 0.05:
                return 'DECREASING'
            else:
                return 'STABLE'
                
        except Exception:
            return 'UNKNOWN'
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache or cache_key not in self.cache_timestamps:
            return False
        
        cache_age = datetime.now() - self.cache_timestamps[cache_key]
        max_age = timedelta(hours=self.config.data.cache_duration_hours)
        
        return cache_age < max_age
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Data cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_items = len(self.cache)
        total_size = sum(df.memory_usage(deep=True).sum() for df in self.cache.values())
        
        return {
            'total_items': total_items,
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_item': min(self.cache_timestamps.values()) if self.cache_timestamps else None,
            'newest_item': max(self.cache_timestamps.values()) if self.cache_timestamps else None
        }

class DataQualityChecker:
    """Comprehensive data quality validation"""
    
    def __init__(self):
        self.quality_thresholds = {
            'max_missing_pct': 0.05,    # Max 5% missing data
            'min_trading_days': 20,      # Minimum trading days
            'max_price_change_pct': 0.20, # Max 20% single-day change
            'min_volume': 10000          # Minimum daily volume
        }
    
    def check_data_quality(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Comprehensive data quality check"""
        quality_report = {
            'symbol': symbol,
            'total_records': len(data),
            'quality_score': 0.0,
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        if data.empty:
            quality_report['issues'].append("No data available")
            return quality_report
        
        # Check for missing data
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        quality_report['metrics']['missing_pct'] = missing_pct
        
        if missing_pct > self.quality_thresholds['max_missing_pct']:
            quality_report['issues'].append(f"High missing data: {missing_pct:.2%}")
        
        # Check minimum data points
        if len(data) < self.quality_thresholds['min_trading_days']:
            quality_report['issues'].append(f"Insufficient data: {len(data)} days")
        
        # Check for extreme price movements
        if 'Close' in data.columns and len(data) > 1:
            price_changes = data['Close'].pct_change().abs()
            extreme_changes = price_changes > self.quality_thresholds['max_price_change_pct']
            extreme_count = extreme_changes.sum()
            quality_report['metrics']['extreme_moves'] = extreme_count
            
            if extreme_count > 0:
                quality_report['warnings'].append(f"Extreme price changes: {extreme_count} days")
        
        # Check volume
        if 'Volume' in data.columns:
            low_volume_days = (data['Volume'] < self.quality_thresholds['min_volume']).sum()
            quality_report['metrics']['low_volume_days'] = low_volume_days
            
            if low_volume_days > len(data) * 0.1:  # More than 10% of days
                quality_report['warnings'].append(f"Low volume days: {low_volume_days}")
        
        # Calculate quality score
        quality_score = 1.0
        quality_score -= len(quality_report['issues']) * 0.3
        quality_score -= len(quality_report['warnings']) * 0.1
        quality_report['quality_score'] = max(0.0, quality_score)
        
        return quality_report
    
    def is_data_usable(self, quality_report: Dict) -> bool:
        """Determine if data quality is sufficient for trading"""
        return quality_report['quality_score'] >= 0.7 and len(quality_report['issues']) == 0