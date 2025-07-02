# Comprehensive Micro Auto-Investor Development Gameplan

## 🎯 Executive Summary

**Project**: Automated Weekly Micro-Investing Python Tool  
**Initial Capital**: $50  
**Timeline**: 20 weeks (5 months)  
**Trading Strategy**: SMA crossover with multi-factor risk filtering  
**Target**: 8-12% annual returns with automated risk management  

## 🛠️ Required MCP Tools Setup

### Essential MCP Tools for This Project

```bash
# Install required MCP tools
# 1. Memory Server - Store project insights and architectural decisions
# 2. Git Server - Version control and commit analysis  
# 3. Browser Tools - Test financial APIs and data sources
# 4. Sequential Thinking - Complex problem solving and debugging
# 5. Filesystem Server - Direct project file access
```

**MCP Integration Strategy:**
- Use memory server to store all architectural decisions and bug fixes
- Use sequential thinking for complex debugging sessions
- Use git server for tracking changes and analyzing commit patterns
- Use browser tools for testing financial data APIs
- Use filesystem server for reading multiple configuration files in parallel

## 📋 Phase-by-Phase Development Plan

---

## **Phase 1: Foundation & Architecture (Weeks 1-2)**

### 🎯 **Objectives**
- Establish robust project foundation
- Set up development environment
- Create modular architecture
- Implement comprehensive logging

### 📋 **Technical Requirements**

**Project Structure:**
```
/auto_investor/
├── main.py                 # Orchestration entry point
├── config.py              # Configuration management
├── requirements.txt       # Dependencies
├── pytest.ini            # Testing configuration
├── .env.template         # Environment variables template
├── /data/
│   ├── __init__.py
│   ├── fetch_data.py     # Market data collection
│   ├── clean_data.py     # Data validation and cleaning
│   ├── sentiment.py      # News sentiment analysis
│   └── calendar.py       # Economic calendar integration
├── /signals/
│   ├── __init__.py
│   ├── strategy.py       # SMA crossover logic
│   ├── indicators.py     # Technical indicators
│   └── filters.py        # Confirmation filters
├── /risk/
│   ├── __init__.py
│   ├── risk_checks.py    # Risk management rules
│   ├── position_sizing.py # Position size calculation
│   └── stop_loss.py      # Stop-loss management
├── /orders/
│   ├── __init__.py
│   ├── broker_api.py     # Broker integration
│   ├── order_manager.py  # Order execution logic
│   └── paper_trading.py  # Paper trading simulator
├── /logging/
│   ├── __init__.py
│   ├── logger.py         # Logging configuration
│   └── performance.py    # Performance tracking
├── /backtest/
│   ├── __init__.py
│   ├── backtest.py       # Backtesting engine
│   └── analysis.py       # Performance analysis
├── /utils/
│   ├── __init__.py
│   ├── notifier.py       # Email/SMS notifications
│   ├── database.py       # Database operations
│   └── validators.py     # Input validation
├── /tests/
│   ├── __init__.py
│   ├── test_data/        # Test data fixtures
│   ├── test_signals/     # Strategy testing
│   ├── test_risk/        # Risk management testing
│   └── integration/      # Integration tests
└── /knowledge_bank/
    ├── strategies/       # Strategy documentation
    ├── datasets/         # Historical data
    ├── results/          # Performance results
    └── changelogs/       # Version tracking
```

**Core Dependencies:**
```python
# requirements.txt
pandas>=1.5.3
numpy>=1.24.3
yfinance>=0.2.18
requests>=2.31.0
sqlalchemy>=2.0.15
python-dotenv>=1.0.0
boto3>=1.26.137
textblob>=0.17.1
matplotlib>=3.7.1
pytest>=7.3.1
pytest-mock>=3.10.0
pytest-asyncio>=0.21.0
hypothesis>=6.75.1
responses>=0.23.0
freezegun>=1.2.2
factory-boy>=3.2.1
```

**Configuration Management:**
```python
# config.py
import os
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path

@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    stop_loss_pct: float = 0.05
    max_atr_pct: float = 0.04
    sma_short: int = 2
    sma_long: int = 5
    position_size_pct: float = 1.0
    cooling_period_days: int = 7
    max_positions: int = 1
    min_volume: int = 100000

@dataclass
class DataConfig:
    """Data source configuration"""
    alpha_vantage_key: str = field(default_factory=lambda: os.getenv('ALPHA_VANTAGE_KEY'))
    news_api_key: str = field(default_factory=lambda: os.getenv('NEWS_API_KEY'))
    economic_calendar_key: str = field(default_factory=lambda: os.getenv('ECON_CALENDAR_KEY'))
    cache_duration_hours: int = 1
    max_retries: int = 3
    request_timeout: int = 30

@dataclass
class BrokerConfig:
    """Broker API configuration"""
    api_key: str = field(default_factory=lambda: os.getenv('BROKER_API_KEY'))
    secret_key: str = field(default_factory=lambda: os.getenv('BROKER_SECRET_KEY'))
    base_url: str = "https://paper-api.alpaca.markets"
    paper_trading: bool = True
    
@dataclass
class Config:
    """Main configuration class"""
    trading: TradingConfig = field(default_factory=TradingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent / "logs")
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "cache")
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        # Create directories if they don't exist
        self.log_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
```

### 🧪 **Testing Strategy**

**Unit Testing Framework:**
```python
# tests/conftest.py
import pytest
import pandas as pd
from unittest.mock import Mock
from auto_investor.config import Config

@pytest.fixture
def config():
    """Test configuration fixture"""
    return Config()

@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing"""
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'open': [100 + i for i in range(30)],
        'high': [105 + i for i in range(30)],
        'low': [95 + i for i in range(30)],
        'close': [102 + i for i in range(30)],
        'volume': [1000000] * 30
    }, index=dates)

@pytest.fixture
def mock_broker_api():
    """Mock broker API for testing"""
    mock = Mock()
    mock.get_account.return_value = {"buying_power": 1000.0}
    mock.get_positions.return_value = []
    return mock
```

**Configuration Testing:**
```python
# tests/test_config.py
import pytest
import os
from auto_investor.config import Config, TradingConfig

def test_config_default_values():
    """Test default configuration values"""
    config = Config()
    assert config.trading.stop_loss_pct == 0.05
    assert config.trading.sma_short == 2
    assert config.trading.sma_long == 5

def test_config_environment_variables(monkeypatch):
    """Test configuration from environment variables"""
    monkeypatch.setenv("BROKER_API_KEY", "test_key")
    config = Config()
    assert config.broker.api_key == "test_key"

def test_config_validation():
    """Test configuration validation"""
    config = Config()
    assert config.trading.sma_short < config.trading.sma_long
    assert 0 < config.trading.stop_loss_pct < 1
```

### 🛡️ **Bug Prevention Measures**

**Input Validation:**
```python
# utils/validators.py
from typing import Union, List
import pandas as pd

class ValidationError(Exception):
    """Custom validation exception"""
    pass

def validate_price_data(data: pd.DataFrame) -> bool:
    """Validate OHLCV price data"""
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Check required columns
    if not all(col in data.columns for col in required_columns):
        raise ValidationError(f"Missing required columns: {required_columns}")
    
    # Check for negative prices
    price_columns = ['open', 'high', 'low', 'close']
    if (data[price_columns] < 0).any().any():
        raise ValidationError("Negative prices detected")
    
    # Check high >= low logic
    if (data['high'] < data['low']).any():
        raise ValidationError("High price below low price detected")
    
    # Check for missing data
    if data[required_columns].isnull().any().any():
        raise ValidationError("Missing data detected")
    
    return True

def validate_trade_signal(signal: dict) -> bool:
    """Validate trading signal structure"""
    required_fields = ['symbol', 'action', 'confidence', 'timestamp']
    
    if not all(field in signal for field in required_fields):
        raise ValidationError(f"Missing signal fields: {required_fields}")
    
    if signal['action'] not in ['BUY', 'SELL', 'HOLD']:
        raise ValidationError(f"Invalid action: {signal['action']}")
    
    if not 0 <= signal['confidence'] <= 1:
        raise ValidationError(f"Invalid confidence: {signal['confidence']}")
    
    return True
```

**Error Handling Patterns:**
```python
# utils/error_handling.py
import functools
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
            
        return wrapper
    return decorator

def handle_api_errors(func: Callable) -> Callable:
    """Decorator for handling API errors gracefully"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise APIError(f"API request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    
    return wrapper
```

### ✅ **Success Criteria**

1. **✅ Project Structure**: All directories and files created with proper imports
2. **✅ Configuration Management**: Environment variables loaded and validated
3. **✅ Logging System**: Structured logging with appropriate levels
4. **✅ Database Schema**: SQLite database created with all required tables
5. **✅ Testing Framework**: pytest configured with fixtures and mock data
6. **✅ Error Handling**: Retry mechanisms and graceful error handling implemented
7. **✅ Documentation**: README and setup instructions completed

### 🔍 **Validation Checkpoints**

- [ ] Run `pytest tests/test_config.py` - All configuration tests pass
- [ ] Run `python -c "from auto_investor.config import Config; print(Config())"` - Configuration loads without errors
- [ ] Check logging output format and levels
- [ ] Verify database schema creation
- [ ] Test error handling with intentional failures

---

## **Phase 2: Data Infrastructure (Weeks 3-4)**

### 🎯 **Objectives**
- Implement robust market data collection
- Build news sentiment analysis pipeline
- Create economic calendar integration
- Establish data validation and caching systems

### 📋 **Technical Requirements**

**Market Data Module:**
```python
# data/fetch_data.py
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    """Market data fetching with error handling and caching"""
    
    def __init__(self, config):
        self.config = config
        self.cache = {}
        
    @retry_with_backoff(max_retries=3)
    @handle_api_errors
    def fetch_ohlcv(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch OHLCV data with validation"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise DataError(f"No data returned for {symbol}")
            
            # Validate data quality
            validate_price_data(data)
            
            # Add symbol column
            data['symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise
    
    def fetch_multiple_symbols(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols in parallel"""
        results = {}
        errors = []
        
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_ohlcv(symbol)
            except Exception as e:
                errors.append(f"{symbol}: {e}")
                
        if errors:
            logger.warning(f"Errors fetching data: {errors}")
            
        return results
    
    def get_etf_list(self) -> List[str]:
        """Get list of liquid ETFs for trading"""
        return [
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO',  # Broad market
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI',  # Sectors
            'EFA', 'EEM', 'VWO', 'GLD', 'TLT'   # International/Commodities
        ]
```

**News Sentiment Analysis:**
```python
# data/sentiment.py
import requests
from textblob import TextBlob
import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """News sentiment analysis for trading signals"""
    
    def __init__(self, config):
        self.config = config
        self.news_api_key = config.data.news_api_key
        
    @retry_with_backoff(max_retries=3)
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Fetch recent news for a symbol"""
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f"{symbol} ETF",
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'sortBy': 'relevancy',
            'language': 'en',
            'apiKey': self.news_api_key
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        return response.json().get('articles', [])
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using TextBlob"""
        blob = TextBlob(text)
        
        return {
            'polarity': blob.sentiment.polarity,      # -1 to 1
            'subjectivity': blob.sentiment.subjectivity  # 0 to 1
        }
    
    def get_symbol_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get overall sentiment score for a symbol"""
        try:
            articles = self.fetch_news(symbol)
            
            if not articles:
                logger.warning(f"No news found for {symbol}")
                return {'sentiment_score': 0.0, 'confidence': 0.0}
            
            sentiments = []
            for article in articles[:10]:  # Limit to top 10 articles
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self.analyze_sentiment(text)
                sentiments.append(sentiment['polarity'])
            
            # Calculate weighted average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments)
            confidence = min(len(sentiments) / 10.0, 1.0)  # Confidence based on article count
            
            return {
                'sentiment_score': avg_sentiment,
                'confidence': confidence,
                'article_count': len(articles)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return {'sentiment_score': 0.0, 'confidence': 0.0}
```

### 🧪 **Testing Strategy**

**Data Fetching Tests:**
```python
# tests/test_data/test_fetch_data.py
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from auto_investor.data.fetch_data import DataFetcher

@pytest.fixture
def data_fetcher(config):
    return DataFetcher(config)

def test_fetch_ohlcv_success(data_fetcher, sample_ohlcv_data):
    """Test successful OHLCV data fetching"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.return_value.history.return_value = sample_ohlcv_data
        
        result = data_fetcher.fetch_ohlcv('SPY')
        
        assert not result.empty
        assert 'symbol' in result.columns
        assert result['symbol'].iloc[0] == 'SPY'

def test_fetch_ohlcv_no_data(data_fetcher):
    """Test handling of no data returned"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        
        with pytest.raises(DataError, match="No data returned"):
            data_fetcher.fetch_ohlcv('INVALID')

def test_fetch_multiple_symbols(data_fetcher, sample_ohlcv_data):
    """Test fetching multiple symbols"""
    with patch.object(data_fetcher, 'fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = sample_ohlcv_data
        
        result = data_fetcher.fetch_multiple_symbols(['SPY', 'QQQ'])
        
        assert len(result) == 2
        assert 'SPY' in result
        assert 'QQQ' in result
```

**Sentiment Analysis Tests:**
```python
# tests/test_data/test_sentiment.py
import pytest
from unittest.mock import Mock, patch
from auto_investor.data.sentiment import SentimentAnalyzer

@pytest.fixture
def sentiment_analyzer(config):
    return SentimentAnalyzer(config)

def test_analyze_sentiment_positive(sentiment_analyzer):
    """Test positive sentiment analysis"""
    result = sentiment_analyzer.analyze_sentiment("This is great news!")
    
    assert result['polarity'] > 0
    assert 0 <= result['subjectivity'] <= 1

def test_analyze_sentiment_negative(sentiment_analyzer):
    """Test negative sentiment analysis"""
    result = sentiment_analyzer.analyze_sentiment("This is terrible news!")
    
    assert result['polarity'] < 0
    assert 0 <= result['subjectivity'] <= 1

@patch('requests.get')
def test_fetch_news_success(mock_get, sentiment_analyzer):
    """Test successful news fetching"""
    mock_response = Mock()
    mock_response.json.return_value = {
        'articles': [
            {'title': 'Test Title', 'description': 'Test Description'}
        ]
    }
    mock_get.return_value = mock_response
    
    result = sentiment_analyzer.fetch_news('SPY')
    
    assert len(result) == 1
    assert result[0]['title'] == 'Test Title'
```

### 🛡️ **Bug Prevention Measures**

**Data Quality Checks:**
```python
# data/quality_checks.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

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
            'warnings': []
        }
        
        # Check for missing data
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > self.quality_thresholds['max_missing_pct']:
            quality_report['issues'].append(f"High missing data: {missing_pct:.2%}")
        
        # Check minimum data points
        if len(data) < self.quality_thresholds['min_trading_days']:
            quality_report['issues'].append(f"Insufficient data: {len(data)} days")
        
        # Check for extreme price movements
        if not data.empty:
            price_changes = data['close'].pct_change().abs()
            extreme_changes = price_changes > self.quality_thresholds['max_price_change_pct']
            if extreme_changes.any():
                quality_report['warnings'].append(f"Extreme price changes detected: {extreme_changes.sum()} days")
        
        # Check volume
        if 'volume' in data.columns:
            low_volume_days = (data['volume'] < self.quality_thresholds['min_volume']).sum()
            if low_volume_days > 0:
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
```

### ✅ **Success Criteria**

1. **✅ Market Data**: Successfully fetch OHLCV data for all ETF symbols
2. **✅ Data Validation**: All fetched data passes quality checks
3. **✅ News Integration**: Sentiment analysis working for major ETFs
4. **✅ Caching System**: Data caching reduces API calls by 80%
5. **✅ Error Handling**: Graceful handling of API failures and rate limits
6. **✅ Performance**: Data fetching completes within 60 seconds for all symbols

### 🔍 **Validation Checkpoints**

- [ ] Run `pytest tests/test_data/` - All data tests pass
- [ ] Fetch live data for SPY, QQQ, IWM - Data quality score > 0.8
- [ ] Test sentiment analysis for recent news - Sentiment scores within expected range
- [ ] Test error handling with invalid symbols - Graceful error handling
- [ ] Performance test with all ETF symbols - Completes within time limit

---

## **Phase 3: Trading Strategy Engine (Weeks 5-6)**

### 🎯 **Objectives**
- Implement SMA crossover strategy
- Build technical indicators library
- Create signal confirmation filters
- Develop backtesting framework

### 📋 **Technical Requirements**

**Strategy Engine:**
```python
# signals/strategy.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TradingStrategy:
    """SMA crossover strategy with multiple confirmations"""
    
    def __init__(self, config):
        self.config = config
        self.sma_short = config.trading.sma_short
        self.sma_long = config.trading.sma_long
        
    def calculate_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data['close'].rolling(window=period).mean()
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range for volatility"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=period).mean()
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Generate trading signals with confirmations"""
        try:
            # Calculate indicators
            data['sma_short'] = self.calculate_sma(data, self.sma_short)
            data['sma_long'] = self.calculate_sma(data, self.sma_long)
            data['atr'] = self.calculate_atr(data)
            
            # Get latest values
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            # Primary signal: SMA crossover
            current_signal = 'BUY' if latest['sma_short'] > latest['sma_long'] else 'SELL'
            prev_signal = 'BUY' if prev['sma_short'] > prev['sma_long'] else 'SELL'
            
            # Check for crossover
            crossover = current_signal != prev_signal
            
            # Volatility filter
            atr_pct = (latest['atr'] / latest['close']) * 100
            volatility_ok = atr_pct <= self.config.trading.max_atr_pct * 100
            
            # Calculate confidence score
            confidence = self._calculate_confidence(data, symbol)
            
            return {
                'symbol': symbol,
                'action': current_signal if crossover else 'HOLD',
                'confidence': confidence,
                'crossover': crossover,
                'volatility_ok': volatility_ok,
                'atr_pct': atr_pct,
                'timestamp': pd.Timestamp.now(),
                'price': latest['close']
            }
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_confidence(self, data: pd.DataFrame, symbol: str) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence_factors = []
        
        # Trend strength
        recent_data = data.tail(10)
        trend_consistency = len(recent_data[recent_data['sma_short'] > recent_data['sma_long']]) / 10
        confidence_factors.append(abs(trend_consistency - 0.5) * 2)  # 0-1 scale
        
        # Volume confirmation
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        recent_volume = data['volume'].iloc[-1]
        volume_factor = min(recent_volume / avg_volume, 2.0) / 2.0  # Cap at 2x average
        confidence_factors.append(volume_factor)
        
        # Price momentum
        returns = data['close'].pct_change().tail(5)
        momentum = abs(returns.mean())
        confidence_factors.append(min(momentum * 100, 1.0))  # Convert to 0-1 scale
        
        return np.mean(confidence_factors)
```

### 🧪 **Testing Strategy**

**Strategy Testing:**
```python
# tests/test_signals/test_strategy.py
import pytest
import pandas as pd
import numpy as np
from auto_investor.signals.strategy import TradingStrategy

@pytest.fixture
def trading_strategy(config):
    return TradingStrategy(config)

def test_sma_calculation(trading_strategy, sample_ohlcv_data):
    """Test SMA calculation accuracy"""
    result = trading_strategy.calculate_sma(sample_ohlcv_data, 5)
    
    # Verify SMA calculation
    expected = sample_ohlcv_data['close'].rolling(5).mean()
    pd.testing.assert_series_equal(result, expected)

def test_crossover_detection(trading_strategy):
    """Test SMA crossover detection"""
    # Create test data with known crossover
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 103, 102, 101, 100, 99],
        'high': [101, 102, 103, 104, 105, 104, 103, 102, 101, 100],
        'low': [99, 100, 101, 102, 103, 102, 101, 100, 99, 98],
        'volume': [1000000] * 10
    }, index=dates)
    
    signal = trading_strategy.generate_signals(data, 'TEST')
    
    assert signal['symbol'] == 'TEST'
    assert signal['action'] in ['BUY', 'SELL', 'HOLD']
    assert 0 <= signal['confidence'] <= 1

def test_volatility_filter(trading_strategy):
    """Test ATR volatility filter"""
    # Create high volatility test data
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    high_vol_data = pd.DataFrame({
        'close': [100 + i * 5 for i in range(20)],  # High volatility
        'high': [105 + i * 5 for i in range(20)],
        'low': [95 + i * 5 for i in range(20)],
        'volume': [1000000] * 20
    }, index=dates)
    
    signal = trading_strategy.generate_signals(high_vol_data, 'VOLATILE')
    
    # Should have high ATR percentage
    assert signal['atr_pct'] > 4.0  # Above our threshold
    assert not signal['volatility_ok']
```

### ✅ **Success Criteria**

1. **✅ SMA Calculations**: SMA values match expected mathematical results
2. **✅ Crossover Detection**: Accurate detection of bullish/bearish crossovers  
3. **✅ Volatility Filtering**: ATR filter correctly identifies high volatility periods
4. **✅ Signal Quality**: Confidence scores correlate with actual performance
5. **✅ Backtesting**: Strategy shows positive results on historical data

---

## **Phase 4: Risk Management System (Weeks 7-8)**

### 🎯 **Objectives**
- Implement trailing stop-loss mechanism
- Build position sizing calculator
- Create risk monitoring dashboard
- Develop trade halting logic

### 📋 **Technical Requirements**

**Risk Management:**
```python
# risk/risk_manager.py
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, config):
        self.config = config
        self.stop_loss_pct = config.trading.stop_loss_pct
        self.max_positions = config.trading.max_positions
        self.cooling_period = config.trading.cooling_period_days
        
    def calculate_position_size(self, account_balance: float, symbol: str, 
                              current_price: float) -> Dict:
        """Calculate appropriate position size"""
        # Use 100% of available balance (as per strategy)
        available_capital = account_balance * self.config.trading.position_size_pct
        
        # Calculate shares (must be whole shares)
        shares = int(available_capital / current_price)
        
        # Calculate actual investment amount
        investment_amount = shares * current_price
        
        # Calculate stop-loss price
        stop_loss_price = current_price * (1 - self.stop_loss_pct)
        
        # Calculate maximum loss
        max_loss = (current_price - stop_loss_price) * shares
        
        return {
            'symbol': symbol,
            'shares': shares,
            'investment_amount': investment_amount,
            'available_capital': available_capital,
            'stop_loss_price': stop_loss_price,
            'max_loss': max_loss,
            'max_loss_pct': (max_loss / investment_amount) * 100 if investment_amount > 0 else 0
        }
    
    def update_stop_loss(self, position: Dict, current_price: float) -> Dict:
        """Update trailing stop-loss price"""
        entry_price = position['entry_price']
        current_stop = position.get('stop_loss_price', entry_price * (1 - self.stop_loss_pct))
        
        # Calculate new trailing stop
        new_stop = current_price * (1 - self.stop_loss_pct)
        
        # Only update if new stop is higher (for long positions)
        if new_stop > current_stop:
            position['stop_loss_price'] = new_stop
            position['stop_updated'] = pd.Timestamp.now()
            logger.info(f"Updated stop-loss for {position['symbol']}: {new_stop:.2f}")
        
        return position
    
    def check_stop_loss_trigger(self, position: Dict, current_price: float) -> bool:
        """Check if stop-loss should be triggered"""
        stop_price = position.get('stop_loss_price')
        
        if stop_price is None:
            return False
        
        # Trigger if current price is below stop-loss
        return current_price <= stop_price
    
    def assess_portfolio_risk(self, positions: List[Dict], 
                            account_balance: float) -> Dict:
        """Assess overall portfolio risk"""
        total_value = sum(pos['current_value'] for pos in positions)
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in positions)
        
        # Calculate portfolio metrics
        portfolio_risk = {
            'total_positions': len(positions),
            'total_value': total_value,
            'account_balance': account_balance,
            'portfolio_value': account_balance + total_unrealized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'unrealized_pnl_pct': (total_unrealized_pnl / account_balance) * 100 if account_balance > 0 else 0,
            'risk_level': self._calculate_risk_level(total_unrealized_pnl, account_balance)
        }
        
        return portfolio_risk
    
    def _calculate_risk_level(self, unrealized_pnl: float, account_balance: float) -> str:
        """Calculate risk level based on unrealized P&L"""
        if account_balance == 0:
            return 'UNKNOWN'
        
        pnl_pct = (unrealized_pnl / account_balance) * 100
        
        if pnl_pct < -10:
            return 'HIGH'
        elif pnl_pct < -5:
            return 'MEDIUM'
        elif pnl_pct > 10:
            return 'LOW'  # Positive territory
        else:
            return 'LOW'
```

### 🧪 **Testing Strategy**

**Risk Management Tests:**
```python
# tests/test_risk/test_risk_manager.py
import pytest
from auto_investor.risk.risk_manager import RiskManager

@pytest.fixture
def risk_manager(config):
    return RiskManager(config)

def test_position_sizing(risk_manager):
    """Test position size calculation"""
    result = risk_manager.calculate_position_size(
        account_balance=1000.0,
        symbol='SPY',
        current_price=400.0
    )
    
    assert result['shares'] == 2  # 1000 / 400 = 2.5, rounded down to 2
    assert result['investment_amount'] == 800.0  # 2 * 400
    assert result['stop_loss_price'] == 380.0  # 400 * 0.95
    assert result['max_loss'] == 40.0  # (400 - 380) * 2

def test_trailing_stop_update(risk_manager):
    """Test trailing stop-loss updates"""
    position = {
        'symbol': 'SPY',
        'entry_price': 400.0,
        'stop_loss_price': 380.0,
        'shares': 2
    }
    
    # Price goes up - stop should update
    updated_position = risk_manager.update_stop_loss(position, 420.0)
    assert updated_position['stop_loss_price'] == 399.0  # 420 * 0.95
    
    # Price goes down - stop should not update
    updated_position = risk_manager.update_stop_loss(updated_position, 410.0)
    assert updated_position['stop_loss_price'] == 399.0  # Unchanged

def test_stop_loss_trigger(risk_manager):
    """Test stop-loss trigger detection"""
    position = {
        'symbol': 'SPY',
        'stop_loss_price': 380.0
    }
    
    # Price above stop - no trigger
    assert not risk_manager.check_stop_loss_trigger(position, 385.0)
    
    # Price at stop - trigger
    assert risk_manager.check_stop_loss_trigger(position, 380.0)
    
    # Price below stop - trigger
    assert risk_manager.check_stop_loss_trigger(position, 375.0)
```

### ✅ **Success Criteria**

1. **✅ Position Sizing**: Correct calculation of shares and investment amounts
2. **✅ Stop-Loss Logic**: Trailing stops update correctly with price movements
3. **✅ Risk Assessment**: Portfolio risk metrics calculated accurately
4. **✅ Trade Halting**: System correctly halts trading after stop-loss events
5. **✅ Performance**: Risk calculations complete within 1 second

---

## **Phase 5: Broker Integration (Weeks 9-10)**

### 🎯 **Objectives**
- Integrate Alpaca broker API
- Implement order management system
- Create paper trading mode
- Build position monitoring

### 📋 **Technical Requirements**

**Broker API Integration:**
```python
# orders/broker_api.py
import alpaca_trade_api as tradeapi
from typing import Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)

class AlpacaBroker:
    """Alpaca broker API integration"""
    
    def __init__(self, config):
        self.config = config
        self.api = tradeapi.REST(
            config.broker.api_key,
            config.broker.secret_key,
            config.broker.base_url,
            api_version='v2'
        )
        
    def get_account(self) -> Dict:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'account_id': account.id,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'day_trading_buying_power': float(account.day_trading_buying_power),
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': int(pos.qty),
                    'side': pos.side,
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price),
                    'avg_entry_price': float(pos.avg_entry_price)
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise
    
    def place_order(self, symbol: str, qty: int, side: str, 
                   order_type: str = 'market', time_in_force: str = 'day',
                   stop_price: Optional[float] = None) -> Dict:
        """Place an order"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                stop_price=stop_price
            )
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'qty': int(order.qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'filled_qty': int(order.filled_qty or 0),
                'filled_avg_price': float(order.filled_avg_price or 0)
            }
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status"""
        try:
            order = self.api.get_order(order_id)
            return {
                'order_id': order.id,
                'status': order.status,
                'filled_qty': int(order.filled_qty or 0),
                'filled_avg_price': float(order.filled_avg_price or 0)
            }
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            raise
```

**Order Manager:**
```python
# orders/order_manager.py
from typing import Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)

class OrderManager:
    """Manages order execution and monitoring"""
    
    def __init__(self, broker, risk_manager):
        self.broker = broker
        self.risk_manager = risk_manager
        self.active_orders = {}
        
    def execute_trade(self, signal: Dict, account_balance: float) -> Dict:
        """Execute a trade based on signal"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            current_price = signal['price']
            
            if action == 'HOLD':
                return {'status': 'no_action', 'reason': 'HOLD signal'}
            
            # Calculate position size
            position_info = self.risk_manager.calculate_position_size(
                account_balance, symbol, current_price
            )
            
            if position_info['shares'] == 0:
                return {'status': 'no_action', 'reason': 'Insufficient funds'}
            
            # Place market order
            order = self.broker.place_order(
                symbol=symbol,
                qty=position_info['shares'],
                side=action.lower(),
                order_type='market'
            )
            
            # Set stop-loss order
            if action == 'BUY':
                stop_order = self.broker.place_order(
                    symbol=symbol,
                    qty=position_info['shares'],
                    side='sell',
                    order_type='stop',
                    stop_price=position_info['stop_loss_price']
                )
                
                order['stop_order_id'] = stop_order['order_id']
            
            # Track order
            self.active_orders[order['order_id']] = {
                'signal': signal,
                'position_info': position_info,
                'order': order,
                'timestamp': time.time()
            }
            
            logger.info(f"Order placed: {order}")
            return {'status': 'success', 'order': order}
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def monitor_orders(self) -> List[Dict]:
        """Monitor active orders and update status"""
        updates = []
        
        for order_id, order_data in list(self.active_orders.items()):
            try:
                status = self.broker.get_order_status(order_id)
                
                if status['status'] in ['filled', 'cancelled', 'rejected']:
                    # Order completed
                    order_data['final_status'] = status
                    updates.append(order_data)
                    del self.active_orders[order_id]
                    
            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")
        
        return updates
```

### 🧪 **Testing Strategy**

**Broker Integration Tests:**
```python
# tests/test_orders/test_broker_api.py
import pytest
from unittest.mock import Mock, patch
from auto_investor.orders.broker_api import AlpacaBroker

@pytest.fixture
def mock_alpaca_api():
    """Mock Alpaca API for testing"""
    with patch('alpaca_trade_api.REST') as mock_api:
        yield mock_api.return_value

@pytest.fixture 
def alpaca_broker(config, mock_alpaca_api):
    return AlpacaBroker(config)

def test_get_account(alpaca_broker, mock_alpaca_api):
    """Test account information retrieval"""
    # Mock account data
    mock_account = Mock()
    mock_account.id = 'test_account'
    mock_account.buying_power = '1000.00'
    mock_account.cash = '1000.00'
    mock_account.portfolio_value = '1000.00'
    mock_alpaca_api.get_account.return_value = mock_account
    
    result = alpaca_broker.get_account()
    
    assert result['account_id'] == 'test_account'
    assert result['buying_power'] == 1000.0
    assert result['cash'] == 1000.0

def test_place_order(alpaca_broker, mock_alpaca_api):
    """Test order placement"""
    # Mock order response
    mock_order = Mock()
    mock_order.id = 'order_123'
    mock_order.symbol = 'SPY'
    mock_order.qty = '2'
    mock_order.side = 'buy'
    mock_order.status = 'accepted'
    mock_alpaca_api.submit_order.return_value = mock_order
    
    result = alpaca_broker.place_order('SPY', 2, 'buy')
    
    assert result['order_id'] == 'order_123'
    assert result['symbol'] == 'SPY'
    assert result['qty'] == 2
    assert result['side'] == 'buy'
```

### ✅ **Success Criteria**

1. **✅ API Connection**: Successful connection to Alpaca API
2. **✅ Order Placement**: Orders placed and tracked correctly
3. **✅ Position Monitoring**: Real-time position updates
4. **✅ Stop-Loss Orders**: Automatic stop-loss placement
5. **✅ Error Handling**: Graceful handling of API errors

---

## **Phase 6: Automation & Orchestration (Weeks 11-12)**

### 🎯 **Objectives**
- Create main orchestration script
- Implement scheduling system
- Build notification system
- Add health checks

### 📋 **Technical Requirements**

**Main Orchestration:**
```python
# main.py
import logging
import sys
from datetime import datetime, timedelta
import pandas as pd

from auto_investor.config import Config
from auto_investor.data.fetch_data import DataFetcher
from auto_investor.data.sentiment import SentimentAnalyzer
from auto_investor.signals.strategy import TradingStrategy
from auto_investor.risk.risk_manager import RiskManager
from auto_investor.orders.broker_api import AlpacaBroker
from auto_investor.orders.order_manager import OrderManager
from auto_investor.utils.notifier import Notifier
from auto_investor.logging.logger import setup_logging

logger = logging.getLogger(__name__)

class AutoInvestor:
    """Main orchestration class for automated investing"""
    
    def __init__(self):
        self.config = Config()
        self.setup_components()
        
    def setup_components(self):
        """Initialize all components"""
        self.data_fetcher = DataFetcher(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.strategy = TradingStrategy(self.config)
        self.risk_manager = RiskManager(self.config)
        self.broker = AlpacaBroker(self.config)
        self.order_manager = OrderManager(self.broker, self.risk_manager)
        self.notifier = Notifier(self.config)
        
    def run_weekly_analysis(self) -> Dict:
        """Run weekly trading analysis and execution"""
        logger.info("Starting weekly analysis...")
        
        try:
            # Get ETF list
            etf_symbols = self.data_fetcher.get_etf_list()
            
            # Fetch market data
            logger.info("Fetching market data...")
            market_data = self.data_fetcher.fetch_multiple_symbols(etf_symbols)
            
            # Get account information
            account_info = self.broker.get_account()
            current_positions = self.broker.get_positions()
            
            # Analyze each ETF
            signals = []
            for symbol, data in market_data.items():
                if data is not None and not data.empty:
                    # Get sentiment
                    sentiment = self.sentiment_analyzer.get_symbol_sentiment(symbol)
                    
                    # Generate trading signal
                    signal = self.strategy.generate_signals(data, symbol)
                    signal['sentiment'] = sentiment
                    
                    # Apply filters
                    signal = self._apply_filters(signal, data)
                    signals.append(signal)
            
            # Rank signals and select best
            best_signal = self._select_best_signal(signals)
            
            # Execute trade if signal is strong enough
            execution_result = None
            if best_signal and best_signal['action'] != 'HOLD':
                execution_result = self.order_manager.execute_trade(
                    best_signal, account_info['buying_power']
                )
            
            # Update positions with stop-losses
            self._update_positions(current_positions)
            
            # Generate report
            report = self._generate_report(signals, best_signal, execution_result, account_info)
            
            # Send notifications
            self.notifier.send_weekly_report(report)
            
            logger.info("Weekly analysis completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error in weekly analysis: {e}")
            self.notifier.send_error_alert(str(e))
            raise
    
    def _apply_filters(self, signal: Dict, data: pd.DataFrame) -> Dict:
        """Apply additional filters to signals"""
        # Sentiment filter
        sentiment_score = signal.get('sentiment', {}).get('sentiment_score', 0)
        if sentiment_score < -0.3:  # Very negative sentiment
            signal['action'] = 'HOLD'
            signal['filter_reason'] = 'Negative sentiment'
        
        # Volume filter
        recent_volume = data['volume'].tail(5).mean()
        avg_volume = data['volume'].mean()
        if recent_volume < avg_volume * 0.5:  # Low volume
            signal['action'] = 'HOLD'
            signal['filter_reason'] = 'Low volume'
        
        return signal
    
    def _select_best_signal(self, signals: List[Dict]) -> Optional[Dict]:
        """Select the best trading signal"""
        # Filter for BUY signals only
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        
        if not buy_signals:
            return None
        
        # Sort by confidence and select highest
        buy_signals.sort(key=lambda x: x['confidence'], reverse=True)
        return buy_signals[0]
    
    def _update_positions(self, positions: List[Dict]):
        """Update trailing stop-losses for existing positions"""
        for position in positions:
            try:
                # Get current price
                current_data = self.data_fetcher.fetch_ohlcv(position['symbol'], '1d')
                current_price = current_data['close'].iloc[-1]
                
                # Update stop-loss
                updated_position = self.risk_manager.update_stop_loss(position, current_price)
                
                # Place new stop-loss order if updated
                if updated_position.get('stop_updated'):
                    self.broker.place_order(
                        symbol=position['symbol'],
                        qty=abs(position['qty']),
                        side='sell',
                        order_type='stop',
                        stop_price=updated_position['stop_loss_price']
                    )
                    
            except Exception as e:
                logger.error(f"Error updating position {position['symbol']}: {e}")
    
    def _generate_report(self, signals: List[Dict], best_signal: Optional[Dict], 
                        execution_result: Optional[Dict], account_info: Dict) -> Dict:
        """Generate comprehensive trading report"""
        return {
            'timestamp': datetime.now(),
            'account_info': account_info,
            'signals_analyzed': len(signals),
            'best_signal': best_signal,
            'execution_result': execution_result,
            'signals': signals[:5]  # Top 5 signals
        }

def main():
    """Main entry point"""
    setup_logging()
    
    try:
        investor = AutoInvestor()
        report = investor.run_weekly_analysis()
        
        print("Weekly analysis completed successfully!")
        print(f"Account balance: ${report['account_info']['buying_power']:.2f}")
        
        if report['best_signal']:
            print(f"Best signal: {report['best_signal']['symbol']} - {report['best_signal']['action']}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### ✅ **Success Criteria**

1. **✅ Orchestration**: All components work together seamlessly
2. **✅ Scheduling**: Weekly execution runs automatically
3. **✅ Error Handling**: System recovers from failures gracefully
4. **✅ Notifications**: Users receive timely updates and alerts
5. **✅ Performance**: Complete analysis finishes within 10 minutes

---

## **Phase 7-10: AWS Deployment, Testing, Compliance & Launch (Weeks 13-20)**

### **Phase 7: AWS Deployment & Infrastructure (Weeks 13-14)**

**AWS Lambda Deployment:**
```python
# deploy/lambda_handler.py
import json
import logging
from main import AutoInvestor

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """AWS Lambda handler for weekly trading execution"""
    try:
        investor = AutoInvestor()
        report = investor.run_weekly_analysis()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Weekly analysis completed successfully',
                'account_balance': report['account_info']['buying_power'],
                'best_signal': report.get('best_signal', {}).get('symbol', 'None')
            })
        }
    except Exception as e:
        logger.error(f"Lambda execution failed: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

**CloudFormation Template:**
```yaml
# infrastructure/cloudformation.yml
AWSTemplateFormatVersion: '2010-09-09'
Resources:
  AutoInvestorFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: micro-auto-investor
      Runtime: python3.9
      Handler: lambda_handler.lambda_handler
      Code:
        ZipFile: |
          # Lambda code deployed here
      Environment:
        Variables:
          BROKER_API_KEY: !Ref BrokerApiKey
          ALPHA_VANTAGE_KEY: !Ref AlphaVantageKey
      Timeout: 300
      
  WeeklySchedule:
    Type: AWS::Events::Rule
    Properties:
      Description: "Weekly trading execution"
      ScheduleExpression: "cron(0 23 ? * FRI *)"  # Fridays at 6 PM ET
      State: ENABLED
      Targets:
        - Arn: !GetAtt AutoInvestorFunction.Arn
          Id: "WeeklyTarget"
```

### **Phase 8: Comprehensive Testing (Weeks 15-16)**

**End-to-End Testing:**
```python
# tests/e2e/test_full_workflow.py
import pytest
from unittest.mock import Mock, patch
from auto_investor.main import AutoInvestor

@pytest.mark.e2e
def test_complete_workflow(config):
    """Test complete weekly analysis workflow"""
    with patch.multiple(
        'auto_investor.main',
        DataFetcher=Mock(),
        SentimentAnalyzer=Mock(),
        AlpacaBroker=Mock()
    ):
        investor = AutoInvestor()
        report = investor.run_weekly_analysis()
        
        assert 'timestamp' in report
        assert 'account_info' in report
        assert 'signals_analyzed' in report

@pytest.mark.performance
def test_performance_benchmarks():
    """Test performance requirements"""
    start_time = time.time()
    
    # Run analysis
    investor = AutoInvestor()
    report = investor.run_weekly_analysis()
    
    execution_time = time.time() - start_time
    assert execution_time < 600  # Must complete within 10 minutes
```

### **Phase 9: Compliance & Documentation (Weeks 17-18)**

**Tax Reporting:**
```python
# utils/tax_reporting.py
import pandas as pd
from typing import List, Dict

class TaxReporter:
    """Generate tax reports for trading activity"""
    
    def generate_annual_report(self, trades: List[Dict]) -> pd.DataFrame:
        """Generate annual tax report"""
        tax_data = []
        
        for trade in trades:
            if trade['status'] == 'filled':
                tax_data.append({
                    'date': trade['filled_at'],
                    'symbol': trade['symbol'],
                    'action': trade['side'],
                    'quantity': trade['qty'],
                    'price': trade['filled_avg_price'],
                    'total_amount': trade['qty'] * trade['filled_avg_price'],
                    'fees': 0.0  # Alpaca commission-free
                })
        
        return pd.DataFrame(tax_data)
```

### **Phase 10: Launch & Monitoring (Weeks 19-20)**

**Production Monitoring:**
```python
# monitoring/dashboard.py
import boto3
import pandas as pd
from typing import Dict, List

class PerformanceDashboard:
    """Monitor system performance and trading results"""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        
    def publish_metrics(self, report: Dict):
        """Publish metrics to CloudWatch"""
        self.cloudwatch.put_metric_data(
            Namespace='AutoInvestor',
            MetricData=[
                {
                    'MetricName': 'AccountBalance',
                    'Value': report['account_info']['buying_power'],
                    'Unit': 'None'
                },
                {
                    'MetricName': 'SignalsAnalyzed',
                    'Value': report['signals_analyzed'],
                    'Unit': 'Count'
                }
            ]
        )
```

## 🧪 **Overall Testing Architecture**

### **Testing Pyramid Structure**

1. **Unit Tests (70%)**: Fast, isolated tests for individual functions
2. **Integration Tests (20%)**: Test component interactions
3. **End-to-End Tests (10%)**: Full workflow testing with paper trading

### **Test Data Management**
```python
# tests/fixtures/market_data.py
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def generate_test_ohlcv(symbol: str, days: int = 30) -> pd.DataFrame:
    """Generate realistic test OHLCV data"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price movements
    np.random.seed(42)  # Reproducible data
    returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily return, 2% volatility
    
    prices = [100]  # Starting price
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.DataFrame({
        'open': [p * 0.999 for p in prices[1:]],
        'high': [p * 1.015 for p in prices[1:]],
        'low': [p * 0.985 for p in prices[1:]],
        'close': prices[1:],
        'volume': np.random.randint(500000, 2000000, days)
    }, index=dates)
```

### **Continuous Integration Setup**
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run unit tests
      run: pytest tests/unit/ -v --cov=auto_investor
    
    - name: Run integration tests
      run: pytest tests/integration/ -v
    
    - name: Run end-to-end tests
      run: pytest tests/e2e/ -v --paper-trading
```

## 🚀 **Deployment Strategy**

### **Local Development Setup**
```bash
# setup_dev.sh
#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup environment
cp .env.template .env
echo "Please configure your API keys in .env file"

# Initialize database
python -c "from auto_investor.utils.database import init_db; init_db()"

# Run initial tests
pytest tests/unit/ -v
```

### **Production Deployment (AWS)**
```python
# deploy/aws_deploy.py
import boto3
import zipfile
import os

def deploy_lambda():
    """Deploy to AWS Lambda with proper configuration"""
    # Create deployment package
    with zipfile.ZipFile('deployment.zip', 'w') as z:
        for root, dirs, files in os.walk('auto_investor'):
            for file in files:
                z.write(os.path.join(root, file))
    
    # Deploy to Lambda
    lambda_client = boto3.client('lambda')
    lambda_client.update_function_code(
        FunctionName='micro-auto-investor',
        ZipFile=open('deployment.zip', 'rb').read()
    )
```

## 🎯 **MCP Tools Integration Plan**

### **Required MCP Tools for This Project**

1. **Memory Server (`mcp_memory`)**
   - Store architectural decisions and design patterns
   - Record bug fixes and their solutions
   - Maintain cross-session development insights
   - Track performance optimizations

2. **Sequential Thinking (`mcp_sequential-thinking`)**
   - Complex debugging of multi-component issues
   - Strategy development and backtesting analysis
   - Root cause analysis for system failures
   - Planning feature implementations step-by-step

3. **Git Server (`mcp_git`)**
   - Version control management
   - Commit history analysis for debugging
   - Branch management for feature development
   - Code review and change tracking

4. **Browser Tools (`mcp_browser`)**
   - Test financial data APIs (Yahoo Finance, Alpha Vantage)
   - Validate broker API endpoints
   - Debug network connectivity issues
   - Test web-based monitoring dashboards

5. **Filesystem Server (`mcp_filesystem`)**
   - Direct file access for configuration management
   - Parallel reading of multiple data files
   - Log file analysis and debugging
   - Backup and restore operations

### **MCP Workflow Integration**

**Development Workflow:**
```
Sequential Thinking → Memory Storage → Git Tracking → Testing
```

**Debugging Workflow:**
```
Browser Testing → Sequential Analysis → Memory Update → Git Commit
```

**Deployment Workflow:**
```
Filesystem Access → Git Management → Memory Documentation
```

## 📊 **Risk Management & Success Metrics**

### **Technical Risk Mitigation**

1. **Data Quality Risks**
   - Implement multiple data source fallbacks
   - Real-time data validation at API boundaries
   - Circuit breaker patterns for external services
   - Comprehensive logging for data issues

2. **Trading Execution Risks**
   - Paper trading validation before live deployment
   - Mandatory stop-loss orders on all positions
   - Position size limits and account balance checks
   - Trade confirmation and reconciliation processes

3. **System Reliability Risks**
   - Redundant AWS deployment across regions
   - Health check monitoring with automatic alerts
   - Graceful degradation for non-critical failures
   - Automated backup and disaster recovery

### **Success Metrics & KPIs**

**Technical Performance:**
- System uptime: >99.5%
- Trade execution latency: <30 seconds
- Data fetch success rate: >95%
- Test coverage: >90%

**Financial Performance:**
- Risk-adjusted returns (Sharpe ratio)
- Maximum drawdown limits (<10%)
- Win rate and profit factor
- Tax efficiency metrics

**Operational Excellence:**
- Mean time to detection (MTTD): <5 minutes
- Mean time to recovery (MTTR): <30 minutes
- False positive alert rate: <5%
- Documentation completeness: 100%

## 🗓️ **Project Timeline & Milestones**

### **Critical Path Analysis**

**Weeks 1-8: Foundation** (High Priority)
- Core architecture and data infrastructure
- Cannot proceed without solid foundation
- Dependency: All subsequent phases

**Weeks 9-12: Integration** (Medium Priority)  
- Broker integration and automation
- Can be developed in parallel with testing framework
- Dependency: Phase 13+ deployment

**Weeks 13-20: Production** (Low Priority)
- AWS deployment and monitoring
- Can be refined iteratively
- Dependency: System launch

### **Go/No-Go Decision Points**

**Week 8: Technical Foundation Review**
- [ ] All unit tests passing (>95% coverage)
- [ ] Data quality validation working
- [ ] Strategy backtesting shows positive results
- [ ] Risk management system functional

**Week 12: Integration Readiness**
- [ ] Broker API integration complete
- [ ] Paper trading successful for 1 week
- [ ] End-to-end testing passing
- [ ] Performance benchmarks met

**Week 16: Production Deployment**
- [ ] AWS infrastructure provisioned
- [ ] Security audit completed
- [ ] Monitoring dashboards operational
- [ ] Disaster recovery tested

**Week 20: Live Trading Launch**
- [ ] Compliance validation complete
- [ ] $50 initial capital deployed
- [ ] Weekly execution schedule active
- [ ] Performance monitoring in place

## 🔧 **Development Best Practices**

### **Code Quality Standards**

```python
# Example: Type hints and documentation standards
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_position_size(
    account_balance: float,
    symbol: str,
    current_price: float,
    risk_percentage: float = 0.05
) -> Dict[str, float]:
    """
    Calculate appropriate position size based on account balance and risk.
    
    Args:
        account_balance: Available trading capital
        symbol: Trading symbol (e.g., 'SPY')
        current_price: Current market price
        risk_percentage: Maximum risk per trade (default 5%)
    
    Returns:
        Dictionary containing position size details
        
    Raises:
        ValueError: If invalid parameters provided
    """
    if account_balance <= 0:
        raise ValueError("Account balance must be positive")
    
    # Implementation with proper error handling
    # ...
```

### **Monitoring & Alerting Strategy**

```python
# monitoring/alerts.py
import boto3
from typing import Dict, List

class AlertManager:
    """Centralized alerting for system monitoring"""
    
    def __init__(self, config):
        self.sns = boto3.client('sns')
        self.topic_arn = config.aws.sns_topic_arn
        
    def send_critical_alert(self, message: str, context: Dict):
        """Send critical system alert"""
        self.sns.publish(
            TopicArn=self.topic_arn,
            Subject="CRITICAL: Auto-Investor Alert",
            Message=f"{message}\n\nContext: {context}"
        )
    
    def send_performance_report(self, metrics: Dict):
        """Send weekly performance summary"""
        # Implementation for performance reporting
        pass
```

## 🎯 **Final Implementation Checklist**

### **Pre-Development Setup**
- [ ] Install all required MCP tools
- [ ] Set up development environment with Python 3.9+
- [ ] Create project structure and version control
- [ ] Configure API keys and environment variables
- [ ] Set up testing framework and CI/CD pipeline

### **Phase-by-Phase Validation**
- [ ] Phase 1: Foundation architecture passes all tests
- [ ] Phase 2: Data infrastructure handles edge cases
- [ ] Phase 3: Trading strategy shows positive backtests
- [ ] Phase 4: Risk management prevents excessive losses
- [ ] Phase 5: Broker integration executes orders correctly
- [ ] Phase 6: Automation runs without manual intervention
- [ ] Phase 7: AWS deployment scales and performs
- [ ] Phase 8: Testing suite covers all critical paths
- [ ] Phase 9: Compliance requirements fully met
- [ ] Phase 10: Production monitoring operational

### **Launch Readiness Criteria**
- [ ] System passes all automated tests
- [ ] Paper trading shows consistent profitability
- [ ] Risk management tested under stress
- [ ] Documentation complete and reviewed
- [ ] Monitoring and alerting verified
- [ ] Backup and recovery procedures tested
- [ ] Initial $50 capital ready for deployment

This comprehensive gameplan provides a detailed roadmap for building a robust, automated micro-investing system with extensive testing, monitoring, and risk management capabilities. The systematic approach ensures reliable operation while maximizing the chances of successful automated trading.
</rewritten_file> 