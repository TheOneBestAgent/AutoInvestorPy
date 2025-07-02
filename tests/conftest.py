import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock
from pathlib import Path
import tempfile

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from signals.strategy import TradingStrategy
from risk.risk_manager import RiskManager
from data.fetch_data import DataFetcher
from data.sentiment import SentimentAnalyzer

@pytest.fixture
def config():
    """Test configuration fixture"""
    return Config()

@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing"""
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    
    # Generate realistic OHLCV data
    np.random.seed(42)  # For reproducible tests
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% volatility
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV from close prices
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    data['Open'] = data['Close'].shift(1).fillna(base_price)
    data['High'] = data[['Open', 'Close']].max(axis=1) * np.random.uniform(1.0, 1.02, len(data))
    data['Low'] = data[['Open', 'Close']].min(axis=1) * np.random.uniform(0.98, 1.0, len(data))
    data['Volume'] = np.random.randint(100000, 2000000, len(data))
    
    return data

@pytest.fixture
def sample_signals():
    """Sample trading signals for testing"""
    return [
        {
            'symbol': 'SPY',
            'action': 'BUY',
            'confidence': 0.8,
            'crossover': True,
            'volatility_ok': True,
            'timestamp': datetime.now(),
            'price': 420.50
        },
        {
            'symbol': 'QQQ',
            'action': 'HOLD',
            'confidence': 0.4,
            'crossover': False,
            'volatility_ok': True,
            'timestamp': datetime.now(),
            'price': 350.25
        }
    ]

@pytest.fixture
def sample_positions():
    """Sample positions for testing"""
    return [
        {
            'symbol': 'SPY',
            'quantity': 10,
            'entry_price': 400.0,
            'current_price': 420.0,
            'stop_loss_price': 380.0,
            'unrealized_pnl': 200.0,
            'unrealized_pnl_pct': 5.0
        }
    ]

@pytest.fixture
def mock_broker_api():
    """Mock broker API for testing"""
    mock = Mock()
    mock.get_account.return_value = {
        'equity': 10000.0,
        'cash': 5000.0,
        'buying_power': 5000.0
    }
    mock.get_positions.return_value = []
    mock.get_current_price.return_value = 100.0
    mock.submit_order.return_value = {
        'id': 'TEST123',
        'status': 'filled',
        'filled_avg_price': '100.00'
    }
    mock.is_market_open.return_value = True
    return mock

@pytest.fixture
def trading_strategy(config):
    """Trading strategy instance for testing"""
    return TradingStrategy(config)

@pytest.fixture
def risk_manager(config):
    """Risk manager instance for testing"""
    return RiskManager(config)

@pytest.fixture
def data_fetcher(config):
    """Data fetcher instance for testing"""
    return DataFetcher(config)

@pytest.fixture
def sentiment_analyzer(config):
    """Sentiment analyzer instance for testing"""
    return SentimentAnalyzer(config)

@pytest.fixture
def mock_market_data():
    """Mock market data for multiple symbols"""
    symbols = ['SPY', 'QQQ', 'IWM']
    data = {}
    
    for symbol in symbols:
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Open': [100 + i for i in range(30)],
            'High': [105 + i for i in range(30)],
            'Low': [95 + i for i in range(30)],
            'Close': [102 + i for i in range(30)],
            'Volume': [1000000] * 30
        }, index=dates)
        data[symbol] = df
    
    return data

@pytest.fixture
def mock_news_data():
    """Mock news data for sentiment testing"""
    return [
        {
            'title': 'SPY ETF shows strong performance',
            'description': 'The SPDR S&P 500 ETF continues to gain momentum',
            'publishedAt': '2023-06-01T10:00:00Z'
        },
        {
            'title': 'Market volatility concerns',
            'description': 'Investors worry about increased market volatility',
            'publishedAt': '2023-06-01T11:00:00Z'
        }
    ]

# Test utilities
class TestDataBuilder:
    """Helper class for building test data"""
    
    @staticmethod
    def create_trending_data(periods: int = 30, trend: str = 'up') -> pd.DataFrame:
        """Create trending price data for testing"""
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        
        if trend == 'up':
            prices = [100 + i * 2 for i in range(periods)]
        elif trend == 'down':
            prices = [100 - i * 2 for i in range(periods)]
        else:  # sideways
            prices = [100 + np.sin(i * 0.1) * 5 for i in range(periods)]
        
        return pd.DataFrame({
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': [1000000] * periods
        }, index=dates)
    
    @staticmethod
    def create_volatile_data(periods: int = 30) -> pd.DataFrame:
        """Create high volatility data for testing"""
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        
        # High volatility price movements
        np.random.seed(42)
        returns = np.random.normal(0, 0.05, periods)  # 5% daily volatility
        prices = [100]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.05 for p in prices],
            'Low': [p * 0.95 for p in prices],
            'Close': prices,
            'Volume': [1000000] * periods
        }, index=dates)

# Performance test helpers
@pytest.fixture
def performance_timer():
    """Timer for performance testing"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.end_time - self.start_time
    
    return Timer()

# Parameterized test data
@pytest.fixture(params=[
    {'sma_short': 2, 'sma_long': 5},
    {'sma_short': 5, 'sma_long': 10},
    {'sma_short': 10, 'sma_long': 20}
])
def sma_parameters(request):
    """Different SMA parameter combinations for testing"""
    return request.param

@pytest.fixture(params=[0.02, 0.05, 0.10])
def stop_loss_percentages(request):
    """Different stop-loss percentages for testing"""
    return request.param