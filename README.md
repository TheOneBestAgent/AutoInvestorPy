# Micro Auto-Investor

An automated weekly micro-investing system that uses SMA crossover strategy with risk management to trade ETFs starting with as little as $50.

## 🎯 Overview

The Micro Auto-Investor is a sophisticated algorithmic trading system designed for small-scale, automated ETF trading. It implements a Simple Moving Average (SMA) crossover strategy enhanced with multiple confirmation signals, comprehensive risk management, and automated execution capabilities.

### Key Features

- **SMA Crossover Strategy**: 2-period vs 5-period SMA with confirmation signals
- **Risk Management**: Automated stop-losses, position sizing, portfolio risk assessment
- **Multi-Factor Analysis**: Technical indicators (RSI, MACD, ATR) + news sentiment
- **Paper Trading**: Built-in mock broker for safe testing
- **Automated Execution**: Weekly analysis and trade execution
- **Comprehensive Logging**: Detailed trade logs and performance tracking
- **Database Storage**: SQLite database for all trading data
- **Notifications**: Email alerts for trades and errors

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- Email account (for notifications)
- API keys for data sources (optional but recommended)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/micro-auto-investor.git
cd micro-auto-investor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**
```bash
cp .env.template .env
# Edit .env with your API keys and email settings
nano .env
```

4. **Test the installation:**
```bash
python main.py
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.template`:

```env
# Data API Keys (optional - will use yfinance as fallback)
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
NEWS_API_KEY=your_news_api_key_here

# Email Notifications
EMAIL_FROM=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
EMAIL_TO=recipient@gmail.com

# Broker API (for live trading - currently uses paper trading)
BROKER_API_KEY=your_alpaca_api_key_here
BROKER_SECRET_KEY=your_alpaca_secret_key_here
```

### Trading Parameters

Key parameters can be modified in `config.py`:

```python
@dataclass
class TradingConfig:
    stop_loss_pct: float = 0.05        # 5% stop-loss
    max_atr_pct: float = 0.04          # 4% max volatility
    sma_short: int = 2                 # Short SMA period
    sma_long: int = 5                  # Long SMA period
    position_size_pct: float = 1.0     # Use 100% of capital
    max_positions: int = 1             # Maximum concurrent positions
    
    # ETF Universe
    etf_universe: List[str] = [
        'SPY', 'QQQ', 'IWM', 'VTI', 'VOO',  # Broad market
        'XLF', 'XLK', 'XLE', 'XLV', 'XLI',  # Sectors
        'EFA', 'EEM', 'VWO', 'GLD', 'TLT'   # International/Commodities
    ]
```

## 📊 How It Works

### Weekly Analysis Cycle

1. **Data Collection**: Fetch OHLCV data for all ETFs in the universe
2. **Sentiment Analysis**: Analyze news sentiment for each ETF
3. **Signal Generation**: Calculate SMA crossovers and technical indicators
4. **Risk Assessment**: Apply volatility filters and risk checks
5. **ETF Ranking**: Score and rank ETFs based on multiple factors
6. **Position Management**: Update stop-losses and check triggers
7. **Trade Execution**: Execute new trades based on signals
8. **Logging & Notification**: Save results and send notifications

### Trading Strategy

**Primary Signal**: SMA(2) vs SMA(5) crossover
- **BUY**: When SMA(2) crosses above SMA(5)
- **SELL**: When SMA(2) crosses below SMA(5)

**Confirmation Filters**:
- Volume confirmation (20% above average)
- RSI not in extreme zones (30-70)
- MACD bullish/bearish alignment
- News sentiment analysis
- Volatility filter (ATR < 4%)

**Risk Management**:
- 5% trailing stop-loss on all positions
- Maximum 1 position at a time
- 7-day cooling period after stop-loss
- Position sizing based on account balance

## 🔧 Usage

### Basic Usage

Run weekly analysis:
```bash
python main.py
```

### Command Line Options

The system supports various operational modes:

```bash
# Run with specific configuration
python main.py --config custom_config.py

# Run in test mode (no actual trades)
python main.py --test-mode

# Run backtest on historical data
python main.py --backtest --start-date 2023-01-01 --end-date 2023-12-31
```

### Paper Trading Mode

By default, the system runs in paper trading mode using mock data:

```python
# In config.py
@dataclass
class BrokerConfig:
    paper_trading: bool = True  # Set to False for live trading
```

### Scheduling Automated Runs

Set up automated weekly runs using cron:

```bash
# Edit crontab
crontab -e

# Add entry for Friday 6 PM ET
0 18 * * 5 cd /path/to/micro-auto-investor && python main.py
```

## 📈 Monitoring & Analysis

### Database Schema

The system stores all data in SQLite database (`auto_investor.db`):

- `weekly_analysis`: Weekly analysis results
- `trades`: All trade executions
- `positions`: Position tracking
- `signals`: Trading signals
- `risk_alerts`: Risk management alerts
- `performance`: Performance metrics

### Viewing Results

```python
from utils.database import DatabaseManager
from config import config

db = DatabaseManager(config)

# Get recent performance
summary = db.get_performance_summary(days=30)
print(summary)

# Get all positions
positions = db.get_positions()
print(positions)
```

### Export Data

```python
# Export all data to CSV
db.export_data(Path("./exports"), format='csv')

# Export to JSON
db.export_data(Path("./exports"), format='json')
```

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_strategy.py
pytest tests/test_risk_manager.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Test Strategy

The codebase includes comprehensive testing:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Mock Data**: Realistic test scenarios
- **Backtesting**: Historical performance validation

### Example Test

```python
# Test SMA calculation
def test_sma_calculation(trading_strategy, sample_data):
    result = trading_strategy.calculate_sma(sample_data, 5)
    expected = sample_data['Close'].rolling(5).mean()
    pd.testing.assert_series_equal(result, expected)
```

## 🔒 Security & Risk Warnings

### Important Disclaimers

⚠️ **TRADING INVOLVES RISK**: This software is for educational purposes. You can lose money trading. Never invest more than you can afford to lose.

⚠️ **NOT INVESTMENT ADVICE**: This software does not provide investment advice. All trading decisions are automated based on technical analysis only.

⚠️ **PAPER TRADING FIRST**: Always test thoroughly in paper trading mode before considering live trading.

### Security Best Practices

1. **API Keys**: Store API keys securely in environment variables
2. **Credentials**: Never commit credentials to version control
3. **Access Control**: Limit broker API permissions to trading only
4. **Monitoring**: Monitor all trades and set up alerts
5. **Backup**: Regular database backups

### Risk Management Features

- **Position Limits**: Maximum 1 position to limit exposure
- **Stop Losses**: Automatic 5% trailing stops
- **Cooling Periods**: 7-day pause after stop-loss triggers
- **Volatility Filters**: Skip high-volatility periods
- **Account Protection**: Never exceed available balance

## 📚 Architecture

### Component Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Strategy Layer   │    │  Execution      │
│                 │    │                  │    │  Layer          │
│ • DataFetcher   │───▶│ • TradingStrategy │───▶│ • OrderManager  │
│ • Sentiment     │    │ • RiskManager    │    │ • BrokerAPI     │
│ • Validators    │    │ • Signals        │    │ • Notifications │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Storage Layer  │    │  Utils Layer     │    │  Config Layer   │
│                 │    │                  │    │                 │
│ • Database      │    │ • Logging        │    │ • Configuration │
│ • Knowledge     │    │ • Error Handling │    │ • Environment   │
│ • Exports       │    │ • Circuit Breaker│    │ • Validation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow

1. **Data Collection**: Market data, news, economic indicators
2. **Processing**: Technical analysis, sentiment scoring
3. **Signal Generation**: SMA crossovers with confirmations
4. **Risk Checks**: Volatility, position limits, stop-losses
5. **Execution**: Order placement and position management
6. **Storage**: Database logging and audit trail
7. **Monitoring**: Notifications and alerts

## 🛠️ Development

### Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run test suite: `pytest`
5. Submit pull request

### Development Setup

```bash
# Clone for development
git clone https://github.com/yourusername/micro-auto-investor.git
cd micro-auto-investor

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Style

- **Black**: Code formatting
- **isort**: Import sorting  
- **pylint**: Code analysis
- **mypy**: Type checking

```bash
# Format code
black .
isort .

# Run linting
pylint auto_investor/
mypy auto_investor/
```

## 📋 Roadmap

### Current Version (v1.0)
- ✅ SMA crossover strategy
- ✅ Risk management system
- ✅ Paper trading
- ✅ Database logging
- ✅ Email notifications

### Planned Features (v2.0)
- [ ] Live broker integration (Alpaca)
- [ ] Additional technical indicators
- [ ] Machine learning signal enhancement
- [ ] Web dashboard
- [ ] Mobile notifications
- [ ] Multi-timeframe analysis

### Future Enhancements
- [ ] Options trading
- [ ] Crypto support  
- [ ] Portfolio optimization
- [ ] Social sentiment integration
- [ ] Cloud deployment (AWS Lambda)

## 🤝 Support

### Getting Help

1. **Documentation**: Check this README and inline code comments
2. **Issues**: Submit issues on GitHub
3. **Discussions**: Join GitHub Discussions for questions
4. **Wiki**: Check the project wiki for advanced topics

### Common Issues

**Import Errors**: Make sure all dependencies are installed
```bash
pip install -r requirements.txt
```

**Database Errors**: Delete and recreate database
```bash
rm auto_investor.db
python main.py  # Will recreate database
```

**API Errors**: Check your API keys and network connection

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is provided for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. The authors and contributors are not responsible for any financial losses incurred through the use of this software.

**Always consult with a qualified financial advisor before making investment decisions.**

---

**Built with ❤️ for the algorithmic trading community**
