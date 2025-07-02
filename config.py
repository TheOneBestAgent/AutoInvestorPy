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
    
    # ETF watchlist
    etf_universe: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'IWM', 'VTI', 'VOO',  # Broad market
        'XLF', 'XLK', 'XLE', 'XLV', 'XLI',  # Sectors
        'EFA', 'EEM', 'VWO', 'GLD', 'TLT'   # International/Commodities
    ])

@dataclass
class DataConfig:
    """Data source configuration"""
    alpha_vantage_key: str = field(default_factory=lambda: os.getenv('ALPHA_VANTAGE_KEY', ''))
    news_api_key: str = field(default_factory=lambda: os.getenv('NEWS_API_KEY', ''))
    economic_calendar_key: str = field(default_factory=lambda: os.getenv('ECON_CALENDAR_KEY', ''))
    cache_duration_hours: int = 1
    max_retries: int = 3
    request_timeout: int = 30

@dataclass
class BrokerConfig:
    """Broker API configuration"""
    api_key: str = field(default_factory=lambda: os.getenv('BROKER_API_KEY', ''))
    secret_key: str = field(default_factory=lambda: os.getenv('BROKER_SECRET_KEY', ''))
    base_url: str = "https://paper-api.alpaca.markets"
    paper_trading: bool = True
    
@dataclass
class NotificationConfig:
    """Notification settings"""
    email_enabled: bool = True
    telegram_enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_from: str = field(default_factory=lambda: os.getenv('EMAIL_FROM', ''))
    email_password: str = field(default_factory=lambda: os.getenv('EMAIL_PASSWORD', ''))
    email_to: str = field(default_factory=lambda: os.getenv('EMAIL_TO', ''))
    telegram_token: str = field(default_factory=lambda: os.getenv('TELEGRAM_TOKEN', ''))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv('TELEGRAM_CHAT_ID', ''))

@dataclass
class Config:
    """Main configuration class"""
    trading: TradingConfig = field(default_factory=TradingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent / "logs")
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "cache")
    knowledge_bank_dir: Path = field(default_factory=lambda: Path(__file__).parent / "knowledge_bank")
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Database
    database_url: str = field(default_factory=lambda: f"sqlite:///{Path(__file__).parent}/auto_investor.db")
    
    def __post_init__(self):
        # Create directories if they don't exist
        self.log_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_bank_dir.mkdir(exist_ok=True)
        
        # Create knowledge bank subdirectories
        for subdir in ['strategies', 'datasets', 'results', 'backtests', 'changelogs']:
            (self.knowledge_bank_dir / subdir).mkdir(exist_ok=True)

# Global config instance
config = Config()