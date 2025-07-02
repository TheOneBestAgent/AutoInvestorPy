import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

def setup_logging(config):
    """Setup comprehensive logging configuration"""
    
    # Create logs directory if it doesn't exist
    config.log_dir.mkdir(exist_ok=True)
    
    # Setup formatters
    formatter = logging.Formatter(config.log_format)
    
    # Setup file handler for all logs
    log_file = config.log_dir / f"auto_investor_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB files, keep 5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, config.log_level))
    
    # Setup error file handler
    error_file = config.log_dir / f"auto_investor_errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file, maxBytes=5*1024*1024, backupCount=3  # 5MB files, keep 3
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(error_handler)
    
    # Setup specific loggers for external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    
    logging.info("Logging system initialized")
    logging.info(f"Log files: {log_file}, {error_file}")

class TradeLogger:
    """Specialized logger for trade-related events"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('trades')
        
        # Setup trade-specific file handler
        trade_file = config.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.log"
        trade_handler = logging.handlers.RotatingFileHandler(
            trade_file, maxBytes=5*1024*1024, backupCount=10
        )
        trade_formatter = logging.Formatter(
            '%(asctime)s - TRADE - %(levelname)s - %(message)s'
        )
        trade_handler.setFormatter(trade_formatter)
        self.logger.addHandler(trade_handler)
        self.logger.setLevel(logging.INFO)
        
    def log_signal(self, symbol: str, signal: dict):
        """Log trading signal"""
        self.logger.info(f"SIGNAL - {symbol}: {signal['action']} (confidence: {signal['confidence']:.2f})")
        
    def log_trade_execution(self, symbol: str, action: str, quantity: int, price: float):
        """Log trade execution"""
        self.logger.info(f"EXECUTION - {action} {quantity} shares of {symbol} @ ${price:.2f}")
        
    def log_position_update(self, symbol: str, position_data: dict):
        """Log position updates"""
        self.logger.info(f"POSITION - {symbol}: {position_data}")
        
    def log_portfolio_summary(self, portfolio_data: dict):
        """Log portfolio summary"""
        self.logger.info(f"PORTFOLIO - Value: ${portfolio_data.get('portfolio_value', 0):.2f}, "
                        f"P&L: ${portfolio_data.get('unrealized_pnl', 0):.2f}")

class PerformanceLogger:
    """Logger for tracking performance metrics"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('performance')
        
        # Setup performance file handler
        perf_file = config.log_dir / f"performance_{datetime.now().strftime('%Y%m')}.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_file, maxBytes=2*1024*1024, backupCount=12  # Monthly files
        )
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERF - %(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        self.logger.addHandler(perf_handler)
        self.logger.setLevel(logging.INFO)
        
    def log_weekly_performance(self, metrics: dict):
        """Log weekly performance metrics"""
        self.logger.info(f"WEEKLY - Return: {metrics.get('weekly_return', 0):.2%}, "
                        f"Volatility: {metrics.get('volatility', 0):.2%}, "
                        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        
    def log_monthly_performance(self, metrics: dict):
        """Log monthly performance summary"""
        self.logger.info(f"MONTHLY - Total Return: {metrics.get('total_return', 0):.2%}, "
                        f"Win Rate: {metrics.get('win_rate', 0):.2%}, "
                        f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")