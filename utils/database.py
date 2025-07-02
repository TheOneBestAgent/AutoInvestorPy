import sqlite3
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    """SQLite database manager for auto-investor data"""
    
    def __init__(self, config):
        self.config = config
        self.db_path = config.project_root / "auto_investor.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create tables
                self._create_tables(cursor)
                conn.commit()
                
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _create_tables(self, cursor):
        """Create all required database tables"""
        
        # Weekly analysis results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weekly_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date DATE NOT NULL,
                portfolio_value REAL,
                account_balance REAL,
                unrealized_pnl REAL,
                unrealized_pnl_pct REAL,
                risk_level TEXT,
                total_positions INTEGER,
                top_etfs TEXT,  -- JSON string
                trade_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trade executions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date DATE NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,  -- BUY, SELL
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                signal_confidence REAL,
                stop_loss_price REAL,
                trade_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Position tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_date DATE NOT NULL,
                entry_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                current_price REAL,
                current_value REAL,
                stop_loss_price REAL,
                unrealized_pnl REAL,
                unrealized_pnl_pct REAL,
                status TEXT DEFAULT 'OPEN',  -- OPEN, CLOSED
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trading signals
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_date DATE NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                crossover BOOLEAN,
                volatility_ok BOOLEAN,
                atr_pct REAL,
                price REAL,
                sma_short REAL,
                sma_long REAL,
                rsi REAL,
                macd REAL,
                confirmations TEXT,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start DATE NOT NULL,
                period_end DATE NOT NULL,
                period_type TEXT NOT NULL,  -- WEEKLY, MONTHLY, QUARTERLY
                total_return REAL,
                annualized_return REAL,
                volatility REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                profit_factor REAL,
                trades_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Risk alerts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_date DATE NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                symbol TEXT,
                alert_value REAL,
                threshold_value REAL,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Market data cache (optional, for offline analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        """)
        
        # Create indices for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_date ON trades(symbol, trade_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_date ON signals(symbol, signal_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date ON market_data_cache(symbol, date)")
    
    def save_weekly_analysis(self, ranked_etfs: List[Dict], trade_results: Dict, 
                           portfolio_assessment: Dict):
        """Save weekly analysis results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare data
                analysis_date = datetime.now().date()
                top_etfs_json = json.dumps([{
                    'symbol': etf['symbol'],
                    'score': etf['score'],
                    'action': etf['signal']['action'],
                    'confidence': etf['signal']['confidence']
                } for etf in ranked_etfs[:5]])
                
                cursor.execute("""
                    INSERT INTO weekly_analysis (
                        analysis_date, portfolio_value, account_balance, unrealized_pnl,
                        unrealized_pnl_pct, risk_level, total_positions, top_etfs, trade_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis_date,
                    portfolio_assessment.get('portfolio_value', 0),
                    portfolio_assessment.get('account_balance', 0),
                    portfolio_assessment.get('unrealized_pnl', 0),
                    portfolio_assessment.get('unrealized_pnl_pct', 0),
                    portfolio_assessment.get('risk_level', 'UNKNOWN'),
                    portfolio_assessment.get('total_positions', 0),
                    top_etfs_json,
                    len(trade_results.get('executed_trades', []))
                ))
                
                conn.commit()
                logger.info("Weekly analysis saved to database")
                
        except Exception as e:
            logger.error(f"Error saving weekly analysis: {e}")
    
    def save_trade(self, trade_data: Dict):
        """Save trade execution data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trades (
                        trade_date, symbol, action, quantity, price, total_value,
                        signal_confidence, stop_loss_price, trade_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().date(),
                    trade_data.get('symbol'),
                    trade_data.get('action'),
                    trade_data.get('quantity', 0),
                    trade_data.get('price', 0),
                    trade_data.get('total_value', 0),
                    trade_data.get('signal_confidence', 0),
                    trade_data.get('stop_loss_price', 0),
                    trade_data.get('reason', '')
                ))
                
                conn.commit()
                logger.info(f"Trade saved: {trade_data.get('action')} {trade_data.get('symbol')}")
                
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
    
    def save_signal(self, signal_data: Dict):
        """Save trading signal data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                confirmations_json = json.dumps(signal_data.get('confirmations', {}))
                
                cursor.execute("""
                    INSERT INTO signals (
                        signal_date, symbol, action, confidence, crossover, volatility_ok,
                        atr_pct, price, sma_short, sma_long, rsi, macd, confirmations
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().date(),
                    signal_data.get('symbol'),
                    signal_data.get('action'),
                    signal_data.get('confidence', 0),
                    signal_data.get('crossover', False),
                    signal_data.get('volatility_ok', True),
                    signal_data.get('atr_pct', 0),
                    signal_data.get('price', 0),
                    signal_data.get('sma_short'),
                    signal_data.get('sma_long'),
                    signal_data.get('rsi'),
                    signal_data.get('macd'),
                    confirmations_json
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
    
    def update_position(self, position_data: Dict):
        """Update or insert position data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if position exists
                cursor.execute("SELECT id FROM positions WHERE symbol = ? AND status = 'OPEN'", 
                             (position_data['symbol'],))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing position
                    cursor.execute("""
                        UPDATE positions SET
                            current_price = ?, current_value = ?, stop_loss_price = ?,
                            unrealized_pnl = ?, unrealized_pnl_pct = ?, last_updated = ?
                        WHERE id = ?
                    """, (
                        position_data.get('current_price'),
                        position_data.get('current_value'),
                        position_data.get('stop_loss_price'),
                        position_data.get('unrealized_pnl'),
                        position_data.get('unrealized_pnl_pct'),
                        datetime.now(),
                        existing[0]
                    ))
                else:
                    # Insert new position
                    cursor.execute("""
                        INSERT INTO positions (
                            symbol, entry_date, entry_price, quantity, current_price,
                            current_value, stop_loss_price, unrealized_pnl, unrealized_pnl_pct
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        position_data.get('symbol'),
                        datetime.now().date(),
                        position_data.get('entry_price'),
                        position_data.get('quantity'),
                        position_data.get('current_price'),
                        position_data.get('current_value'),
                        position_data.get('stop_loss_price'),
                        position_data.get('unrealized_pnl'),
                        position_data.get('unrealized_pnl_pct')
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def close_position(self, symbol: str, close_price: float, close_reason: str):
        """Close a position"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE positions SET 
                        status = 'CLOSED',
                        current_price = ?,
                        last_updated = ?
                    WHERE symbol = ? AND status = 'OPEN'
                """, (close_price, datetime.now(), symbol))
                
                conn.commit()
                logger.info(f"Position closed: {symbol} @ ${close_price:.2f} ({close_reason})")
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def get_positions(self, status: str = 'OPEN') -> List[Dict]:
        """Get current positions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM positions WHERE status = ?
                    ORDER BY entry_date DESC
                """, (status,))
                
                columns = [desc[0] for desc in cursor.description]
                positions = []
                
                for row in cursor.fetchall():
                    position = dict(zip(columns, row))
                    positions.append(position)
                
                return positions
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for the last N days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get recent trades
                trades_df = pd.read_sql_query("""
                    SELECT * FROM trades 
                    WHERE trade_date >= date('now', '-{} days')
                    ORDER BY trade_date DESC
                """.format(days), conn)
                
                # Get recent analysis
                analysis_df = pd.read_sql_query("""
                    SELECT * FROM weekly_analysis 
                    WHERE analysis_date >= date('now', '-{} days')
                    ORDER BY analysis_date DESC
                """.format(days), conn)
                
                # Calculate summary metrics
                summary = {
                    'total_trades': len(trades_df),
                    'buy_trades': len(trades_df[trades_df['action'] == 'BUY']),
                    'sell_trades': len(trades_df[trades_df['action'] == 'SELL']),
                    'total_volume': trades_df['total_value'].sum() if not trades_df.empty else 0,
                    'avg_trade_size': trades_df['total_value'].mean() if not trades_df.empty else 0,
                    'current_portfolio_value': analysis_df['portfolio_value'].iloc[0] if not analysis_df.empty else 0,
                    'current_risk_level': analysis_df['risk_level'].iloc[0] if not analysis_df.empty else 'UNKNOWN',
                    'analysis_count': len(analysis_df)
                }
                
                return summary
                
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def save_risk_alert(self, alert_data: Dict):
        """Save risk alert"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO risk_alerts (
                        alert_date, alert_type, severity, message, symbol, 
                        alert_value, threshold_value
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().date(),
                    alert_data.get('type'),
                    alert_data.get('severity'),
                    alert_data.get('message'),
                    alert_data.get('symbol'),
                    alert_data.get('value'),
                    alert_data.get('threshold')
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving risk alert: {e}")
    
    def export_data(self, output_dir: Path, format: str = 'csv'):
        """Export all data to files"""
        try:
            output_dir.mkdir(exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                tables = ['weekly_analysis', 'trades', 'positions', 'signals', 'performance', 'risk_alerts']
                
                for table in tables:
                    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    
                    if format.lower() == 'csv':
                        df.to_csv(output_dir / f"{table}.csv", index=False)
                    elif format.lower() == 'json':
                        df.to_json(output_dir / f"{table}.json", orient='records', date_format='iso')
                
                logger.info(f"Data exported to {output_dir}")
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")

class MockDatabaseManager:
    """Mock database manager for testing"""
    
    def __init__(self, config):
        self.config = config
        self.data = {
            'weekly_analysis': [],
            'trades': [],
            'positions': [],
            'signals': [],
            'risk_alerts': []
        }
    
    def save_weekly_analysis(self, ranked_etfs: List[Dict], trade_results: Dict, 
                           portfolio_assessment: Dict):
        self.data['weekly_analysis'].append({
            'date': datetime.now().date(),
            'portfolio_assessment': portfolio_assessment,
            'ranked_etfs': ranked_etfs,
            'trade_results': trade_results
        })
    
    def save_trade(self, trade_data: Dict):
        self.data['trades'].append({**trade_data, 'date': datetime.now().date()})
    
    def save_signal(self, signal_data: Dict):
        self.data['signals'].append({**signal_data, 'date': datetime.now().date()})
    
    def update_position(self, position_data: Dict):
        # Simple mock implementation
        pass
    
    def close_position(self, symbol: str, close_price: float, close_reason: str):
        # Simple mock implementation
        pass
    
    def get_positions(self, status: str = 'OPEN') -> List[Dict]:
        return []
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        return {'total_trades': len(self.data['trades'])}
    
    def save_risk_alert(self, alert_data: Dict):
        self.data['risk_alerts'].append({**alert_data, 'date': datetime.now().date()})