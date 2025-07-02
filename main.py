#!/usr/bin/env python3
"""
Micro Auto-Investor - Main Entry Point
Automated weekly micro-investing system with SMA crossover strategy
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import config
from logging.logger import setup_logging
from data.fetch_data import DataFetcher
from data.sentiment import SentimentAnalyzer
from signals.strategy import TradingStrategy
from risk.risk_manager import RiskManager
from orders.broker_api import BrokerAPI
from orders.order_manager import OrderManager
from utils.notifier import Notifier
from utils.database import DatabaseManager

logger = logging.getLogger(__name__)

class AutoInvestor:
    """Main orchestrator for the automated investing system"""
    
    def __init__(self):
        self.config = config
        
        # Initialize components
        self.data_fetcher = DataFetcher(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.trading_strategy = TradingStrategy(self.config)
        self.risk_manager = RiskManager(self.config)
        self.broker_api = BrokerAPI(self.config)
        self.order_manager = OrderManager(self.config, self.broker_api, self.risk_manager)
        self.notifier = Notifier(self.config)
        self.database = DatabaseManager(self.config)
        
        logger.info("AutoInvestor initialized successfully")
    
    def run_weekly_analysis(self):
        """Run the complete weekly analysis and trading cycle"""
        try:
            logger.info("=" * 60)
            logger.info("Starting weekly auto-investor analysis")
            logger.info(f"Timestamp: {datetime.now()}")
            logger.info("=" * 60)
            
            # Step 1: Fetch market data for all ETFs
            logger.info("Step 1: Fetching market data...")
            market_data = self.data_fetcher.fetch_multiple_symbols(
                self.config.trading.etf_universe
            )
            
            if not market_data:
                logger.error("No market data retrieved. Aborting analysis.")
                return False
            
            logger.info(f"Successfully fetched data for {len(market_data)} symbols")
            
            # Step 2: Analyze sentiment for each ETF
            logger.info("Step 2: Analyzing market sentiment...")
            sentiment_data = {}
            for symbol in market_data.keys():
                sentiment_data[symbol] = self.sentiment_analyzer.get_symbol_sentiment(symbol)
            
            # Step 3: Generate trading signals
            logger.info("Step 3: Generating trading signals...")
            signals = {}
            for symbol, data in market_data.items():
                signal = self.trading_strategy.generate_signals(data, symbol)
                
                # Add sentiment to signal
                signal['sentiment'] = sentiment_data.get(symbol, {'sentiment_score': 0.0})
                signals[symbol] = signal
            
            # Step 4: Rank ETFs and select top candidates
            logger.info("Step 4: Ranking ETFs and selecting candidates...")
            ranked_etfs = self._rank_etfs(signals)
            
            # Step 5: Check current positions and assess portfolio
            logger.info("Step 5: Assessing current portfolio...")
            current_positions = self.broker_api.get_positions()
            account_info = self.broker_api.get_account()
            
            portfolio_assessment = self.risk_manager.assess_portfolio_risk(
                current_positions, account_info['equity']
            )
            
            # Step 6: Execute trading decisions
            logger.info("Step 6: Executing trading decisions...")
            trade_results = self.order_manager.execute_weekly_trades(
                ranked_etfs, current_positions, account_info
            )
            
            # Step 7: Log results and send notifications
            logger.info("Step 7: Logging results and sending notifications...")
            self._log_weekly_results(ranked_etfs, trade_results, portfolio_assessment)
            
            # Send summary notification
            self.notifier.send_weekly_summary(
                ranked_etfs, trade_results, portfolio_assessment
            )
            
            logger.info("Weekly analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in weekly analysis: {e}", exc_info=True)
            self.notifier.send_error_alert(f"Weekly analysis failed: {e}")
            return False
    
    def _rank_etfs(self, signals: dict) -> list:
        """Rank ETFs based on signal strength and sentiment"""
        etf_scores = []
        
        for symbol, signal in signals.items():
            # Base score from signal confidence
            score = signal.get('confidence', 0.0)
            
            # Adjust for sentiment
            sentiment_score = signal.get('sentiment', {}).get('sentiment_score', 0.0)
            score += sentiment_score * 0.2  # 20% weight to sentiment
            
            # Penalize high volatility
            if not signal.get('volatility_ok', True):
                score *= 0.5
            
            # Bonus for crossover signals
            if signal.get('crossover', False) and signal.get('action') == 'BUY':
                score *= 1.2
            
            etf_scores.append({
                'symbol': symbol,
                'score': score,
                'signal': signal
            })
        
        # Sort by score descending and return top 3
        etf_scores.sort(key=lambda x: x['score'], reverse=True)
        return etf_scores[:3]
    
    def _log_weekly_results(self, ranked_etfs: list, trade_results: dict, 
                          portfolio_assessment: dict):
        """Log weekly results to database and files"""
        # Save to database
        self.database.save_weekly_analysis(
            ranked_etfs, trade_results, portfolio_assessment
        )
        
        # Log summary
        logger.info("=== WEEKLY RESULTS SUMMARY ===")
        logger.info(f"Top ETFs: {[etf['symbol'] for etf in ranked_etfs]}")
        logger.info(f"Trades executed: {len(trade_results.get('executed_trades', []))}")
        logger.info(f"Portfolio value: ${portfolio_assessment.get('portfolio_value', 0):.2f}")
        logger.info(f"Risk level: {portfolio_assessment.get('risk_level', 'UNKNOWN')}")

def main():
    """Main entry point"""
    # Setup logging
    setup_logging(config)
    
    try:
        # Create auto-investor instance
        auto_investor = AutoInvestor()
        
        # Run weekly analysis
        success = auto_investor.run_weekly_analysis()
        
        if success:
            logger.info("Auto-investor completed successfully")
            sys.exit(0)
        else:
            logger.error("Auto-investor failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()