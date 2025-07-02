import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class Notifier:
    """Handle email and SMS notifications"""
    
    def __init__(self, config):
        self.config = config
        self.email_enabled = config.notifications.email_enabled
        self.telegram_enabled = config.notifications.telegram_enabled
        
    def send_weekly_summary(self, ranked_etfs: List[Dict], trade_results: Dict, 
                           portfolio_assessment: Dict):
        """Send weekly analysis summary"""
        try:
            subject = f"Auto-Investor Weekly Summary - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Create summary content
            summary = self._create_weekly_summary(ranked_etfs, trade_results, portfolio_assessment)
            
            if self.email_enabled:
                self._send_email(subject, summary)
                
            logger.info("Weekly summary notification sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending weekly summary: {e}")
    
    def send_error_alert(self, error_message: str):
        """Send error alert notification"""
        try:
            subject = "Auto-Investor Error Alert"
            content = f"""
ERROR ALERT
===========
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Error: {error_message}

Please check the system logs for more details.
            """
            
            if self.email_enabled:
                self._send_email(subject, content)
                
            logger.warning(f"Error alert sent: {error_message}")
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
    
    def send_trade_notification(self, trade_details: Dict):
        """Send trade execution notification"""
        try:
            action = trade_details.get('action', 'UNKNOWN')
            symbol = trade_details.get('symbol', 'UNKNOWN')
            quantity = trade_details.get('quantity', 0)
            price = trade_details.get('price', 0)
            
            subject = f"Trade Executed: {action} {symbol}"
            content = f"""
TRADE NOTIFICATION
==================
Action: {action}
Symbol: {symbol}
Quantity: {quantity} shares
Price: ${price:.2f}
Total Value: ${quantity * price:.2f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            if self.email_enabled:
                self._send_email(subject, content)
                
            logger.info(f"Trade notification sent: {action} {quantity} {symbol}")
            
        except Exception as e:
            logger.error(f"Error sending trade notification: {e}")
    
    def _create_weekly_summary(self, ranked_etfs: List[Dict], trade_results: Dict, 
                              portfolio_assessment: Dict) -> str:
        """Create formatted weekly summary content"""
        
        summary = f"""
AUTO-INVESTOR WEEKLY SUMMARY
============================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO OVERVIEW
------------------
Portfolio Value: ${portfolio_assessment.get('portfolio_value', 0):.2f}
Account Balance: ${portfolio_assessment.get('account_balance', 0):.2f}
Unrealized P&L: ${portfolio_assessment.get('unrealized_pnl', 0):.2f} ({portfolio_assessment.get('unrealized_pnl_pct', 0):.1f}%)
Risk Level: {portfolio_assessment.get('risk_level', 'UNKNOWN')}
Total Positions: {portfolio_assessment.get('total_positions', 0)}

TOP RANKED ETFs
---------------
"""
        
        for i, etf in enumerate(ranked_etfs[:5], 1):
            signal = etf['signal']
            summary += f"{i}. {etf['symbol']}: {signal['action']} (Score: {etf['score']:.2f}, Confidence: {signal['confidence']:.2f})\n"
        
        summary += f"""

TRADES EXECUTED
---------------
"""
        
        executed_trades = trade_results.get('executed_trades', [])
        if executed_trades:
            for trade in executed_trades:
                summary += f"- {trade.get('action', 'UNKNOWN')} {trade.get('quantity', 0)} {trade.get('symbol', 'UNKNOWN')} @ ${trade.get('price', 0):.2f}\n"
        else:
            summary += "No trades executed this week.\n"
        
        # Add recommendations if any
        recommendations = portfolio_assessment.get('recommendations', [])
        if recommendations:
            summary += f"""

RECOMMENDATIONS
---------------
"""
            for rec in recommendations:
                summary += f"- {rec}\n"
        
        return summary
    
    def _send_email(self, subject: str, content: str):
        """Send email notification"""
        try:
            if not self.config.notifications.email_from or not self.config.notifications.email_to:
                logger.warning("Email configuration incomplete, skipping email notification")
                return
                
            msg = MIMEMultipart()
            msg['From'] = self.config.notifications.email_from
            msg['To'] = self.config.notifications.email_to
            msg['Subject'] = subject
            
            msg.attach(MIMEText(content, 'plain'))
            
            # Connect to server and send email
            server = smtplib.SMTP(self.config.notifications.smtp_server, self.config.notifications.smtp_port)
            server.starttls()
            server.login(self.config.notifications.email_from, self.config.notifications.email_password)
            
            text = msg.as_string()
            server.sendmail(self.config.notifications.email_from, self.config.notifications.email_to, text)
            server.quit()
            
            logger.info(f"Email sent successfully to {self.config.notifications.email_to}")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")

class MockNotifier:
    """Mock notifier for testing purposes"""
    
    def __init__(self, config):
        self.config = config
        self.sent_notifications = []
        
    def send_weekly_summary(self, ranked_etfs: List[Dict], trade_results: Dict, 
                           portfolio_assessment: Dict):
        """Mock weekly summary"""
        self.sent_notifications.append({
            'type': 'weekly_summary',
            'timestamp': datetime.now(),
            'data': {
                'ranked_etfs': ranked_etfs,
                'trade_results': trade_results,
                'portfolio_assessment': portfolio_assessment
            }
        })
        logger.info("Mock: Weekly summary notification sent")
    
    def send_error_alert(self, error_message: str):
        """Mock error alert"""
        self.sent_notifications.append({
            'type': 'error_alert',
            'timestamp': datetime.now(),
            'data': {'error_message': error_message}
        })
        logger.info(f"Mock: Error alert sent - {error_message}")
    
    def send_trade_notification(self, trade_details: Dict):
        """Mock trade notification"""
        self.sent_notifications.append({
            'type': 'trade_notification',
            'timestamp': datetime.now(),
            'data': trade_details
        })
        logger.info(f"Mock: Trade notification sent")
    
    def get_sent_notifications(self) -> List[Dict]:
        """Get list of sent notifications for testing"""
        return self.sent_notifications
    
    def clear_notifications(self):
        """Clear notification history"""
        self.sent_notifications.clear()