import requests
from textblob import TextBlob
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
import time

from utils.error_handling import retry_with_backoff, handle_api_errors, APIError, DataError
from utils.validators import validate_sentiment_data

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """News sentiment analysis for trading signals"""
    
    def __init__(self, config):
        self.config = config
        self.news_api_key = config.data.news_api_key
        
    @retry_with_backoff(max_retries=3, exceptions=(APIError,))
    @handle_api_errors
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Fetch recent news for a symbol"""
        if not self.news_api_key:
            logger.warning("No News API key configured, skipping news fetch")
            return []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f"{symbol} ETF",
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'sortBy': 'relevancy',
            'language': 'en',
            'apiKey': self.news_api_key,
            'pageSize': 20  # Limit results
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            logger.info(f"Fetched {len(articles)} news articles for {symbol}")
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch news for {symbol}: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using TextBlob"""
        try:
            if not text or not text.strip():
                return {'polarity': 0.0, 'subjectivity': 0.0}
            
            blob = TextBlob(text)
            
            return {
                'polarity': blob.sentiment.polarity,      # -1 to 1
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def get_symbol_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get overall sentiment score for a symbol"""
        try:
            articles = self.fetch_news(symbol)
            
            if not articles:
                logger.debug(f"No news found for {symbol}, returning neutral sentiment")
                return {
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'article_count': 0,
                    'source': 'no_news'
                }
            
            sentiments = []
            processed_articles = 0
            
            for article in articles[:10]:  # Limit to top 10 articles
                title = article.get('title', '')
                description = article.get('description', '')
                
                if not title and not description:
                    continue
                
                text = f"{title} {description}".strip()
                sentiment = self.analyze_sentiment(text)
                
                if sentiment['polarity'] != 0.0:  # Only include non-neutral sentiments
                    sentiments.append(sentiment['polarity'])
                    processed_articles += 1
            
            if not sentiments:
                return {
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'article_count': len(articles),
                    'source': 'neutral_news'
                }
            
            # Calculate weighted average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            # Confidence based on article count and sentiment consistency
            confidence = min(len(sentiments) / 10.0, 1.0)  # Max confidence with 10+ articles
            
            # Adjust confidence based on sentiment consistency
            if len(sentiments) > 1:
                sentiment_std = pd.Series(sentiments).std()
                consistency_factor = max(0.0, 1.0 - sentiment_std)
                confidence *= consistency_factor
            
            result = {
                'sentiment_score': float(avg_sentiment),
                'confidence': float(confidence),
                'article_count': len(articles),
                'processed_articles': processed_articles,
                'source': 'news_api'
            }
            
            # Validate result
            validate_sentiment_data(result)
            
            logger.debug(f"Sentiment for {symbol}: {avg_sentiment:.3f} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'source': 'error'
            }
    
    def get_market_sentiment(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get sentiment analysis for multiple symbols"""
        sentiment_data = {}
        
        for symbol in symbols:
            try:
                sentiment_data[symbol] = self.get_symbol_sentiment(symbol)
                # Small delay to respect API rate limits
                time.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Failed to get sentiment for {symbol}: {e}")
                sentiment_data[symbol] = {
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'article_count': 0,
                    'source': 'error'
                }
        
        return sentiment_data
    
    def calculate_market_mood(self, sentiment_data: Dict[str, Dict]) -> Dict:
        """Calculate overall market mood from individual sentiments"""
        if not sentiment_data:
            return {'market_mood': 'NEUTRAL', 'confidence': 0.0}
        
        # Weight sentiments by confidence
        weighted_sentiments = []
        total_confidence = 0.0
        
        for symbol, sentiment in sentiment_data.items():
            score = sentiment.get('sentiment_score', 0.0)
            confidence = sentiment.get('confidence', 0.0)
            
            if confidence > 0:
                weighted_sentiments.append(score * confidence)
                total_confidence += confidence
        
        if not weighted_sentiments or total_confidence == 0:
            return {'market_mood': 'NEUTRAL', 'confidence': 0.0}
        
        # Calculate weighted average
        avg_sentiment = sum(weighted_sentiments) / total_confidence
        overall_confidence = total_confidence / len(sentiment_data)
        
        # Determine mood
        if avg_sentiment > 0.1:
            mood = 'BULLISH'
        elif avg_sentiment < -0.1:
            mood = 'BEARISH'
        else:
            mood = 'NEUTRAL'
        
        return {
            'market_mood': mood,
            'sentiment_score': avg_sentiment,
            'confidence': overall_confidence,
            'symbols_analyzed': len(sentiment_data)
        }

class AlternativeSentimentProvider:
    """Alternative sentiment provider for when News API is not available"""
    
    def __init__(self, config):
        self.config = config
        
    def get_symbol_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment using alternative methods (RSS, web scraping, etc.)"""
        # Placeholder for alternative sentiment sources
        # Could implement RSS feed parsing, Reddit sentiment, etc.
        
        logger.debug(f"Using alternative sentiment provider for {symbol}")
        
        return {
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'article_count': 0,
            'source': 'alternative'
        }

class SentimentAggregator:
    """Aggregate sentiment from multiple sources"""
    
    def __init__(self, config):
        self.config = config
        self.primary_analyzer = SentimentAnalyzer(config)
        self.alternative_analyzer = AlternativeSentimentProvider(config)
        
    def get_comprehensive_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment from multiple sources and aggregate"""
        
        # Try primary source first
        primary_sentiment = self.primary_analyzer.get_symbol_sentiment(symbol)
        
        # If primary source has low confidence, try alternative
        if primary_sentiment['confidence'] < 0.3:
            alternative_sentiment = self.alternative_analyzer.get_symbol_sentiment(symbol)
            
            # Combine sentiments if both are available
            if alternative_sentiment['confidence'] > 0:
                combined_score = (
                    primary_sentiment['sentiment_score'] * primary_sentiment['confidence'] +
                    alternative_sentiment['sentiment_score'] * alternative_sentiment['confidence']
                ) / (primary_sentiment['confidence'] + alternative_sentiment['confidence'])
                
                return {
                    'sentiment_score': combined_score,
                    'confidence': (primary_sentiment['confidence'] + alternative_sentiment['confidence']) / 2,
                    'article_count': primary_sentiment['article_count'] + alternative_sentiment['article_count'],
                    'source': 'combined'
                }
        
        return primary_sentiment