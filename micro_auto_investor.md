# Comprehensive Automated Weekly Micro-Investor Plan (Starting Capital: \$50)

## 1. Data Sourcing & Research

- **Data Feeds**: Pull weekly OHLCV data from Yahoo Finance or Alpha Vantage.
- **News Sentiment**: Weekly scrape of ETF-specific news headlines; sentiment score determines risk.
- **Earnings & Dividends**: API check for ETF holdings’ key dates.
- **Volatility & ATR**: Compute weekly ATR; skip trades if ATR >4%.
- **Sector & Relative Strength**: Rank sectors monthly; select top 3 ETFs by 4-week performance.
- **Macro Calendar**: Fetch events like FOMC, CPI, GDP from economic calendar APIs to pause trading if major events occur.

## 2. Trading Strategy

- **Signal**: Weekly SMA2 > SMA5 crossover triggers buy; SMA2 < SMA5 triggers sell.
- **Confirmation**: Trade only if ETF is in top 3 relative strength and not in negative sentiment.
- **Position Sizing**: Allocate 100% capital per trade for compounding.

## 3. Risk Management

- One position max.
- Trailing stop-loss at -5%.
- Halt new trades for a week after stop-loss is hit.
- Avoid weeks with >4% ATR or negative news sentiment.

## 4. Portfolio Management

- Quarterly rebalance: close positions outside top 3 ETFs or >±10% from entry.
- Adjust watchlist quarterly based on liquidity and sector performance.

## 5. Automation & Execution

- **Scheduling**: AWS CloudWatch Events triggers Python script Fridays 6 PM ET.
- **Modules**:
  - Data Collector: gathers prices, news, calendar, earnings.
  - Signal Generator: calculates SMAs, ranks ETFs.
  - Risk Checker: applies volatility, sentiment, macro filters.
  - Order Executor: places/cancels orders via broker API.
  - Logger: writes to SQLite/Cloud database.
- **Fail-safes**:
  - Email/SMS alerts for errors.
  - Daily health checks for data and broker API.

## 6. Compliance

- Avoids pattern day trading by trading only weekly.
- Records all trades for tax reporting.
- Cash account only (no margin).
- Uses SEC-compliant U.S. brokers.

## 7. Knowledge Bank

- `/strategies/` for logic documents.
- `/datasets/` for raw & cleaned data.
- `/results/` for logs and account snapshots.
- `/backtests/` for historical performance.
- `/changelogs/` for version tracking.

## 8. Changelog Template

```markdown
## [Version YYYY-MM-DD]
### Changed
- Details of code or strategy updates.

### Added
- New research or functionality.

### Removed
- Deprecated logic or data sources.

### Notes
- Reasoning behind the changes.
```

## 9. Development Script Tree

```plaintext
/auto_investor/
  main.py            # Entry point, orchestrates components
  config.py          # Settings, API keys, thresholds
  /data/
    fetch_data.py    # Download OHLCV, news, earnings, macro data
    clean_data.py    # Normalize & validate data
  /signals/
    strategy.py      # SMA crossover, sector ranking, filters
  /risk/
    risk_checks.py   # Apply volatility, sentiment, macro filters
  /orders/
    broker_api.py    # Execute trades via broker
  /logging/
    logger.py        # Store all actions
  /backtest/
    backtest.py      # Run strategy on historical data
  /utils/
    notifier.py      # Email/SMS alerts
  /knowledge_bank/
    ...              # Strategy docs, datasets, logs
```

## 10. Key Features for Hands-Free Operation

- Automated data collection, signal generation, risk checks, and execution.
- Self-updating watchlist and sector rotation.
- Integrated macro-event awareness to skip risky weeks.
- Automatic quarterly rebalancing and performance review.
- Comprehensive logging and error notification.

This plan ensures a fully autonomous investor capable of researching, deciding, and trading with minimal oversight.

