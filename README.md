# Crypto Funding Rate Arbitrage Strategy (Hedge Frame)
An enterprise-grade, production-ready quantitative backtesting framework built in Python to isolate and exploit structural pricing inefficiencies and funding rate differentials between highly correlated crypto perpetual swap contracts (BTCUSD / BCHUSD) using production data structures mapped from Delta Exchange India.

📈 Strategic Backtest Performance Overview

A comprehensive 507-day backtest was executed across real-world historical market structures. The system uncovered a powerful mathematical anomaly: while engineered as a low-beta, delta-neutral cash-flow harvest engine, the edge return from asset spread convergence violently outpaced baseline funding accumulation.

Performance Breakdown
========================================================================

Trade Summary Report: BTCUSD / BCHUSD Funding-Hedge Strategy

========================================================================

Backtest Period           : 2025-01-10 20:30:00 to 2026-06-01 20:30:00

Backtest Length (Days)    : 507.0 Days

Total Executed Trades     : 20 Positions

Average Holding Window    : 538.80 Hours (22.45 Days)

Return Allocations:
  - Funding Return Total  : +34.1652% (Annualized: 24.5963%)
  - Hedge Return Total    : +83.5396% (Annualized: 60.1419%)

👉 Cumulative Strategy Return: +117.7048% (Annualized CAGR: 84.7382%)

========================================================================

🧠 The Mathematical Plot Twist

The core hypothesis focused on farming the Funding Rate payout intervals. However, because the entry rule triggers exclusively during absolute valuation discrepancies (Spread ≥ 1%), the relative pricing spread between BTCUSD and BCHUSD snapped back to historical mean values with remarkable force. The Hedge Return generated 60.14% annualized, proving that structural entry thresholds act as a natural mean-reversion filter.

⚙️ Core Architecture & Operational Mechanics

1. The Funding Mechanism (Alpha Capture): Perpetual swap contracts balance price tracking against spot indices using an 8-hour payment window. Overextended directional biases drive funding rates into extremes, which this framework actively harvests.
2. Delta-Neutral Insulated Hedging (Risk Mitigation): By matching long and short positions across tightly correlated crypto pairs, beta market risk is theoretically neutralized. Total portfolio protection is maintained throughout severe macroeconomic macro swings.
3. Execution Logic:
   1. Entry: Triggered if Net Funding Rate = |Funding_sym2 - Funding_sym1| >= 0.01  
   2. Exit: Governed by the Directional Flip Exit Rule—positions liquidate instantly when the net spread returns to zero or crosses into opposite polarity.

