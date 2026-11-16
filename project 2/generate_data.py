"""
Generates a realistic historical stock prices CSV file for testing and analysis.
Uses a Geometric Brownian Motion model with realistic drift, volatility, and market shocks.
"""

import math
import random
import csv
from datetime import date, timedelta

def generate_stock_csv(filename: str = "prices.csv", days: int = 1000, start_price: float = 120.0, seed: int = 42):
    random.seed(seed)
    
    current_date = date(2022, 1, 3)
    price = start_price
    records = []
    
    # GBM parameters (annualized: drift = 12%, vol = 25%)
    dt = 1.0 / 252.0
    mu = 0.12
    sigma = 0.25
    
    for _ in range(days):
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            
        z = random.gauss(0, 1)
        # Periodic market shocks / events for realism
        shock = 0.0
        if random.random() < 0.02:
            shock = random.uniform(-0.06, 0.06)
            
        daily_return = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z + shock
        open_price = price
        close_price = price * math.exp(daily_return)
        
        # Intra-day high and low
        high_price = max(open_price, close_price) * (1.0 + abs(random.gauss(0, 0.008)))
        low_price = min(open_price, close_price) * (1.0 - abs(random.gauss(0, 0.008)))
        volume = int(random.lognormvariate(16.5, 0.4))
        
        records.append({
            "Date": current_date.isoformat(),
            "Open": round(open_price, 2),
            "High": round(high_price, 2),
            "Low": round(low_price, 2),
            "Close": round(close_price, 2),
            "Volume": volume
        })
        
        price = close_price
        current_date += timedelta(days=1)
        
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Open", "High", "Low", "Close", "Volume"])
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Generated {len(records)} trading days of stock data in '{filename}'. Initial: ${start_price:.2f}, Final: ${records[-1]['Close']:.2f}")

if __name__ == "__main__":
    generate_stock_csv("prices.csv", days=1000, start_price=150.0)
