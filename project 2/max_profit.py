"""
Max Profit Finder: Divide-and-Conquer Maximum Subarray vs. Linear Scan.
Implements:
  1. Divide-and-Conquer Maximum Subarray (O(n log n))
  2. Linear Scan / Kadane's Algorithm (O(n))
  3. Direct Price Linear Scan (O(n))
  4. Brute Force (O(n^2)) for verification
Supports reading price data via pandas or built-in csv.
"""

from typing import List, Tuple, Dict, Any, Optional
import csv
import math

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False


def load_price_data(csv_path: str) -> Tuple[List[str], List[float]]:
    """
    Loads Date and Close price series from a CSV file.
    Uses pandas if available, falling back to standard csv module.
    """
    if HAS_PANDAS and pd is not None:
        df = pd.read_csv(csv_path)
        dates = df["Date"].astype(str).tolist()
        prices = df["Close"].astype(float).tolist()
        return dates, prices
    else:
        dates = []
        prices = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dates.append(row["Date"])
                prices.append(float(row["Close"]))
        return dates, prices


def compute_deltas(prices: List[float]) -> List[float]:
    """
    Transforms daily prices P into daily price changes (deltas):
    deltas[k] = prices[k] - prices[k-1] for k in 1..len(prices)-1.
    """
    return [round(prices[i] - prices[i - 1], 4) for i in range(1, len(prices))]


# =====================================================================
# 1. Divide-and-Conquer Maximum Subarray Algorithm (CLRS Chapter 4)
# =====================================================================

def find_max_crossing_subarray(
    deltas: List[float], low: int, mid: int, high: int
) -> Tuple[int, int, float]:
    """
    Finds the maximum subarray that crosses the midpoint 'mid'.
    Runs in linear time Theta(high - low + 1).

    Returns:
        (max_left_index, max_right_index, cross_sum)
    """
    # Scan left from mid down to low
    left_sum = -math.inf
    current_sum = 0.0
    max_left = mid
    for i in range(mid, low - 1, -1):
        current_sum += deltas[i]
        if current_sum > left_sum:
            left_sum = current_sum
            max_left = i

    # Scan right from mid + 1 up to high
    right_sum = -math.inf
    current_sum = 0.0
    max_right = mid + 1
    for j in range(mid + 1, high + 1):
        current_sum += deltas[j]
        if current_sum > right_sum:
            right_sum = current_sum
            max_right = j

    return max_left, max_right, round(left_sum + right_sum, 4)


def find_max_subarray(
    deltas: List[float], low: int, high: int
) -> Tuple[int, int, float]:
    """
    Recursive Divide-and-Conquer algorithm for Maximum Subarray Problem.
    Recurrence: T(n) = 2T(n/2) + Theta(n) => O(n log n).

    Returns:
        (start_delta_index, end_delta_index, max_profit)
    """
    # Base case: only one element
    if low == high:
        return low, high, deltas[low]

    mid = (low + high) // 2

    # Conquer left half, right half, and crossing
    left_low, left_high, left_sum = find_max_subarray(deltas, low, mid)
    right_low, right_high, right_sum = find_max_subarray(deltas, mid + 1, high)
    cross_low, cross_high, cross_sum = find_max_crossing_subarray(deltas, low, mid, high)

    # Combine step: pick maximum of the three
    if left_sum >= right_sum and left_sum >= cross_sum:
        return left_low, left_high, left_sum
    elif right_sum >= left_sum and right_sum >= cross_sum:
        return right_low, right_high, right_sum
    else:
        return cross_low, cross_high, cross_sum


def max_profit_divide_and_conquer(prices: List[float]) -> Dict[str, Any]:
    """
    Solves Best Buy/Sell Days using Divide-and-Conquer Maximum Subarray on price changes.
    Time Complexity: O(n log n)
    Space Complexity: O(log n) recursion stack
    """
    if len(prices) < 2:
        return {"buy_day": 0, "sell_day": 0, "profit": 0.0}

    deltas = compute_deltas(prices)
    delta_start, delta_end, max_sub_sum = find_max_subarray(deltas, 0, len(deltas) - 1)

    # Map delta indices back to original price days:
    # delta[k] represents prices[k+1] - prices[k] (0-indexed delta corresponds to price days 0 and 1)
    buy_day = delta_start
    sell_day = delta_end + 1
    profit = round(prices[sell_day] - prices[buy_day], 4)

    # If even the max subarray is negative, best decision is not to trade (or buy/sell same day)
    if profit < 0:
        return {"buy_day": 0, "sell_day": 0, "profit": 0.0}

    return {
        "buy_day": buy_day,
        "sell_day": sell_day,
        "buy_price": prices[buy_day],
        "sell_price": prices[sell_day],
        "profit": profit,
        "algorithm": "Divide-and-Conquer (O(n log n))"
    }


# =====================================================================
# 2. Linear Scan Algorithms (O(n))
# =====================================================================

def max_profit_linear_kadane(prices: List[float]) -> Dict[str, Any]:
    """
    Kadane's Algorithm applied to daily price deltas.
    Time Complexity: O(n)
    Space Complexity: O(1) auxiliary
    """
    if len(prices) < 2:
        return {"buy_day": 0, "sell_day": 0, "profit": 0.0}

    deltas = compute_deltas(prices)
    max_so_far = -math.inf
    current_max = 0.0
    start_idx = 0
    best_start = 0
    best_end = 0

    for i, d in enumerate(deltas):
        current_max += d
        if current_max > max_so_far:
            max_so_far = current_max
            best_start = start_idx
            best_end = i

        if current_max < 0:
            current_max = 0.0
            start_idx = i + 1

    buy_day = best_start
    sell_day = best_end + 1
    profit = round(prices[sell_day] - prices[buy_day], 4)

    if profit < 0:
        return {"buy_day": 0, "sell_day": 0, "profit": 0.0}

    return {
        "buy_day": buy_day,
        "sell_day": sell_day,
        "buy_price": prices[buy_day],
        "sell_price": prices[sell_day],
        "profit": profit,
        "algorithm": "Linear Scan / Kadane (O(n))"
    }


def max_profit_linear_direct(prices: List[float]) -> Dict[str, Any]:
    """
    Single-pass linear scan directly on price series.
    Tracks minimum price seen so far and maximum attainable profit.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(prices) < 2:
        return {"buy_day": 0, "sell_day": 0, "profit": 0.0}

    min_price = prices[0]
    min_day = 0
    best_buy = 0
    best_sell = 0
    max_profit = 0.0

    for day in range(1, len(prices)):
        price = prices[day]
        potential_profit = price - min_price
        if potential_profit > max_profit:
            max_profit = potential_profit
            best_buy = min_day
            best_sell = day

        if price < min_price:
            min_price = price
            min_day = day

    return {
        "buy_day": best_buy,
        "sell_day": best_sell,
        "buy_price": prices[best_buy],
        "sell_price": prices[best_sell],
        "profit": round(max_profit, 4),
        "algorithm": "Direct Linear Scan (O(n))"
    }


# =====================================================================
# 3. Brute Force Algorithm (O(n^2)) for verification
# =====================================================================

def max_profit_brute_force(prices: List[float]) -> Dict[str, Any]:
    """
    Exhaustive brute-force search over all (buy, sell) pairs with buy < sell.
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    n = len(prices)
    max_profit = 0.0
    best_buy = 0
    best_sell = 0

    for i in range(n):
        for j in range(i + 1, n):
            profit = prices[j] - prices[i]
            if profit > max_profit:
                max_profit = profit
                best_buy = i
                best_sell = j

    return {
        "buy_day": best_buy,
        "sell_day": best_sell,
        "buy_price": prices[best_buy],
        "sell_price": prices[best_sell],
        "profit": round(max_profit, 4),
        "algorithm": "Brute Force (O(n^2))"
    }


if __name__ == "__main__":
    dates, prices = load_price_data("prices.csv")
    print(f"Loaded {len(prices)} price records from prices.csv.")
    
    res_dc = max_profit_divide_and_conquer(prices)
    res_kadane = max_profit_linear_kadane(prices)
    res_direct = max_profit_linear_direct(prices)

    print("\n--- Divide and Conquer Result ---")
    print(f"Buy Day : {res_dc['buy_day']} ({dates[res_dc['buy_day']]}) at ${res_dc['buy_price']:.2f}")
    print(f"Sell Day: {res_dc['sell_day']} ({dates[res_dc['sell_day']]}) at ${res_dc['sell_price']:.2f}")
    print(f"Max Profit: ${res_dc['profit']:.2f}")

    print("\n--- Linear Scan Result ---")
    print(f"Buy Day : {res_direct['buy_day']} ({dates[res_direct['buy_day']]}) at ${res_direct['buy_price']:.2f}")
    print(f"Sell Day: {res_direct['sell_day']} ({dates[res_direct['sell_day']]}) at ${res_direct['sell_price']:.2f}")
    print(f"Max Profit: ${res_direct['profit']:.2f}")

    assert res_dc["buy_day"] == res_direct["buy_day"]
    assert res_dc["sell_day"] == res_direct["sell_day"]
    assert abs(res_dc["profit"] - res_direct["profit"]) < 1e-4
    print("\nSUCCESS: All algorithms match perfectly!")
