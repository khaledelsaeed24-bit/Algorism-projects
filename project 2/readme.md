# Project 2: Max Profit Finder
### Best Buy and Sell Days via Divide-and-Conquer Maximum Subarray vs. Linear Scan

**Course:** Algorithms: Analysis and Design  
**Tools:** Python 3, pandas / csv, historical price CSV (`prices.csv`)

---

## Executive Summary

The **Max Profit Finder** problem (also known as the *Stock Buy/Sell Timing Problem*) seeks the optimal pair of buy and sell days to maximize financial return from a single purchase and subsequent sale over a historical price time series:
$$\max_{0 \le i < j < n} (P[j] - P[i])$$

This project implements, mathematically proves, and empirically benchmarks two foundational algorithmic paradigms to solve this problem:
1. **Divide-and-Conquer Maximum Subarray ($\mathcal{O}(n \log n)$)** via transformation to daily price changes (CLRS Chapter 4).
2. **Linear Scan / Dynamic Programming ($\mathcal{O}(n)$)** using Kadane's algorithm on price differences and direct single-pass minimum price tracking.
3. **Brute Force ($\mathcal{O}(n^2)$)** as an exhaustive baseline for verification.

All approaches are benchmarked across datasets scaling up to **$n = 50,000$**, and demonstrated on a realistic 1,000-day historical stock price series (`prices.csv`).

---

## 1. Problem Formulation & Subarray Transformation

Let $P = [P[0], P[1], \dots, P[n-1]]$ represent the closing stock price on $n$ consecutive trading days.

### The Telescoping Transformation
Instead of examining price levels directly, we compute the sequence of daily price differentials:
$$\Delta[k] = P[k] - P[k-1] \quad \text{for } k = 1, 2, \dots, n-1$$

For any buy day $i$ and sell day $j$ ($i < j$), the total profit is a **telescoping sum**:
$$\sum_{k=i+1}^j \Delta[k] = (P[i+1] - P[i]) + (P[i+2] - P[i+1]) + \dots + (P[j] - P[j-1]) = P[j] - P[i]$$

**Theorem:** Finding a buy day $i$ and sell day $j$ that maximizes profit $P[j] - P[i]$ is isomorphic to finding a contiguous subarray of $\Delta$ with the maximum sum.
- If the maximum subarray of $\Delta$ starts at index $s$ and ends at index $e$, then:
  - **Optimal Buy Day:** $i = s - 1$
  - **Optimal Sell Day:** $j = e$
  - **Maximum Profit:** $\sum_{k=s}^e \Delta[k] = P[j] - P[i]$

---

## 2. Algorithmic Implementations

### Approach 1: Divide-and-Conquer ($\mathcal{O}(n \log n)$)

Following Cormen et al. (CLRS):
1. **Divide:** Find the midpoint $mid = \lfloor(low + high)/2\rfloor$.
2. **Conquer:** Recursively solve for the maximum subarray in:
   - Left half: $\Delta[low \dots mid]$
   - Right half: $\Delta[mid+1 \dots high]$
3. **Combine:** Find the maximum subarray that **crosses** $mid$ using `find_max_crossing_subarray` in $\Theta(high - low + 1) = \Theta(k)$ time.
4. **Return:** The maximum of the left, right, and cross subarrays.

```python
def find_max_crossing_subarray(deltas, low, mid, high):
    left_sum, current_sum, max_left = -math.inf, 0.0, mid
    for i in range(mid, low - 1, -1):
        current_sum += deltas[i]
        if current_sum > left_sum:
            left_sum, max_left = current_sum, i

    right_sum, current_sum, max_right = -math.inf, 0.0, mid + 1
    for j in range(mid + 1, high + 1):
        current_sum += deltas[j]
        if current_sum > right_sum:
            right_sum, max_right = current_sum, j

    return max_left, max_right, left_sum + right_sum

def find_max_subarray(deltas, low, high):
    if low == high:
        return low, high, deltas[low]
    mid = (low + high) // 2
    l_low, l_high, l_sum = find_max_subarray(deltas, low, mid)
    r_low, r_high, r_sum = find_max_subarray(deltas, mid + 1, high)
    c_low, c_high, c_sum = find_max_crossing_subarray(deltas, low, mid, high)

    if l_sum >= r_sum and l_sum >= c_sum:
        return l_low, l_high, l_sum
    elif r_sum >= l_sum and r_sum >= c_sum:
        return r_low, r_high, r_sum
    else:
        return c_low, c_high, c_sum
```

#### Recurrence Relation & Master Theorem Proof
The recurrence is:
$$T(n) = 2T(n/2) + \Theta(n)$$
Applying the Master Theorem ($T(n) = aT(n/b) + f(n)$):
- $a = 2$, $b = 2$, $f(n) = \Theta(n)$
- $n^{\log_b a} = n^{\log_2 2} = n^1 = n$
- Since $f(n) = \Theta(n^{\log_b a}) = \Theta(n)$, **Case 2** applies:
$$T(n) = \Theta(n^{\log_b a} \log n) = \mathbf{\Theta(n \log n)}$$
- Auxiliary Stack Depth: $\mathcal{O}(\log n)$.

---

### Approach 2: Linear Scan ($\mathcal{O}(n)$)

#### A. Kadane's Algorithm on Deltas
Maintains a running sum `current_sum`. If `current_sum < 0`, it resets to $0$ and advances the candidate start index. Tracks the global maximum sum `max_so_far` in a single pass.

#### B. Direct Price Tracking (Single Pass)
Tracks the minimum price observed so far (`min_price_so_far`). At each day $j$, the maximum profit attainable by selling on day $j$ is $P[j] - \text{min\_price\_so\_far}$.

```python
def max_profit_linear_direct(prices: list[float]) -> dict:
    min_price = prices[0]
    min_day = best_buy = best_sell = 0
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

    return {"buy_day": best_buy, "sell_day": best_sell, "profit": max_profit}
```
- **Time Complexity:** $\mathbf{\Theta(n)}$ (exactly $n-1$ iterations, constant operations per iteration).
- **Space Complexity:** $\mathbf{\mathcal{O}(1)}$ (only scalar variables).

---

## 3. Case Study: 1,000 Trading Days (`prices.csv`)

We applied all algorithms to `prices.csv` spanning 1,000 trading days (over 4 calendar years).

### Optimal Trade Result
- **Optimal Buy Day:** **Day 580** (Date: `2024-03-25`) at price **\$86.85**
- **Optimal Sell Day:** **Day 717** (Date: `2024-10-02`) at price **\$115.23**
- **Maximum Profit:** **+\$28.38 per share (+32.68% ROI)**
- **Verification:** Both Divide-and-Conquer, Kadane's, and Direct Linear Scan identified the exact identical days and profit.

### Visualized Trade Execution
The chart below illustrates the full price trajectory, marking the optimal buy/sell positions and holding window:

![Stock Price History and Optimal Buy/Sell Trade](stock_price_profit.svg)

---

## 4. Empirical Benchmark & Runtime Comparison

All three algorithms were timed across input sizes ranging from $n = 100$ to $n = 50,000$:

| Input Size ($n$) | Linear Scan $\mathcal{O}(n)$ | Divide & Conquer $\mathcal{O}(n \log n)$ | Brute Force $\mathcal{O}(n^2)$ | Speedup (Linear vs D&C) |
| :---: | :---: | :---: | :---: | :---: |
| **100** | 0.027 ms | 0.728 ms | 0.618 ms | $27.0\times$ |
| **250** | 0.016 ms | 0.472 ms | 1.074 ms | $29.5\times$ |
| **500** | 0.021 ms | 0.801 ms | 3.507 ms | $38.1\times$ |
| **1,000** | 0.036 ms | 1.294 ms | 20.172 ms | $35.9\times$ |
| **2,500** | 0.100 ms | 3.824 ms | 144.020 ms | $38.2\times$ |
| **5,000** | 0.468 ms | 13.273 ms | 670.535 ms | $28.4\times$ |
| **10,000** | 1.781 ms | 29.355 ms | *Omitted ($>2.5$ s)* | $16.5\times$ |
| **25,000** | 1.096 ms | 77.189 ms | *Omitted ($>16$ s)* | $70.4\times$ |
| **50,000** | **2.514 ms** | **143.175 ms** | *Omitted ($>67$ s)* | **$57.0\times$** |

### Runtime Scaling Chart

![Runtime Comparison](runtime_comparison.svg)

---

## 5. In-Depth Comparative Analysis

| Evaluation Criterion | Divide-and-Conquer | Linear Scan (Kadane / Direct) | Brute Force |
| :--- | :--- | :--- | :--- |
| **Asymptotic Time** | $\mathcal{O}(n \log n)$ | $\mathbf{\mathcal{O}(n)}$ | $\mathcal{O}(n^2)$ |
| **Auxiliary Memory** | $\mathcal{O}(\log n)$ (call stack) | $\mathbf{\mathcal{O}(1)}$ (strictly in-place) | $\mathcal{O}(1)$ |
| **Streaming / Online Ability** | No (requires full array upfront) | **Yes** (processes each tick in real time) | No |
| **Cache Locality** | Moderate (jumps across halves) | **Optimal** (single linear memory scan) | Low |
| **Constant Factors** | High (recursion frames, tuple returns) | **Very Low** (simple primitive comparisons) | Moderate |
| **Pedagogical Value** | Core example of Master Theorem Case 2 | Classic DP / greedy transition | Naive baseline |

### Why Linear Scan Outperforms Divide-and-Conquer:
1. **Asymptotic Dominance:** $\mathcal{O}(n)$ grows strictly slower than $\mathcal{O}(n \log n)$. At $n = 50,000$, $\log_2(50000) \approx 15.6$ operations per element.
2. **Elimination of Subproblem Redundancy:** Divide-and-conquer repeatedly scans overlapping boundary regions in `find_max_crossing_subarray`, whereas the linear scan processes each element exactly once.
3. **Hardware / Cache Efficiency:** Linear scan accesses contiguous memory addresses sequentially, enabling CPU hardware prefetching and L1/L2 cache hits with zero function call overhead.
4. **Online Financial Trading:** In real-world algorithmic trading, stock prices arrive sequentially as a live stream. The linear scan updates optimal decisions on every incoming price tick in $\mathcal{O}(1)$ time, making it ideal for streaming systems.

---

## 6. Project Directory & Reproduction Guide

### Directory Structure
```
project 2/
│── prices.csv               # 1,000 days of stock price history
│── generate_data.py         # Geometric Brownian Motion price generator
│── max_profit.py            # Core D&C, Linear Scan, and Brute Force implementations
│── benchmark.py             # Timing benchmarks & SVG chart generator
│── stock_price_profit.svg   # High-resolution stock price chart with trade markers
│── runtime_comparison.svg   # Runtime scaling chart against theoretical curves
│── benchmark_results.json   # Raw benchmark measurements
└── readme.md                # This complete report
```

### How to Run

```powershell
# 1. Run algorithm verification and view optimal trade on prices.csv
python max_profit.py

# 2. Run scaling benchmarks and regenerate vector SVG charts
python benchmark.py
```
