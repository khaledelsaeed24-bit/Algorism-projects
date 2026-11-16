"""
Benchmark and Visualization Suite for Project 2: Max Profit Finder.
Generates:
  1. stock_price_profit.svg (Price trajectory with buy/sell markers)
  2. runtime_comparison.svg (Empirical runtime vs n for D&C, Linear, and Brute Force)
  3. benchmark_results.json
"""

import json
import math
import os
import random
import time
from typing import List, Dict, Any

from max_profit import (
    load_price_data,
    max_profit_divide_and_conquer,
    max_profit_linear_direct,
    max_profit_linear_kadane,
    max_profit_brute_force,
)


def run_benchmarks() -> Dict[str, Any]:
    # Sizes for benchmarking
    sizes = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]
    random.seed(42)

    results = []
    print(f"{'Size (n)':>8} | {'Linear Scan (s)':>16} | {'Divide & Conquer (s)':>21} | {'Brute Force (s)':>16}")
    print("-" * 70)

    for n in sizes:
        # Generate random price walk of size n
        prices = [100.0]
        for _ in range(n - 1):
            prices.append(round(prices[-1] * (1.0 + random.gauss(0.0005, 0.015)), 2))

        # 1. Linear Scan
        trials_lin = 20 if n <= 5000 else 5
        t0 = time.perf_counter()
        for _ in range(trials_lin):
            res_lin = max_profit_linear_direct(prices)
        t_lin = (time.perf_counter() - t0) / trials_lin

        # 2. Divide & Conquer
        trials_dc = 10 if n <= 5000 else 3
        t0 = time.perf_counter()
        for _ in range(trials_dc):
            res_dc = max_profit_divide_and_conquer(prices)
        t_dc = (time.perf_counter() - t0) / trials_dc

        # Verification of matching result
        assert res_lin["profit"] == res_dc["profit"], f"Mismatch at n={n}"

        # 3. Brute Force (only for n <= 5000 due to O(n^2))
        t_bf = None
        if n <= 5000:
            trials_bf = 1 if n >= 2500 else 3
            t0 = time.perf_counter()
            for _ in range(trials_bf):
                res_bf = max_profit_brute_force(prices)
            t_bf = (time.perf_counter() - t0) / trials_bf
            assert res_bf["profit"] == res_lin["profit"], f"BF Mismatch at n={n}"

        bf_str = f"{t_bf:>16.6f}" if t_bf is not None else f"{'N/A (too slow)':>16}"
        print(f"{n:>8} | {t_lin:>16.6f} | {t_dc:>21.6f} | {bf_str}")

        results.append({
            "n": n,
            "linear_time": t_lin,
            "dc_time": t_dc,
            "bf_time": t_bf,
        })

    return {"sizes": sizes, "data": results}


def generate_stock_chart_svg(
    dates: List[str],
    prices: List[float],
    buy_day: int,
    sell_day: int,
    filename: str = "stock_price_profit.svg",
    width: int = 900,
    height: int = 500
):
    """
    Renders the historical stock price curve with Buy / Sell annotations.
    """
    margin_l = 80
    margin_r = 60
    margin_t = 60
    margin_b = 60

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    min_p = min(prices) * 0.95
    max_p = max(prices) * 1.05
    n = len(prices)

    def map_x(i):
        return margin_l + (i / (n - 1)) * plot_w

    def map_y(p):
        return margin_t + (1.0 - (p - min_p) / (max_p - min_p)) * plot_h

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">')

    # Gradient for price area
    svg.append("""
    <defs>
      <linearGradient id="priceAreaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
      </linearGradient>
      <linearGradient id="profitWindowGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#22c55e" stop-opacity="0.25"/>
        <stop offset="100%" stop-color="#22c55e" stop-opacity="0.05"/>
      </linearGradient>
    </defs>
    """)

    # Background
    svg.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="10"/>')
    svg.append(f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#1e293b" stroke="#334155" stroke-width="1.5" rx="4"/>')

    # Title
    svg.append(f'<text x="{margin_l + plot_w/2}" y="35" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="600">Stock Price History & Optimal Buy/Sell Trade Execution</text>')

    # Grid & Y-ticks
    y_ticks = 6
    for i in range(y_ticks + 1):
        p_val = min_p + (max_p - min_p) * (i / y_ticks)
        sy = map_y(p_val)
        svg.append(f'<line x1="{margin_l}" y1="{sy}" x2="{margin_l + plot_w}" y2="{sy}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l - 12}" y="{sy + 4}" text-anchor="end" fill="#94a3b8" font-size="12">${p_val:.1f}</text>')

    # Grid & X-ticks (Dates)
    x_ticks = 5
    for i in range(x_ticks + 1):
        idx = int(i * (n - 1) / x_ticks)
        sx = map_x(idx)
        svg.append(f'<line x1="{sx}" y1="{margin_t}" x2="{sx}" y2="{margin_t + plot_h}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>')
        d_str = dates[idx]
        svg.append(f'<text x="{sx}" y="{margin_t + plot_h + 20}" text-anchor="middle" fill="#94a3b8" font-size="12">{d_str}</text>')

    # Profit Window Shading between buy and sell
    bx = map_x(buy_day)
    sx = map_x(sell_day)
    svg.append(f'<rect x="{bx}" y="{margin_t}" width="{sx - bx}" height="{plot_h}" fill="url(#profitWindowGrad)"/>')

    # Price Area Path
    area_d = [f"M{map_x(0):.1f},{margin_t + plot_h:.1f}"]
    for idx, p in enumerate(prices):
        area_d.append(f"L{map_x(idx):.1f},{map_y(p):.1f}")
    area_d.append(f"L{map_x(n-1):.1f},{margin_t + plot_h:.1f} Z")
    svg.append(f'<path d="{" ".join(area_d)}" fill="url(#priceAreaGrad)"/>')

    # Price Line Path
    line_d = []
    for idx, p in enumerate(prices):
        cmd = "M" if idx == 0 else "L"
        line_d.append(f"{cmd}{map_x(idx):.1f},{map_y(p):.1f}")
    svg.append(f'<path d="{" ".join(line_d)}" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linejoin="round"/>')

    # Buy Day Marker
    by = map_y(prices[buy_day])
    svg.append(f'<line x1="{bx}" y1="{margin_t}" x2="{bx}" y2="{margin_t + plot_h}" stroke="#22c55e" stroke-dasharray="4,3" stroke-width="1.8"/>')
    svg.append(f'<circle cx="{bx}" cy="{by}" r="6" fill="#22c55e" stroke="#ffffff" stroke-width="2"/>')
    
    # Sell Day Marker
    sy = map_y(prices[sell_day])
    svg.append(f'<line x1="{sx}" y1="{margin_t}" x2="{sx}" y2="{margin_t + plot_h}" stroke="#ef4444" stroke-dasharray="4,3" stroke-width="1.8"/>')
    svg.append(f'<circle cx="{sx}" cy="{sy}" r="6" fill="#ef4444" stroke="#ffffff" stroke-width="2"/>')

    # Annotations / Badges
    buy_text = f"BUY: ${prices[buy_day]:.2f} ({dates[buy_day]})"
    sell_text = f"SELL: ${prices[sell_day]:.2f} ({dates[sell_day]})"
    profit = prices[sell_day] - prices[buy_day]
    return_pct = (profit / prices[buy_day]) * 100
    profit_text = f"Max Profit: +${profit:.2f} (+{return_pct:.1f}%)"

    # Buy Callout Card
    svg.append(f'<rect x="{bx - 60}" y="{by + 15}" width="140" height="26" fill="#14532d" stroke="#22c55e" rx="4"/>')
    svg.append(f'<text x="{bx + 10}" y="{by + 32}" text-anchor="middle" fill="#f0fdf4" font-size="11" font-weight="600">{buy_text}</text>')

    # Sell Callout Card
    svg.append(f'<rect x="{sx - 70}" y="{sy - 40}" width="140" height="26" fill="#7f1d1d" stroke="#ef4444" rx="4"/>')
    svg.append(f'<text x="{sx}" y="{sy - 23}" text-anchor="middle" fill="#fef2f2" font-size="11" font-weight="600">{sell_text}</text>')

    # Profit Summary Banner in Chart
    banner_w = 260
    banner_h = 36
    banner_x = margin_l + plot_w - banner_w - 15
    banner_y = margin_t + 15
    svg.append(f'<rect x="{banner_x}" y="{banner_y}" width="{banner_w}" height="{banner_h}" fill="#0f172a" stroke="#22c55e" stroke-width="1.5" rx="6" opacity="0.95"/>')
    svg.append(f'<text x="{banner_x + banner_w/2}" y="{banner_y + 23}" text-anchor="middle" fill="#4ade80" font-size="14" font-weight="700">{profit_text}</text>')

    svg.append('</svg>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"Generated chart: {filename}")


def generate_runtime_plot_svg(
    benchmark_data: List[Dict[str, Any]],
    filename: str = "runtime_comparison.svg",
    width: int = 850,
    height: int = 500
):
    """
    Renders runtime vs input size comparing O(n) Linear Scan, O(n log n) Divide & Conquer, and O(n^2) Brute Force.
    """
    margin_l = 85
    margin_r = 190
    margin_t = 60
    margin_b = 60

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    max_n = 50000
    # Let's use log scale or clean high range up to max time
    times = [d["dc_time"] for d in benchmark_data] + [d["linear_time"] for d in benchmark_data]
    max_t = max(times) * 1.15

    def map_x(n):
        return margin_l + (n / max_n) * plot_w

    def map_y(t):
        return margin_t + (1.0 - (t / max_t)) * plot_h

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">')

    # Background
    svg.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="10"/>')
    svg.append(f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#1e293b" stroke="#334155" stroke-width="1.5" rx="4"/>')

    # Title
    svg.append(f'<text x="{margin_l + plot_w/2}" y="35" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="600">Runtime Comparison: Divide-and-Conquer vs. Linear Scan</text>')

    # Grid & X-ticks
    x_steps = 5
    for i in range(x_steps + 1):
        n_val = int((max_n / x_steps) * i)
        sx = map_x(n_val)
        svg.append(f'<line x1="{sx}" y1="{margin_t}" x2="{sx}" y2="{margin_t + plot_h}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>')
        svg.append(f'<text x="{sx}" y="{margin_t + plot_h + 20}" text-anchor="middle" fill="#94a3b8" font-size="12">{n_val:,}</text>')

    # Grid & Y-ticks
    y_steps = 5
    for i in range(y_steps + 1):
        t_val = (max_t / y_steps) * i
        sy = map_y(t_val)
        svg.append(f'<line x1="{margin_l}" y1="{sy}" x2="{margin_l + plot_w}" y2="{sy}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l - 12}" y="{sy + 4}" text-anchor="end" fill="#94a3b8" font-size="12">{t_val*1000:.1f} ms</text>')

    # Axis Labels
    svg.append(f'<text x="{margin_l + plot_w / 2}" y="{height - 15}" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="500">Input Size (n)</text>')
    svg.append(f'<text x="25" y="{margin_t + plot_h / 2}" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="500" transform="rotate(-90 25 {margin_t + plot_h / 2})">Execution Time (milliseconds)</text>')

    # Series 1: Divide & Conquer
    dc_pts = [(d["n"], d["dc_time"]) for d in benchmark_data]
    dc_d = []
    for idx, (px, py) in enumerate(dc_pts):
        cmd = "M" if idx == 0 else "L"
        dc_d.append(f"{cmd}{map_x(px):.1f},{map_y(py):.1f}")
    svg.append(f'<path d="{" ".join(dc_d)}" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round"/>')
    for px, py in dc_pts:
        svg.append(f'<circle cx="{map_x(px):.1f}" cy="{map_y(py):.1f}" r="3.5" fill="#f59e0b" stroke="#0f172a" stroke-width="1.5"/>')

    # Series 2: Linear Scan
    lin_pts = [(d["n"], d["linear_time"]) for d in benchmark_data]
    lin_d = []
    for idx, (px, py) in enumerate(lin_pts):
        cmd = "M" if idx == 0 else "L"
        lin_d.append(f"{cmd}{map_x(px):.1f},{map_y(py):.1f}")
    svg.append(f'<path d="{" ".join(lin_d)}" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round"/>')
    for px, py in lin_pts:
        svg.append(f'<circle cx="{map_x(px):.1f}" cy="{map_y(py):.1f}" r="3.5" fill="#10b981" stroke="#0f172a" stroke-width="1.5"/>')

    # Theoretical Scaled Lines
    c_lin = lin_pts[-1][1] / 50000
    c_dc = dc_pts[-1][1] / (50000 * math.log2(50000))

    dense_n = [1000 + i * 2000 for i in range(25)]
    th_lin_d = [f"{'M' if i==0 else 'L'}{map_x(n):.1f},{map_y(c_lin * n):.1f}" for i, n in enumerate(dense_n)]
    th_dc_d = [f"{'M' if i==0 else 'L'}{map_x(n):.1f},{map_y(c_dc * n * math.log2(n)):.1f}" for i, n in enumerate(dense_n)]

    svg.append(f'<path d="{" ".join(th_lin_d)}" fill="none" stroke="#6ee7b7" stroke-width="1.5" stroke-dasharray="4,4"/>')
    svg.append(f'<path d="{" ".join(th_dc_d)}" fill="none" stroke="#fde68a" stroke-width="1.5" stroke-dasharray="4,4"/>')

    # Legend
    leg_x = margin_l + plot_w + 15
    leg_y = margin_t + 10
    svg.append(f'<rect x="{leg_x}" y="{leg_y}" width="{margin_r - 25}" height="140" fill="#1e293b" stroke="#334155" rx="6"/>')
    svg.append(f'<text x="{leg_x + 10}" y="{leg_y + 20}" fill="#94a3b8" font-size="11" font-weight="600" text-transform="uppercase">Legend</text>')

    items = [
        ("D&C O(n log n)", "#f59e0b", ""),
        ("D&C Theory Curve", "#fde68a", "4,4"),
        ("Linear Scan O(n)", "#10b981", ""),
        ("Linear Theory Curve", "#6ee7b7", "4,4"),
    ]
    for idx, (lbl, col, dash) in enumerate(items):
        iy = leg_y + 44 + idx * 24
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="{leg_x + 10}" y1="{iy - 4}" x2="{leg_x + 30}" y2="{iy - 4}" stroke="{col}" stroke-width="2.5"{dash_attr}/>')
        if not dash:
            svg.append(f'<circle cx="{leg_x + 20}" cy="{iy - 4}" r="3" fill="{col}"/>')
        svg.append(f'<text x="{leg_x + 38}" y="{iy}" fill="#f1f5f9" font-size="11">{lbl}</text>')

    svg.append('</svg>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"Generated plot: {filename}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "prices.csv")
    dates, prices = load_price_data(csv_file)
    sol = max_profit_divide_and_conquer(prices)
    
    chart_svg = os.path.join(script_dir, "stock_price_profit.svg")
    generate_stock_chart_svg(
        dates=dates,
        prices=prices,
        buy_day=sol["buy_day"],
        sell_day=sol["sell_day"],
        filename=chart_svg
    )

    # 2. Run scaling benchmarks
    benchmarks = run_benchmarks()
    json_path = os.path.join(script_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)

    # 3. Generate runtime plot
    runtime_svg = os.path.join(script_dir, "runtime_comparison.svg")
    generate_runtime_plot_svg(benchmarks["data"], filename=runtime_svg)

    print("\nBenchmark and visualization suite completed successfully!")


if __name__ == "__main__":
    main()
