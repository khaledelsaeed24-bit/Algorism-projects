"""
Benchmark and Visualization Suite for Project 3: Top-K Frequent Words.
Generates:
  1. top_k_barchart.svg (Top-15 words with frequency counts and percentages)
  2. runtime_vs_k.svg (Runtime as k varies: Min-Heap O(M log k) vs Full Sort O(M log M))
  3. runtime_vs_n.svg (Runtime as text length N scales up to 100,000 words)
  4. zipf_distribution.svg (Zipf's Law empirical frequency vs rank)
  5. benchmark_results.json
"""

import json
import math
import os
import random
import re
import time
from collections import Counter
from typing import List, Dict, Any, Tuple

from top_k_words import (
    tokenize_text,
    top_k_min_heap,
    top_k_full_sort,
    top_k_bucket_sort,
    top_k_counter_most_common,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS_FILE = os.path.join(SCRIPT_DIR, "corpus.txt")


def load_corpus(filepath: str = DEFAULT_CORPUS_FILE) -> List[str]:
    if not os.path.isabs(filepath) and not os.path.exists(filepath):
        alt_path = os.path.join(SCRIPT_DIR, filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return tokenize_text(text)


def benchmark_k_scaling(tokens: List[str]) -> List[Dict[str, Any]]:
    k_values = [5, 10, 25, 50, 100, 250, 500, 1000, 1500]
    trials = 10
    results = []

    print(f"\n--- Benchmark 1: Scaling k on fixed corpus (N={len(tokens)}, M={len(set(tokens))}) ---")
    print(f"{'k':>6} | {'Min-Heap (ms)':>16} | {'Full Sort (ms)':>16} | {'Counter.most_common (ms)':>26}")
    print("-" * 75)

    for k in k_values:
        # Min-Heap
        t0 = time.perf_counter()
        for _ in range(trials):
            top_k_min_heap(tokens, k)
        t_heap = ((time.perf_counter() - t0) / trials) * 1000

        # Full Sort
        t0 = time.perf_counter()
        for _ in range(trials):
            top_k_full_sort(tokens, k)
        t_sort = ((time.perf_counter() - t0) / trials) * 1000

        # Most common
        t0 = time.perf_counter()
        for _ in range(trials):
            top_k_counter_most_common(tokens, k)
        t_mc = ((time.perf_counter() - t0) / trials) * 1000

        print(f"{k:>6} | {t_heap:>16.3f} | {t_sort:>16.3f} | {t_mc:>26.3f}")
        results.append({
            "k": k,
            "min_heap_ms": t_heap,
            "full_sort_ms": t_sort,
            "counter_ms": t_mc
        })

    return results


def benchmark_n_scaling(tokens: List[str]) -> List[Dict[str, Any]]:
    n_values = [1000, 2500, 5000, 10000, 25000, 50000]
    # Multiply tokens to get 100,000 tokens
    extended_tokens = tokens + tokens
    if len(extended_tokens) >= 100000:
        n_values.append(100000)

    k = 20
    trials = 5
    results = []

    print(f"\n--- Benchmark 2: Scaling text size N (fixed k={k}) ---")
    print(f"{'N':>8} | {'Min-Heap (ms)':>16} | {'Full Sort (ms)':>16} | {'Bucket Sort (ms)':>18}")
    print("-" * 65)

    for n in n_values:
        sub_tokens = extended_tokens[:n]

        t0 = time.perf_counter()
        for _ in range(trials):
            top_k_min_heap(sub_tokens, k)
        t_heap = ((time.perf_counter() - t0) / trials) * 1000

        t0 = time.perf_counter()
        for _ in range(trials):
            top_k_full_sort(sub_tokens, k)
        t_sort = ((time.perf_counter() - t0) / trials) * 1000

        t0 = time.perf_counter()
        for _ in range(trials):
            top_k_bucket_sort(sub_tokens, k)
        t_bucket = ((time.perf_counter() - t0) / trials) * 1000

        print(f"{n:>8} | {t_heap:>16.3f} | {t_sort:>16.3f} | {t_bucket:>18.3f}")
        results.append({
            "n": n,
            "min_heap_ms": t_heap,
            "full_sort_ms": t_sort,
            "bucket_ms": t_bucket
        })

    return results


# =====================================================================
# SVG Plot Generators
# =====================================================================

def generate_barchart_svg(
    top_words: List[Tuple[str, int]],
    total_tokens: int,
    filename: str = "top_k_barchart.svg",
    width: int = 850,
    height: int = 550
):
    """
    Renders horizontal bar chart of top-k words.
    """
    margin_l = 130
    margin_r = 90
    margin_t = 60
    margin_b = 40

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    k = len(top_words)
    max_freq = top_words[0][1] * 1.15

    bar_gap = 6
    bar_h = (plot_h - (k - 1) * bar_gap) / k

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">')

    # Gradients
    svg.append("""
    <defs>
      <linearGradient id="barGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#3b82f6"/>
        <stop offset="100%" stop-color="#38bdf8"/>
      </linearGradient>
    </defs>
    """)

    # Background
    svg.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="10"/>')
    svg.append(f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#1e293b" stroke="#334155" stroke-width="1.5" rx="4"/>')

    # Title
    svg.append(f'<text x="{margin_l + plot_w/2}" y="35" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="600">Top-{k} Most Frequent Words in Corpus (N = {total_tokens:,})</text>')

    # Vertical gridlines
    x_ticks = 5
    for i in range(x_ticks + 1):
        freq_val = int((max_freq / x_ticks) * i)
        sx = margin_l + (freq_val / max_freq) * plot_w
        svg.append(f'<line x1="{sx}" y1="{margin_t}" x2="{sx}" y2="{margin_t + plot_h}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>')
        svg.append(f'<text x="{sx}" y="{margin_t + plot_h + 18}" text-anchor="middle" fill="#94a3b8" font-size="11">{freq_val:,}</text>')

    # Bars
    for idx, (word, count) in enumerate(top_words):
        by = margin_t + idx * (bar_h + bar_gap)
        bw = (count / max_freq) * plot_w
        pct = (count / total_tokens) * 100

        # Bar
        svg.append(f'<rect x="{margin_l}" y="{by}" width="{bw}" height="{bar_h}" fill="url(#barGrad)" rx="3"/>')
        # Word label (Y-axis)
        svg.append(f'<text x="{margin_l - 12}" y="{by + bar_h/2 + 4}" text-anchor="end" fill="#f1f5f9" font-size="12" font-weight="500">{word}</text>')
        # Frequency & Pct label
        svg.append(f'<text x="{margin_l + bw + 8}" y="{by + bar_h/2 + 4}" fill="#38bdf8" font-size="11" font-weight="600">{count:,} <tspan fill="#94a3b8" font-weight="400">({pct:.2f}%)</tspan></text>')

    svg.append('</svg>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"Generated chart: {filename}")


def generate_line_plot(
    title: str,
    x_label: str,
    y_label: str,
    series: List[Dict[str, Any]],
    x_max: float,
    y_max: float,
    filename: str,
    width: int = 850,
    height: int = 500
):
    margin_l = 85
    margin_r = 190
    margin_t = 60
    margin_b = 60

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def map_x(x):
        return margin_l + (x / x_max) * plot_w

    def map_y(y):
        return margin_t + (1.0 - (y / y_max)) * plot_h

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">')

    svg.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="10"/>')
    svg.append(f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#1e293b" stroke="#334155" stroke-width="1.5" rx="4"/>')

    # Title
    svg.append(f'<text x="{margin_l + plot_w/2}" y="35" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="600">{title}</text>')

    # Grid & X Ticks
    x_steps = 5
    for i in range(x_steps + 1):
        x_val = (x_max / x_steps) * i
        sx = map_x(x_val)
        svg.append(f'<line x1="{sx}" y1="{margin_t}" x2="{sx}" y2="{margin_t + plot_h}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>')
        svg.append(f'<text x="{sx}" y="{margin_t + plot_h + 20}" text-anchor="middle" fill="#94a3b8" font-size="12">{int(x_val):,}</text>')

    # Grid & Y Ticks
    y_steps = 5
    for i in range(y_steps + 1):
        y_val = (y_max / y_steps) * i
        sy = map_y(y_val)
        svg.append(f'<line x1="{margin_l}" y1="{sy}" x2="{margin_l + plot_w}" y2="{sy}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l - 12}" y="{sy + 4}" text-anchor="end" fill="#94a3b8" font-size="12">{y_val:.2f} ms</text>')

    # Axis Labels
    svg.append(f'<text x="{margin_l + plot_w / 2}" y="{height - 15}" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="500">{x_label}</text>')
    svg.append(f'<text x="25" y="{margin_t + plot_h / 2}" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="500" transform="rotate(-90 25 {margin_t + plot_h / 2})">{y_label}</text>')

    # Lines
    legend_items = []
    for s in series:
        pts = s["points"]
        col = s["color"]
        name = s["name"]
        dash = s.get("dash", "")

        path_d = [f"{'M' if idx==0 else 'L'}{map_x(px):.1f},{map_y(py):.1f}" for idx, (px, py) in enumerate(pts)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<path d="{" ".join(path_d)}" fill="none" stroke="{col}" stroke-width="2.5"{dash_attr} stroke-linejoin="round"/>')

        if not s.get("hide_dots", False):
            for px, py in pts:
                svg.append(f'<circle cx="{map_x(px):.1f}" cy="{map_y(py):.1f}" r="3.5" fill="{col}" stroke="#0f172a" stroke-width="1.5"/>')

        legend_items.append((name, col, dash))

    # Legend
    leg_x = margin_l + plot_w + 15
    leg_y = margin_t + 10
    svg.append(f'<rect x="{leg_x}" y="{leg_y}" width="{margin_r - 25}" height="{len(legend_items)*26 + 30}" fill="#1e293b" stroke="#334155" rx="6"/>')
    svg.append(f'<text x="{leg_x + 10}" y="{leg_y + 20}" fill="#94a3b8" font-size="11" font-weight="600" text-transform="uppercase">Legend</text>')

    for idx, (name, col, dash) in enumerate(legend_items):
        iy = leg_y + 44 + idx * 24
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="{leg_x + 10}" y1="{iy - 4}" x2="{leg_x + 30}" y2="{iy - 4}" stroke="{col}" stroke-width="2.5"{dash_attr}/>')
        if not dash:
            svg.append(f'<circle cx="{leg_x + 20}" cy="{iy - 4}" r="3" fill="{col}"/>')
        svg.append(f'<text x="{leg_x + 38}" y="{iy}" fill="#f1f5f9" font-size="11">{name}</text>')

    svg.append('</svg>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"Generated plot: {filename}")


def generate_zipf_plot_svg(
    word_counts: Counter,
    filename: str = "zipf_distribution.svg",
    width: int = 850,
    height: int = 500
):
    """
    Generates log-log plot demonstrating Zipf's Law (log(Rank) vs log(Frequency)).
    """
    sorted_counts = [c for w, c in word_counts.most_common()]
    ranks = list(range(1, len(sorted_counts) + 1))

    log_ranks = [math.log10(r) for r in ranks]
    log_freqs = [math.log10(f) for f in sorted_counts]

    margin_l = 85
    margin_r = 160
    margin_t = 60
    margin_b = 60

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    max_log_r = math.log10(len(ranks))
    max_log_f = math.log10(sorted_counts[0]) * 1.05

    def map_x(lr):
        return margin_l + (lr / max_log_r) * plot_w

    def map_y(lf):
        return margin_t + (1.0 - (lf / max_log_f)) * plot_h

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">')

    svg.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="10"/>')
    svg.append(f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#1e293b" stroke="#334155" stroke-width="1.5" rx="4"/>')

    svg.append(f'<text x="{margin_l + plot_w/2}" y="35" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="600">Zipf\'s Law: Empirical Word Frequency vs. Rank (Log-Log Scale)</text>')

    # Ticks
    for i in range(int(max_log_r) + 1):
        sx = map_x(i)
        svg.append(f'<line x1="{sx}" y1="{margin_t}" x2="{sx}" y2="{margin_t + plot_h}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>')
        svg.append(f'<text x="{sx}" y="{margin_t + plot_h + 20}" text-anchor="middle" fill="#94a3b8" font-size="12">10^{i}</text>')

    for i in range(int(max_log_f) + 1):
        sy = map_y(i)
        svg.append(f'<line x1="{margin_l}" y1="{sy}" x2="{margin_l + plot_w}" y2="{sy}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l - 12}" y="{sy + 4}" text-anchor="end" fill="#94a3b8" font-size="12">10^{i}</text>')

    svg.append(f'<text x="{margin_l + plot_w / 2}" y="{height - 15}" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="500">Log10(Rank)</text>')
    svg.append(f'<text x="25" y="{margin_t + plot_h / 2}" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="500" transform="rotate(-90 25 {margin_t + plot_h / 2})">Log10(Frequency)</text>')

    # Empirical Curve
    pts_d = []
    # Sample points to keep SVG compact
    step = max(1, len(ranks) // 300)
    for i in range(0, len(ranks), step):
        sx = map_x(log_ranks[i])
        sy = map_y(log_freqs[i])
        pts_d.append(f"{'M' if i==0 else 'L'}{sx:.1f},{sy:.1f}")

    svg.append(f'<path d="{" ".join(pts_d)}" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linejoin="round"/>')

    # Theoretical Zipfian line (slope ~ -1.0)
    th_pts = [
        (0.0, max_log_f * 0.95),
        (max_log_r, max_log_f * 0.95 - max_log_r * 0.98)
    ]
    th_d = f"M{map_x(th_pts[0][0]):.1f},{map_y(th_pts[0][1]):.1f} L{map_x(th_pts[1][0]):.1f},{map_y(th_pts[1][1]):.1f}"
    svg.append(f'<path d="{th_d}" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="5,4"/>')

    # Legend
    leg_x = margin_l + plot_w + 15
    leg_y = margin_t + 15
    svg.append(f'<rect x="{leg_x}" y="{leg_y}" width="{margin_r - 25}" height="90" fill="#1e293b" stroke="#334155" rx="6"/>')
    svg.append(f'<line x1="{leg_x + 10}" y1="{leg_y + 30}" x2="{leg_x + 30}" y2="{leg_y + 30}" stroke="#38bdf8" stroke-width="2.5"/>')
    svg.append(f'<text x="{leg_x + 38}" y="{leg_y + 34}" fill="#f1f5f9" font-size="11">Empirical Data</text>')
    svg.append(f'<line x1="{leg_x + 10}" y1="{leg_y + 60}" x2="{leg_x + 30}" y2="{leg_y + 60}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="5,4"/>')
    svg.append(f'<text x="{leg_x + 38}" y="{leg_y + 64}" fill="#f1f5f9" font-size="11">Zipf Ideal (~1/r)</text>')

    svg.append('</svg>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"Generated plot: {filename}")


def main():
    tokens = load_corpus(DEFAULT_CORPUS_FILE)
    print(f"Loaded corpus with {len(tokens):,} tokens. Unique words: {len(set(tokens)):,}")

    # 1. Generate Top-15 words bar chart
    top15 = top_k_min_heap(tokens, 15)
    barchart_path = os.path.join(SCRIPT_DIR, "top_k_barchart.svg")
    generate_barchart_svg(top15, total_tokens=len(tokens), filename=barchart_path)

    # 2. Benchmark k-scaling
    res_k = benchmark_k_scaling(tokens)

    # 3. Benchmark N-scaling
    res_n = benchmark_n_scaling(tokens)

    # Save benchmark json
    benchmarks = {
        "k_scaling": res_k,
        "n_scaling": res_n
    }
    json_path = os.path.join(SCRIPT_DIR, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)

    # 4. Generate runtime vs k plot
    pts_heap_k = [(d["k"], d["min_heap_ms"]) for d in res_k]
    pts_sort_k = [(d["k"], d["full_sort_ms"]) for d in res_k]
    pts_mc_k = [(d["k"], d["counter_ms"]) for d in res_k]
    max_t_k = max(d["full_sort_ms"] for d in res_k) * 1.3

    k_plot_path = os.path.join(SCRIPT_DIR, "runtime_vs_k.svg")
    generate_line_plot(
        title="Runtime vs. k (Top-k Selection, N = 50,000, M = 1,614)",
        x_label="Number of Top Words (k)",
        y_label="Execution Time (ms)",
        series=[
            {"name": "Min-Heap O(M log k)", "points": pts_heap_k, "color": "#10b981"},
            {"name": "Counter.most_common", "points": pts_mc_k, "color": "#38bdf8"},
            {"name": "Full Sort O(M log M)", "points": pts_sort_k, "color": "#ef4444", "dash": "4,3"},
        ],
        x_max=max(d["k"] for d in res_k),
        y_max=max_t_k,
        filename=k_plot_path
    )

    # 5. Generate runtime vs N plot
    pts_heap_n = [(d["n"], d["min_heap_ms"]) for d in res_n]
    pts_sort_n = [(d["n"], d["full_sort_ms"]) for d in res_n]
    pts_bucket_n = [(d["n"], d["bucket_ms"]) for d in res_n]
    max_t_n = max(d["full_sort_ms"] for d in res_n) * 1.25

    n_plot_path = os.path.join(SCRIPT_DIR, "runtime_vs_n.svg")
    generate_line_plot(
        title="Runtime vs. Text Length N (Fixed k = 20)",
        x_label="Corpus Size (N words)",
        y_label="Execution Time (ms)",
        series=[
            {"name": "Min-Heap (k=20)", "points": pts_heap_n, "color": "#10b981"},
            {"name": "Bucket Sort O(N+M)", "points": pts_bucket_n, "color": "#f59e0b"},
            {"name": "Full Sort O(M log M)", "points": pts_sort_n, "color": "#ef4444"},
        ],
        x_max=max(d["n"] for d in res_n),
        y_max=max_t_n,
        filename=n_plot_path
    )

    # 6. Generate Zipf distribution plot
    zipf_path = os.path.join(SCRIPT_DIR, "zipf_distribution.svg")
    generate_zipf_plot_svg(Counter(tokens), filename=zipf_path)

    print("\nAll Project 3 benchmarks and plots completed successfully!")


if __name__ == "__main__":
    main()
