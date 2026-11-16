"""
Empirical Analysis and Experimentation Suite for Insertion Sort and Merge Sort.
Conducts timing and operation count experiments across input sizes n = 100 to 5000.
Generates publication-quality SVG plots and comprehensive markdown benchmark tables.
"""

import math
import random
import time
from typing import Dict, List, Any
import json
import os

from insertion_sort import insertion_sort, insertion_sort_instrumented, merge_sort

def run_benchmarks(sizes: List[int], trials: int = 5) -> Dict[str, Any]:
    """
    Executes empirical experiments across specified array sizes and distributions.
    """
    random.seed(42)
    results = []

    print(f"{'Size (n)':>8} | {'Type':<14} | {'Time (s)':>10} | {'Comparisons':>12} | {'Shifts':>12} | {'MergeSort (s)':>14}")
    print("-" * 80)

    for n in sizes:
        # 1. Sorted input
        sorted_arr = list(range(n))
        # Time clean insertion sort
        t0 = time.perf_counter()
        for _ in range(trials):
            insertion_sort(sorted_arr)
        t_sorted = (time.perf_counter() - t0) / trials
        _, comp_sorted, shift_sorted = insertion_sort_instrumented(sorted_arr)

        # Merge sort on sorted
        t0 = time.perf_counter()
        for _ in range(trials):
            merge_sort(sorted_arr)
        t_merge_sorted = (time.perf_counter() - t0) / trials

        # 2. Reverse-sorted input
        rev_arr = list(range(n, 0, -1))
        t0 = time.perf_counter()
        for _ in range(trials):
            insertion_sort(rev_arr)
        t_rev = (time.perf_counter() - t0) / trials
        _, comp_rev, shift_rev = insertion_sort_instrumented(rev_arr)

        # Merge sort on reverse-sorted
        t0 = time.perf_counter()
        for _ in range(trials):
            merge_sort(rev_arr)
        t_merge_rev = (time.perf_counter() - t0) / trials

        # 3. Random input (averaged over trials)
        random_times = []
        random_comps = []
        random_shifts = []
        merge_rand_times = []

        for _ in range(trials):
            rand_arr = [random.randint(1, 10 * n) for _ in range(n)]
            t0 = time.perf_counter()
            insertion_sort(rand_arr)
            random_times.append(time.perf_counter() - t0)

            _, comps, shifts = insertion_sort_instrumented(rand_arr)
            random_comps.append(comps)
            random_shifts.append(shifts)

            t0 = time.perf_counter()
            merge_sort(rand_arr)
            merge_rand_times.append(time.perf_counter() - t0)

        t_rand = sum(random_times) / len(random_times)
        comp_rand = int(sum(random_comps) / len(random_comps))
        shift_rand = int(sum(random_shifts) / len(random_shifts))
        t_merge_rand = sum(merge_rand_times) / len(merge_rand_times)

        entry = {
            "n": n,
            "sorted": {
                "time": t_sorted,
                "comparisons": comp_sorted,
                "shifts": shift_sorted,
                "merge_time": t_merge_sorted,
            },
            "reverse": {
                "time": t_rev,
                "comparisons": comp_rev,
                "shifts": shift_rev,
                "merge_time": t_merge_rev,
            },
            "random": {
                "time": t_rand,
                "comparisons": comp_rand,
                "shifts": shift_rand,
                "merge_time": t_merge_rand,
            }
        }
        results.append(entry)

        print(f"{n:>8} | {'Sorted':<14} | {t_sorted:>10.6f} | {comp_sorted:>12} | {shift_sorted:>12} | {t_merge_sorted:>14.6f}")
        print(f"{n:>8} | {'Random':<14} | {t_rand:>10.6f} | {comp_rand:>12} | {shift_rand:>12} | {t_merge_rand:>14.6f}")
        print(f"{n:>8} | {'Reverse-Sorted':<14} | {t_rev:>10.6f} | {comp_rev:>12} | {shift_rev:>12} | {t_merge_rev:>14.6f}")
        print("-" * 80)

    return {"sizes": sizes, "data": results}


def generate_svg_plot(
    title: str,
    x_label: str,
    y_label: str,
    series: List[Dict[str, Any]],
    x_max: float,
    y_max: float,
    width: int = 800,
    height: int = 500,
    filename: str = "plot.svg",
    use_log_y: bool = False,
    y_min: float = 0.0,
):
    """
    Renders an SVG graph for multiple data series with axes, gridlines, and legend.
    """
    margin_left = 90
    margin_right = 180
    margin_top = 60
    margin_bottom = 60

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def map_x(val):
        return margin_left + (val / x_max) * plot_w

    def map_y(val):
        if use_log_y:
            log_min = math.log10(max(y_min, 1e-6))
            log_max = math.log10(max(y_max, 1e-6))
            val_clamp = max(val, 1e-6)
            norm = (math.log10(val_clamp) - log_min) / (log_max - log_min)
            return margin_top + (1.0 - norm) * plot_h
        else:
            return margin_top + (1.0 - (val / y_max)) * plot_h

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">')

    # Styles
    svg.append("""
    <defs>
      <linearGradient id="gridGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#1e293b" stop-opacity="0.8"/>
        <stop offset="100%" stop-color="#0f172a" stop-opacity="0.8"/>
      </linearGradient>
    </defs>
    """)

    # Background frame
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0f172a" rx="10"/>')
    svg.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="url(#gridGrad)" stroke="#334155" stroke-width="1.5" rx="4"/>')

    # Title
    svg.append(f'<text x="{margin_left + plot_w / 2}" y="35" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="600">{title}</text>')

    # Grid lines and X ticks
    x_steps = 5
    for i in range(x_steps + 1):
        x_val = (x_max / x_steps) * i
        sx = map_x(x_val)
        svg.append(f'<line x1="{sx}" y1="{margin_top}" x2="{sx}" y2="{margin_top + plot_h}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>')
        svg.append(f'<text x="{sx}" y="{margin_top + plot_h + 20}" text-anchor="middle" fill="#94a3b8" font-size="12">{int(x_val)}</text>')

    # Y ticks
    y_steps = 5
    for i in range(y_steps + 1):
        if use_log_y:
            log_min = math.log10(max(y_min, 1e-6))
            log_max = math.log10(max(y_max, 1e-6))
            cur_log = log_min + (log_max - log_min) * (i / y_steps)
            y_val = 10 ** cur_log
            y_label_str = f"{y_val:.1e}"
        else:
            y_val = (y_max / y_steps) * i
            if y_max >= 1e6:
                y_label_str = f"{y_val/1e6:.1f}M"
            elif y_max >= 1e3:
                y_label_str = f"{y_val/1e3:.1f}k"
            elif y_max < 0.1:
                y_label_str = f"{y_val:.4f}"
            else:
                y_label_str = f"{y_val:.2f}"
        sy = map_y(y_val)
        svg.append(f'<line x1="{margin_left}" y1="{sy}" x2="{margin_left + plot_w}" y2="{sy}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>')
        svg.append(f'<text x="{margin_left - 12}" y="{sy + 4}" text-anchor="end" fill="#94a3b8" font-size="12">{y_label_str}</text>')

    # Axis Labels
    svg.append(f'<text x="{margin_left + plot_w / 2}" y="{height - 15}" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="500">{x_label}</text>')
    svg.append(f'<text x="25" y="{margin_top + plot_h / 2}" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="500" transform="rotate(-90 25 {margin_top + plot_h / 2})">{y_label}</text>')

    # Plot Series
    legend_items = []
    for s in series:
        pts = s["points"]
        color = s.get("color", "#38bdf8")
        stroke_width = s.get("width", 2.5)
        dash = s.get("dash", "")
        name = s.get("name", "Series")

        path_d = []
        for idx, (px, py) in enumerate(pts):
            sx = map_x(px)
            sy = map_y(py)
            cmd = "M" if idx == 0 else "L"
            path_d.append(f"{cmd}{sx:.1f},{sy:.1f}")

        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<path d="{" ".join(path_d)}" fill="none" stroke="{color}" stroke-width="{stroke_width}"{dash_attr} stroke-linejoin="round" stroke-linecap="round"/>')

        # Add data points
        if not s.get("hide_dots", False):
            for px, py in pts:
                sx = map_x(px)
                sy = map_y(py)
                svg.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3.5" fill="{color}" stroke="#0f172a" stroke-width="1.5"/>')

        legend_items.append((name, color, dash))

    # Legend
    leg_x = margin_left + plot_w + 15
    leg_y = margin_top + 10
    svg.append(f'<rect x="{leg_x - 5}" y="{leg_y - 10}" width="{margin_right - 20}" height="{len(legend_items) * 24 + 18}" fill="#1e293b" stroke="#334155" rx="6" opacity="0.95"/>')
    svg.append(f'<text x="{leg_x + 5}" y="{leg_y + 6}" fill="#94a3b8" font-size="11" font-weight="600" text-transform="uppercase">Legend</text>')

    for idx, (name, color, dash) in enumerate(legend_items):
        item_y = leg_y + 26 + idx * 22
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="{leg_x + 5}" y1="{item_y - 4}" x2="{leg_x + 25}" y2="{item_y - 4}" stroke="{color}" stroke-width="2.5"{dash_attr}/>')
        svg.append(f'<circle cx="{leg_x + 15}" cy="{item_y - 4}" r="3" fill="{color}"/>')
        svg.append(f'<text x="{leg_x + 32}" y="{item_y}" fill="#f1f5f9" font-size="11">{name}</text>')

    svg.append('</svg>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"Generated plot: {filename}")


def run():
    # Range from 100 to 5000 as requested
    test_sizes = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    print(f"Starting experiments for n in {test_sizes}...")
    benchmarks = run_benchmarks(test_sizes, trials=3)

    data = benchmarks["data"]
    sizes = benchmarks["sizes"]

    # Save benchmark json
    os.makedirs(r"c:\Users\Khaled\OneDrive\Desktop\Algorism projects\project 1", exist_ok=True)
    json_path = r"c:\Users\Khaled\OneDrive\Desktop\Algorism projects\project 1\benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)

    # Prepare data series for plots
    # 1. Running time plot
    time_sorted = [(d["n"], d["sorted"]["time"]) for d in data]
    time_rev = [(d["n"], d["reverse"]["time"]) for d in data]
    time_rand = [(d["n"], d["random"]["time"]) for d in data]
    time_merge = [(d["n"], d["random"]["merge_time"]) for d in data]

    # Theoretical curves scaled
    # Let's fit c_worst * n^2 based on n=5000
    c_worst = time_rev[-1][1] / (5000 ** 2)
    c_rand = time_rand[-1][1] / (5000 ** 2)
    c_best = time_sorted[-1][1] / 5000
    c_merge = time_merge[-1][1] / (5000 * math.log2(5000))

    dense_n = [100 + i * 200 for i in range(26)]
    th_on2_worst = [(n, c_worst * (n ** 2)) for n in dense_n]
    th_on2_rand = [(n, c_rand * (n ** 2)) for n in dense_n]
    th_on_best = [(n, c_best * n) for n in dense_n]

    max_t = max(time_rev[-1][1], th_on2_worst[-1][1]) * 1.05

    series_time = [
        {"name": "Worst (Reverse)", "points": time_rev, "color": "#ef4444", "width": 2.5},
        {"name": "Theory O(n²) Worst", "points": th_on2_worst, "color": "#fca5a5", "width": 1.8, "dash": "5,4", "hide_dots": True},
        {"name": "Avg (Random)", "points": time_rand, "color": "#f59e0b", "width": 2.5},
        {"name": "Theory O(n²) Avg", "points": th_on2_rand, "color": "#fde68a", "width": 1.8, "dash": "5,4", "hide_dots": True},
        {"name": "Merge Sort", "points": time_merge, "color": "#06b6d4", "width": 2.2},
        {"name": "Best (Sorted)", "points": time_sorted, "color": "#10b981", "width": 2.5},
        {"name": "Theory O(n) Best", "points": th_on_best, "color": "#6ee7b7", "width": 1.8, "dash": "5,4", "hide_dots": True},
    ]

    generate_svg_plot(
        title="Insertion Sort Running Time vs. Input Size (n = 100 to 5,000)",
        x_label="Input Size (n)",
        y_label="Execution Time (seconds)",
        series=series_time,
        x_max=5000,
        y_max=max_t,
        filename=r"c:\Users\Khaled\OneDrive\Desktop\Algorism projects\project 1\runtime_comparison.svg",
    )

    # 2. Operations plot (Comparisons & Shifts)
    comp_sorted = [(d["n"], d["sorted"]["comparisons"]) for d in data]
    comp_rand = [(d["n"], d["random"]["comparisons"]) for d in data]
    comp_rev = [(d["n"], d["reverse"]["comparisons"]) for d in data]

    shift_sorted = [(d["n"], d["sorted"]["shifts"]) for d in data]
    shift_rand = [(d["n"], d["random"]["shifts"]) for d in data]
    shift_rev = [(d["n"], d["reverse"]["shifts"]) for d in data]

    th_comp_worst = [(n, n * (n - 1) / 2) for n in dense_n]
    th_comp_avg = [(n, n * (n - 1) / 4) for n in dense_n]
    th_comp_best = [(n, n - 1) for n in dense_n]

    max_ops = (5000 * 4999 / 2) * 1.05

    series_ops = [
        {"name": "Comparisons (Worst)", "points": comp_rev, "color": "#ef4444", "width": 2.5},
        {"name": "Shifts (Worst)", "points": shift_rev, "color": "#f87171", "width": 2.0, "dash": "4,3"},
        {"name": "Exact n(n-1)/2", "points": th_comp_worst, "color": "#cbd5e1", "width": 1.5, "dash": "2,2", "hide_dots": True},
        {"name": "Comparisons (Random)", "points": comp_rand, "color": "#f59e0b", "width": 2.5},
        {"name": "Shifts (Random)", "points": shift_rand, "color": "#fbbf24", "width": 2.0, "dash": "4,3"},
        {"name": "Exact ~n(n-1)/4", "points": th_comp_avg, "color": "#fde68a", "width": 1.5, "dash": "2,2", "hide_dots": True},
        {"name": "Comparisons (Sorted)", "points": comp_sorted, "color": "#10b981", "width": 2.5},
        {"name": "Shifts (Sorted = 0)", "points": shift_sorted, "color": "#34d399", "width": 2.0, "dash": "4,3"},
    ]

    generate_svg_plot(
        title="Operation Counts (Comparisons & Shifts) vs. Input Size (n)",
        x_label="Input Size (n)",
        y_label="Total Operations Count",
        series=series_ops,
        x_max=5000,
        y_max=max_ops,
        filename=r"c:\Users\Khaled\OneDrive\Desktop\Algorism projects\project 1\operations_comparison.svg",
    )

    # 3. Direct comparison: Insertion Sort vs Merge Sort
    series_vs = [
        {"name": "Insertion (Random)", "points": time_rand, "color": "#f59e0b", "width": 2.5},
        {"name": "Insertion (Reverse)", "points": time_rev, "color": "#ef4444", "width": 2.5},
        {"name": "Insertion (Sorted)", "points": time_sorted, "color": "#10b981", "width": 2.5},
        {"name": "Merge Sort (Random)", "points": time_merge, "color": "#38bdf8", "width": 2.5},
    ]

    generate_svg_plot(
        title="Insertion Sort vs. Merge Sort Empirical Runtime",
        x_label="Input Size (n)",
        y_label="Execution Time (seconds)",
        series=series_vs,
        x_max=5000,
        y_max=max_t,
        filename=r"c:\Users\Khaled\OneDrive\Desktop\Algorism projects\project 1\insertion_vs_merge.svg",
    )

    print("All experiments and plots completed successfully!")

if __name__ == "__main__":
    run()
