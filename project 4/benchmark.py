"""
Benchmark and Visualization Suite for Project 4: Word Ladder.
Generates:
  1. word_ladder_graph.svg (Visual graph of word transformations and shortest path)
  2. bfs_vs_bidirectional.svg (Search space reduction: BFS vs Bidirectional BFS)
  3. benchmark_results.json
"""

import json
import os
import time
from typing import List, Dict, Any, Tuple

from word_ladder import (
    load_word_list,
    word_ladder_bfs,
    word_ladder_bidirectional_bfs,
    word_ladder_dfs,
)


BENCHMARK_PAIRS = [
    ("lead", "gold", "Lead to Gold"),
    ("cold", "warm", "Cold to Warm"),
    ("cat", "dog", "Cat to Dog"),
    ("head", "tail", "Head to Tail"),
    ("fool", "sage", "Fool to Sage"),
    ("hate", "love", "Hate to Love"),
    ("fire", "cold", "Fire to Cold"),
    ("beer", "wine", "Beer to Wine"),
]


def run_benchmarks(words: set) -> List[Dict[str, Any]]:
    results = []
    print(f"\n{'Puzzle':<18} | {'Ladder':<28} | {'Len':>3} | {'BFS Nodes':>10} | {'Bi-BFS Nodes':>12} | {'Pruning %':>10} | {'BFS Time':>9} | {'Bi Time':>9}")
    print("-" * 115)

    trials = 5
    for start, target, label in BENCHMARK_PAIRS:
        # 1. BFS
        t0 = time.perf_counter()
        for _ in range(trials):
            res_bfs = word_ladder_bfs(start, target, words)
        t_bfs = ((time.perf_counter() - t0) / trials) * 1000

        # 2. Bidirectional BFS
        t0 = time.perf_counter()
        for _ in range(trials):
            res_bi = word_ladder_bidirectional_bfs(start, target, words)
        t_bi = ((time.perf_counter() - t0) / trials) * 1000

        # Verify correctness
        assert res_bfs["length"] == res_bi["length"], f"Mismatch for {start}->{target}"

        nodes_bfs = res_bfs["visited_count"]
        nodes_bi = res_bi["visited_count"]
        pruning_pct = ((nodes_bfs - nodes_bi) / nodes_bfs * 100) if nodes_bfs > 0 else 0.0
        ladder_str = " -> ".join([w.upper() for w in res_bfs["path"]])

        print(f"{label:<18} | {ladder_str:<28} | {res_bfs['length']:>3} | {nodes_bfs:>10} | {nodes_bi:>12} | {pruning_pct:>9.1f}% | {t_bfs:>7.2f} ms | {t_bi:>7.2f} ms")

        results.append({
            "start": start,
            "target": target,
            "label": label,
            "path": res_bfs["path"],
            "length": res_bfs["length"],
            "transformations": res_bfs["transformations"],
            "bfs_nodes": nodes_bfs,
            "bi_nodes": nodes_bi,
            "pruning_pct": round(pruning_pct, 2),
            "bfs_time_ms": round(t_bfs, 3),
            "bi_time_ms": round(t_bi, 3),
        })

    return results


# =====================================================================
# SVG Plot 1: Word Ladder Graph Visualization
# =====================================================================

def generate_ladder_graph_svg(filename: str = "word_ladder_graph.svg", width: int = 900, height: int = 520):
    """
    Renders an SVG graph displaying the shortest path and branching exploration.
    Visualizes the COLD -> WARM and LEAD -> GOLD solutions.
    """
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">')

    # Definitions
    svg.append("""
    <defs>
      <linearGradient id="pathNodeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#0284c7"/>
        <stop offset="100%" stop-color="#0369a1"/>
      </linearGradient>
      <linearGradient id="startNodeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#16a34a"/>
        <stop offset="100%" stop-color="#15803d"/>
      </linearGradient>
      <linearGradient id="endNodeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#dc2626"/>
        <stop offset="100%" stop-color="#b91c1c"/>
      </linearGradient>
      <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
    </defs>
    """)

    # Background
    svg.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="10"/>')

    # Title & Subtitle
    svg.append(f'<text x="{width/2}" y="35" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="700">Word Ladder: BFS Shortest Path Tree &amp; State Space Exploration</text>')
    svg.append(f'<text x="{width/2}" y="56" text-anchor="middle" fill="#94a3b8" font-size="12">Unweighted Graph Shortest Path via Level-by-Level Breadth-First Search</text>')

    # Section 1: COLD -> WARM (Top half)
    svg.append('<text x="50" y="90" fill="#38bdf8" font-size="14" font-weight="600">Ladder 1: COLD → WARM (4 Transformations, 5 Words)</text>')

    cold_nodes = [
        {"word": "COLD", "x": 90, "y": 140, "type": "start", "branches": ["BOLD", "HOLD", "GOLD", "COLT"]},
        {"word": "CORD", "x": 270, "y": 140, "type": "path", "branches": ["CARD", "CORK", "CORN", "LORD"]},
        {"word": "WORD", "x": 450, "y": 140, "type": "path", "branches": ["WORM", "WORE", "WORK", "FORD"]},
        {"word": "WARD", "x": 630, "y": 140, "type": "path", "branches": ["HARD", "YARD", "WARS", "WARN"]},
        {"word": "WARM", "x": 810, "y": 140, "type": "end", "branches": []},
    ]

    # Draw branch nodes (alternative BFS candidates explored)
    for idx, node in enumerate(cold_nodes[:-1]):
        nx, ny = node["x"], node["y"]
        for b_idx, b_word in enumerate(node["branches"][:2]):
            bx = nx + 40
            by = ny + 55 + (b_idx * 26)
            # Edge to branch
            svg.append(f'<line x1="{nx}" y1="{ny}" x2="{bx}" y2="{by}" stroke="#475569" stroke-width="1.2" stroke-dasharray="3,3"/>')
            svg.append(f'<rect x="{bx-22}" y="{by-11}" width="44" height="22" fill="#1e293b" stroke="#475569" rx="3"/>')
            svg.append(f'<text x="{bx}" y="{by+4}" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="monospace">{b_word}</text>')

    # Draw main path edges
    for i in range(len(cold_nodes) - 1):
        x1, y1 = cold_nodes[i]["x"], cold_nodes[i]["y"]
        x2, y2 = cold_nodes[i + 1]["x"], cold_nodes[i + 1]["y"]
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#38bdf8" stroke-width="3.5" filter="url(#glow)"/>')
        # Arrowhead or diff label
        mx = (x1 + x2) / 2
        svg.append(f'<circle cx="{mx}" cy="{y1}" r="3" fill="#38bdf8"/>')

    # Draw main path nodes
    for node in cold_nodes:
        x, y = node["x"], node["y"]
        fill_grad = "startNodeGrad" if node["type"] == "start" else ("endNodeGrad" if node["type"] == "end" else "pathNodeGrad")
        stroke_col = "#22c55e" if node["type"] == "start" else ("#ef4444" if node["type"] == "end" else "#38bdf8")
        
        svg.append(f'<rect x="{x-35}" y="{y-20}" width="70" height="40" fill="url(#{fill_grad})" stroke="{stroke_col}" stroke-width="2" rx="6" filter="url(#glow)"/>')
        svg.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="700" font-family="monospace">{node["word"]}</text>')

    # Section 2: LEAD -> GOLD (Bottom half)
    svg.append('<text x="50" y="290" fill="#38bdf8" font-size="14" font-weight="600">Ladder 2: LEAD → GOLD (3 Transformations, 4 Words)</text>')

    lead_nodes = [
        {"word": "LEAD", "x": 140, "y": 340, "type": "start", "branches": ["HEAD", "DEAD", "LOAD", "LEAF"]},
        {"word": "LOAD", "x": 370, "y": 340, "type": "path", "branches": ["TOAD", "ROAD", "GOAD", "LOAN"]},
        {"word": "GOAD", "x": 600, "y": 340, "type": "path", "branches": ["GOOD", "GOAT", "GOLD"]},
        {"word": "GOLD", "x": 810, "y": 340, "type": "end", "branches": []},
    ]

    for idx, node in enumerate(lead_nodes[:-1]):
        nx, ny = node["x"], node["y"]
        for b_idx, b_word in enumerate(node["branches"][:2]):
            bx = nx + 50
            by = ny + 55 + (b_idx * 26)
            svg.append(f'<line x1="{nx}" y1="{ny}" x2="{bx}" y2="{by}" stroke="#475569" stroke-width="1.2" stroke-dasharray="3,3"/>')
            svg.append(f'<rect x="{bx-22}" y="{by-11}" width="44" height="22" fill="#1e293b" stroke="#475569" rx="3"/>')
            svg.append(f'<text x="{bx}" y="{by+4}" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="monospace">{b_word}</text>')

    for i in range(len(lead_nodes) - 1):
        x1, y1 = lead_nodes[i]["x"], lead_nodes[i]["y"]
        x2, y2 = lead_nodes[i + 1]["x"], lead_nodes[i + 1]["y"]
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#10b981" stroke-width="3.5" filter="url(#glow)"/>')
        mx = (x1 + x2) / 2
        svg.append(f'<circle cx="{mx}" cy="{y1}" r="3" fill="#10b981"/>')

    for node in lead_nodes:
        x, y = node["x"], node["y"]
        fill_grad = "startNodeGrad" if node["type"] == "start" else ("endNodeGrad" if node["type"] == "end" else "pathNodeGrad")
        stroke_col = "#22c55e" if node["type"] == "start" else ("#ef4444" if node["type"] == "end" else "#10b981")
        
        svg.append(f'<rect x="{x-35}" y="{y-20}" width="70" height="40" fill="url(#{fill_grad})" stroke="{stroke_col}" stroke-width="2" rx="6" filter="url(#glow)"/>')
        svg.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="700" font-family="monospace">{node["word"]}</text>')

    # Legend at bottom
    leg_y = height - 45
    svg.append(f'<rect x="180" y="{leg_y}" width="540" height="34" fill="#1e293b" stroke="#334155" rx="6"/>')
    svg.append(f'<circle cx="210" cy="{leg_y+17}" r="7" fill="#16a34a"/>')
    svg.append(f'<text x="225" y="{leg_y+21}" fill="#cbd5e1" font-size="11">Start Word</text>')
    svg.append(f'<circle cx="320" cy="{leg_y+17}" r="7" fill="#0284c7"/>')
    svg.append(f'<text x="335" y="{leg_y+21}" fill="#cbd5e1" font-size="11">Shortest Path Step</text>')
    svg.append(f'<circle cx="470" cy="{leg_y+17}" r="7" fill="#dc2626"/>')
    svg.append(f'<text x="485" y="{leg_y+21}" fill="#cbd5e1" font-size="11">Target Word</text>')
    svg.append(f'<rect x="585" y="{leg_y+10}" width="26" height="14" fill="#1e293b" stroke="#475569" rx="2"/>')
    svg.append(f'<text x="620" y="{leg_y+21}" fill="#94a3b8" font-size="11">Branch Explored</text>')

    svg.append('</svg>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"Generated diagram: {filename}")


# =====================================================================
# SVG Plot 2: BFS vs Bidirectional BFS Nodes Visited
# =====================================================================

def generate_comparison_barchart_svg(
    results: List[Dict[str, Any]],
    filename: str = "bfs_vs_bidirectional.svg",
    width: int = 900,
    height: int = 500
):
    """
    Renders grouped bar chart comparing nodes visited in BFS vs Bidirectional BFS.
    """
    margin_l = 90
    margin_r = 160
    margin_t = 60
    margin_b = 60

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    n_bars = len(results)
    group_w = plot_w / n_bars
    bar_w = group_w * 0.35

    max_nodes = max(d["bfs_nodes"] for d in results) * 1.15

    def map_y(val):
        return margin_t + (1.0 - (val / max_nodes)) * plot_h

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">')

    # Gradients
    svg.append("""
    <defs>
      <linearGradient id="bfsBarGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#ef4444"/>
        <stop offset="100%" stop-color="#b91c1c"/>
      </linearGradient>
      <linearGradient id="biBarGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#10b981"/>
        <stop offset="100%" stop-color="#047857"/>
      </linearGradient>
    </defs>
    """)

    # Background
    svg.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="10"/>')
    svg.append(f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#1e293b" stroke="#334155" stroke-width="1.5" rx="4"/>')

    # Title
    svg.append(f'<text x="{margin_l + plot_w/2}" y="35" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="600">Search Space Reduction: Standard BFS vs. Bidirectional BFS</text>')

    # Y Ticks
    y_steps = 5
    for i in range(y_steps + 1):
        v = (max_nodes / y_steps) * i
        sy = map_y(v)
        svg.append(f'<line x1="{margin_l}" y1="{sy}" x2="{margin_l + plot_w}" y2="{sy}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l - 12}" y="{sy + 4}" text-anchor="end" fill="#94a3b8" font-size="12">{int(v)}</text>')

    # Bars
    for idx, d in enumerate(results):
        gx = margin_l + idx * group_w
        cx = gx + group_w / 2

        # 1. BFS Bar (Red)
        bfs_h = (d["bfs_nodes"] / max_nodes) * plot_h
        bfs_y = margin_t + plot_h - bfs_h
        bx1 = cx - bar_w - 2
        svg.append(f'<rect x="{bx1}" y="{bfs_y}" width="{bar_w}" height="{bfs_h}" fill="url(#bfsBarGrad)" rx="3"/>')
        svg.append(f'<text x="{bx1 + bar_w/2}" y="{bfs_y - 6}" text-anchor="middle" fill="#fca5a5" font-size="10" font-weight="600">{d["bfs_nodes"]}</text>')

        # 2. Bidirectional BFS Bar (Green)
        bi_h = (d["bi_nodes"] / max_nodes) * plot_h
        bi_y = margin_t + plot_h - bi_h
        bx2 = cx + 2
        svg.append(f'<rect x="{bx2}" y="{bi_y}" width="{bar_w}" height="{bi_h}" fill="url(#biBarGrad)" rx="3"/>')
        svg.append(f'<text x="{bx2 + bar_w/2}" y="{bi_y - 6}" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="600">{d["bi_nodes"]}</text>')

        # X label
        svg.append(f'<text x="{cx}" y="{margin_t + plot_h + 20}" text-anchor="middle" fill="#e2e8f0" font-size="11" font-weight="500">{d["start"].upper()}→{d["target"].upper()}</text>')
        # Pruning label
        svg.append(f'<text x="{cx}" y="{margin_t + plot_h + 34}" text-anchor="middle" fill="#38bdf8" font-size="10">(-{d["pruning_pct"]:.0f}%)</text>')

    # Axis Labels
    svg.append(f'<text x="{margin_l + plot_w / 2}" y="{height - 12}" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="500">Word Ladder Benchmark Puzzles</text>')
    svg.append(f'<text x="25" y="{margin_t + plot_h / 2}" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="500" transform="rotate(-90 25 {margin_t + plot_h / 2})">Nodes Visited (Search Space)</text>')

    # Legend
    leg_x = margin_l + plot_w + 15
    leg_y = margin_t + 15
    svg.append(f'<rect x="{leg_x}" y="{leg_y}" width="{margin_r - 25}" height="100" fill="#1e293b" stroke="#334155" rx="6"/>')
    svg.append(f'<text x="{leg_x + 10}" y="{leg_y + 20}" fill="#94a3b8" font-size="11" font-weight="600" text-transform="uppercase">Legend</text>')

    svg.append(f'<rect x="{leg_x + 10}" y="{leg_y + 35}" width="16" height="12" fill="url(#bfsBarGrad)" rx="2"/>')
    svg.append(f'<text x="{leg_x + 34}" y="{leg_y + 45}" fill="#f1f5f9" font-size="11">Standard BFS O(bᵈ)</text>')

    svg.append(f'<rect x="{leg_x + 10}" y="{leg_y + 65}" width="16" height="12" fill="url(#biBarGrad)" rx="2"/>')
    svg.append(f'<text x="{leg_x + 34}" y="{leg_y + 75}" fill="#f1f5f9" font-size="11">Bi-BFS O(2·bᵈ/²)</text>')

    svg.append('</svg>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"Generated chart: {filename}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    words_file = os.path.join(script_dir, "words.txt")
    words = load_word_list(words_file)
    print(f"Loaded {len(words)} dictionary words from: {words_file}")

    # 1. Generate ladder graph SVG
    graph_svg = os.path.join(script_dir, "word_ladder_graph.svg")
    generate_ladder_graph_svg(filename=graph_svg)

    # 2. Run benchmarks
    results = run_benchmarks(words)
    json_path = os.path.join(script_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 3. Generate comparison bar chart
    chart_svg = os.path.join(script_dir, "bfs_vs_bidirectional.svg")
    generate_comparison_barchart_svg(results, filename=chart_svg)

    print("\nWord Ladder benchmarks and visualizations completed successfully!")


if __name__ == "__main__":
    main()
