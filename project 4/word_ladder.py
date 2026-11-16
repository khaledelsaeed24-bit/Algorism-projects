"""
Word Ladder: Finding Shortest Transformation Sequences via BFS.
Tools: Python, collections.deque, word list

Implements:
  1. Standard One-Directional Breadth-First Search (BFS) - Guaranteed Shortest Path
  2. Optimized Bidirectional BFS - Exponential Search Space Reduction (O(b^(d/2)))
  3. Depth-First Search (DFS) - Baseline to demonstrate lack of shortest-path guarantee
  4. Wildcard pattern graph builder and dynamic character mutation generator
"""

from collections import deque, defaultdict
import os
import time
from typing import List, Tuple, Dict, Set, Optional, Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORDS_FILE = os.path.join(SCRIPT_DIR, "words.txt")


def load_word_list(filepath: str = DEFAULT_WORDS_FILE, target_len: Optional[int] = None) -> Set[str]:
    """
    Loads words from file into a set for O(1) membership lookup.
    Optionally filters by word length.
    Automatically resolves relative paths against the script's directory.
    """
    if not os.path.isabs(filepath) and not os.path.exists(filepath):
        alt_path = os.path.join(SCRIPT_DIR, filepath)
        if os.path.exists(alt_path):
            filepath = alt_path

    words = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w.isalpha():
                if target_len is None or len(w) == target_len:
                    words.add(w)
    return words


def build_pattern_map(word_list: Set[str]) -> Dict[str, List[str]]:
    """
    Precomputes wildcard patterns mapping to candidate neighbors.
    Example: 'c*t' -> ['cat', 'cot', 'cut']
    Time Complexity: O(V * L^2)
    Space Complexity: O(V * L^2)
    """
    patterns = defaultdict(list)
    for word in word_list:
        for i in range(len(word)):
            pattern = word[:i] + "*" + word[i + 1:]
            patterns[pattern].append(word)
    return patterns


# =====================================================================
# 1. Standard One-Directional BFS (Guaranteed Shortest Path)
# =====================================================================

def word_ladder_bfs(
    begin_word: str,
    end_word: str,
    word_list: Set[str]
) -> Dict[str, Any]:
    """
    Finds the shortest transformation sequence using Breadth-First Search.
    
    Time Complexity: O(V * L^2) where V is vocabulary size and L is word length.
    Space Complexity: O(V * L) for queue and visited set.
    """
    begin_word = begin_word.lower()
    end_word = end_word.lower()
    L = len(begin_word)

    if len(end_word) != L:
        return {"path": [], "length": 0, "transformations": 0, "visited_count": 0, "found": False, "time_ms": 0.0, "algorithm": "One-Directional BFS"}

    if begin_word == end_word:
        return {"path": [begin_word], "length": 1, "transformations": 0, "visited_count": 1, "found": True, "time_ms": 0.0, "algorithm": "One-Directional BFS"}

    # end_word must exist in the valid word dictionary
    if end_word not in word_list:
        return {"path": [], "length": 0, "transformations": 0, "visited_count": 0, "found": False, "time_ms": 0.0, "algorithm": "One-Directional BFS"}

    t0 = time.perf_counter()

    queue = deque([[begin_word]])
    visited = {begin_word}
    nodes_visited = 0

    while queue:
        path = queue.popleft()
        current_word = path[-1]
        nodes_visited += 1

        if current_word == end_word:
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "path": path,
                "length": len(path),
                "transformations": len(path) - 1,
                "visited_count": nodes_visited,
                "found": True,
                "time_ms": elapsed,
                "algorithm": "One-Directional BFS"
            }

        # Generate neighbors by mutating each of the L character positions
        for i in range(L):
            curr_char = current_word[i]
            for c_code in range(ord('a'), ord('z') + 1):
                c = chr(c_code)
                if c == curr_char:
                    continue
                neighbor = current_word[:i] + c + current_word[i + 1:]
                if neighbor in word_list and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "path": [],
        "length": 0,
        "transformations": 0,
        "visited_count": nodes_visited,
        "found": False,
        "time_ms": elapsed,
        "algorithm": "One-Directional BFS"
    }


# =====================================================================
# 2. Optimized Bidirectional BFS
# =====================================================================

def word_ladder_bidirectional_bfs(
    begin_word: str,
    end_word: str,
    word_list: Set[str]
) -> Dict[str, Any]:
    """
    Bidirectional BFS searching simultaneously from begin_word and end_word.
    Drastically prunes the search space from O(b^d) down to O(2 * b^(d/2)).
    """
    begin_word = begin_word.lower()
    end_word = end_word.lower()
    L = len(begin_word)

    if len(end_word) != L or end_word not in word_list:
        return {"path": [], "length": 0, "transformations": 0, "visited_count": 0, "found": False, "time_ms": 0.0, "algorithm": "Bidirectional BFS"}

    if begin_word == end_word:
        return {"path": [begin_word], "length": 1, "transformations": 0, "visited_count": 1, "found": True, "time_ms": 0.0, "algorithm": "Bidirectional BFS"}

    t0 = time.perf_counter()

    # Augmented dictionary ensuring begin_word is reachable if backward frontier touches it
    valid_words = word_list | {begin_word}

    # Forward and Backward parent mappings
    front_parents: Dict[str, Optional[str]] = {begin_word: None}
    back_parents: Dict[str, Optional[str]] = {end_word: None}

    front_queue = {begin_word}
    back_queue = {end_word}
    nodes_visited = 0
    meeting_word = None

    while front_queue and back_queue:
        # Always expand the smaller frontier to minimize branching factor
        if len(front_queue) > len(back_queue):
            front_queue, back_queue = back_queue, front_queue
            front_parents, back_parents = back_parents, front_parents

        next_frontier = set()
        for word in front_queue:
            nodes_visited += 1
            for i in range(L):
                curr_c = word[i]
                for c_code in range(ord('a'), ord('z') + 1):
                    c = chr(c_code)
                    if c == curr_c:
                        continue
                    neighbor = word[:i] + c + word[i + 1:]
                    if neighbor in valid_words:
                        # Check collision with opposite frontier
                        if neighbor in back_parents:
                            front_parents[neighbor] = word
                            meeting_word = neighbor
                            break

                        if neighbor not in front_parents:
                            front_parents[neighbor] = word
                            next_frontier.add(neighbor)

                if meeting_word:
                    break
            if meeting_word:
                break

        if meeting_word:
            break
        front_queue = next_frontier

    elapsed = (time.perf_counter() - t0) * 1000

    if not meeting_word:
        return {
            "path": [],
            "length": 0,
            "transformations": 0,
            "visited_count": nodes_visited,
            "found": False,
            "time_ms": elapsed,
            "algorithm": "Bidirectional BFS"
        }

    # Reconstruct path through collision node
    def trace_path(parents, start_node):
        p = []
        curr = start_node
        while curr is not None:
            p.append(curr)
            curr = parents[curr]
        return p

    if begin_word in front_parents:
        path_from_begin = trace_path(front_parents, meeting_word)[::-1]
        path_from_end = trace_path(back_parents, meeting_word)[1:]
        full_path = path_from_begin + path_from_end
    else:
        path_from_begin = trace_path(back_parents, meeting_word)[::-1]
        path_from_end = trace_path(front_parents, meeting_word)[1:]
        full_path = path_from_begin + path_from_end

    return {
        "path": full_path,
        "length": len(full_path),
        "transformations": len(full_path) - 1,
        "visited_count": nodes_visited,
        "found": True,
        "time_ms": elapsed,
        "algorithm": "Bidirectional BFS"
    }


# =====================================================================
# 3. Depth-First Search (DFS) for Comparative Analysis
# =====================================================================

def word_ladder_dfs(
    begin_word: str,
    end_word: str,
    word_list: Set[str],
    max_depth: int = 15
) -> Dict[str, Any]:
    """
    Depth-First Search (DFS) baseline.
    Demonstrates that DFS does not guarantee the shortest path and easily
    takes sub-optimal routes or risks deep recursion stacks.
    """
    begin_word = begin_word.lower()
    end_word = end_word.lower()
    L = len(begin_word)

    if len(end_word) != L or end_word not in word_list:
        return {"path": [], "length": 0, "transformations": 0, "visited_count": 0, "found": False, "time_ms": 0.0, "algorithm": "Depth-First Search (DFS)"}

    t0 = time.perf_counter()
    nodes_visited = 0
    best_path = None

    visited = {begin_word}

    def dfs(curr_word, current_path):
        nonlocal nodes_visited, best_path
        nodes_visited += 1

        if curr_word == end_word:
            best_path = list(current_path)
            return True

        if len(current_path) >= max_depth:
            return False

        for i in range(L):
            curr_c = curr_word[i]
            for c_code in range(ord('a'), ord('z') + 1):
                c = chr(c_code)
                if c == curr_c:
                    continue
                neighbor = curr_word[:i] + c + curr_word[i + 1:]
                if neighbor in word_list and neighbor not in visited:
                    visited.add(neighbor)
                    if dfs(neighbor, current_path + [neighbor]):
                        return True
                    visited.remove(neighbor)
        return False

    dfs(begin_word, [begin_word])
    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "path": best_path or [],
        "length": len(best_path) if best_path else 0,
        "transformations": len(best_path) - 1 if best_path else 0,
        "visited_count": nodes_visited,
        "found": best_path is not None,
        "time_ms": elapsed,
        "algorithm": "Depth-First Search (DFS)"
    }


def format_ladder_diff(path: List[str]) -> str:
    """
    Formats the step-by-step ladder showing which letter changed at each step.
    """
    lines = []
    for idx, word in enumerate(path):
        if idx == 0:
            lines.append(f"  Step 0: {word.upper()} (Start)")
        else:
            prev = path[idx - 1]
            diff_idx = next(i for i in range(len(word)) if word[i] != prev[i])
            char_from = prev[diff_idx].upper()
            char_to = word[diff_idx].upper()
            lines.append(f"  Step {idx}: {word.upper()} [{char_from} -> {char_to} at pos {diff_idx + 1}]")
    return "\n".join(lines)


if __name__ == "__main__":
    words = load_word_list(DEFAULT_WORDS_FILE)
    print(f"Loaded dictionary with {len(words)} words from: {DEFAULT_WORDS_FILE}")

    test_pairs = [
        ("cold", "warm"),
        ("lead", "gold"),
        ("cat", "dog"),
        ("fool", "sage"),
    ]

    for start, target in test_pairs:
        print(f"\n==========================================")
        print(f"Solving Word Ladder: {start.upper()} -> {target.upper()}")
        print(f"==========================================")
        
        res_bfs = word_ladder_bfs(start, target, words)
        res_bi = word_ladder_bidirectional_bfs(start, target, words)

        print(f"[BFS]          Length: {res_bfs['length']} words ({res_bfs['transformations']} steps), Visited: {res_bfs['visited_count']} nodes, Time: {res_bfs['time_ms']:.2f} ms")
        print(f"[Bidirectional] Length: {res_bi['length']} words ({res_bi['transformations']} steps), Visited: {res_bi['visited_count']} nodes, Time: {res_bi['time_ms']:.2f} ms")
        print(f"\nPath Transformation:")
        print(format_ladder_diff(res_bfs["path"]))

        assert res_bfs["length"] == res_bi["length"], "BFS and Bidirectional length mismatch!"
