"""
Top-K Frequent Words: Min-Heap of size k vs. Alternative Paradigms.
Tools: Python, collections.Counter, heapq

Implements:
  1. Min-Heap of size k (O(N + M log k)) - Core assignment algorithm
  2. Full Sort (O(N + M log M)) - Baseline comparison
  3. Max-Heap / nlargest (O(N + M + k log M)) - Python Counter.most_common
  4. Bucket Sort (O(N + M)) - Linear-time frequency distribution
"""

from collections import Counter
import heapq
import re
from typing import List, Tuple, Dict, Any


class HeapItem:
    """
    Min-heap element wrapper for (frequency, word).
    Implements comparison where:
      - Primary key: frequency (ascending, so smaller count is evicted first)
      - Secondary key: word (descending lexicographically, so alphabetically later words
        are evicted first on ties, preserving alphabetical preference for top-k).
    """
    __slots__ = ("count", "word")

    def __init__(self, count: int, word: str):
        self.count = count
        self.word = word

    def __lt__(self, other: "HeapItem") -> bool:
        if self.count != other.count:
            return self.count < other.count
        return self.word > other.word

    def __eq__(self, other: "HeapItem") -> bool:
        return self.count == other.count and self.word == other.word

    def to_tuple(self) -> Tuple[str, int]:
        return self.word, self.count


def tokenize_text(text: str) -> List[str]:
    """
    Splits text into lower-cased alphanumeric word tokens.
    Time Complexity: O(N) where N is text length in characters.
    """
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


# =====================================================================
# 1. Min-Heap of Size k (Primary Assignment Algorithm)
# =====================================================================

def top_k_min_heap(words: List[str], k: int) -> List[Tuple[str, int]]:
    """
    Finds the k most frequent words using a Min-Heap of size k.
    
    Complexity:
      - Counting frequencies: O(N) time, O(M) space where M is unique words count.
      - Heap maintenance: O(M log k) time, O(k) auxiliary heap space.
      - Final sort of k items: O(k log k) time.
      - Total Time: O(N + M log k)
      - Total Space: O(M + k)
    """
    if k <= 0:
        return []

    # Step 1: Count word frequencies with collections.Counter (O(N))
    counts = Counter(words)
    k = min(k, len(counts))

    # Step 2: Maintain a min-heap of size k (O(M log k))
    heap: List[HeapItem] = []

    for word, count in counts.items():
        item = HeapItem(count, word)
        if len(heap) < k:
            heapq.heappush(heap, item)
        else:
            # If item is greater than the smallest in the heap, replace it
            if item > heap[0]:
                heapq.heapreplace(heap, item)

    # Step 3: Extract and sort the top k items (O(k log k))
    # Higher frequency first; on ties, alphabetically smaller first
    result = [item.to_tuple() for item in heap]
    result.sort(key=lambda x: (-x[1], x[0]))
    return result


# =====================================================================
# 2. Full Sorting Approach (Baseline)
# =====================================================================

def top_k_full_sort(words: List[str], k: int) -> List[Tuple[str, int]]:
    """
    Sorts all unique words by frequency and returns the top k.
    Total Time: O(N + M log M)
    Total Space: O(M)
    """
    counts = Counter(words)
    k = min(k, len(counts))
    # Sort all M items
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return sorted_items[:k]


# =====================================================================
# 3. collections.Counter.most_common(k) (heapq.nlargest)
# =====================================================================

def top_k_counter_most_common(words: List[str], k: int) -> List[Tuple[str, int]]:
    """
    Uses Python's built-in Counter.most_common(k), which under the hood
    invokes heapq.nlargest on the unique items.
    """
    counts = Counter(words)
    return counts.most_common(k)


# =====================================================================
# 4. Bucket Sort / Linear Distribution (O(N + M))
# =====================================================================

def top_k_bucket_sort(words: List[str], k: int) -> List[Tuple[str, int]]:
    """
    Bucket sort exploiting maximum possible frequency bounded by N.
    Total Time: O(N + M)
    Total Space: O(N + M)
    """
    counts = Counter(words)
    n = len(words)
    k = min(k, len(counts))

    # Buckets from frequency 0 to n
    buckets: List[List[str]] = [[] for _ in range(n + 1)]
    for word, count in counts.items():
        buckets[count].append(word)

    result = []
    # Iterate from highest frequency bucket down to 1
    for freq in range(n, 0, -1):
        if not buckets[freq]:
            continue
        # Sort words in same bucket alphabetically
        buckets[freq].sort()
        for word in buckets[freq]:
            result.append((word, freq))
            if len(result) == k:
                return result

    return result


if __name__ == "__main__":
    sample_text = """
    Data structures and algorithms are fundamental to computer science.
    A heap is a specialized tree-based data structure.
    Min-heap keeps the smallest element at the root.
    Algorithms with min-heap achieve optimal time complexity for top-k queries.
    Algorithms, data, and structures power modern computing.
    """
    tokens = tokenize_text(sample_text)
    print(f"Total tokens: {len(tokens)}, Unique words: {len(set(tokens))}")
    
    k = 5
    top_heap = top_k_min_heap(tokens, k)
    top_sort = top_k_full_sort(tokens, k)
    top_bucket = top_k_bucket_sort(tokens, k)

    print(f"\nTop-{k} via Min-Heap (size k):")
    for rank, (w, c) in enumerate(top_heap, 1):
        print(f"  {rank}. {w:<12} (freq: {c})")

    print(f"\nTop-{k} via Full Sort:")
    for rank, (w, c) in enumerate(top_sort, 1):
        print(f"  {rank}. {w:<12} (freq: {c})")

    assert top_heap == top_sort == top_bucket
    print("\nSUCCESS: All top-k algorithms produced identical results!")
