"""
Insertion Sort and Merge Sort Implementations.
Provides both standard (optimized) and instrumented (tracking comparisons and shifts) variants.
"""

from typing import List, Tuple

def insertion_sort(arr: List[int]) -> List[int]:
    """
    Standard insertion sort implementation.
    Sorts array in non-decreasing order in-place on a shallow copy.
    """
    a = list(arr)
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a

def insertion_sort_instrumented(arr: List[int]) -> Tuple[List[int], int, int]:
    """
    Instrumented insertion sort that counts key comparisons and element shifts.

    Definitions:
      - comparisons: Number of times an element in the sorted prefix is compared against 'key'.
      - shifts: Number of times an element is moved to the right (a[j+1] = a[j]) to make room for 'key'.

    Returns:
        (sorted_list, comparisons, shifts)
    """
    a = list(arr)
    n = len(a)
    comparisons = 0
    shifts = 0

    for i in range(1, n):
        key = a[i]
        j = i - 1
        while j >= 0:
            comparisons += 1
            if a[j] > key:
                a[j + 1] = a[j]
                shifts += 1
                j -= 1
            else:
                break
        a[j + 1] = key

    return a, comparisons, shifts

def merge_sort(arr: List[int]) -> List[int]:
    """
    Standard divide-and-conquer Merge Sort implementation for benchmark comparison.
    Time complexity: O(n log n) in all cases.
    Space complexity: O(n) auxiliary.
    """
    if len(arr) <= 1:
        return list(arr)
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    merged: List[int] = []
    i = j = 0
    len_left = len(left)
    len_right = len(right)
    
    while i < len_left and j < len_right:
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            
    if i < len_left:
        merged.extend(left[i:])
    if j < len_right:
        merged.extend(right[j:])
        
    return merged

if __name__ == "__main__":
    sample = [5, 2, 9, 1, 5, 6]
    sorted_sample, comps, shifts = insertion_sort_instrumented(sample)
    print("Original:", sample)
    print("Sorted:  ", sorted_sample)
    print(f"Comparisons: {comps}, Shifts: {shifts}")
    print("Merge Sort:", merge_sort(sample))
