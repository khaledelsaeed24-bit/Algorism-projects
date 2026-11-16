"""
Generates a realistic text corpus for Top-K Frequent Words benchmark.
Employs a Zipfian power-law distribution over a natural English vocabulary
to simulate real-world text corpora (articles, books, web documents).
"""

import random
import string
import os

SAMPLE_WORDS = [
    # Top stop / functional words
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    # Topic & domain vocabulary (algorithms, computer science, data)
    "algorithm", "data", "system", "heap", "complexity", "time", "space", "search",
    "tree", "node", "graph", "network", "memory", "sort", "array", "binary",
    "function", "structure", "optimal", "process", "analysis", "frequency", "element",
    "divide", "conquer", "linear", "logarithmic", "iteration", "recursion", "dynamic",
    "programming", "performance", "execution", "model", "parameter", "index", "pointer",
    "benchmark", "empirical", "theory", "distribution", "stream", "queue", "priority",
    "storage", "computer", "science", "software", "hardware", "cache", "machine", "learning",
    "dataset", "token", "string", "hash", "table", "collision", "design", "pattern",
    "interface", "variable", "constant", "condition", "loop", "invariant", "proof",
    "theorem", "asymptotic", "notation", "bound", "worst", "average", "best", "case"
]

def generate_zipf_corpus(filename: str = "corpus.txt", total_words: int = 50000, seed: int = 42):
    random.seed(seed)
    
    # Generate Zipfian weights: weight ~ 1 / (rank ^ 1.05)
    vocab = list(SAMPLE_WORDS)
    # Add additional synthetic vocabulary words to expand unique count M
    for i in range(1, 1500):
        vocab.append(f"term_{i}")
        
    num_vocab = len(vocab)
    weights = [1.0 / ((i + 1) ** 1.02) for i in range(num_vocab)]
    total_w = sum(weights)
    cum_weights = []
    c = 0.0
    for w in weights:
        c += w
        cum_weights.append(c)
        
    # Sample words using random choices
    tokens = random.choices(vocab, weights=weights, k=total_words)
    
    # Format into realistic paragraphs with punctuation and capitalization
    sentences = []
    current_sentence = []
    
    for word in tokens:
        current_sentence.append(word)
        if len(current_sentence) >= random.randint(8, 20):
            s = " ".join(current_sentence)
            # Capitalize first letter and add period
            s = s[0].upper() + s[1:] + "."
            sentences.append(s)
            current_sentence = []
            
    if current_sentence:
        s = " ".join(current_sentence)
        s = s[0].upper() + s[1:] + "."
        sentences.append(s)
        
    # Join into paragraphs
    paragraphs = []
    p_len = random.randint(3, 7)
    curr_p = []
    for s in sentences:
        curr_p.append(s)
        if len(curr_p) >= p_len:
            paragraphs.append(" ".join(curr_p))
            curr_p = []
            p_len = random.randint(3, 7)
    if curr_p:
        paragraphs.append(" ".join(curr_p))
        
    text = "\n\n".join(paragraphs)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
        
    print(f"Generated text corpus with {total_words} words ({num_vocab} potential vocab) in '{filename}'. Size: {os.path.getsize(filename)/1024:.1f} KB")

if __name__ == "__main__":
    generate_zipf_corpus("corpus.txt", total_words=50000)
