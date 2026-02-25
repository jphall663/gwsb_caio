### imports and configs #######################################################

import re

### constants #################################################################

AI_PATTERNS = [
    r"\bAI\b",
    r"artificial intelligence",
    r"machine learning",
    r"deep learning",
    r"neural network",
    r"natural language processing|\bNLP\b",
    r"large language model|\bLLM\b",
    r"generative AI|genAI",
    r"computer vision",
    r"reinforcement learning",
]

AI_REGEX = re.compile("|".join(f"(?:{p})" for p in AI_PATTERNS), re.IGNORECASE)

### utilities #################################################################

def set_ai_patterns(patterns):
    global AI_PATTERNS, AI_REGEX
    AI_PATTERNS = list(patterns)
    AI_REGEX = re.compile("|".join(f"(?:{p})" for p in AI_PATTERNS), re.IGNORECASE)

def get_ai_patterns():
    return list(AI_PATTERNS)

def get_ai_regex():
    return AI_REGEX
