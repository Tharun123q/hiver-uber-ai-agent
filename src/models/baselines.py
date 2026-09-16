import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.nlp.taxonomy import INTENT_NAMES, match_intent_keywords


class RandomBaselineClassifier:
    """
    Baseline 1: Random Classifier.
    Selects an intent uniformly at random or according to empirical prior distributions.
    Assigns pseudo-confidence.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        self.classes = INTENT_NAMES

    def fit(self, texts: List[str], labels: List[str]) -> "RandomBaselineClassifier":
        """Calculates class prior distributions from training data."""
        self.classes = sorted(list(set(labels)))
        return self

    def predict(self, text: str) -> Tuple[str, float]:
        """Returns random intent selection with baseline confidence."""
        predicted = self.rng.choice(self.classes)
        # Random confidence centered around 1 / num_classes
        base_prob = 1.0 / len(self.classes)
        conf = round(self.rng.uniform(base_prob * 0.8, base_prob * 1.5), 2)
        return (predicted, conf)


class KeywordBaselineClassifier:
    """
    Baseline 2: Keyword Matching Classifier.
    Uses regex dictionary lookup. Falls back to majority class ("General Inquiry")
    if no keywords match.
    """

    def __init__(self, default_intent: str = "General Inquiry"):
        self.default_intent = default_intent

    def fit(self, texts: List[str], labels: List[str]) -> "KeywordBaselineClassifier":
        """No training required for heuristic keyword lookup."""
        return self

    def predict(self, text: str) -> Tuple[str, float]:
        """Matches customer text against keywords."""
        matched = match_intent_keywords(text)
        if matched:
            return (matched, 0.85)
        return (self.default_intent, 0.40)
