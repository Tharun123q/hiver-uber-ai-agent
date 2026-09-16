from typing import Dict, List, Optional
import re

from src.nlp.intent_discovery import DISCOVERED_INTENTS

INTENT_NAMES: List[str] = list(DISCOVERED_INTENTS.keys())

INTENT_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(INTENT_NAMES)}
ID_TO_INTENT: Dict[int, str] = {idx: name for idx, name in enumerate(INTENT_NAMES)}


def match_intent_keywords(text: str) -> Optional[str]:
    """Matches text against keyword regex patterns of discovered intents."""
    lowered = text.lower()
    # Check Safety Concern first because it has highest priority
    if re.search(DISCOVERED_INTENTS["Safety Concern"]["regex"], lowered, re.IGNORECASE):
        return "Safety Concern"
        
    for intent, data in DISCOVERED_INTENTS.items():
        if re.search(data["regex"], lowered, re.IGNORECASE):
            return intent
    return None
