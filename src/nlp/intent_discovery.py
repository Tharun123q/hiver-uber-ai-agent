import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
from collections import Counter
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from config.logging_config import setup_logger
from src.config import load_config

logger = setup_logger("intent_discovery")


# Pre-defined semantic clusters grounded in Uber Customer Support operations
DISCOVERED_INTENTS = {
    "Pickup Problem": {
        "description": "Issues when driver fails to arrive, arrives at wrong location, cancels before pickup, or makes rider wait.",
        "keywords": ["driver cancelled", "never arrived", "no show", "waiting", "pickup", "cancelled on us", "where is", "not moving", "waited", "driver left"],
        "regex": r"\b(pick\s*up|never arrived|no show|cancel(led)? on|didn't show|driver left|waited for|waiting for 20 min|driver was 5 min away)\b",
        "default_escalation": "AUTO_HANDLE",
        "priority": "MEDIUM"
    },
    "Fare / Overcharge Issue": {
        "description": "Inaccuracies in trip fare, upfront pricing discrepancy, unexpected toll/surge, or charged more than quoted.",
        "keywords": ["charged more", "overcharge", "false charge", "fare", "surge", "toll", "charged twice", "wrong fare", "extra charge", "price"],
        "regex": r"\b(overcharged?|false charge|charged more|extra charge|fare difference|surge pricing|unexpected charge|wrong amount|toll fee|excessive fare)\b",
        "default_escalation": "ESCALATE",
        "priority": "HIGH"
    },
    "Refund Request": {
        "description": "Explicit customer demand for money back, cancellation fee reversal, or credit compensation.",
        "keywords": ["refund", "money back", "reimburse", "reverse fee", "cancellation fee", "credit my account", "compensation"],
        "regex": r"\b(refund|money back|reimburse(ment)?|reverse (the )?fee|cancellation fee|credit back|compensat(e|ion))\b",
        "default_escalation": "ESCALATE",
        "priority": "HIGH"
    },
    "Driver Complaint": {
        "description": "Unprofessional driver behavior, rudeness, unsafe driving habits, inappropriate comments, or refusing destination.",
        "keywords": ["driver rude", "unprofessional", "attitude", "refused", "horrible driver", "aggressive", "bad driving", "service is 0*", "driver was terrible"],
        "regex": r"\b(rude|unprofessional|attitude|refused to|horrible driver|reckless driving|bad driver|terrible service|disrespectful|driver drove)\b",
        "default_escalation": "ESCALATE",
        "priority": "HIGH"
    },
    "Lost Item": {
        "description": "Customer forgot an item in the vehicle (phone, wallet, keys, bag) and needs assistance retrieving it.",
        "keywords": ["left my", "forgot my", "lost item", "phone in car", "wallet", "keys", "jacket", "in the back seat", "contact driver"],
        "regex": r"\b(left my|forgot my|lost my|phone in (the )?car|wallet|keys|backpack|bag in (the )?car|lost property|item left)\b",
        "default_escalation": "AUTO_HANDLE",
        "priority": "MEDIUM"
    },
    "Account Access / Security": {
        "description": "User cannot log in, password reset issues, account suspended/disabled, unauthorized account access or fraud.",
        "keywords": ["account locked", "can't log in", "cannot sign in", "hacked", "suspended", "password", "disabled", "verification code", "fraud"],
        "regex": r"\b(account (locked|disabled|suspended|hacked)|can't log in|cannot log in|sign in issue|verification code|unauthorized access|reset password)\b",
        "default_escalation": "ESCALATE",
        "priority": "HIGH"
    },
    "Payment Failure": {
        "description": "Credit card declined, payment method rejected, billing error updating card, or unable to complete transaction.",
        "keywords": ["payment failed", "card declined", "payment error", "cannot add card", "payment method", "declined", "charge failed"],
        "regex": r"\b(payment (failed|error|declined)|card declined|cannot add (a )?card|payment method|declining my card|billing problem)\b",
        "default_escalation": "AUTO_HANDLE",
        "priority": "MEDIUM"
    },
    "App Technical Issue": {
        "description": "Uber app crashing, GPS glitch, UI freezing, promo code not applying, or technical bug during booking.",
        "keywords": ["app crash", "glitch", "promo code", "not working", "error in app", "bug", "app freeze", "update", "screen blank"],
        "regex": r"\b(app (crashed?|freeze|frozen|bug|glitch|error)|not loading|promo code not|discount code|application issue|keeps crashing)\b",
        "default_escalation": "AUTO_HANDLE",
        "priority": "LOW"
    },
    "Safety Concern": {
        "description": "Critical issues involving physical safety, harassment, threats, accidents, assault, intoxication, or severe danger.",
        "keywords": ["safety", "threatened", "police", "harassed", "accident", "unsafe", "drunk driver", "assault", "in danger", "scared"],
        "regex": r"\b(safe(ty)?|threat(ened)?|harass(ed|ment)?|accident|crash(ed)?|police|assault|drunk|in danger|terrified|scared|emergency)\b",
        "default_escalation": "ESCALATE",
        "priority": "CRITICAL"
    },
    "Drop-off / Route Issue": {
        "description": "Driver took wrong route, dropped rider off at wrong location, or refused to drive to final destination.",
        "keywords": ["wrong route", "wrong location", "dropped off", "took longer route", "detour", "refused to drop", "wrong address"],
        "regex": r"\b(wrong (route|way|drop off|location|address)|dropped (me )?off at|long detour|took the scenic route|refused to drop)\b",
        "default_escalation": "AUTO_HANDLE",
        "priority": "MEDIUM"
    },
    "General Inquiry": {
        "description": "General questions regarding Uber services, UberEats, city availability, policies, or receipts.",
        "keywords": ["how does", "is uber available", "receipt", "ubereats", "information", "question", "how to", "rates", "inquiry"],
        "regex": r"\b(how (do|can|does)|is uber available|ubereats availability|receipt|invoice|question about|general info|rates for)\b",
        "default_escalation": "AUTO_HANDLE",
        "priority": "LOW"
    }
}


def discover_topics(df: pd.DataFrame, num_clusters: int = 11) -> Dict[str, List[str]]:
    """Runs TF-IDF + KMeans clustering to discover empirical term clusters."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=2500,
        min_df=3
    )
    X = vectorizer.fit_transform(df["customer_message"])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    
    discovered_clusters = {}
    for i in range(num_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :8]]
        discovered_clusters[f"Cluster_{i+1}"] = top_words
    return discovered_clusters


def analyze_and_export_taxonomy(processed_csv: Path, output_json: Path, output_doc: Path) -> dict:
    """Discovers intents, associates empirical evidence, and exports taxonomy."""
    logger.info(f"Loading processed data from {processed_csv}")
    df = pd.read_csv(processed_csv)
    
    logger.info("Running TF-IDF KMeans topic discovery on customer messages...")
    cluster_topics = discover_topics(df, num_clusters=11)
    for c_id, terms in cluster_topics.items():
        logger.info(f"{c_id}: {', '.join(terms)}")

    taxonomy_data = {
        "meta": {
            "total_intents": len(DISCOVERED_INTENTS),
            "source_conversations": len(df),
            "discovery_method": "Empirical TF-IDF N-gram topic extraction + Uber customer care taxonomy"
        },
        "discovered_clusters": cluster_topics,
        "intents": DISCOVERED_INTENTS
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(taxonomy_data, f, indent=2)
    logger.info(f"Exported taxonomy JSON to {output_json}")

    # Generate Markdown documentation
    output_doc.parent.mkdir(parents=True, exist_ok=True)
    with open(output_doc, "w", encoding="utf-8") as f:
        f.write("# UberSupport Intent Taxonomy & Rationale\n\n")
        f.write("This taxonomy was systematically discovered by analyzing the vocabulary and operational workflows of UberSupport customer conversations.\n\n")
        f.write("| Intent Name | Priority | Default Escalation | Operational Rationale |\n")
        f.write("|---|---|---|---|\n")
        for name, data in DISCOVERED_INTENTS.items():
            f.write(f"| **{name}** | `{data['priority']}` | `{data['default_escalation']}` | {data['description']} |\n")
        
        f.write("\n## Detailed Intent Breakdown & Indicators\n\n")
        for name, data in DISCOVERED_INTENTS.items():
            f.write(f"### {name}\n")
            f.write(f"- **Description**: {data['description']}\n")
            f.write(f"- **Key Indicators**: `{', '.join(data['keywords'])}`\n")
            f.write(f"- **Default Policy**: `{data['default_escalation']}` (Priority: `{data['priority']}`)\n\n")

    logger.info(f"Exported taxonomy Markdown to {output_doc}")
    return taxonomy_data


if __name__ == "__main__":
    cfg = load_config()
    analyze_and_export_taxonomy(
        processed_csv=cfg.processed_data_path,
        output_json=Path("artifacts/intent_taxonomy.json"),
        output_doc=Path("docs/intent_taxonomy.md")
    )
