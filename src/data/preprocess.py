import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
import html
import unicodedata
from typing import Dict, List, Optional, Tuple
import pandas as pd

from config.logging_config import setup_logger
from src.config import AppConfig, load_config

logger = setup_logger("preprocess")


def clean_tweet_text(text: str) -> str:
    """
    Cleans raw tweet text by:
    - Unescaping HTML entities (&amp; -> &, etc.)
    - Replacing replacement characters (\ufffd -> ')
    - Removing URLs (http://t.co/...)
    - Removing Twitter handles (@username)
    - Normalizing unicode whitespace
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.replace("\ufffd", "'")
    text = unicodedata.normalize("NFKC", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove Twitter handles (@user, @Uber_Support, etc.)
    text = re.sub(r"@[A-Za-z0-9_]+\b", "", text)
    # Normalize multiple whitespace, tabs, and newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_conversation(customer_msg: str, support_reply: str, min_len: int = 20) -> bool:
    """Check if conversation contains substantive issue description and reply."""
    if len(customer_msg) < min_len or len(support_reply) < min_len:
        return False
    lowered_c = customer_msg.lower()
    if lowered_c in ["deleted", "null", "none", "[deleted]", "dm sent", "thanks", "thank you"]:
        return False
    # Must have alphabetic content
    if not re.search(r"[a-zA-Z]{3,}", customer_msg):
        return False
    if not re.search(r"[a-zA-Z]{3,}", support_reply):
        return False
    return True


def preprocess_dataset(
    raw_csv_path: Path,
    output_csv_path: Path,
    target_count: int = 950,
    min_len: int = 20,
    brand_id: str = "Uber_Support",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for Twitter Customer Support dataset.
    Extracts, cleans, pairs, deduplicates, and structures UberSupport conversations.
    """
    logger.info(f"Starting preprocessing on raw data: {raw_csv_path}")
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_csv_path}")

    # Read chunks to find Uber_Support tweets and their parent tweet IDs
    chunksize = 150000
    all_chunks = []
    
    rows_read = 0
    max_rows = 350000
    for chunk in pd.read_csv(raw_csv_path, chunksize=chunksize, low_memory=False):
        all_chunks.append(chunk)
        rows_read += len(chunk)
        logger.info(f"Read {rows_read} rows from {raw_csv_path.name}...")
        if rows_read >= max_rows:
            break

    df_raw = pd.concat(all_chunks, ignore_index=True)
    logger.info(f"Loaded {len(df_raw)} raw tweet records for filtering.")

    # Map tweet_id to tweet row for instant parent lookup
    tweet_map: Dict[int, dict] = {}
    for _, row in df_raw.iterrows():
        try:
            tid = int(row["tweet_id"])
            tweet_map[tid] = {
                "author_id": str(row.get("author_id", "")),
                "inbound": bool(row.get("inbound", False)),
                "created_at": str(row.get("created_at", "")),
                "text": str(row.get("text", "")),
            }
        except (ValueError, TypeError):
            continue

    # Filter Uber_Support replies that respond to a parent tweet
    uber_mask = (df_raw["author_id"] == brand_id) & (df_raw["in_response_to_tweet_id"].notna())
    df_uber_replies = df_raw[uber_mask].copy()
    logger.info(f"Found {len(df_uber_replies)} {brand_id} replies in the sampled dataset.")

    conversations: List[dict] = []
    seen_customer_messages = set()

    for _, u_row in df_uber_replies.iterrows():
        try:
            parent_id = int(u_row["in_response_to_tweet_id"])
            support_id = int(u_row["tweet_id"])
        except (ValueError, TypeError):
            continue

        if parent_id not in tweet_map:
            continue

        parent_tweet = tweet_map[parent_id]
        
        # Parent must be inbound customer tweet
        if not parent_tweet["inbound"]:
            continue

        raw_c_text = parent_tweet["text"]
        raw_s_text = str(u_row.get("text", ""))

        clean_c = clean_tweet_text(raw_c_text)
        clean_s = clean_tweet_text(raw_s_text)

        if not is_valid_conversation(clean_c, clean_s, min_len=min_len):
            continue

        # Deduplicate identical customer messages
        c_hash = clean_c.lower()
        if c_hash in seen_customer_messages:
            continue
        seen_customer_messages.add(c_hash)

        conv_id = f"UBER_{len(conversations) + 1:04d}"
        conversations.append({
            "conversation_id": conv_id,
            "customer_tweet_id": parent_id,
            "support_tweet_id": support_id,
            "created_at": str(parent_tweet["created_at"]),
            "customer_message": clean_c,
            "support_reply": clean_s,
            "char_length_customer": len(clean_c),
            "char_length_reply": len(clean_s)
        })

    logger.info(f"Extracted {len(conversations)} valid, deduplicated conversation pairs.")

    df_convs = pd.DataFrame(conversations)

    if len(df_convs) > target_count:
        df_convs = df_convs.sample(n=target_count, random_state=random_seed).reset_index(drop=True)
        df_convs["conversation_id"] = [f"UBER_{i+1:04d}" for i in range(len(df_convs))]
        logger.info(f"Sampled to target {target_count} structured conversations.")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_convs.to_csv(output_csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved processed dataset to {output_csv_path} ({len(df_convs)} rows)")

    root_processed = Path("processed_data.csv")
    df_convs.to_csv(root_processed, index=False, encoding="utf-8")
    logger.info(f"Saved copy to {root_processed}")

    return df_convs


if __name__ == "__main__":
    config = load_config()
    preprocess_dataset(
        raw_csv_path=config.raw_data_path,
        output_csv_path=config.processed_data_path,
        target_count=config.target_conversations,
        min_len=config.min_customer_len,
        brand_id=config.brand_id,
        random_seed=config.random_seed
    )
