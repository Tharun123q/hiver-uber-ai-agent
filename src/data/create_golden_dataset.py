import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
import json
import pandas as pd
from typing import List, Dict, Tuple

from config.logging_config import setup_logger
from src.config import load_config
from src.nlp.taxonomy import DISCOVERED_INTENTS

logger = setup_logger("golden_dataset")


def assign_intent_and_escalation(customer_msg: str, support_reply: str) -> Tuple[str, str, str, str]:
    """
    Expert annotation rule engine for golden dataset labeling.
    Identifies nuanced intent and escalation policies.
    """
    lowered = customer_msg.lower()
    
    # 1. Safety Concern (Critical)
    if re.search(r"\b(safety|threat|threaten|harass|police|assault|drunk|accident|crash|emergency|in danger|scared|terrified|unsafe|uncomfortable|illegal)\b", lowered):
        return (
            "Safety Concern",
            "ESCALATE",
            "safety, investigate, direct message, contact details, support team",
            "Critical safety alert involving passenger safety, harassment, or physical accident requiring immediate human investigation."
        )

    # 2. Refund Request
    if re.search(r"\b(refund|money back|reimburse|reimbursement|reverse fee|cancellation fee|credit back|give me my money|chargeback|cancel free of charge)\b", lowered):
        return (
            "Refund Request",
            "ESCALATE",
            "refund, review, trip details, fare adjustment, DM",
            "Explicit demand for monetary refund or fee reversal requiring billing team action."
        )

    # 3. Lost Item
    if re.search(r"\b(left my|forgot my|lost my|lost item|phone in (the )?car|stolen my phone|wallet|keys|backpack|jacket|purse|glasses|sunglasses|left something|lost property)\b", lowered):
        return (
            "Lost Item",
            "AUTO_HANDLE",
            "help section, lost item, driver contact, app, retrieve",
            "Standard self-service lost item retrieval workflow via Uber app Help section."
        )

    # 4. Account Access / Security
    if re.search(r"\b(account (locked|disabled|suspended|hacked)|can't log in|cannot log in|unable to log in|sign in|password|verification code|fraud|compromised|reset code|two factor|2fa|brand new account|gift card.*account)\b", lowered):
        return (
            "Account Access / Security",
            "ESCALATE",
            "account email, phone number, sign in, verify, DM",
            "Account credentials, lockouts, or security issues requiring human agent identity verification."
        )

    # 5. Drop-off / Route Issue
    if re.search(r"\b(wrong (route|way|drop off|location|address)|dropped (me )?off|detour|longer route|wrong destination|scenic route|missed turn|ended trip early|driving in circles|navigation for your drivers|driver is lost|long routes)\b", lowered):
        return (
            "Drop-off / Route Issue",
            "AUTO_HANDLE",
            "route review, trip history, fare review, DM",
            "Inefficient route or incorrect drop-off location report."
        )

    # 6. Pickup Problem
    if re.search(r"\b(driver cancelled|never arrived|no show|cancelled on|didn't show|driver left|waited for|waiting for|5 min away for|stuck in place|not moving|could not find driver|driver didn't come|sat for \d+ min|longer than eta|where is my ride|driver cancel)\b", lowered):
        return (
            "Pickup Problem",
            "AUTO_HANDLE",
            "apologize, contact us, DM, trip details, team connect",
            "Driver dispatch, no-show, or delayed arrival inquiry handled via standard assistance."
        )

    # 7. Fare / Overcharge Issue
    if re.search(r"\b(overcharge|charged (more|twice|extra|\$\d+)|false charge|fare difference|surge|toll|wrong amount|excessive fare|unexpected charge|charged me|high fare|quoted|charges from uber|random.*charge|charged on my (debit|credit|card))\b", lowered):
        return (
            "Fare / Overcharge Issue",
            "ESCALATE",
            "fare review, account details, trip fare, DM, assist",
            "Pricing discrepancy or overcharge requiring trip log verification and adjustment."
        )

    # 8. Payment Failure
    if re.search(r"\b(payment (failed|error|declined)|card declined|cannot add (a )?card|payment method|declining|credit card error|apple pay|cannot pay|cvv|bank error|declined on my|unable to pay)\b", lowered):
        return (
            "Payment Failure",
            "AUTO_HANDLE",
            "payment method, bank, update card, app settings, help",
            "Card processing failure or bank decline handled via self-service payment update steps."
        )

    # 9. App Technical Issue
    if re.search(r"\b(app (crashed?|freeze|frozen|bug|glitch|error)|promo code|code expired|discount code|not loading|application issue|ui not working|blank screen|gps not loading|update app|error messages)\b", lowered):
        return (
            "App Technical Issue",
            "AUTO_HANDLE",
            "update app, restart, clear cache, promo details, support note",
            "Technical app bug, UI crash, or promo code application error."
        )

    # 10. Driver Complaint
    if re.search(r"\b(rude|unprofessional|attitude|refused|horrible driver|reckless|bad driver|terrible service|disrespectful|0\*|rating|driving terribly|driver cancelled on purpose|smelled|yelled|terrible experience)\b", lowered):
        return (
            "Driver Complaint",
            "ESCALATE",
            "driver feedback, safety team, apologize, report, DM",
            "Interpersonal conduct or quality complaint against driver partner requiring review."
        )

    # 11. General Inquiry (Default fallback)
    return (
        "General Inquiry",
        "AUTO_HANDLE",
        "happy to help, contact us, DM, information, assistance",
        "General inquiry regarding service policies, receipt requests, or product availability."
    )


def build_golden_dataset(
    processed_csv: Path,
    output_csv: Path,
    target_size: int = 200,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Constructs a high-quality 200-sample balanced golden benchmark dataset.
    Stratifies across all discovered intents.
    """
    logger.info(f"Loading processed data from {processed_csv}")
    df = pd.read_csv(processed_csv)

    annotated_rows = []
    for idx, row in df.iterrows():
        c_msg = str(row["customer_message"])
        s_rep = str(row["support_reply"])
        intent, esc, keywords, notes = assign_intent_and_escalation(c_msg, s_rep)
        annotated_rows.append({
            "original_id": row["conversation_id"],
            "customer_message": c_msg,
            "support_reply": s_rep,
            "intent": intent,
            "escalation": esc,
            "reply_keywords": keywords,
            "ground_truth_notes": notes
        })

    df_ann = pd.DataFrame(annotated_rows)
    logger.info("Intent distribution across candidate pool:")
    logger.info(df_ann["intent"].value_counts().to_string())

    # Save full annotated dataset as well for training the ML classifier
    labeled_full_path = Path("data/labeled_processed_data.csv")
    df_ann.to_csv(labeled_full_path, index=False, encoding="utf-8")
    logger.info(f"Saved full annotated dataset to {labeled_full_path}")

    unique_intents = list(DISCOVERED_INTENTS.keys())
    per_intent_target = target_size // len(unique_intents)
    remainder = target_size % len(unique_intents)

    sampled_dfs = []
    for i, intent_name in enumerate(unique_intents):
        sub_df = df_ann[df_ann["intent"] == intent_name]
        n_to_sample = per_intent_target + (1 if i < remainder else 0)
        
        if len(sub_df) >= n_to_sample:
            sample = sub_df.sample(n=n_to_sample, random_state=random_seed)
        else:
            sample = sub_df
        sampled_dfs.append(sample)

    df_golden = pd.concat(sampled_dfs, ignore_index=True)
    
    if len(df_golden) < target_size:
        remaining_needed = target_size - len(df_golden)
        remaining_pool = df_ann[~df_ann["original_id"].isin(df_golden["original_id"])]
        fill_sample = remaining_pool.sample(n=remaining_needed, random_state=random_seed)
        df_golden = pd.concat([df_golden, fill_sample], ignore_index=True)

    df_golden = df_golden.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    df_golden["conversation_id"] = [f"GOLDEN_{i+1:03d}" for i in range(len(df_golden))]
    final_cols = ["conversation_id", "customer_message", "intent", "escalation", "reply_keywords", "ground_truth_notes"]
    df_golden = df_golden[final_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_golden.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"Successfully generated golden dataset at {output_csv} ({len(df_golden)} rows)")
    
    logger.info("Golden dataset intent distribution:")
    logger.info(df_golden["intent"].value_counts().to_string())
    logger.info("Golden dataset escalation distribution:")
    logger.info(df_golden["escalation"].value_counts().to_string())

    return df_golden


if __name__ == "__main__":
    cfg = load_config()
    build_golden_dataset(
        processed_csv=cfg.processed_data_path,
        output_csv=cfg.golden_dataset_path,
        target_size=200,
        random_seed=cfg.random_seed
    )
