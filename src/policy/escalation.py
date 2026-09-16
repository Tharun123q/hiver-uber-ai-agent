import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import re
import argparse
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from config.logging_config import setup_logger
from src.config import load_config

logger = setup_logger("escalation")


@dataclass
class EscalationDecision:
    decision: str  # "AUTO_HANDLE" or "ESCALATE"
    reason: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "confidence": round(self.confidence, 2)
        }


class EscalationEngine:
    """
    Production Escalation Policy Engine for UberSupport AI Agent.
    Evaluates customer message, predicted intent, and model confidence to decide
    whether to automatically handle the interaction or route to a human support agent.
    """

    def __init__(
        self,
        low_confidence_threshold: float = 0.55,
        billing_threshold: float = 0.60
    ):
        self.low_confidence_threshold = low_confidence_threshold
        self.billing_threshold = billing_threshold

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float
    ) -> EscalationDecision:
        """
        Determines escalation decision, detailed reason, and decision confidence.
        """
        lowered = customer_message.lower()

        # 1. Critical Safety Concerns (Absolute highest priority)
        if (
            predicted_intent == "Safety Concern" or
            re.search(r"\b(safety|threat|harass|police|assault|drunk|accident|crash|emergency|in danger|scared|terrified|unsafe)\b", lowered)
        ):
            return EscalationDecision(
                decision="ESCALATE",
                reason="Critical safety incident requiring emergency safety team review and immediate human intervention.",
                confidence=0.98
            )

        # 2. Billing / Fare / Overcharge Issues
        if (
            predicted_intent in ["Fare / Overcharge Issue", "Refund Request"] or
            re.search(r"\b(charged twice|false charge|overcharg|double charge|unauthorized charge|refund|money back|reimburse|stole my money)\b", lowered)
        ):
            return EscalationDecision(
                decision="ESCALATE",
                reason="Possible billing issue requiring account verification.",
                confidence=0.94
            )

        # 3. Account Access / Security Issues
        if (
            predicted_intent == "Account Access / Security" or
            re.search(r"\b(hacked|locked out|can't log in|cannot sign in|account suspended|fraud|stolen account)\b", lowered)
        ):
            return EscalationDecision(
                decision="ESCALATE",
                reason="Account credentials or security compromise requiring human identity verification.",
                confidence=0.92
            )

        # 4. Severe Driver Misconduct
        if (
            predicted_intent == "Driver Complaint" or
            re.search(r"\b(aggressive|abusive|yelling|threatened|refused destination|reckless driver)\b", lowered)
        ):
            return EscalationDecision(
                decision="ESCALATE",
                reason="Severe driver partner conduct complaint requiring internal review.",
                confidence=0.89
            )

        # 5. Low-Confidence Classification / Ambiguous Inquiries
        if intent_confidence < self.low_confidence_threshold:
            return EscalationDecision(
                decision="ESCALATE",
                reason=f"Ambiguous customer inquiry with low classification confidence ({intent_confidence:.2f}) requiring human clarification.",
                confidence=0.75
            )

        # 6. Standard Operational / Self-Service Inquiries (Pickup, Lost Item, App, General)
        if predicted_intent == "Pickup Problem":
            return EscalationDecision(
                decision="AUTO_HANDLE",
                reason="Historical conversations show standard resolution.",
                confidence=0.92
            )

        if predicted_intent == "Lost Item":
            return EscalationDecision(
                decision="AUTO_HANDLE",
                reason="Standard self-service lost item protocol available via app Help section.",
                confidence=0.93
            )

        if predicted_intent in ["Payment Failure", "App Technical Issue", "Drop-off / Route Issue", "General Inquiry"]:
            return EscalationDecision(
                decision="AUTO_HANDLE",
                reason="Inquiry follows standard automated assistance workflows and self-service troubleshooting.",
                confidence=0.88
            )

        # Default fallback
        return EscalationDecision(
            decision="AUTO_HANDLE",
            reason="Standard inquiry resolvable via historical resolution protocols.",
            confidence=0.80
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UberSupport Escalation Engine")
    parser.add_argument("--message", type=str, required=True, help="Customer message text")
    parser.add_argument("--intent", type=str, default="General Inquiry", help="Predicted intent")
    parser.add_argument("--confidence", type=float, default=0.85, help="Intent confidence")
    args = parser.parse_args()

    engine = EscalationEngine()
    result = engine.evaluate(args.message, args.intent, args.confidence)

    print(f"\nCustomer:\n\"{args.message}\"")
    print(f"Decision:\n{result.decision}")
    print(f"Reason:\n{result.reason}")
    print(f"Confidence:\n{result.confidence:.2f}")
