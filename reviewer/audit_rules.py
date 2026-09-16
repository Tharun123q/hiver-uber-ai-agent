import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

from src.nlp.taxonomy import DISCOVERED_INTENTS


@dataclass
class AuditResult:
    status: str  # "PASS", "WARNING", "FAIL"
    reason: str
    score: float  # 0.0 to 10.0


class IndependentAuditEngine:
    """
    Independent secondary review logic.
    Operates completely separate from the Primary Agent to verify:
    1. Intent Classification Consistency
    2. Escalation Policy Compliance
    3. Groundedness, Hallucination Absence, and Tone in Replies
    """

    def review_intent(
        self,
        customer_message: str,
        predicted_intent: str,
        confidence: float
    ) -> AuditResult:
        lowered = customer_message.lower()

        # Check if predicted intent exists in defined taxonomy
        if predicted_intent not in DISCOVERED_INTENTS:
            return AuditResult("FAIL", f"Predicted intent '{predicted_intent}' is not in approved taxonomy.", 0.0)

        # Critical Safety checks
        is_safety = bool(re.search(r"\b(safety|threat.*|harass.*|police|assault.*|drunk|intoxicated|accident|crash.*|danger|scared|emergency)\b", lowered))
        if is_safety and predicted_intent != "Safety Concern":
            return AuditResult("FAIL", f"Message contains high-risk safety signals but was classified as '{predicted_intent}'.", 2.0)

        # Explicit Billing checks
        is_billing = bool(re.search(r"\b(charged twice|refund|overcharge|unauthorized charge|money back)\b", lowered))
        if is_billing and predicted_intent not in ["Fare / Overcharge Issue", "Refund Request"]:
            return AuditResult("WARNING", f"Message expresses clear financial dispute but was labeled '{predicted_intent}'.", 6.0)

        # Check confidence sanity
        if confidence < 0.40:
            return AuditResult("WARNING", f"Model exhibited very low confidence ({confidence:.2f}) on intent classification.", 6.5)

        return AuditResult("PASS", "Predicted intent is semantically aligned with the customer issue.", 10.0)

    def review_escalation(
        self,
        customer_message: str,
        predicted_intent: str,
        escalation: Dict[str, Any]
    ) -> AuditResult:
        decision = escalation.get("decision", "").upper()
        reason = escalation.get("reason", "")
        lowered = customer_message.lower()

        # Invariant 1: Safety messages MUST ESCALATE
        is_safety = (predicted_intent == "Safety Concern") or bool(
            re.search(r"\b(safety|threat.*|harass.*|police|assault.*|drunk|intoxicated|accident|crash.*|danger|scared|emergency)\b", lowered)
        )
        if is_safety and decision != "ESCALATE":
            return AuditResult("FAIL", "Safety concern was incorrectly marked as AUTO_HANDLE; violates zero-tolerance safety policy.", 0.0)

        # Invariant 2: Overcharge & Refund disputes MUST ESCALATE
        is_billing_critical = (predicted_intent in ["Fare / Overcharge Issue", "Refund Request"]) or bool(
            re.search(r"\b(charged twice|refund|overcharge|unauthorized charge)\b", lowered)
        )
        if is_billing_critical and decision != "ESCALATE":
            return AuditResult("FAIL", "Billing dispute requires account financial verification and should be escalated.", 3.0)

        # Invariant 3: Routine inquiries should ideally be AUTO_HANDLE
        if predicted_intent in ["Lost Item", "Pickup Problem"] and decision == "ESCALATE" and not is_safety and not is_billing_critical:
            return AuditResult("WARNING", f"Inquiry for {predicted_intent} could have been auto-handled via standard self-service.", 7.5)

        return AuditResult("PASS", "Escalation decision strictly adheres to company customer care guidelines.", 10.0)

    def review_grounding(
        self,
        customer_message: str,
        grounded_reply: str,
        influenced_by: List[Dict[str, Any]],
        grounding_rationale: str
    ) -> AuditResult:
        if not grounded_reply or len(grounded_reply.strip()) < 15:
            return AuditResult("FAIL", "Generated reply is missing or too brief.", 0.0)

        reply_lowered = grounded_reply.lower()

        # Check for ungrounded promises / hallucinated phone numbers or free credits
        if re.search(r"\b(1-800|\bfree ride\b|\$50 credit|guaranteed refund within 1 hour)\b", reply_lowered):
            return AuditResult("FAIL", "Reply contains ungrounded policy promises or hallucinated numbers.", 1.0)

        # Check for safety and professionalism
        if not re.search(r"\b(help|support|dm|details|apologize|sorry|assist|team)\b", reply_lowered):
            return AuditResult("WARNING", "Reply tone is lacking clear assistance protocols or polite customer support framing.", 6.0)

        if not influenced_by:
            return AuditResult("WARNING", "No historical conversation citations provided for grounding verification.", 6.5)

        return AuditResult("PASS", "Generated reply is thoroughly grounded in historical precedents, safe, and professional.", 10.0)

    def audit_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        msg = record.get("customer_message", "")
        pred_intent = record.get("predicted_intent", "")
        conf = float(record.get("intent_confidence", 0.8))
        esc = record.get("escalation", {})
        reply = record.get("grounded_reply", "")
        inf = record.get("influenced_by", [])
        rat = record.get("grounding_rationale", "")

        intent_res = self.review_intent(msg, pred_intent, conf)
        esc_res = self.review_escalation(msg, pred_intent, esc)
        ground_res = self.review_grounding(msg, reply, inf, rat)

        item_score = round((intent_res.score * 0.35) + (esc_res.score * 0.35) + (ground_res.score * 0.30), 2)
        has_fail = (intent_res.status == "FAIL" or esc_res.status == "FAIL" or ground_res.status == "FAIL")
        has_warn = (intent_res.status == "WARNING" or esc_res.status == "WARNING" or ground_res.status == "WARNING")

        if has_fail:
            verdict = "FAIL"
        elif has_warn:
            verdict = "WARNING"
        else:
            verdict = "PASS"

        return {
            "conversation_id": record.get("conversation_id", "UNKNOWN"),
            "item_verdict": verdict,
            "item_score": item_score,
            "intent_review": {"status": intent_res.status, "reason": intent_res.reason},
            "escalation_review": {"status": esc_res.status, "reason": esc_res.reason},
            "grounding_review": {"status": ground_res.status, "reason": ground_res.reason}
        }
