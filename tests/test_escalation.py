import pytest
from src.policy.escalation import EscalationEngine


def test_escalation_safety_always_escalates():
    engine = EscalationEngine()
    decision = engine.evaluate(
        customer_message="Driver was drunk and crashed into a pole! Police were called.",
        predicted_intent="Safety Concern",
        intent_confidence=0.98
    )
    assert decision.decision == "ESCALATE"
    assert "safety" in decision.reason.lower()
    assert decision.confidence >= 0.95


def test_escalation_billing_escalates():
    engine = EscalationEngine()
    decision = engine.evaluate(
        customer_message="I was charged twice.",
        predicted_intent="Fare / Overcharge Issue",
        intent_confidence=0.94
    )
    assert decision.decision == "ESCALATE"
    assert "billing" in decision.reason.lower()


def test_escalation_routine_auto_handles():
    engine = EscalationEngine()
    decision = engine.evaluate(
        customer_message="My driver never arrived.",
        predicted_intent="Pickup Problem",
        intent_confidence=0.88
    )
    assert decision.decision == "AUTO_HANDLE"
    assert "standard" in decision.reason.lower()


def test_escalation_low_confidence_escalates():
    engine = EscalationEngine(low_confidence_threshold=0.60)
    decision = engine.evaluate(
        customer_message="Hmm something weird happened today with that thing",
        predicted_intent="General Inquiry",
        intent_confidence=0.45
    )
    assert decision.decision == "ESCALATE"
    assert "low classification confidence" in decision.reason.lower()
