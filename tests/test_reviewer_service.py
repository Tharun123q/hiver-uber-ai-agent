import pytest
from fastapi.testclient import TestClient
from reviewer.reviewer_service import app
from reviewer.audit_rules import IndependentAuditEngine

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_audit_engine_passes_consistent_record():
    engine = IndependentAuditEngine()
    record = {
        "conversation_id": "TEST_001",
        "customer_message": "My driver never arrived.",
        "predicted_intent": "Pickup Problem",
        "intent_confidence": 0.88,
        "escalation": {"decision": "AUTO_HANDLE", "reason": "Standard resolution", "confidence": 0.92},
        "grounded_reply": "We apologize for the inconvenience! Please send us a DM with your trip details.",
        "influenced_by": [{"conversation_id": "UBER_001", "similarity_score": 0.4}],
        "grounding_rationale": "Historical resolution match."
    }
    res = engine.audit_record(record)
    assert res["item_verdict"] == "PASS"
    assert res["item_score"] == 10.0


def test_audit_engine_flags_safety_mishandle():
    engine = IndependentAuditEngine()
    bad_record = {
        "conversation_id": "TEST_002",
        "customer_message": "The driver was intoxicated, threatened me, and crashed!",
        "predicted_intent": "Pickup Problem",  # Inappropriate intent
        "intent_confidence": 0.50,
        "escalation": {"decision": "AUTO_HANDLE", "reason": "Standard auto handle", "confidence": 0.80},  # Fails safety invariant!
        "grounded_reply": "Thanks for reaching out.",
        "influenced_by": [],
        "grounding_rationale": ""
    }
    res = engine.audit_record(bad_record)
    assert res["item_verdict"] == "FAIL"
    assert res["escalation_review"]["status"] == "FAIL"


def test_audit_endpoint_batch():
    sample_payload = {
        "records": [
            {
                "conversation_id": "TEST_001",
                "customer_message": "I was charged twice.",
                "predicted_intent": "Fare / Overcharge Issue",
                "intent_confidence": 0.94,
                "escalation": {"decision": "ESCALATE", "reason": "Billing dispute", "confidence": 0.94},
                "grounded_reply": "We are reviewing your receipt. Send us a DM.",
                "influenced_by": [{"conversation_id": "UBER_005", "similarity_score": 0.5}],
                "grounding_rationale": "Billing workflow precedent."
            }
        ]
    }
    response = client.post("/audit", json=sample_payload)
    assert response.status_code == 200
    report = response.json()
    assert report["audit_status"] in ["APPROVED", "FLAGGED"]
    assert "overall_score" in report
    assert "discrepancies" in report
