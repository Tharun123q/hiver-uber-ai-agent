import pytest
from src.rag.responder import GroundedReplyGenerator
from src.policy.escalation import EscalationDecision


def test_responder_generates_grounded_reply():
    responder = GroundedReplyGenerator()
    retrieved = [
        {
            "conversation_id": "HIST_001",
            "similarity_score": 0.55,
            "support_reply": "We apologize for the delay. Please send us a DM with your account phone number."
        }
    ]
    esc = EscalationDecision("AUTO_HANDLE", "Standard resolution", 0.90)

    res = responder.generate(
        customer_message="My driver never arrived.",
        predicted_intent="Pickup Problem",
        retrieved_convs=retrieved,
        escalation=esc
    )

    assert res.is_grounded is True
    assert len(res.reply) > 20
    assert len(res.influenced_by) == 1
    assert res.influenced_by[0]["conversation_id"] == "HIST_001"
    assert "HIST_001" in res.grounding_rationale


def test_responder_no_citations_fallback():
    responder = GroundedReplyGenerator()
    res = responder.generate(
        customer_message="Unseen query",
        predicted_intent="General Inquiry",
        retrieved_convs=[],
        escalation=None
    )
    assert res.is_grounded is False
    assert len(res.influenced_by) == 0
    assert "direct message" in res.reply.lower()
