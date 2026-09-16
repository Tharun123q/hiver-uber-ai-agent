import pytest
from src.rag.retriever import TFIDFRetriever


def test_retriever_search_and_similarity():
    sample_data = [
        {"conversation_id": "U1", "customer_message": "Driver never showed up.", "support_reply": "Send us a DM."},
        {"conversation_id": "U2", "customer_message": "Overcharged on my fare.", "support_reply": "We will review your fare."},
        {"conversation_id": "U3", "customer_message": "I lost my backpack in the cab.", "support_reply": "Check the app help section."}
    ]
    retriever = TFIDFRetriever()
    retriever.fit(sample_data)

    results = retriever.search("I left my backpack in the car", top_k=2)
    assert len(results) == 2
    assert results[0]["conversation_id"] == "U3"
    assert results[0]["similarity_score"] > 0.0
    assert results[0]["rank"] == 1
    assert "support_reply" in results[0]


def test_retriever_empty_query():
    retriever = TFIDFRetriever()
    retriever.fit([{"conversation_id": "U1", "customer_message": "Test message", "support_reply": "Test reply"}])
    assert retriever.search("") == []


def test_retriever_applies_similarity_threshold():
    retriever = TFIDFRetriever(similarity_threshold=0.5)
    retriever.fit([
        {"conversation_id": "U1", "customer_message": "Driver never showed up.", "support_reply": "Send us a DM."}
    ])

    assert retriever.search("unrelated payment issue", top_k=3) == []
