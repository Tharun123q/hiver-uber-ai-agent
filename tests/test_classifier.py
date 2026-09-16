import pytest
from pathlib import Path
from src.models.classifier import IntentClassifier
from src.config import load_config


def test_classifier_fit_and_predict():
    texts = [
        "My driver never showed up and left me waiting.",
        "Why was I charged twice for my trip?",
        "Driver was extremely rude and yelled at me.",
        "I left my wallet in the back seat.",
        "The driver was swerving dangerously, emergency!"
    ]
    labels = [
        "Pickup Problem",
        "Fare / Overcharge Issue",
        "Driver Complaint",
        "Lost Item",
        "Safety Concern"
    ]
    clf = IntentClassifier(random_seed=42)
    clf.fit(texts, labels)

    intent, conf = clf.predict("Driver never arrived.")
    assert isinstance(intent, str)
    assert 0.0 <= conf <= 1.0
    assert intent == "Pickup Problem"


def test_classifier_save_and_load(tmp_path):
    texts = ["Left my phone in car", "Charged extra money"]
    labels = ["Lost Item", "Fare / Overcharge Issue"]
    clf = IntentClassifier()
    clf.fit(texts, labels)

    save_file = tmp_path / "test_model.pkl"
    clf.save(save_file)
    assert save_file.exists()

    loaded = IntentClassifier.load(save_file)
    intent, conf = loaded.predict("I forgot my phone")
    assert intent in labels
    assert 0.0 <= conf <= 1.0
