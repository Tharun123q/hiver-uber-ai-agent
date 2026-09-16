import pytest
from src.data.preprocess import clean_tweet_text, is_valid_conversation


def test_clean_tweet_text_removes_urls_and_handles():
    raw = "@Uber_Support @115872 I was overcharged for my ride! http://t.co/xyz123 &amp; driver was rude"
    cleaned = clean_tweet_text(raw)
    assert "@Uber_Support" not in cleaned
    assert "@115872" not in cleaned
    assert "http://t.co/xyz123" not in cleaned
    assert "&" in cleaned  # Unescaped &amp;
    assert "I was overcharged for my ride!" in cleaned


def test_clean_tweet_text_normalizes_whitespace():
    raw = "My   driver \n\n  never   arrived    "
    cleaned = clean_tweet_text(raw)
    assert cleaned == "My driver never arrived"


def test_is_valid_conversation_rejects_empty_or_deleted():
    assert not is_valid_conversation("", "")
    assert not is_valid_conversation("deleted", "We are sorry.")
    assert not is_valid_conversation("hi", "hello")
    assert not is_valid_conversation("12345678901234567890", "12345678901234567890")  # No alphabet


def test_is_valid_conversation_accepts_valid_pairs():
    cust = "My driver took a long detour and charged me double the fare."
    supp = "We are sorry to hear this. Please DM us your account email so we can help."
    assert is_valid_conversation(cust, supp, min_len=20)
