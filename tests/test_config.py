from src.config import load_config


def test_load_config_reads_yaml_settings():
    config = load_config()

    assert config.target_conversations == 950
    assert config.retrieval_top_k == 3
    assert config.retrieval_similarity_threshold == 0.15
    assert config.classifier_test_size == 0.25
    assert config.escalation_low_confidence_threshold == 0.55