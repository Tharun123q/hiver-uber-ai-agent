import os
from pathlib import Path
from dataclasses import dataclass
import yaml


@dataclass
class AppConfig:
    random_seed: int = 42
    raw_data_path: Path = Path("twcs.csv")
    processed_data_path: Path = Path("data/processed_data.csv")
    golden_dataset_path: Path = Path("data/golden_dataset.csv")
    retrieval_index_path: Path = Path("artifacts/retrieval_index.pkl")
    classifier_model_path: Path = Path("artifacts/classifier_model.pkl")
    primary_eval_path: Path = Path("artifacts/primary_eval.json")
    review_payload_path: Path = Path("artifacts/review_payload.json")
    final_audit_report_path: Path = Path("artifacts/final_audit_report.json")
    metrics_path: Path = Path("artifacts/metrics.json")
    classification_report_path: Path = Path("artifacts/classification_report.json")
    confusion_matrix_path: Path = Path("artifacts/confusion_matrix.png")
    failure_analysis_path: Path = Path("reports/failure_analysis.md")
    decision_log_path: Path = Path("docs/decision_log.md")
    report_path: Path = Path("reports/report.md")

    target_conversations: int = 950
    min_customer_len: int = 20
    min_reply_len: int = 20
    brand_id: str = "Uber_Support"

    retrieval_top_k: int = 3
    retrieval_similarity_threshold: float = 0.15
    retrieval_ngram_range: tuple[int, int] = (1, 2)
    retrieval_max_features: int = 5000

    classifier_test_size: float = 0.25
    confidence_threshold: float = 0.65
    classifier_ngram_range: tuple[int, int] = (1, 2)
    classifier_max_features: int = 8000
    escalation_low_confidence_threshold: float = 0.55
    escalation_review_billing_threshold: float = 0.60

    reviewer_host: str = "127.0.0.1"
    reviewer_port: int = 8000
    reviewer_audit_url: str = "http://127.0.0.1:8000/audit"
    reviewer_health_url: str = "http://127.0.0.1:8000/health"


def load_config(config_file: str = "config/config.yaml") -> AppConfig:
    cfg = AppConfig()
    path = Path(config_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        paths = data.get("paths", {})
        path_fields = {
            "raw_data": "raw_data_path",
            "processed_data": "processed_data_path",
            "golden_dataset": "golden_dataset_path",
            "retrieval_index": "retrieval_index_path",
            "classifier_model": "classifier_model_path",
            "primary_eval": "primary_eval_path",
            "review_payload": "review_payload_path",
            "final_audit_report": "final_audit_report_path",
            "metrics": "metrics_path",
            "classification_report": "classification_report_path",
            "confusion_matrix": "confusion_matrix_path",
            "failure_analysis": "failure_analysis_path",
            "decision_log": "decision_log_path",
            "report": "report_path",
        }
        for yaml_name, field_name in path_fields.items():
            if yaml_name in paths:
                setattr(cfg, field_name, Path(paths[yaml_name]))

        proj = data.get("project", {})
        if "random_seed" in proj:
            cfg.random_seed = int(proj["random_seed"])

        prep = data.get("preprocessing", {})
        for name in ("target_conversations", "min_customer_len", "min_reply_len"):
            if name in prep:
                setattr(cfg, name, int(prep[name]))
        if "brand_id" in prep:
            cfg.brand_id = str(prep["brand_id"])

        ret = data.get("retrieval", {})
        if "top_k" in ret:
            cfg.retrieval_top_k = int(ret["top_k"])
        if "similarity_threshold" in ret:
            cfg.retrieval_similarity_threshold = float(ret["similarity_threshold"])
        if "ngram_range" in ret:
            cfg.retrieval_ngram_range = tuple(int(v) for v in ret["ngram_range"])
        if "max_features" in ret:
            cfg.retrieval_max_features = int(ret["max_features"])

        classifier = data.get("classifier", {})
        if "test_size" in classifier:
            cfg.classifier_test_size = float(classifier["test_size"])
        if "confidence_threshold" in classifier:
            cfg.confidence_threshold = float(classifier["confidence_threshold"])
        if "ngram_range" in classifier:
            cfg.classifier_ngram_range = tuple(int(v) for v in classifier["ngram_range"])
        if "max_features" in classifier:
            cfg.classifier_max_features = int(classifier["max_features"])

        escalation = data.get("escalation", {})
        if "low_confidence_threshold" in escalation:
            cfg.escalation_low_confidence_threshold = float(escalation["low_confidence_threshold"])
        if "review_billing_threshold" in escalation:
            cfg.escalation_review_billing_threshold = float(escalation["review_billing_threshold"])

        rev = data.get("reviewer", {})
        if "host" in rev:
            cfg.reviewer_host = str(rev["host"])
        if "port" in rev:
            cfg.reviewer_port = int(rev["port"])
        cfg.reviewer_audit_url = str(rev.get("audit_endpoint", f"http://{cfg.reviewer_host}:{cfg.reviewer_port}/audit"))
        cfg.reviewer_health_url = str(rev.get("health_endpoint", f"http://{cfg.reviewer_host}:{cfg.reviewer_port}/health"))

        if not cfg.raw_data_path.exists():
            candidates = [
                Path("data") / "twcs.csv",
                Path("data") / "uber.csv",
                Path("twcs.csv"),
            ]
            cfg.raw_data_path = next(
                (candidate for candidate in candidates if candidate.exists()),
                cfg.raw_data_path,
            )

    return cfg
