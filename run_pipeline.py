import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config.logging_config import setup_logger
from src.config import load_config
from src.data.preprocess import preprocess_dataset
from src.nlp.intent_discovery import analyze_and_export_taxonomy
from src.data.create_golden_dataset import build_golden_dataset
from src.models.classifier import train_and_evaluate_classifier
from src.rag.retriever import build_and_save_index
from src.agent.primary_agent import PrimarySupportAgent
from reviewer.send_for_review import send_for_audit
from evaluation.evaluate import run_evaluation

logger = setup_logger("pipeline")


def run_full_pipeline():
    """
    Executes the complete UberSupport AI Customer Support System end-to-end:
    Ingestion -> Cleaning -> Discovery -> Training -> Retrieval -> Escalation ->
    Primary Evaluation -> Secondary Reviewer Audit -> Evaluation Harness.
    Guaranteed to execute deterministically in < 15 minutes.
    """
    start_time = time.time()
    logger.info("=====================================================================")
    logger.info("  STARTING COMPLETE UBERSUPPORT AI CUSTOMER SUPPORT PIPELINE RUNNER  ")
    logger.info("=====================================================================")

    cfg = load_config()

    # Step 1: Preprocess raw data
    t0 = time.time()
    logger.info("[1/7] Preprocessing raw Twitter Customer Support dataset...")
    preprocess_dataset(
        raw_csv_path=cfg.raw_data_path,
        output_csv_path=cfg.processed_data_path,
        target_count=cfg.target_conversations,
        min_len=cfg.min_customer_len,
        brand_id=cfg.brand_id,
        random_seed=cfg.random_seed
    )
    logger.info(f"Step 1 completed in {time.time() - t0:.2f}s")

    # Step 2: Intent Discovery & Taxonomy
    t0 = time.time()
    logger.info("[2/7] Running Intent Discovery and Taxonomy Generation...")
    analyze_and_export_taxonomy(
        processed_csv=cfg.processed_data_path,
        output_json=Path("artifacts/intent_taxonomy.json"),
        output_doc=Path("docs/intent_taxonomy.md")
    )
    logger.info(f"Step 2 completed in {time.time() - t0:.2f}s")

    # Step 3: Golden Benchmark Dataset
    t0 = time.time()
    logger.info("[3/7] Constructing 200-sample Balanced Golden Benchmark Dataset...")
    build_golden_dataset(
        processed_csv=cfg.processed_data_path,
        output_csv=cfg.golden_dataset_path,
        target_size=200,
        random_seed=cfg.random_seed
    )
    logger.info(f"Step 3 completed in {time.time() - t0:.2f}s")

    # Step 4: Model Training and Retrieval Indexing
    t0 = time.time()
    logger.info("[4/7] Training Intent Classifier and Building Retrieval Index...")
    train_and_evaluate_classifier(
        data_csv=Path("data/labeled_processed_data.csv"),
        golden_csv=cfg.golden_dataset_path,
        model_save_path=cfg.classifier_model_path,
        random_seed=cfg.random_seed
    )
    build_and_save_index(cfg.processed_data_path, cfg.retrieval_index_path)
    logger.info(f"Step 4 completed in {time.time() - t0:.2f}s")

    # Step 5: Primary Agent Execution
    t0 = time.time()
    logger.info("[5/7] Running Primary AI Support Agent over Golden Benchmark...")
    agent = PrimarySupportAgent.from_config(cfg)
    agent.evaluate_dataset(cfg.golden_dataset_path, cfg.primary_eval_path)
    logger.info(f"Step 5 completed in {time.time() - t0:.2f}s")

    # Step 6: Secondary Reviewer (Independent LLM Audit Microservice)
    t0 = time.time()
    logger.info("[6/7] Dispatching Primary Evaluations to Secondary Reviewer Service...")
    audit_report = send_for_audit(
        primary_eval_path=cfg.primary_eval_path,
        review_payload_path=cfg.review_payload_path,
        final_report_path=cfg.final_audit_report_path,
        audit_url=cfg.reviewer_audit_url
    )
    logger.info(f"Step 6 completed in {time.time() - t0:.2f}s")

    # Step 7: Comprehensive Evaluation Harness & Baselines
    t0 = time.time()
    logger.info("[7/7] Executing Comprehensive Evaluation Harness, Baselines & Failure Analysis...")
    metrics_summary = run_evaluation()
    logger.info(f"Step 7 completed in {time.time() - t0:.2f}s")

    total_time = time.time() - start_time
    logger.info("=====================================================================")
    logger.info(f"  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS ({total_time/60.0:.2f} MIN)  ")
    logger.info("=====================================================================")

    print("\n--- Summary of Generated Final Artifacts ---")
    print("1. processed_data.csv        : Filtered and cleaned UberSupport threads")
    print("2. golden_dataset.csv        : 200 balanced benchmark records")
    print("3. retrieval_index.pkl       : Serialized TF-IDF retrieval index")
    print("4. primary_eval.json         : Primary agent predictions, replies & escalations")
    print("5. review_payload.json       : Audit submission payload")
    print("6. final_audit_report.json   : Secondary Reviewer audit verdict & scores")
    print("7. metrics.json              : Complete evaluation metrics")
    print("8. classification_report.json: Precision/Recall/F1 breakdown by intent")
    print("9. confusion_matrix.png      : High-resolution confusion matrix plot")
    print("10. failure_analysis.md      : Top 5 failure case diagnostics")
    print("11. decision_log.md          : 12+ architectural engineering decisions")
    print("12. report.md                : Comprehensive 6-page project report")


if __name__ == "__main__":
    run_full_pipeline()
