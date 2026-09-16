import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import time
import subprocess
import requests
from typing import Any, Dict, List, Optional

from config.logging_config import setup_logger
from src.config import load_config
from reviewer.audit_rules import IndependentAuditEngine

logger = setup_logger("send_for_review")


def ensure_reviewer_service_running(host: str = "127.0.0.1", port: int = 8000, timeout: int = 15) -> Optional[subprocess.Popen]:
    """
    Checks if the reviewer FastAPI service is live on port 8000.
    If not, automatically launches reviewer_service.py in a background subprocess.
    """
    health_url = f"http://{host}:{port}/health"
    try:
        resp = requests.get(health_url, timeout=1.5)
        if resp.status_code == 200:
            logger.info(f"Reviewer service is already active at {health_url}")
            return None
    except Exception:
        pass

    logger.info(f"Reviewer service not detected. Launching reviewer_service.py on port {port}...")
    python_exe = sys.executable
    service_script = Path(__file__).parent / "reviewer_service.py"
    
    proc = subprocess.Popen(
        [python_exe, str(service_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for service to become responsive
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(health_url, timeout=1.0)
            if resp.status_code == 200:
                logger.info("Reviewer service launched successfully and responding to health checks.")
                return proc
        except Exception:
            time.sleep(0.5)

    logger.warning("Reviewer service did not respond via HTTP in time; fallback mode available.")
    return proc


def send_for_audit(
    primary_eval_path: Path,
    review_payload_path: Path,
    final_report_path: Path,
    audit_url: str = "http://127.0.0.1:8000/audit"
) -> Dict[str, Any]:
    """
    STEP 2 & 3:
    1. Parses primary_eval.json
    2. Constructs review_payload.json
    3. POSTs payload to Secondary Reviewer service
    4. Generates final_audit_report.json
    """
    logger.info(f"Loading primary evaluation records from {primary_eval_path}...")
    if not primary_eval_path.exists():
        raise FileNotFoundError(f"Primary evaluation file not found: {primary_eval_path}")

    with open(primary_eval_path, "r", encoding="utf-8") as f:
        primary_records = json.load(f)

    logger.info(f"Loaded {len(primary_records)} records from primary evaluation.")

    # Create review payload structure
    review_payload = {
        "metadata": {
            "source": "Primary Agent",
            "timestamp": time.time(),
            "records_count": len(primary_records)
        },
        "records": primary_records
    }

    # Save review_payload.json
    review_payload_path.parent.mkdir(parents=True, exist_ok=True)
    with open(review_payload_path, "w", encoding="utf-8") as f:
        json.dump(review_payload, f, indent=2)
    logger.info(f"Saved review payload to {review_payload_path}")

    # Also save copy to root 'review_payload.json'
    root_payload = Path("review_payload.json")
    with open(root_payload, "w", encoding="utf-8") as f:
        json.dump(review_payload, f, indent=2)
    logger.info(f"Saved copy to {root_payload}")

    # Ensure reviewer service is running
    cfg = load_config()
    proc = ensure_reviewer_service_running(host=cfg.reviewer_host, port=cfg.reviewer_port)

    # POST payload to HTTP audit endpoint with fallback
    report_data = None
    try:
        logger.info(f"Submitting payload to Secondary Reviewer endpoint: {audit_url}...")
        resp = requests.post(audit_url, json=review_payload, timeout=30)
        if resp.status_code == 200:
            report_data = resp.json()
            logger.info("Secondary Reviewer audit completed successfully via HTTP API.")
        else:
            logger.error(f"Reviewer returned HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.warning(f"HTTP request to reviewer failed: {e}. Executing in-process audit...")

    if report_data is None:
        # Programmatic in-process execution fallback if HTTP network socket is restricted
        logger.info("Running independent audit engine in-process...")
        from reviewer.reviewer_service import audit_batch, AuditPayload
        report_data = audit_batch(AuditPayload(records=primary_records))

    # Save final_audit_report.json
    final_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Saved final audit report to {final_report_path}")

    # Also save copy to root 'final_audit_report.json'
    root_report = Path("final_audit_report.json")
    with open(root_report, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Saved copy to {root_report}")

    return report_data


if __name__ == "__main__":
    cfg = load_config()
    report = send_for_audit(
        primary_eval_path=cfg.primary_eval_path,
        review_payload_path=cfg.review_payload_path,
        final_report_path=cfg.final_audit_report_path,
        audit_url=cfg.reviewer_audit_url
    )

    print("\n========================================================")
    print("           SECONDARY REVIEWER FINAL AUDIT REPORT        ")
    print("========================================================")
    print(json.dumps(report, indent=2))
