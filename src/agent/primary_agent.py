import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import argparse
from typing import Any, Dict, List, Optional
import pandas as pd

from config.logging_config import setup_logger
from src.config import AppConfig, load_config
from src.models.classifier import IntentClassifier
from src.rag.retriever import TFIDFRetriever
from src.policy.escalation import EscalationEngine, EscalationDecision
from src.rag.responder import GroundedReplyGenerator, GroundedResponse

logger = setup_logger("primary_agent")


class PrimarySupportAgent:
    """
    Unified Primary Customer Support AI Agent.
    Coordinates Intent Classification, Context Retrieval, Grounded Generation,
    and Escalation Policy Engine to formulate trustworthy support decisions.
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        retriever: TFIDFRetriever,
        escalator: EscalationEngine,
        responder: GroundedReplyGenerator
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.escalator = escalator
        self.responder = responder

    @classmethod
    def from_config(cls, cfg: Optional[AppConfig] = None) -> "PrimarySupportAgent":
        """Factory method initializing agent components from configuration."""
        if cfg is None:
            cfg = load_config()

        logger.info("Initializing PrimarySupportAgent components...")
        if not cfg.classifier_model_path.exists():
            raise FileNotFoundError(f"Classifier model missing at {cfg.classifier_model_path}")
        if not cfg.retrieval_index_path.exists():
            raise FileNotFoundError(f"Retrieval index missing at {cfg.retrieval_index_path}")

        classifier = IntentClassifier.load(cfg.classifier_model_path)
        retriever = TFIDFRetriever.load(cfg.retrieval_index_path)
        escalator = EscalationEngine(
            low_confidence_threshold=cfg.escalation_low_confidence_threshold
        )
        responder = GroundedReplyGenerator()
        agent = cls(classifier, retriever, escalator, responder)
        agent.default_top_k = cfg.retrieval_top_k
        return agent

    def handle_message(
        self,
        customer_message: str,
        conversation_id: str = "LIVE_0001",
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Processes an incoming customer tweet through the full primary support pipeline.
        """
        # 1. Intent Classification
        predicted_intent, intent_confidence = self.classifier.predict(customer_message)

        # 2. Historical Retrieval
        retrieved_convs = self.retriever.search(
            customer_message,
            top_k=top_k if top_k is not None else getattr(self, "default_top_k", 3)
        )

        # 3. Escalation Policy
        escalation = self.escalator.evaluate(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence
        )

        # 4. Grounded Reply Generation
        response = self.responder.generate(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            retrieved_convs=retrieved_convs,
            escalation=escalation
        )

        return {
            "conversation_id": conversation_id,
            "customer_message": customer_message,
            "predicted_intent": predicted_intent,
            "intent_confidence": intent_confidence,
            "escalation": escalation.to_dict(),
            "grounded_reply": response.reply,
            "influenced_by": response.influenced_by,
            "grounding_rationale": response.grounding_rationale,
            "is_grounded": response.is_grounded
        }

    def evaluate_dataset(
        self,
        dataset_path: Path,
        output_json_path: Path,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates the primary agent over a dataset (e.g. golden dataset)
        and persists results to primary_eval.json.
        """
        logger.info(f"Evaluating Primary Agent on dataset {dataset_path}...")
        df = pd.read_csv(dataset_path)
        if limit:
            df = df.head(limit)

        eval_records = []
        for idx, row in df.iterrows():
            cid = str(row.get("conversation_id", f"EVAL_{idx:04d}"))
            msg = str(row.get("customer_message", ""))
            
            output = self.handle_message(msg, conversation_id=cid)
            # Add ground truth if available in dataset
            if "intent" in row:
                output["ground_truth_intent"] = str(row["intent"])
            if "escalation" in row:
                output["ground_truth_escalation"] = str(row["escalation"])
            if "reply_keywords" in row:
                output["expected_reply_keywords"] = str(row["reply_keywords"])
                
            eval_records.append(output)

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_records, f, indent=2)
        logger.info(f"Saved primary evaluation to {output_json_path} ({len(eval_records)} records).")

        # Also write copy to root 'primary_eval.json'
        root_path = Path("primary_eval.json")
        with open(root_path, "w", encoding="utf-8") as f:
            json.dump(eval_records, f, indent=2)
        logger.info(f"Saved copy to {root_path}")

        return eval_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UberSupport Primary Agent Runner")
    parser.add_argument("--eval_dataset", type=str, help="Path to evaluation dataset (e.g. golden_dataset.csv)")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--message", type=str, help="Single query message")
    args = parser.parse_args()

    cfg = load_config()
    agent = PrimarySupportAgent.from_config(cfg)

    if args.message:
        res = agent.handle_message(args.message)
        print("\n--- Primary Agent Single Query Output ---")
        print(json.dumps(res, indent=2))
    else:
        dataset_path = Path(args.eval_dataset) if args.eval_dataset else cfg.golden_dataset_path
        out_path = Path(args.output) if args.output else cfg.primary_eval_path
        agent.evaluate_dataset(dataset_path, out_path)
