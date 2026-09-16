import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.logging_config import setup_logger
from src.config import load_config
from src.rag.base import BaseRetriever

logger = setup_logger("retriever")


class TFIDFRetriever(BaseRetriever):
    """
    TF-IDF + Cosine Similarity Retriever.
    Indexes historical UberSupport inquiries and retrieves Top-K semantically
    and lexically similar historical dialogues.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 5000,
        sublinear_tf: bool = True,
        similarity_threshold: float = 0.0
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            lowercase=True,
            stop_words="english"
        )
        self.corpus_vectors: Optional[np.ndarray] = None
        self.corpus_records: List[Dict[str, Any]] = []

    def fit(self, conversations: List[Dict[str, Any]]) -> "TFIDFRetriever":
        """Index historical customer conversations."""
        logger.info(f"Indexing {len(conversations)} historical conversations...")
        self.corpus_records = conversations
        
        # Build document text combining customer message and context
        doc_texts = [
            f"{c.get('customer_message', '')} {c.get('support_reply', '')}"
            for c in conversations
        ]
        self.corpus_vectors = self.vectorizer.fit_transform(doc_texts)
        logger.info(f"TF-IDF Index created with shape {self.corpus_vectors.shape}.")
        return self

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve Top-K historical conversations matching the query."""
        if self.corpus_vectors is None or not self.corpus_records:
            raise RuntimeError("Retriever index is empty. Call fit() or load() first.")

        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        query_vec = self.vectorizer.transform([cleaned_query])
        sim_scores = cosine_similarity(query_vec, self.corpus_vectors)[0]

        # Top-K indices sorted in descending order
        top_indices = [
            idx for idx in np.argsort(sim_scores)[::-1]
            if sim_scores[idx] >= self.similarity_threshold
        ][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            record = self.corpus_records[idx]
            score = float(sim_scores[idx])
            results.append({
                "conversation_id": record.get("conversation_id", f"HIST_{idx}"),
                "customer_message": record.get("customer_message", ""),
                "support_reply": record.get("support_reply", ""),
                "similarity_score": round(score, 4),
                "rank": rank
            })
        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes the retrieval index and metadata."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "corpus_vectors": self.corpus_vectors,
                "corpus_records": self.corpus_records,
                "ngram_range": self.ngram_range,
                "max_features": self.max_features,
                "sublinear_tf": self.sublinear_tf,
                "similarity_threshold": self.similarity_threshold
            }, f)
        logger.info(f"Saved retrieval index to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TFIDFRetriever":
        """Loads a persisted retrieval index from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Retrieval index not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(
            ngram_range=data["ngram_range"],
            max_features=data["max_features"],
            sublinear_tf=data["sublinear_tf"],
            similarity_threshold=data.get("similarity_threshold", 0.0)
        )
        instance.vectorizer = data["vectorizer"]
        instance.corpus_vectors = data["corpus_vectors"]
        instance.corpus_records = data["corpus_records"]
        logger.info(f"Loaded retrieval index from {path} with {len(instance.corpus_records)} documents.")
        return instance


def build_and_save_index(processed_csv: Path, output_index_path: Path) -> TFIDFRetriever:
    """Builds retrieval index from processed conversations and saves to disk."""
    logger.info(f"Loading conversations from {processed_csv}")
    df = pd.read_csv(processed_csv)
    records = df.to_dict(orient="records")
    
    cfg = load_config()
    retriever = TFIDFRetriever(
        ngram_range=cfg.retrieval_ngram_range,
        max_features=cfg.retrieval_max_features,
        similarity_threshold=cfg.retrieval_similarity_threshold
    )
    retriever.fit(records)
    retriever.save(output_index_path)
    return retriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UberSupport TF-IDF Retriever")
    parser.add_argument("--build", action="store_true", help="Build and save index")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K results")
    args = parser.parse_args()

    cfg = load_config()
    index_path = cfg.retrieval_index_path

    if args.query:
        if not index_path.exists():
            logger.info("Index not found. Building first...")
            retriever = build_and_save_index(cfg.processed_data_path, index_path)
        else:
            retriever = TFIDFRetriever.load(index_path)
        results = retriever.search(args.query, top_k=args.top_k)
        print(f"\nQuery: \"{args.query}\"")
        print(f"Top {len(results)} Retrieved Conversations:\n")
        for res in results:
            print(f"[{res['rank']}] ID: {res['conversation_id']} | Sim Score: {res['similarity_score']}")
            print(f"    Customer: {res['customer_message'][:80]}...")
            print(f"    Support : {res['support_reply'][:80]}...\n")
    else:
        retriever = build_and_save_index(cfg.processed_data_path, index_path)
        print("Retrieval index built and verified.")
