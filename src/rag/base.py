from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union


class BaseRetriever(ABC):
    """
    Abstract interface for conversation retrieval in the RAG pipeline.
    Allows swappable implementations: TF-IDF, Sentence Transformers, FAISS, Hybrid.
    """

    @abstractmethod
    def fit(self, conversations: List[Dict[str, Any]]) -> "BaseRetriever":
        """Index historical customer support conversations."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve Top-K most relevant historical conversations for a given customer query.
        
        Returns a list of dicts with keys:
            - conversation_id
            - customer_message
            - support_reply
            - similarity_score
            - rank
        """
        pass

    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """Persist index and metadata to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseRetriever":
        """Load index and metadata from disk."""
        pass
